"""Resumable fixed-CFG consistency training. Run from the assembled FastGen root."""
from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import random
import shutil
import signal
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from teacher.config import Config
from teacher.diffusion import Diffusion
from teacher.utils import load_checkpoint, make_model, save_checkpoint, save_image_grid, set_seed, write_json
from .config import Settings, load_settings
from .core import loss, sample, sampling_times


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "consistency").glob("*.py"))
    files += sorted((root / "teacher").glob("*.py"))
    files += [root / name for name in ("dataset.py", "src/utils.py") if (root / name).exists()]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(file_digest(path)))
    return digest.hexdigest()


@torch.no_grad()
def update_average(target: torch.nn.Module, student: torch.nn.Module, decay: float) -> None:
    for a, b in zip(target.parameters(), student.parameters(), strict=True):
        a.lerp_(b.detach(), 1 - decay)
    for a, b in zip(target.buffers(), student.buffers(), strict=True):
        a.copy_(b)


def epoch_loader(dataset, settings: Settings, epoch: int) -> DataLoader:
    # Separate generators keep DataLoader construction from consuming model RNG
    # during resume. Augmentation is prohibited for this first controlled run.
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(settings.seed + epoch))
    return DataLoader(Subset(dataset, order.tolist()), batch_size=settings.batch_size,
                      num_workers=settings.num_workers, drop_last=False,
                      pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(settings.seed + epoch + 1000000))


class StopRequest:
    def __init__(self):
        self.requested = False

    def __call__(self, signum, frame):
        self.requested = True
        print(f"Signal {signum}: saving after the current optimizer update.", flush=True)


def run_training(settings: Settings, teacher_state: dict, dataset, output: str | Path,
                 teacher_sha256: str, device: torch.device,
                 max_updates_this_run: int | None = None,
                 stop: StopRequest | None = None) -> dict:
    """Core runner also accepts a synthetic dataset for deterministic resume tests.

    Real-data validation and teacher-file provenance are enforced by main().
    This routine never changes protected evaluator/data files or submission weights.
    """
    cfg = Config(**teacher_state["config"])
    settings.validate(cfg.diffusion_steps)
    if max_updates_this_run is not None and max_updates_this_run < 1:
        raise ValueError("max_updates_this_run must be positive")
    if cfg.prediction_type != "epsilon" or teacher_state.get("prediction_type", "epsilon") != "epsilon":
        raise ValueError("Use the original epsilon diffusion teacher, not a progressive student")
    if cfg.noise_schedule != "linear" or cfg.dropout != 0:
        raise ValueError("This controlled baseline requires the linear, dropout-free teacher")
    if any((cfg.horizontal_flip_probability, cfg.augmentation_degrees,
            cfg.augmentation_translate, cfg.augmentation_scale_delta)):
        raise ValueError("Use the selected no-augmentation teacher for this baseline")
    if dataset.classes != teacher_state["class_names"]:
        raise ValueError("Teacher/data class mapping differs")
    data_digest = hashlib.sha256("\n".join(dataset.paths).encode()).hexdigest()
    if data_digest != teacher_state["train_paths_sha256"]:
        raise ValueError("Teacher/student training manifests differ")
    if len(dataset) == 0:
        raise ValueError("Empty training dataset")
    if device.type == "cuda" and settings.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; use an explicit amp_dtype=none config")

    run = Path(output)
    run.mkdir(parents=True, exist_ok=True)
    # flock is released by the OS even after SIGKILL, unlike a persistent PID file.
    with (run / ".train.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another trainer owns {run}") from error
        return _train_locked(settings, teacher_state, cfg, dataset, run, teacher_sha256,
                             data_digest, device, max_updates_this_run, stop or StopRequest())


def _train_locked(settings, teacher_state, cfg, dataset, run, teacher_sha256,
                  data_digest, device, max_updates_this_run, stop):
    set_seed(settings.seed)
    teacher = make_model(cfg, len(dataset.classes)).to(device).eval().requires_grad_(False)
    teacher.load_state_dict(teacher_state["ema"])
    student = copy.deepcopy(teacher).train().requires_grad_(True)
    # Do not reinterpret an epsilon head as a consistency-residual head.
    torch.nn.init.zeros_(student.out_conv.weight)
    torch.nn.init.zeros_(student.out_conv.bias)
    target = copy.deepcopy(student).eval().requires_grad_(False)
    ema = copy.deepcopy(student).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=settings.learning_rate, weight_decay=0)
    parameters = sum(p.numel() for p in student.parameters())
    if parameters >= 60000000:
        raise ValueError("Student exceeds the <60M parameter budget")
    alpha = Diffusion(cfg.diffusion_steps, cfg.beta_start, cfg.beta_end, device).alpha_bar
    grid = torch.linspace(0, cfg.diffusion_steps - 1, settings.num_scales, device=device).round().long()
    provenance = {"method": settings.method, "settings": settings.to_dict(),
                  "config": teacher_state["config"], "class_names": dataset.classes,
                  "teacher_sha256": teacher_sha256, "train_paths_sha256": data_digest,
                  "implementation_sha256": implementation_digest()}
    step = epoch = cursor = images_seen = 0
    elapsed_before = 0.0
    previous_peak = 0
    last_loss = None
    latest = run / "latest.pt"
    if latest.exists():
        saved = load_checkpoint(latest)
        for key, value in provenance.items():
            if saved.get(key) != value:
                raise ValueError(f"Resume rejected: {key} changed; use a new experiment directory")
        student.load_state_dict(saved["model"])
        target.load_state_dict(saved["target"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        step, epoch, cursor = saved["step"], saved["epoch"], saved["batch_in_epoch"]
        if step > settings.num_train_steps:
            raise ValueError("Checkpoint exceeds configured budget")
        images_seen, elapsed_before = saved["images_seen"], saved["elapsed_seconds"]
        previous_peak, last_loss = saved["peak_vram_bytes"] or 0, saved["last_loss"]
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        if saved["device_type"] != device.type:
            raise ValueError("Resume on the same device type to preserve RNG semantics")
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        del saved
    else:
        if any((run / name).exists() for name in ("config.json", "train.log", "final.pt")):
            raise ValueError("Run artifacts exist without latest.pt; use a fresh output directory")
        root = Path(__file__).resolve().parents[1]
        snapshot = run / "source_snapshot"
        for name in ("consistency", "teacher"):
            shutil.copytree(root / name, snapshot / name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__"))
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
        write_json(run / "config.json", {**provenance,
                                         "git_commit": git.stdout.strip() if git.returncode == 0 else None,
                                         "parameter_count": parameters})
    start_step = step
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.monotonic()

    def amp():
        if device.type == "cuda" and settings.amp_dtype == "bf16":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def peak():
        return max(previous_peak, torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None

    def state():
        return {**provenance, "model": student.state_dict(), "target": target.state_dict(),
                "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "epoch": epoch, "batch_in_epoch": cursor,
                "images_seen": images_seen, "last_loss": last_loss, "device_type": device.type,
                "elapsed_seconds": elapsed_before + time.monotonic() - start_time,
                "peak_vram_bytes": peak(), "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}

    # Even interruption before the first update has a restartable initial state.
    if not latest.exists():
        save_checkpoint(latest, state())
    while step < settings.num_train_steps and not stop.requested:
        loader = epoch_loader(dataset, settings, epoch)
        for batch_index, (images, labels) in enumerate(loader):
            if batch_index < cursor:
                continue
            if stop.requested:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            for offset in range(0, len(labels), settings.microbatch_size):
                x, c = images[offset:offset + settings.microbatch_size], labels[offset:offset + settings.microbatch_size]
                with amp():
                    value = loss(student, target, teacher, x, c, alpha, grid,
                                 settings.guidance_scale, settings.sigma_data, settings.clip_teacher)
                    weighted = value * (len(c) / len(labels))
                if not torch.isfinite(weighted):
                    raise FloatingPointError(f"Non-finite CD loss at update {step + 1}")
                weighted.backward()
                accumulated += float(weighted.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), settings.grad_clip,
                                                       error_if_nonfinite=True)
            optimizer.step()
            update_average(target, student, settings.target_ema_decay)
            update_average(ema, student, settings.sampling_ema_decay)
            step += 1
            cursor = batch_index + 1
            images_seen += len(labels)
            last_loss = accumulated
            used = peak()
            if used is not None and used >= 20000000000:
                save_checkpoint(latest, state())
                raise RuntimeError("Peak allocated VRAM exceeds the 20GB training limit")
            record = {"step": step, "loss": last_loss, "grad_norm": float(grad_norm),
                      "learning_rate": settings.learning_rate, "peak_vram_bytes": used,
                      "elapsed_seconds": elapsed_before + time.monotonic() - start_time}
            with (run / "train.log").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if step % settings.log_interval == 0:
                print(json.dumps(record), flush=True)
            if step % settings.save_interval == 0:
                save_checkpoint(latest, state())
            if step in settings.milestone_steps:
                save_checkpoint(run / f"step_{step:06d}.pt", state())
            if step % settings.sample_interval == 0 and not stop.requested:
                preview_labels = torch.arange(min(8, len(dataset.classes)), device=device)
                with amp():
                    preview = sample(ema, alpha, sampling_times(cfg.diffusion_steps, 1, device),
                                     preview_labels, settings.seed, settings.sigma_data, cfg.image_size)
                save_image_grid(preview, run / f"samples_step_{step:07d}.png")
            if (step >= settings.num_train_steps or stop.requested
                    or (max_updates_this_run is not None and step - start_step >= max_updates_this_run)):
                break
        if cursor == len(loader):
            epoch += 1
            cursor = 0
        if max_updates_this_run is not None and step - start_step >= max_updates_this_run:
            break
    saved = state()
    save_checkpoint(latest, saved)
    completed = step == settings.num_train_steps
    if completed:
        save_checkpoint(run / "final.pt", saved)
    metrics = {"status": "complete" if completed else "paused", "step": step,
               "target_steps": settings.num_train_steps, "method": settings.method,
               "guidance_scale": settings.guidance_scale, "parameter_count": parameters,
               "last_loss": last_loss, "peak_vram_bytes": peak(),
               "elapsed_seconds": saved["elapsed_seconds"], "fid": None,
               "teacher_sha256": teacher_sha256, "train_paths_sha256": data_digest}
    write_json(run / "metrics.json", metrics)
    print(json.dumps(metrics), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-updates-this-run", type=int,
                        help="Save and stop after this many additional updates, for GPU preflight")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Real training requires a CUDA GPU; CPU numerical tests use consistency.tests")
    # Import the protected upstream dataset only when actually training.
    from teacher.data import TrainImages
    settings = load_settings(args.config)
    source = load_checkpoint(args.teacher_checkpoint)
    cfg = Config(**source["config"])
    if (cfg.image_size, cfg.in_channels, cfg.num_classes, cfg.diffusion_steps) != (64, 3, 151, 1000):
        raise ValueError("Use the inspected 64x64/151-class/1000-step teacher")
    if not cfg.train_manifest or not cfg.development_reference_manifest:
        raise ValueError("Training-only development manifests are required")
    dataset = TrainImages(cfg.data_root, cfg.split_dir, 64, 151, split_file=cfg.train_manifest)
    holdout = TrainImages(cfg.data_root, cfg.split_dir, 64, 151, split_file=cfg.development_reference_manifest)
    if set(dataset.paths) & set(holdout.paths):
        raise ValueError("Training and development holdout overlap")
    stop = StopRequest()
    old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        run_training(settings, source, dataset, args.output_dir, file_digest(args.teacher_checkpoint),
                     torch.device("cuda"), args.max_updates_this_run, stop)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
