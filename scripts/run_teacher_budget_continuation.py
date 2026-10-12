"""Original noaug teacher continuation with an external online W&B observer.

The teacher trainer/sampler/loss are unchanged. No inference-only checkpoint or
fresh model is substituted for the required 100K full training checkpoint.
"""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teacher.config import Config, load_config

BASE = ROOT / "configs/teacher/teacher_b64_noaug_constant_100k.json"
RUN_ROOT = ROOT / "experiments/teacher_budget_noaug_100k_200k"
DEFAULT_CHECKPOINT = ROOT / "experiments/teacher_b64_noaug_constant_100k/final.pt"
BOOKKEEPING = {"experiment_name", "output_dir", "num_train_steps"}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def effective_config(raw: dict) -> dict:
    return json.loads(json.dumps(dataclasses.asdict(Config(**raw))))


def check_recipe(previous: dict, current: dict) -> None:
    old, new = effective_config(previous), effective_config(current)
    changed = {key: (old[key], new[key]) for key in old if key not in BOOKKEEPING and old[key] != new[key]}
    if changed:
        raise ValueError(f"Budget-only recipe mismatch: {changed}")
    if new["learning_rate_schedule"] != "constant" or new["warmup_steps"] != 0:
        raise ValueError("Continuation requires constant LR and no warmup")
    if any(new[k] for k in ("horizontal_flip_probability", "augmentation_degrees",
                            "augmentation_translate", "augmentation_scale_delta")):
        raise ValueError("Continuation must remain no-augmentation")


def prepare_configs() -> dict[int, Path]:
    _, base = load_config(BASE)
    paths = {}
    for budget in (150000, 200000):
        raw = json.loads(json.dumps(base))
        raw.update(experiment_name=f"teacher_b64_noaug_constant_{budget//1000}k",
                   num_train_steps=budget,
                   output_dir=str(RUN_ROOT.relative_to(ROOT)))
        check_recipe(base, raw)
        path = ROOT / f"configs/teacher/teacher_b64_noaug_constant_{budget//1000}k.json"
        if path.exists():
            _, existing = load_config(path)
            if effective_config(existing) != effective_config(raw):
                raise FileExistsError(f"Different config preserved: {path}")
        else:
            write_json(path, raw)
        paths[budget] = path
    return paths


def checkpoint_step(path: Path, expected_range: tuple[int, int]) -> tuple[int, str]:
    import torch
    from teacher.utils import make_model
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {"config", "model", "ema", "ema_tracks", "optimizer", "scaler", "step", "epoch",
                "batch_in_epoch", "images_seen", "class_names", "train_paths_sha256",
                "torch_rng", "python_rng", "cuda_rng"}
    if not required <= state.keys():
        raise ValueError(f"Full teacher checkpoint required; missing fields: {sorted(required-state.keys())}")
    _, base = load_config(BASE)
    check_recipe(state["config"], base)
    step = int(state["step"])
    if not expected_range[0] <= step <= expected_range[1]:
        raise ValueError(f"Expected step in {expected_range}, received {step}")
    mapping = json.loads((ROOT / base["split_dir"] / "category_to_id.json").read_text())
    names = [name for name, _ in sorted(mapping.items(), key=lambda pair: pair[1])]
    if state["class_names"] != names:
        raise ValueError("Class mapping differs")
    paths = [line.strip() for line in (ROOT / base["train_manifest"]).read_text().splitlines() if line.strip()]
    if state["train_paths_sha256"] != hashlib.sha256("\n".join(paths).encode()).hexdigest():
        raise ValueError("Optimizer training manifest differs")
    if set(state["ema_tracks"]) != {f"{decay:g}" for decay in base["extra_ema_decays"]}:
        raise ValueError("Secondary EMA tracks differ")
    cfg, _ = load_config(BASE)
    with torch.device("meta"):
        expected = make_model(cfg, 151).state_dict()
    for label, weights in [("model", state["model"]), ("ema", state["ema"]), *state["ema_tracks"].items()]:
        if weights.keys() != expected.keys() or any(weights[key].shape != expected[key].shape for key in expected):
            raise ValueError(f"Teacher {label} architecture differs")
        if sum(tensor.numel() for tensor in weights.values()) != 48_883_107:
            raise ValueError("Unexpected teacher parameter count")
    optimizer = state["optimizer"]
    if not optimizer.get("state") or len(optimizer.get("param_groups", [])) != 1:
        raise ValueError("Trained AdamW state absent")
    group = optimizer["param_groups"][0]
    for key, target in [("lr", cfg.learning_rate), ("weight_decay", cfg.weight_decay),
                        ("eps", 1e-8), ("betas", (0.9, 0.999))]:
        value = tuple(group[key]) if key == "betas" else group[key]
        if value != target:
            raise ValueError(f"Optimizer {key} mismatch")
    if state["cuda_rng"] is None or not state["cuda_rng"]:
        raise ValueError("Original CUDA RNG state absent")
    del state, expected
    return step, digest(path)


def assert_original_teacher_source() -> dict:
    manifest = json.loads((ROOT / "OVERLAY_MANIFEST.json").read_text())
    hashes = {}
    for name, record in manifest["files"].items():
        if name.startswith("teacher/") or name in {"dataset.py", "evaluate.py"}:
            hashes[name] = digest(ROOT / name)
            if hashes[name] != record["sha256"]:
                raise ValueError(f"Original teacher source changed: {name}")
    for name in ("dataset.py", "download_dataset.py", "evaluate.py", "src/utils.py", "requirements.txt",
                 "data/pokemon-generation-one-22k/train_split.txt",
                 "data/pokemon-generation-one-22k/val_split.txt",
                 "data/pokemon-generation-one-22k/category_to_id.json"):
        original = subprocess.check_output(["git", "show", f"{manifest['upstream_commit']}:{name}"], cwd=ROOT)
        if (ROOT / name).read_bytes() != original:
            raise ValueError(f"Protected upstream source changed: {name}")
        hashes[name] = digest(ROOT / name)
    return hashes


class Observer:
    def __init__(self, run, wandb):
        self.run, self.wandb = run, wandb
        self.global_step = 100000
        self.logged_images = set()

    def status(self, phase: str, **extra) -> None:
        data = {"phase": phase, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "global_step": self.global_step, "wandb_url": self.run.url, **extra}
        write_json(RUN_ROOT / "status.json", data)
        self.run.summary["phase"] = phase
        print(json.dumps(data, ensure_ascii=False), flush=True)

    def training_line(self, line: str, stage_dir: Path) -> None:
        try:
            values = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return
        if not isinstance(values, dict) or not {"step", "loss", "lr"} <= values.keys():
            return
        step = int(values["step"])
        self.global_step = max(self.global_step, step)
        payload = {"global_step": step, **{f"train/{key}": value for key, value in values.items()}}
        self.run.log(payload)
        if values.get("peak_vram_bytes", 0) >= 20_000_000_000:
            raise RuntimeError("Measured peak allocation exceeded 20GB; stop continuation")
        self.status("training", stage_checkpoint_dir=str(stage_dir.relative_to(ROOT)))
        for image in sorted(stage_dir.glob("samples_step_*.png")):
            if image in self.logged_images:
                continue
            self.run.log({"global_step": step, "preview/cfg1_ddpm50": self.wandb.Image(str(image))})
            self.logged_images.add(image)


def command(args: list[str], log: Path, observer: Observer | None = None, stage_dir: Path | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    process = subprocess.Popen(args, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    try:
        with log.open("a") as stream:
            assert process.stdout is not None
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
                if observer is not None and stage_dir is not None:
                    observer.training_line(line, stage_dir)
            result = process.wait()
        if result != 0:
            raise subprocess.CalledProcessError(result, args)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def evaluate(checkpoint: Path, step: int, observer: Observer) -> dict:
    checkpoint_sha = digest(checkpoint)
    reports = {}
    observer.global_step = step
    for steps in (32, 50):
        out = RUN_ROOT / f"evaluation/step_{step:06d}/ddim{steps}_cfg2"
        metric_file = out / "metrics.json"
        generated = out / "generated"
        existing = json.loads(metric_file.read_text()) if metric_file.exists() else None
        args = [sys.executable, "-u", "-m", "teacher.evaluate", "--checkpoint", str(checkpoint),
                "--output-dir", str(out), "--reference-split", "development",
                "--reference-dir", "data/development/seed42_holdout5/reference",
                "--weights", "ema", "--sampler", "ddim", "--clip-denoised",
                "--steps", str(steps), "--guidance-scale", "2", "--seed", "42",
                "--batch-size", "16", "--precision", "bf16"]
        fingerprint = out / "checkpoint_sha256.json"
        if fingerprint.exists() and json.loads(fingerprint.read_text())["sha256"] != checkpoint_sha:
            raise ValueError("Evaluation checkpoint content changed; prior result preserved")
        write_json(fingerprint, {"sha256": checkpoint_sha})
        observer.status("evaluating", milestone=step, ddim_steps=steps)
        if existing is None or existing.get("fid") is None:
            count = len(list(generated.glob("*.png")))
            if count:
                if count != 3020 or existing is None:
                    raise FileExistsError(f"Partial evaluation preserved; inspect {out}")
                args.append("--score-only")
            command(args, out / "console.log")
            existing = json.loads(metric_file.read_text())
        expectations = {"checkpoint": str(checkpoint), "checkpoint_step": step, "weights": "ema",
                        "reference_split": "development", "seed": 42, "batch_size": 16,
                        "precision": "bf16", "steps": steps, "guidance_scale": 2.0,
                        "sample_count": 3020, "reference_count": 755, "sampler": "ddim",
                        "clip_denoised": True, "nfe_per_sample": 2 * steps}
        if any(existing.get(key) != value for key, value in expectations.items()) or existing.get("fid") is None:
            raise ValueError(f"Frozen evaluation metadata differs: {out}")
        expected_names = {f"{label:04d}_{i:02d}.png" for label in range(151) for i in range(20)}
        if {path.name for path in generated.glob("*.png")} != expected_names:
            raise ValueError("Generated set incomplete")
        reports[f"ddim{steps}_cfg2"] = existing
        observer.run.log({"global_step": step, f"eval/ddim{steps}_cfg2_fid": existing["fid"]})
    write_json(RUN_ROOT / f"milestone_{step:06d}.json", reports)
    return reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--entity", default="fv2z6n929c-kaist-digital-humanities-and-social-sciences-")
    parser.add_argument("--project", default="pokemon-diffusion")
    parser.add_argument("--wait-for-checkpoint", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    lock = (RUN_ROOT / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    configs = prepare_configs()
    source_hashes = assert_original_teacher_source()
    checkpoint = args.checkpoint.resolve()
    plan = {"baseline_checkpoint": str(checkpoint), "milestone_steps": [100000, 150000, 200000],
            "additional_updates": 100000, "recipe_changes": "budget/output bookkeeping only",
            "evaluation": {"split": "training-only development", "reference_count": 755,
                "samples": 3020, "steps": [32, 50], "guidance_scale": 2, "seed": 42,
                "batch_size": 16, "weights": "primary EMA", "sampler": "clipped DDIM"},
            "teacher_source_sha256": source_hashes,
            "wandb": {"entity": args.entity, "project": args.project, "mode": "online"}}
    plan_path = RUN_ROOT / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing experiment plan differs; preserve this run")
    write_json(plan_path, plan)
    if args.preflight_only:
        if not checkpoint.is_file():
            print(json.dumps({"status": "WAITING_FOR_FULL_TEACHER_CHECKPOINT", "checkpoint": str(checkpoint)}))
            return 2
        step, sha = checkpoint_step(checkpoint, (100000, 100000))
        print(json.dumps({"status": "PREFLIGHT_PASS", "step": step, "sha256": sha}))
        return 0
    if not checkpoint.is_file() and not args.wait_for_checkpoint:
        raise FileNotFoundError(f"Full noaug teacher checkpoint required: {checkpoint}")
    import wandb
    import torch
    identity_path = RUN_ROOT / "wandb_identity.json"
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {"id": uuid.uuid4().hex[:8]}
    write_json(identity_path, identity)
    run = wandb.init(entity=args.entity, project=args.project, id=identity["id"], resume="allow",
                     name="teacher_noaug_100k_to_150k_to_200k_seed42",
                     mode="online", dir=str(RUN_ROOT), config=plan, save_code=False,
                     tags=["teacher", "noaug", "budget-only", "seed42"],
                     settings=wandb.Settings(init_timeout=60, x_disable_meta=True))
    observer = Observer(run, wandb)
    run.define_metric("global_step")
    run.define_metric("train/*", step_metric="global_step")
    run.define_metric("eval/*", step_metric="global_step")
    identity["url"] = run.url
    write_json(identity_path, identity)
    try:
        last_stat = None
        while True:
            if checkpoint.is_file():
                stat = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
                if not args.wait_for_checkpoint or stat == last_stat:
                    break
                last_stat = stat
            observer.status("waiting_for_teacher_checkpoint", checkpoint=str(checkpoint), training_started=False)
            time.sleep(20)
        step, sha = checkpoint_step(checkpoint, (100000, 100000))
        write_json(RUN_ROOT / "baseline_checkpoint.json", {"step": step, "sha256": sha, "path": str(checkpoint)})
        if not torch.cuda.is_available():
            raise RuntimeError("Continuation requires CUDA; no CPU/fresh fallback")
        observer.status("preflight_pass", baseline_sha256=sha)
        milestones = {"100000": evaluate(checkpoint, 100000, observer)}
        source_checkpoint = checkpoint
        for target in (150000, 200000):
            _, cfg = load_config(configs[target])
            stage_dir = ROOT / cfg["output_dir"] / cfg["experiment_name"]
            final = stage_dir / "final.pt"
            if not final.exists():
                resume = stage_dir / "latest.pt" if (stage_dir / "latest.pt").exists() else source_checkpoint
                lower = 100000 if target == 150000 else 150000
                resume_step, _ = checkpoint_step(resume, (lower, target))
                observer.global_step = resume_step
                observer.status("training", target_step=target, resumed_from=str(resume))
                assert_original_teacher_source()
                if resume_step == target:
                    # Recover an interruption between latest.pt and final.pt saves.
                    # Original trainer's completed-step no-op does not create final.pt.
                    shutil.copy2(resume, final)
                else:
                    command([sys.executable, "-u", "-m", "teacher.train", "--config", str(configs[target]),
                             "--resume", str(resume)], stage_dir / "console.log", observer, stage_dir)
            checkpoint_step(final, (target, target))
            milestones[str(target)] = evaluate(final, target, observer)
            source_checkpoint = final
        chosen = min(milestones, key=lambda value: milestones[value]["ddim32_cfg2"]["fid"])
        report = {"status": "completed", "milestones": milestones, "selected_step": int(chosen),
                  "selection_policy": "Lowest DDIM32 CFG2 development FID; DDIM50 secondary; no official tuning"}
        write_json(RUN_ROOT / "summary.json", report)
        run.summary["selected_step"] = int(chosen)
        run.summary["selected_ddim32_cfg2_fid"] = milestones[chosen]["ddim32_cfg2"]["fid"]
        observer.status("completed", selected_step=int(chosen))
        run.finish()
        return 0
    except BaseException as error:
        observer.status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error_type=type(error).__name__, error=str(error)[:500])
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
