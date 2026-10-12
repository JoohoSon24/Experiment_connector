"""Evaluate CD with the protected FastGen generation/FID functions."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from teacher.config import Config
from teacher.diffusion import Diffusion
from teacher.utils import load_checkpoint, make_model, save_checkpoint, write_json
from .core import sample, sampling_times
from .train import file_digest


class Adapter(nn.Module):
    def __init__(self, network, alpha, times, sigma_data, seed):
        super().__init__()
        self.backbone = network
        self.register_buffer("alpha_bar", alpha)
        self.register_buffer("sampling_times", times)
        self.sigma_data, self.seed, self.batch_index = sigma_data, seed, 0

    @torch.inference_mode()
    def sample(self, shape, *, device="cuda", category=None, **kwargs):
        if tuple(shape[1:]) != (3, 64, 64) or category is None or len(category) != shape[0]:
            raise ValueError("Expected categories and [B,3,64,64]")
        device = torch.device(device)
        category = category.to(device=device, dtype=torch.long)
        if torch.any((category < 0) | (category >= self.backbone.num_classes)):
            raise ValueError("Invalid categories")
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
        with amp:
            result = sample(self.backbone, self.alpha_bar, self.sampling_times, category,
                            self.seed + self.batch_index, self.sigma_data)
        self.batch_index += 1
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--reference-split", choices=("development", "official"), default="development")
    parser.add_argument("--nfe", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--select-best", action="store_true",
                        help="Record best_fid.pt beside the source using development 1-NFE FID only")
    args = parser.parse_args()
    if args.select_best and (args.reference_split != "development" or args.nfe != 1):
        parser.error("Model selection is restricted to development 1-NFE FID")
    from evaluate import compute_fid, generate_samples
    from teacher.data import TrainImages
    state = load_checkpoint(args.checkpoint)
    if state.get("method") != "vp_consistency_l2":
        raise ValueError("Expected a consistency training checkpoint")
    cfg = Config(**state["config"])
    reference = Path(args.reference_dir)
    files = sorted(reference.glob("*.png"))
    if args.reference_split == "development":
        if not cfg.train_manifest or not cfg.development_reference_manifest:
            raise ValueError("Missing development manifests")
        holdout = TrainImages(cfg.data_root, cfg.split_dir, 64, 151, split_file=cfg.development_reference_manifest)
        train = TrainImages(cfg.data_root, cfg.split_dir, 64, 151, split_file=cfg.train_manifest)
        if set(holdout.paths) & set(train.paths):
            raise ValueError("Training and holdout overlap")
        expected_count = len(holdout)
    else:
        expected_count = 3020
    if len(files) != expected_count:
        raise ValueError("Reference PNGs missing/incomplete; prepare them with teacher.evaluate")
    refs_hash = hashlib.sha256()
    for path in files:
        refs_hash.update(path.name.encode())
        refs_hash.update(bytes.fromhex(file_digest(path)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = make_model(cfg, 151).to(device).eval()
    network.load_state_dict(state["ema"])
    alpha = Diffusion(cfg.diffusion_steps, cfg.beta_start, cfg.beta_end, device).alpha_bar
    times = sampling_times(cfg.diffusion_steps, args.nfe, device)
    model = Adapter(network, alpha, times, state["settings"]["sigma_data"], args.seed).to(device).eval()
    metrics = {"checkpoint_sha256": file_digest(args.checkpoint), "checkpoint_step": state["step"],
               "method": state["method"], "settings": state["settings"],
               "teacher_sha256": state["teacher_sha256"], "implementation_sha256": state["implementation_sha256"],
               "train_paths_sha256": state["train_paths_sha256"], "reference_split": args.reference_split,
               "reference_sha256": refs_hash.hexdigest(), "reference_count": expected_count,
               "seed": args.seed, "sample_count": 3020, "batch_size": 16,
               "nfe": args.nfe, "sampling_times": times.tolist(), "weights": "ema",
               "precision": "bf16" if device.type == "cuda" else "fp32", "device_type": device.type,
               "parameter_count": sum(p.numel() for p in model.parameters())}
    output = Path(args.output_dir)
    generated, metadata = output / "generated", output / "metrics.json"
    previous = json.loads(metadata.read_text()) if metadata.exists() else None
    if previous is not None and any(previous.get(k) != v for k, v in metrics.items()):
        raise ValueError("Evaluation inputs changed; choose a fresh output directory")
    if previous is not None and previous.get("fid") is not None:
        metrics = previous
    else:
        if generated.exists() and any(generated.iterdir()):
            expected = {f"{c:04d}_{i:02d}.png" for c in range(151) for i in range(20)}
            if previous is None or {p.name for p in generated.iterdir()} != expected:
                raise ValueError("Incomplete/untracked generation; restart in a fresh evaluation directory")
        else:
            write_json(metadata, metrics)
            generate_samples(model, generated, torch.arange(151).repeat_interleave(20), 16, device)
        metrics["fid"] = compute_fid(generated, reference, device, 16)
        write_json(metadata, metrics)
    if args.select_best:
        run = Path(args.checkpoint).parent
        best_path = run / "best_fid.json"
        best = json.loads(best_path.read_text()) if best_path.exists() else None
        signature = ("method", "settings", "teacher_sha256", "implementation_sha256", "train_paths_sha256",
                     "reference_sha256", "seed", "nfe", "sample_count", "batch_size", "precision", "device_type")
        if best is not None and any(best[k] != metrics[k] for k in signature):
            raise ValueError("Best-FID selection requires identical evaluation and training settings")
        if best is None or metrics["fid"] < best["fid"]:
            save_checkpoint(run / "best_fid.pt", state)
            write_json(best_path, metrics)
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
