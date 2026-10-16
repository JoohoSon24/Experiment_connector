"""Training-only qualitative diagnostics; never accesses validation images."""
from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teacher.config import Config
from teacher.data import TrainImages
from teacher.diffusion import Diffusion
from teacher.utils import load_checkpoint, make_model, save_image_grid, set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(args.checkpoint, "cpu")
    raw = checkpoint["config"].copy()
    raw["channel_mult"] = tuple(raw["channel_mult"])
    raw["attention_resolutions"] = tuple(raw["attention_resolutions"])
    cfg = Config(**raw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    model = make_model(cfg, len(checkpoint["class_names"])).to(device).eval()
    diffusion = Diffusion(cfg.diffusion_steps, cfg.beta_start, cfg.beta_end, device)
    labels = torch.tensor([0, 13, 103, 133], device=device)
    variants = [
        ("ema_50_s1_bf16", "ema", 50, 1.0, "bf16"),
        ("ema_250_s1_bf16", "ema", 250, 1.0, "bf16"),
        ("ema_1000_s1_bf16", "ema", 1000, 1.0, "bf16"),
        ("ema_250_s2_bf16", "ema", 250, 2.0, "bf16"),
        ("raw_250_s1_bf16", "model", 250, 1.0, "bf16"),
        ("ema_250_s1_fp32", "ema", 250, 1.0, "float32"),
    ]
    results = {"checkpoint_step": checkpoint["step"], "seed": 42,
               "classes": [checkpoint["class_names"][i] for i in labels.tolist()], "variants": []}
    for name, weights, steps, scale, precision in variants:
        model.load_state_dict(checkpoint[weights])
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and precision == "bf16" else contextlib.nullcontext()
        with amp:
            images, nfe = diffusion.sample(model, labels, cfg.image_size, cfg.in_channels, steps, scale, 42)
        save_image_grid(images, output / f"{name}.png")
        results["variants"].append({"name": name, "weights": weights, "steps": steps,
                                    "guidance_scale": scale, "precision": precision, "nfe": nfe})
        print(name, "NFE", nfe, flush=True)
    dataset = TrainImages(cfg.data_root, cfg.split_dir, cfg.image_size, cfg.num_classes, split_file=cfg.train_manifest)
    counts = Counter(dataset.categories)
    results["training_class_counts"] = {"min": min(counts.values()), "max": max(counts.values()),
                                        "median": statistics.median(counts.values())}
    real = []
    for label in labels.tolist():
        indices = [i for i, c in enumerate(dataset.category_ids) if c == label][:4]
        real.extend(dataset[i][0] for i in indices)
    save_image_grid(torch.stack(real), output / "real_training_examples.png")
    write_json(output / "results.json", results)
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
