"""Audit restored handoff, data, real checkpoints and a disposable GPU update.

Run from the repository root: .venv/bin/python environment/check_restoration.py
No production weight or protected upstream file is modified.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import PokemonDataModule, read_split
from distillation.core import coefficients, ddim_step, merged_target, target_velocity, velocity_loss
from evaluate import prepare_reference_set
from model import Model
from teacher.cfg import apply_cfg
from teacher.config import load_config
from teacher.data import TrainImages
from teacher.diffusion import Diffusion
from teacher.train import drop_classes, update_ema
from teacher.utils import make_model, save_image_grid, set_seed

COMMIT = "e3f5120000cd91037017d86f23993492195101a5"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    import os
    os.chdir(ROOT)
    output = ROOT / "environment"
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "python_executable": sys.executable}
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == COMMIT
    protected = ["dataset.py", "download_dataset.py", "evaluate.py", "src/utils.py",
                 "requirements.txt", "data/pokemon-generation-one-22k/train_split.txt",
                 "data/pokemon-generation-one-22k/val_split.txt",
                 "data/pokemon-generation-one-22k/category_to_id.json"]
    for name in protected:
        assert Path(name).read_bytes() == subprocess.check_output(["git", "show", f"{COMMIT}:{name}"])
    original = subprocess.check_output(["git", "show", f"{COMMIT}:model.py"], text=True)
    assert original.split("class ModelOneNFE")[0] == Path("model.py").read_text().split("class ModelOneNFE")[0]
    manifest = json.loads(Path("OVERLAY_MANIFEST.json").read_text())
    assert manifest["upstream_commit"] == COMMIT
    for name, info in manifest["files"].items():
        assert digest(Path(name)) == info["sha256"], name
    report["source"] = {"base_commit": COMMIT, "protected_files_unchanged": protected,
                        "base_Model_unchanged": True, "overlay_files_verified": len(manifest["files"])}
    for line in Path("environment/requirements.lock.txt").read_text().splitlines():
        name, expected = line.split("==", 1)
        assert metadata.version(name) == expected, name
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    report["packages"] = {name: metadata.version(name) for name in
                          ["torch", "torchvision", "torchaudio", "kagglehub", "Pillow", "clean-fid", "numpy", "scipy"]}
    base = Path("data/pokemon-generation-one-22k")
    train = read_split(base / "train_split.txt")
    val = read_split(base / "val_split.txt")
    development = Path("data/development/seed42_holdout5")
    internal_train = read_split(development / "train.txt")
    holdout = read_split(development / "holdout.txt")
    assert (len(train), len(val), len(internal_train), len(holdout)) == (17079, 3020, 16324, 755)
    assert len(set(train + val)) == 20099
    assert set(internal_train).isdisjoint(holdout)
    assert set(internal_train) | set(holdout) == set(train)
    assert set(val).isdisjoint(internal_train + holdout)
    assert len(Counter(Path(p).parent.name for p in holdout)) == 151
    assert set(Counter(Path(p).parent.name for p in holdout).values()) == {5}
    split_metadata = json.loads((development / "metadata.json").read_text())
    assert digest(base / "train_split.txt") == split_metadata["source_train_sha256"]
    assert digest(base / "val_split.txt") == split_metadata["source_validation_sha256"]
    aggregate = hashlib.sha256()
    print("Checking all 20,099 image files...", flush=True)
    for relative in sorted(train + val):
        path = base / relative
        assert path.is_file(), relative
        with Image.open(path) as image:
            image.verify()
        aggregate.update(relative.encode() + b"\0" + bytes.fromhex(digest(path)))
    report["data"] = {"official_train_count": len(train), "official_validation_count": len(val),
                      "optimizer_train_count": len(internal_train), "holdout_count": len(holdout),
                      "holdout_per_class": 5, "image_verification": "PASS",
                      "image_manifest_aggregate_sha256": aggregate.hexdigest(),
                      "train_holdout_validation_disjoint": True}
    data = PokemonDataModule(batch_size=16, num_workers=0, return_category=True)
    reference = Path("results/reference")
    dev_reference = development / "reference"
    for folder, module, expected in [
        (reference, data, 3020),
        (dev_reference, SimpleNamespace(val_dataloader=lambda: DataLoader(
            TrainImages(base, base, 64, 151, split_file=development / "holdout.txt"),
            batch_size=16, num_workers=0)), 755),
    ]:
        existing = len(list(folder.glob("*.png")))
        assert existing in (0, expected), f"Partial references preserved: {folder}"
        if existing == 0:
            prepare_reference_set(module, folder)
        assert len(list(folder.glob("*.png"))) == expected
    report["references"] = {"official": 3020, "development": 755,
                            "method": "unchanged evaluate.prepare_reference_set and protected dataset transform"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "GPU unavailable; run with GPU device access. No GPU proof claimed."
    report["gpu"] = {"name": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
                     "memory_bytes": torch.cuda.get_device_properties(0).total_memory}
    set_seed(42)
    models = []
    for mode, expected in [("one_nfe", 1), ("few_nfe", 4)]:
        checkpoint_path = Path("checkpoints") / f"{mode}.ckpt"
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        provenance = {k: v for k, v in raw.items() if k != "state_dict"}
        del raw
        cpu = Model.load_checkpoint(str(checkpoint_path), mode, device="cpu")
        assert cpu.count_parameters() == 48_883_107
        model = cpu.to(device)
        torch.cuda.reset_peak_memory_stats()
        count = [0]
        hook = model.backbone.register_forward_hook(lambda *_: count.__setitem__(0, count[0] + 1))
        labels = torch.tensor([0, 13], device=device)
        images = model.sample((2, 3, 64, 64), device=device, category=labels, seed=42)
        assert count[0] == expected
        assert images.shape == (2, 3, 64, 64) and torch.isfinite(images).all()
        assert images.min() >= -1 and images.max() <= 1
        count[0] = 0
        repeat = model.sample((2, 3, 64, 64), device=device, category=labels, seed=42)
        assert count[0] == expected and torch.equal(images, repeat)
        hook.remove()
        save_image_grid(images, output / f"{mode}_smoke.png", columns=2)
        models.append({"mode": mode, "measured_nfe": expected, "parameters": model.count_parameters(),
                       "shape_range_finite": "PASS", "explicit_seed_repeat": "PASS",
                       "checkpoint_sha256": digest(checkpoint_path), "provenance": provenance,
                       "inference_peak_allocated_bytes": torch.cuda.max_memory_allocated()})
        del cpu, model, images, repeat
        gc.collect()
        torch.cuda.empty_cache()
    report["inference_models"] = models
    alpha = Diffusion(1000, .0001, .02, torch.device("cpu")).alpha_bar.double()
    t = torch.tensor([999, 700, 300])
    u = torch.tensor([500, 200, -1])
    a, sigma = coefficients(alpha, t)
    au, su = coefficients(alpha, u)
    x = torch.randn(3, 3, 64, 64, dtype=torch.float64)
    endpoint = torch.randn_like(x)
    y0 = merged_target(x, endpoint, a, sigma, au, su)
    error = (ddim_step(x, y0, a, sigma, au, su) - endpoint).abs().max().item()
    assert error < 1e-10
    target_v = target_velocity(x, y0, a, sigma)
    assert torch.allclose(a * x - sigma * target_v, y0)
    prediction = target_v + .01 * torch.randn_like(x)
    lhs = velocity_loss(prediction, target_v, a, sigma)
    rhs = (torch.maximum(a.square()/sigma.square(), torch.ones_like(a)) *
           (a*x-sigma*prediction-y0).square()).mean()
    assert torch.allclose(lhs.double(), rhs, atol=1e-8, rtol=1e-5)
    report["distillation_math"] = {"merged_endpoint_max_error": error, "v_and_loss_identities": "PASS",
                                   "full_pipeline_smoke": "NOT RUN: full teacher checkpoint absent"}
    print("Disposable full-size teacher update with real training batch=64...", flush=True)
    cfg, _ = load_config("configs/teacher/teacher_b64_noaug_constant_100k.json")
    dataset = TrainImages(cfg.data_root, cfg.split_dir, 64, 151, split_file=cfg.train_manifest)
    images, labels = next(iter(DataLoader(dataset, batch_size=64, num_workers=0)))
    assert images.shape == (64, 3, 64, 64) and images.min() >= -1 and images.max() <= 1
    images, labels = images.to(device), labels.to(device)
    torch.cuda.reset_peak_memory_stats()
    network = make_model(cfg, 151).to(device)
    ema = make_model(cfg, 151).to(device).eval().requires_grad_(False)
    ema.load_state_dict(network.state_dict())
    optimizer = torch.optim.AdamW(network.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = Diffusion(1000, .0001, .02, device).loss(network, images,
                    drop_classes(labels, 151, cfg.class_dropout))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
    norm = torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.grad_clip)
    optimizer.step()
    update_ema(ema, network, cfg.ema_decay)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        times = torch.zeros_like(labels[:2])
        conditional = network(images[:2], times, labels[:2])
        unconditional = network(images[:2], times, torch.full_like(labels[:2], 151))
        assert conditional.shape == unconditional.shape == images[:2].shape
        assert torch.equal(apply_cfg(unconditional, conditional, 0), unconditional.float())
    peak = torch.cuda.max_memory_allocated()
    assert peak < 20_000_000_000
    report["disposable_teacher_update"] = {"batch_size": 64, "updates": 1, "parameters": sum(p.numel() for p in network.parameters()),
        "loss": loss.item(), "gradient_norm": norm.item(), "finite_loss_gradient": "PASS", "ema_update": "PASS",
        "conditional_unconditional_shape_cfg0": "PASS", "peak_allocated_bytes": peak,
        "production_weights_changed": False, "scope": "random-init smoke, not quality or exact resume proof"}
    report["unavailable_training_artifacts"] = [
        "experiments/teacher_b64_noaug_constant_100k/final.pt",
        "experiments/fixed_cfg2_progressive/{n4,n2,n1}/final.pt",
        "experiments/low_nfe_budget_100k/n1/final.pt",
        "experiments/low_nfe_budget_100k/n2/step_075000.pt or final.pt"]
    report["status"] = "PASS: restored inference, data, protected code and disposable GPU training update"
    (output / "restoration_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
