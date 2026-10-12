"""Check actual exported models on the assembled upstream repo, without image data."""
from __future__ import annotations

import argparse
import json

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("one_nfe", "few_nfe"), required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from model import Model
    device = torch.device(args.device)
    model = Model.load_checkpoint(args.checkpoint, args.mode, device=device)
    labels = torch.tensor([0, 13], device=device)
    expected = 1 if args.mode == "one_nfe" else 4
    calls = []
    hook = model.backbone.register_forward_hook(lambda *unused: calls.append(1))
    try:
        first = model.sample((2, 3, 64, 64), device=device, category=labels, seed=42)
        assert len(calls) == expected
        calls.clear()
        repeat = model.sample((2, 3, 64, 64), device=device, category=labels, seed=42)
        assert len(calls) == expected
        assert first.shape == (2, 3, 64, 64)
        assert torch.isfinite(first).all() and ((first >= -1) & (first <= 1)).all()
        assert torch.equal(first, repeat)
        assert model.count_parameters() < 60000000
    finally:
        hook.remove()
    print(json.dumps({"status": "PASS", "checkpoint": args.checkpoint,
                      "nfe": expected, "parameters": model.count_parameters(),
                      "deterministic": True, "device": str(device)}))


if __name__ == "__main__":
    main()
