"""Export an explicitly selected consistency checkpoint without replacing baselines."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from teacher.diffusion import Diffusion
from teacher.utils import load_checkpoint, save_checkpoint
from .core import sampling_times
from .train import file_digest


def export(checkpoint: str, output: str, mode: str) -> None:
    from model import Model, ModelOneNFE, ModelFewNFE
    if mode not in ("one_nfe", "few_nfe"):
        raise ValueError("Expected one_nfe or few_nfe")
    if Path(output).exists():
        raise FileExistsError("Choose a new output path; existing checkpoints are preserved")
    state = load_checkpoint(checkpoint)
    if state.get("method") != "vp_consistency_l2":
        raise ValueError("Expected a consistency training checkpoint")
    cfg = state["config"]
    architecture = (cfg["image_size"], cfg["in_channels"], cfg["num_classes"], cfg["model_channels"],
                    tuple(cfg["channel_mult"]), cfg["num_res_blocks"], tuple(cfg["attention_resolutions"]))
    if architecture != (64, 3, 151, 96, (1, 2, 3, 4), 2, (16,)) or cfg["diffusion_steps"] != 1000:
        raise ValueError("Checkpoint differs from the fixed submission architecture")
    nfe = 1 if mode == "one_nfe" else 4
    model = (ModelOneNFE if nfe == 1 else ModelFewNFE)()
    model.backbone.load_state_dict(state["ema"])
    model.alpha_bar.copy_(Diffusion(1000, cfg["beta_start"], cfg["beta_end"], torch.device("cpu")).alpha_bar)
    model.sampling_times.copy_(sampling_times(1000, nfe))
    model.sampling_seed.fill_(state["settings"]["seed"])
    model.register_buffer("consistency_sigma_data", torch.tensor(state["settings"]["sigma_data"]))
    save_checkpoint(Path(output), {"state_dict": model.state_dict(), "method": state["method"],
                                   "source_checkpoint": checkpoint, "source_sha256": file_digest(checkpoint),
                                   "nfe": nfe, "parameter_count": model.count_parameters(),
                                   "fixed_guidance_scale": state["settings"]["guidance_scale"]})
    loaded = Model.load_checkpoint(output, mode, device="cpu")
    if loaded.count_parameters() != 48883107:
        raise ValueError("Unexpected submission parameter count")
    print(f"Export/load PASS: {output}; {nfe} NFE; {loaded.count_parameters()} parameters", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("one_nfe", "few_nfe"), required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.mode)


if __name__ == "__main__":
    main()
