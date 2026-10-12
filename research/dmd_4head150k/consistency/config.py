"""Explicit settings shared by training, evaluation and resume checks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    method: str = "vp_consistency_l2"
    guidance_scale: float = 2.0
    num_train_steps: int = 100000
    batch_size: int = 64
    microbatch_size: int = 16
    num_workers: int = 4
    learning_rate: float = 0.0002
    learning_rate_schedule: str = "constant"
    warmup_steps: int = 0
    target_ema_decay: float = 0.999
    sampling_ema_decay: float = 0.999
    grad_clip: float = 1.0
    sigma_data: float = 0.5
    num_scales: int = 128
    clip_teacher: bool = True
    amp_dtype: str = "bf16"
    seed: int = 42
    save_interval: int = 500
    log_interval: int = 20
    sample_interval: int = 1000
    milestone_steps: tuple[int, ...] = (25000, 50000, 75000)

    def validate(self, diffusion_steps: int) -> None:
        if self.method != "vp_consistency_l2":
            raise ValueError("Unsupported method")
        if self.learning_rate_schedule != "constant" or self.warmup_steps != 0:
            raise ValueError("Project policy requires constant LR and no warmup")
        if self.amp_dtype not in ("bf16", "none"):
            raise ValueError("Use bf16 on CUDA or none for CPU diagnostics")
        if not 2 <= self.num_scales <= diffusion_steps:
            raise ValueError("num_scales must be between 2 and diffusion_steps")
        if min(self.num_train_steps, self.batch_size, self.microbatch_size,
               self.save_interval, self.log_interval, self.sample_interval) < 1:
            raise ValueError("Step counts, batch sizes and intervals must be positive")
        if self.num_workers < 0 or self.learning_rate <= 0 or self.sigma_data <= 0 or self.grad_clip <= 0:
            raise ValueError("Invalid training settings")
        if not 0 <= self.guidance_scale <= 6:
            raise ValueError("Use a fixed guidance scale in [0, 6]")
        if not all(0 <= x < 1 for x in (self.target_ema_decay, self.sampling_ema_decay)):
            raise ValueError("EMA decay must be in [0, 1)")

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))


def load_settings(path: str | Path) -> Settings:
    raw = json.loads(Path(path).read_text())
    if "milestone_steps" in raw:
        raw["milestone_steps"] = tuple(raw["milestone_steps"])
    return Settings(**raw)
