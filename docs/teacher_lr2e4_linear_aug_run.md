# Teacher restart: linear learning-rate decay, 50,000 updates

Completed normally: 50,000 updates, 11,261.44 seconds, exit code 0, peak allocated
VRAM 2.0748 GiB. Original DDIM-50 / CFG-1 FID: 189.94; clipped DDPM-50 / CFG-1
FID on the same checkpoint: 82.99, also at 50 NFE. See
[teacher_review_50k.md](/root/workspace_distillation/docs/teacher_review_50k.md) for
sampling diagnosis, quantitative reporting and next experiments.

The user requested stopping the LR-5e-5 run, restoring initial LR to 2e-4,
adding linear LR decay, and deleting previous experiment folders including
smoke/debug outputs. The running trainer and its loader workers were stopped.
All 11 previous children of `experiments/` were deleted, including the original
100,000-step checkpoint, diagnostic samples, FID artifacts, and probes. Existing
documents retain historical measurements, but their old artifact paths no longer
exist. Dataset, environment, source, configs, and official evaluator are preserved.

## New settings

Config: `configs/teacher/teacher_lr2e4_linear_aug_50k.json`.

- Fresh random initialization, seed 42; 50,000 optimizer updates.
- Initial LR 2e-4, no warmup. Update index k=0,...,49,999 uses
  `lr = 2e-4 * (1 - k / 49,999)`: first update 2e-4, final update zero.
- `learning_rate_schedule="linear"` is distinct from diffusion
  `noise_schedule="linear"`; the 1,000-step beta noise schedule is unchanged.
- Resume reconstructs LR from saved optimizer step, so decay does not restart.
- Same 48,883,107-parameter class-conditional U-Net, microbatch 8 × accumulation
  2, bf16, EMA 0.9999, class dropout 0.1, and epsilon MSE objective.
- Same mild train-only augmentation: flip probability 0.5, rotation ±5°,
  translation ±2%, scale 0.98–1.02, bilinear interpolation with white fill.
- Entire official training split only; no validation-based tuning or new holdout.
- Diagnostic sampling stays 50 DDIM steps at CFG scale 1 (50 NFE).

## Launch and records

```bash
cd /root/workspace_distillation
tmux new-session -d -s fastgen_teacher_lr2e4_linear_aug 'bash scripts/train_teacher_lr2e4_linear_aug.sh'
tmux attach -t fastgen_teacher_lr2e4_linear_aug
tail -f experiments/teacher_lr2e4_linear_aug_50k/console.log
```

`console.log` captures stdout/stderr, and `train.log` records loss and actual LR
on every optimizer update. `config.json`, `source_snapshot`, and `metrics.json`
record settings/source/measurements. Checkpoints are saved every 500 updates;
samples every 1,000; `final.pt` at completion. Relaunch resumes only this run's
own `latest.pt`, if present.

## Checks

LR boundary checks verified initial 2e-4, midpoint approximately 1e-4, final 0,
and unchanged behavior of constant-LR configurations. A temporary three-update
full-size augmented training check verified actual LRs [2e-4, 1e-4, 0], finite
forward/backward, checkpoint metadata/load, and sample generation. Temporary
probe artifacts were removed automatically; no smoke/debug output folder remains.
