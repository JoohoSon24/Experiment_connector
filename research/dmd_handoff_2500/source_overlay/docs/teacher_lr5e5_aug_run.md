# Teacher: lower LR and mild augmentation, 50,000 steps

User requested a fresh run, not continuation of the 100,000-step baseline.
The new model and EMA are randomly initialized with seed 42. No old checkpoint
is loaded on the first launch. The original baseline and its results are preserved.

## Settings

- Config: `configs/teacher/teacher_lr5e5_aug_50k.json`.
- 50,000 optimizer updates, constant learning rate `5e-5`, no warmup.
- Original 48,883,107-parameter 64×64 class-conditional U-Net.
- Microbatch 8, gradient accumulation 2, effective batch 16; bf16.
- EMA 0.9999, class dropout 0.1, epsilon MSE, 1,000-step linear noise schedule.
- Training-only augmentation: horizontal flip probability 0.5; uniform rotation
  within ±5°; translation within ±2% of image dimensions; scale 0.98–1.02.
  Affine transforms use bilinear interpolation and white fill. No color jitter
  or strong crops. Augmentation is applied before normalization to [-1,1].
- Sampling remains 50 DDIM steps, guidance scale 1: 50 NFE. Training steps and
  sampling steps are distinct settings.
- All 17,079 official training images and 151 classes are used. There is no new
  internal holdout in this run; official validation never enters training.

LR and augmentation change together at the user's request. This is a comparison
of combined settings, not an ablation attributing improvement to either change.
Fewer updates or lower LR do not guarantee better generation; assess after training.

## Holdout clarification

An internal holdout would require splitting the official training manifest before
training, omitting those images from optimizer updates, and using them for
development decisions such as CFG comparisons. The old teacher saw the entire
training split, so a subset drawn afterwards is not an unseen holdout for it.
No new split is implemented here. Official validation FID remains reporting-only.

## Launch and logs

```bash
cd /root/workspace_distillation
tmux new-session -d -s fastgen_teacher_lr5e5_aug 'bash scripts/train_teacher_lr5e5_aug.sh'
tmux attach -t fastgen_teacher_lr5e5_aug
tail -f experiments/teacher_lr5e5_aug_50k/console.log
```

All stdout/stderr is appended to `console.log`. Every optimizer update is
recorded in `train.log` as JSONL. `metrics.json`, `config.json`, and a teacher
source snapshot are saved. `latest.pt` is saved every 500 updates, `final.pt`
at 50,000, and samples every 1,000. Relaunch resumes this run's own `latest.pt`
if it exists; it never automatically loads the old baseline.

## Verification

Augmented real training images have shape (3,64,64), preserve labels and range,
repeat with a fixed seed, and vary across random draws. A two-update full-size
training probe checks finite forward/backward, saving, and sample generation
before the sustained run. Its measurements are in
`experiments/teacher_lr5e5_aug_probe/metrics.json`.

The probe passed: losses 0.999755 and 1.000056, finite gradients, LR `5e-5`,
checkpoint model/EMA reload passed, and a two-step sample was generated.
Peak allocated VRAM was 2,225,084,928 bytes (~2.07 GiB).
The sustained run launched in tmux at 2026-09-28 01:47:46 UTC; step 20 had
loss 0.898511, LR `5e-5`, and peak allocated VRAM 2,227,620,352 bytes (~2.07 GiB).
This confirms execution, not final sample quality. Completion is tracked by
`metrics.json` and the script's `exit_code` file.
