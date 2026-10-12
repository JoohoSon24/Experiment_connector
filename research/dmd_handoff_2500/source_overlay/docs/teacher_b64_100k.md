# Fresh teacher baseline: batch 64, 100,000 updates

Completed with exit code 0. Official FID: **60.61 at CFG 1 / 50 NFE**, and
**32.98 at training-holdout-selected CFG 2 / 100 NFE**. The sampler/EMA/CFG
development comparisons also completed. Results and the now-active next
experiments are in [teacher_b64_results_and_next.md](/root/workspace_distillation/docs/teacher_b64_results_and_next.md).

The user requested a fresh 100,000-update baseline, preserving the update count
when increasing batch size. After a batch-128 dual-EMA probe ran out of memory,
the user explicitly requested reducing microbatch to **64**. The brief B128 run
was stopped. The active B64 run starts from random initialization with seed 42.
Activation checkpointing is **disabled** in B64; no checkpoint from an older run
is loaded. B128 configs and preflight logs are historical, inactive artifacts.

## Settings

Config: `configs/teacher/teacher_b64_100k.json`.

- Microbatch 64, accumulation 1, **100,000 optimizer updates**; no update reduction.
- Initial LR 2e-4, linear decay to zero over 100,000 updates, no warmup.
- `torch.optim.AdamW`, default betas (0.9,0.999), epsilon 1e-8, weight decay 0,
  amsgrad False. No custom optimizer or explicitly enabled fused implementation.
  At zero weight decay its update follows Adam without weight decay.
- Same 48,883,107-parameter 64×64 U-Net, bf16, epsilon MSE, gradient clip 1,
  class dropout 0.1, and 1,000-step linear diffusion noise schedule.
- Existing mild augmentation: horizontal flip 0.5, rotation ±5°, translation ±2%,
  scale 0.98–1.02, bilinear interpolation and white fill.
- Primary EMA 0.9999 plus EMA 0.999 updated from the **same optimizer trajectory**.
  Extra tracking adds weights only, no second training network or loss.
- Diagnostic samples use clipped DDPM-50 / CFG-1, seed 42.

The larger batch and longer run imply roughly 6.4 million image exposures;
actual exposures are recorded (partial final epoch batches are retained).
This is a new baseline, not an isolated comparison with the former batch-16 run.

## Training-only development split

Before initialization, `teacher.splits` deterministically separated 5 images from
each of 151 classes, using seed 42, from **official training paths only**.

- Optimizer training: 16,324 images, all 151 classes.
- Unseen development holdout: 755 images, exactly 5 per class.
- Original 17,079 training paths and 3,020 validation paths remain unchanged.
- Manifests and source hashes: `data/development/seed42_holdout5/`.
- Manifest adapter checks uniqueness and membership in official training.
  Checkpoints record training-path hash; resume rejects a changed training split.

The holdout is omitted from all optimizer updates and is available for CFG,
sampler and EMA development comparisons. It is not official validation. A FID
reference of 755 images is small and produces biased/noisy scores; compare only
methods sharing this exact development split, and do not compare its numerical
scale directly with official 3,020-reference FID.

## Ordered experiments and logging

1. **Fresh baseline training:** active B64/100K run. Ordinary samples every 1,000
   updates; every 5,000, paired raw/EMA-0.999/EMA-0.9999 × DDIM/DDPM/clipping grids
   and a small CFG check are written. Identical class labels and initial seeds
   accompany metadata. No official validation is used during training.
2. **Sampler and EMA comparisons after training:** the launcher automatically
   runs `scripts/review_teacher_run.py`. It reports 7 raw/EMA/sampler settings
   plus CFG scales 0,0.5,2,3,4,6 on the fixed training holdout. Scale 1 is already
   included. Every setting has 20 generated images/class and records NFE, weights,
   seed, solver and clipping. Intermediate `development_comparison.json` is
   saved after each evaluation. These do not select on official validation.
3. **Next, after inspecting comparison evidence:** a small-set memorization
   check to separate learning/implementation problems from full-dataset problems.
4. **Then controlled training ablations:** augmentation, LR decay, and diffusion
   noise schedule, one variable at a time, on the same development split.
   Full teacher comparison runs retain batch 64 and 100,000 updates as requested.
   No architecture search or distillation is introduced.

Stages 3–4 are intentionally not launched before stages 1–2 can be inspected.
Their experiment selection depends on the preceding evidence. The first two
stages execute in order through the current tmux launcher.

```bash
cd /root/workspace_distillation
tmux new-session -d -s fastgen_teacher_b64 'bash scripts/train_teacher_b64.sh'
tmux attach -t fastgen_teacher_b64
tail -f experiments/teacher_b64_100k/console.log
```

All stdout/stderr is captured. `train.log` records every optimizer step, actual
LR, loss, gradient norm, images seen and peak allocated VRAM. `metrics.json`
records optimizer, split and EMA metadata. Every 500 steps, `latest.pt` stores
raw weights, both EMAs, optimizer/scaler, RNG and data cursor; `final.pt` is saved
at 100,000. Checkpoint loading accepts `--weights model`, `ema`, or `ema:0.999`.
The script resumes only its own latest checkpoint if restarted.

## Preflight history

- B128 + primary EMA alone passed with peak 16.95 GiB.
- B128 + two EMAs without checkpointing failed at backward due to memory limits.
- The vGPU rejected allocator expandable-segments mode; it is not enabled.
- Selective activation checkpointing at 64×64 was implemented and verified to
  preserve forward output and all parameter gradients on trained weights.
- B128 with selective checkpointing and two EMAs passed three real-data updates,
  checkpoint resume and sampler diagnostics at peak 11.68 GiB. This option is
  available but **not used** in the active B64 run, per the user's correction.
- B64 final preflight results are saved under its run directory. Temporary probe
  checkpoints and images are removed automatically; only logs/metrics persist.
  Three updates passed with finite losses 0.998045, 0.989116 and 0.977592;
  peak allocated VRAM was **9,777,104,896 bytes (9.1056 GiB)** with both EMAs
  and no activation checkpointing. The DDPM sample and full checkpoint were saved.

Protected dataset/evaluator/base Model files and upstream dependencies are unchanged.
Source snapshot and copied config accompany the active run.
