# B64/100K results and the next experiments

## Completed baseline

`teacher_b64_100k` completed 100,000 optimizer updates with exit code 0.
Training cost: 16,676.91 seconds (~4h38m), 6,376,600 image exposures.
Peak allocated VRAM: 9,777,302,016 bytes (~9.106 GiB).
Mean loss was 0.05465 in updates 1–1,000, 0.01655 in 49,001–50,000,
and 0.01560 in 99,001–100,000. Parameter count: 48,883,107.

Official validation reporting, 3,020 generated images / 3,020 reference images,
seed 42, batch 16, EMA 0.9999, clipped DDPM-50, bf16 backbone:

| Guidance scale | NFE | Official FID |
| --- | ---: | ---: |
| 1 | 50 | **60.60930082682705** |
| 2 | 100 | **32.98045168862393** |

Scale 1 was the frozen baseline. Scale 2 was chosen using the *training-only*
development sweep before its official reporting, not by validation tuning.
Already generated development-evaluation images were hard-linked into separate
official reporting directories and scored against official reference images.
Generation settings and counts were verified; no regeneration or retraining was
needed. Original development reports and PNGs remain unchanged.
Reports/logs are under `official_fid_ddpm50/` and `official_fid_ddpm50_cfg2/`.
Neither result is an eligible 1/4-NFE student submission.

## Training-holdout observations

The automatic comparison completed all 13 settings with exit code 0.
Its reference set is 755 unseen images drawn only from official training.
Small-reference FID has substantial finite-sample bias; these values are for
comparisons within the same split, not directly against official FID.

| Settings | Development FID |
| --- | ---: |
| Raw / clipped DDIM / scale 1 | 63.06 |
| EMA 0.9999 / clipped DDIM / scale 1 | 61.59 |
| EMA 0.999 / clipped DDIM / scale 1 | 63.16 |
| Raw / clipped DDPM / scale 1 | 67.92 |
| EMA 0.9999 / clipped DDPM / scale 1 | 67.24 |
| EMA 0.999 / clipped DDPM / scale 1 | 67.91 |
| EMA 0.9999 / original DDIM / scale 1 | 61.49 |
| EMA 0.9999 / clipped DDPM / scale 0 | 77.57 |
| Same, scale 0.5 | 86.55 |
| Same, scale 2 | 50.54 |
| Same, scale 3 | 52.63 |
| Same, scale 4 | 58.22 |
| Same, scale 6 | 66.20 |

EMA 0.9999 was slightly better than the other tracks here. The old 50K run's
severe DDIM regression did not persist in this new 100K run: DDIM is competitive
on the holdout. This weakens a claim of a universally broken DDIM implementation
and supports investigating learned-score accuracy, training duration and CFG.
Small score differences warrant further seeds rather than a universal ranking.

## New persistent LR instruction

The user requested **no LR schedule in future experiments**. New configs use
`learning_rate_schedule="constant"`, `warmup_steps=0`, LR 2e-4.
`CODEX.md` records this persistent override. Historical linear-LR configs,
completed checkpoints and their resume behavior remain reproducible.
The diffusion **noise** schedule stays linear; it is distinct from the LR policy.

## Stage 3: 64-image memorization sanity test

Config: `configs/teacher/teacher_memorization64_constant.json`.

- Fresh random initialization, seed 42, same architecture and 151-label mapping.
- 16 seen-training images each for Abra, Charizard, Pikachu and Tauros (64 total),
  selected deterministically from the *optimizer training* manifest. No internal
  holdout or official validation image is used.
- Batch 64, accumulation 1, **2,000 diagnostic updates**, LR 2e-4 constant.
  This short sanity test is separate from the 100,000-update full-data experiment.
- No augmentation. Primary EMA 0.99 and extra EMA 0.999 are tracked; raw weights
  are also inspected. Short-run EMA settings avoid relying on a largely stale
  0.9999 average after only 2K updates.
- Training images are cached in CPU RAM after deterministic preprocessing; all
  64 cached tensors and labels were verified equal to uncached data. Noise and
  timesteps remain freshly sampled. Caching is rejected with augmentation.
- Grids every 250 updates, paired sampler/EMA/CFG checks every 500, and a final
  training-only denoising/oracle audit. No official FID is used for this subset.

This checks basic small-set learning and denoising. Finite decreasing loss alone
does not prove memorization; inspect reconstructions, generated class cues and
noise-level errors. A weak result after 2K updates is not sufficient evidence of
an algorithm bug without considering optimization duration and the audit results.

## Stage 4: full-data constant-LR comparison

After the sanity training and audit exit successfully, the launcher starts
`configs/teacher/teacher_b64_constant_100k.json` from random initialization.
It preserves the previous full-data model, seed, exact development split,
batch 64, 100,000 updates, augmentation, both EMAs, and sampling diagnostics.
**Only the LR policy changes**: linear decay is replaced by constant 2e-4.
It does not load the memorization checkpoint. The end-of-run training-holdout
comparison runs automatically again. No additional full-run hyperparameter
change or distillation is queued.

```bash
cd /root/workspace_distillation
tmux new-session -d -s fastgen_teacher_next 'bash scripts/run_teacher_next_experiments.sh'
tmux attach -t fastgen_teacher_next
tail -f experiments/teacher_memorization64_constant/console.log
# Once the short diagnostic completes:
tail -f experiments/teacher_b64_constant_100k/console.log
```

All stdout/stderr, per-update loss/LR, configs/source snapshots, checkpoints,
measurements and diagnostic images are logged under the respective run directory.

## Memorization sanity result (2000 updates)

Training and audit completed with exit code 0. Mean epsilon loss decreased
from 0.27055 (first 100 updates) to 0.00894 (last 100); the single final
update loss was 0.00424. Seen-image reconstruction at t=500 preserves the
training images clearly, with residual noise (EMA x0 MSE 0.01048).
At t=100, correct-label epsilon MSE was 0.00676 versus 0.08329 with wrong
labels, supporting that class conditioning is learned. The sampler oracle
matched the independent reference, max absolute error 3.73e-9.

Visual inspection: clipped DDIM-50/CFG-2 yields recognizable class cues but
several samples remain noisy or malformed. DDPM-1000/CFG-1 produces recognizable
Abra images; that particular grid covers Abra only. Thus basic learning,
conditioning and arithmetic checks pass, while complete dataset memorization
and consistently clean random-noise generation have not been demonstrated.
No nearest-neighbor matching or generalization claim is made.
