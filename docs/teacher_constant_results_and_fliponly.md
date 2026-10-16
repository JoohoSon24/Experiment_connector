# Constant-LR teacher results and flip-only ablation (2026-09-29)

## Completed experiments

`teacher_b64_constant_100k`: training and 13-setting development comparison both
exited with code 0. Completed 100,000 updates from fresh initialization; batch
64, 6,376,600 image exposures, elapsed 16,681.62 seconds (~4h38m).
Parameters: 48,883,107. Peak allocated training VRAM: 9,777,302,016 bytes (9.106 GiB).
AdamW, betas (0.9, 0.999), eps 1e-8, weight decay 0; bf16; fixed LR 2e-4,
no warmup/decay; primary EMA 0.9999 and secondary EMA 0.999.

### Official reporting

All rows use 3,020 generated images and 3,020 official validation references,
seed 42, batch 16, primary EMA, DDPM 50 steps, clipped x0 and bf16 backbone.
Official validation was used only to report the frozen baseline and CFG 2
selected on the predeclared training-only holdout. Existing generated PNGs were
hard-linked into separate official-report folders; original development results
were preserved. `scripts/report_teacher_official.py` records reuse provenance.

| Training LR policy | CFG | NFE | Official FID |
| --- | ---: | ---: | ---: |
| Previous linear decay | 1 | 50 | 60.60930 |
| Constant 2e-4 | 1 | 50 | **52.62624** |
| Previous linear decay | 2 | 100 | 32.98045 |
| Constant 2e-4 | 2 | 100 | **31.50218** |

Differences: -7.98306 at CFG 1, -1.47827 at CFG 2. One training and sampling seed
is insufficient to establish significance or a universal ranking. These teachers
remain outside the 1/4-NFE submission interface.

### Training-only holdout results

The reference contains 755 official-training images excluded from optimizer
updates in both compared runs. Its FID is biased by its small reference size;
compare within this split, not against official FID.

| Primary EMA setting | Previous linear LR | Constant LR |
| --- | ---: | ---: |
| Clipped DDIM 50, CFG 1 | 61.59 | 59.20 |
| Clipped DDPM 50, CFG 1 | 67.24 | 61.87 |
| Clipped DDPM 50, CFG 2 | 50.54 | 51.86 |
| Clipped DDPM 50, CFG 3 | 52.63 | 56.85 |

Constant LR improved unguided conditional generation on this development set;
CFG 2 retained the best DDPM guidance value but its holdout FID worsened slightly.
This differs from official reporting and argues against interpreting the small
CFG 2 improvement as conclusive. Raw DDPM FID 78.15 vs primary EMA 61.87 and
raw clipped DDIM 86.83 vs EMA 59.20 indicate strong benefit from averaging under
constant LR. This is compatible with optimizer fluctuations, not proof of a bug.

## Qualitative review

Inspected paired final diagnostic grids and a fixed 12-class x 4-sample contact
sheet from the actual FID generation folders, at CFG 2. Selected classes:
Abra, Bulbasaur, Charizard, Dragonair, Eevee, Gastly, Jynx, Mew, Pikachu,
Psyduck, Spearow, Tauros. Class identities, colors and silhouettes are mostly
recognizable. Bulbasaur and Pikachu are fairly coherent; Charizard's wings/limbs,
Tauros's facial details and some Mew samples remain malformed or soft. Some
images have muddy backgrounds or strong color saturation. The constant-LR set
has mixed per-image changes rather than a universal visible improvement.
Repeated frontal poses occur, but 4 images/class cannot quantify mode collapse
or prove memorization. Resizing for the contact sheet uses nearest-neighbor;
original generated images are 64x64.

Artifacts:
- `experiments/teacher_b64_constant_100k/quality_review/teacher_b64_constant_100k_cfg2.png`
- `experiments/teacher_b64_constant_100k/quality_review/teacher_b64_100k_cfg2.png`
- `official_fid_ddpm50/metrics.json` and `official_fid_ddpm50_cfg2/metrics.json`
  within the constant-LR experiment directory.

## Small-set sanity audit

The preceding 64-image, 2,000-update memorization run and audit both exited 0.
Its DDPM 1,000-step grid can generate recognizable class-specific examples.
Seen-data conditional epsilon MSE at t=250: 0.00203 vs wrong-class 0.02686;
at t=500: 0.000884 vs wrong-class 0.01130. Oracle inversion max absolute
error: 3.7253e-9, independent reference check passed. These support functioning
conditioning and diffusion arithmetic. They do not establish generalization,
prove exact training-image memorization or exclude every implementation error.
High-noise x0 errors remain large; epsilon MSE alone cannot certify sample quality.
Full results: `experiments/teacher_memorization64_constant/audit/results.json`.

## Next experiment: remove affine, preserve horizontal flips

Config: `configs/teacher/teacher_b64_fliponly_constant_100k.json`.
Run: `experiments/teacher_b64_fliponly_constant_100k`.

**Method:** fresh initialization, same seed 42 and architecture; same 16,324
training images and 755 held-out training images; batch 64, accumulation 1;
100,000 updates, LR 2e-4 constant/no warmup, same EMA tracks/optimizer/noise
schedule/objective/sampling. Preserve horizontal flip probability 0.5.
Remove the affine pipeline as one ablation factor: rotation +/-5 degrees,
translation +/-2%, scale +/-2% all become zero.

**Purpose:** isolate whether bilinear resampling and unconditioned geometric
augmentation contribute to soft edges and distribution mismatch. The current
pipeline resizes to 64x64 before affine transforms, so resampling can modify
already-small line details. This is a hypothesis, not an established cause.

**Expected benefit:** sharper edges and better fidelity to original training
images while keeping horizontal-flip regularization. Possible cost: less
geometric diversity and more overfitting. Improvement is not guaranteed.
EDM's official implementation explicitly supplies augmentation labels to its
network when applying augmentations; our baseline does not condition on affine
parameters. That motivates inspecting augmentation as a factor, rather than
assuming the architectures have equivalent augmentation behavior. We are not
implementing EDM augmentation conditioning or changing diffusion formulation:
[EDM official loss implementation](https://github.com/NVlabs/edm/blob/main/training/loss.py).

**Decision:** compare final primary-EMA DDPM 50 at frozen CFG 1/2, plus the same
13-setting training-holdout matrix and paired diagnostic grids. Inspect
quality/diversity together. Keep constant LR per user instruction regardless of
which historical schedule scored better. A seed repeat is needed before making
strong causal claims. Official validation remains reporting-only.

### Execution and validation

CPU preflight passed: real dataset batch [64,3,64,64], labels within 0..150,
pixels [-1,1], 16,324 training images; config diff checked to allow only experiment
name and affine values, fixed LR/no warmup and 100,000 updates verified.
Results recorded in `docs/teacher_fliponly_preflight.json`.
Runner bash syntax and official-report Python compilation passed. Protected
upstream files remained unchanged. The first actual optimizer updates are the
GPU execution check using the unchanged training code.

```bash
cd /root/workspace_distillation
tmux attach -t fastgen_teacher_fliponly
tail -f experiments/teacher_b64_fliponly_constant_100k/console.log
# Per-update loss, LR, gradient norm and peak allocated VRAM:
tail -f experiments/teacher_b64_fliponly_constant_100k/train.log
```

The tmux run started after official scoring released the GPU. It saves config,
source snapshots, both EMAs, resume checkpoints and samples, then automatically
runs the training-holdout comparisons. Expected training time is roughly 4h38m
based on the previous run, followed by evaluation; actual duration may differ.

Startup confirmed at update 510: finite loss 0.025255, LR 0.0002, peak allocated VRAM 9.106 GiB. No OOM or non-finite-loss error observed at this check.
