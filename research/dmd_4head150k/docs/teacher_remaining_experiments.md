# Remaining teacher experiments and the diversity question

## Completed

- B64/100K teacher, linear vs constant LR (future LR remains constant).
- Primary/secondary EMA and raw-weight comparisons.
- DDIM/DDPM-50, clipping and CFG comparisons including missing DDIM CFG 2/3.
- Small-set 64-image/2K-update learning and oracle audit.
- Full-data affine+flip vs flip-only at fixed LR, split, architecture and updates.
- Batch/VRAM throughput probes; B128 OOM, user-selected B64 retained.

Latest holdout best among tested settings: flip-only primary EMA, clipped
DDIM-50 CFG 2, FID 44.36898. Constant-LR affine counterpart: 48.40064.
These scores use 755 training-only reference images and are not official FID.

## FID and augmentation

FID compares the means and covariances of Inception features for real and
synthetic image distributions. A lower distance can reflect better realism,
coverage, or both; it does not separately identify fidelity vs diversity and
is not an objective to maximize raw variation or feature variance.
[Original FID paper/code](https://github.com/bioinf-jku/TTUR),
[separate precision/recall evaluation](https://arxiv.org/abs/1904.06991).

Augmentation can reduce overfitting and improve useful distribution coverage.
It can also move the training target away from the unaugmented evaluation
distribution or alter thin lines through interpolation. Thus increasing its
strength is a hypothesis to test, not a consequence of FID measuring diversity.
Current flip-only already retains p=0.5 horizontal flipping. Its better holdout
FID than affine+flip does not prove diversity improved or affine is always bad.
One training seed and only 20 generation samples/class limit conclusions.

## Recommended next order (not launched by this reporting task)

1. **Separate quality/coverage and memorization diagnostics, no retraining:**
   fixed checkpoint generation; precision/recall or density/coverage using
   evaluation-only features, per-class grids and nearest seen-training neighbors.
   Evaluate against the training-only holdout for development. Detect repetition,
   lost coverage, and apparent training-image copying. Small sample/class sizes
   make per-class statistics unstable, so avoid strong per-class FID rankings.
2. **Sampling step/seed robustness, no retraining:** compare DDIM CFG 2 at
   20/50/100 steps and several sampling seeds on the holdout. Separate sampler
   approximation error and Monte Carlo variation from teacher learning quality.
   Cost is 40/100/200 NFE respectively under the existing two-forward CFG.
3. **Flip-only vs no augmentation:** fresh B64/100K, exact split/optimizer/LR/
   EMA/architecture, only flip probability changes 0.5 to 0. Tests whether
   remaining flip regularization improves coverage/generalization. Keep all
   reported prior results; tune on training-only holdout, not validation.
4. **Repeat key training comparison with another initialization seed:** test
   stability of affine vs flip-only while keeping the predeclared manifest
   fixed. Several sampling seeds alone do not estimate training variability.
5. **Linear vs cosine diffusion noise schedule (optional next learning axis):**
   fresh teacher, identical budgets and constant LR. Cosine is not implemented
   in current teacher; it requires consistently replacing train/sample schedule
   construction and verifying it. First use noise-bin diagnostics to decide
   whether schedule/score learning is the likely bottleneck. Do not change LR,
   augmentation, prediction parameterization or architecture simultaneously.

No architecture search or further batch enlargement is required for this
teacher diagnosis. Distillation/consistency/LoRA/etc remain outside the current
Phase 0–1 scope; later project stages need a separate scope expansion.

## Official DDIM report

The selected latest flip-only EMA DDIM-50 CFG 2 setting is frozen before
reporting. Existing 3,020 generated PNGs are hard-linked to
`experiments/teacher_b64_fliponly_constant_100k/official_fid_ddim50_cfg2/generated`
and compared with all 3,020 official validation references through the unchanged
FastGen evaluator. This is a single report, not a validation hyperparameter sweep.
Log and exit status are in that directory. Metadata records checkpoint, seed,
batch, precision, clipping, EMA and 100 NFE/image. No retraining is launched here.

Official scoring completed with exit code 0: **FID 25.12627732**, 3,020 generated/reference images, seed 42, EMA 0.9999, clipped DDIM-50 CFG 2, 100 NFE. Previous affine constant-LR DDPM-50 CFG 2 official FID was 31.50218; this comparison changes both augmentation and sampler, so cannot attribute its difference to either factor alone.

## Authorized progression

The user has now requested experiment 2, then experiment 3, then 1-/few-NFE distillation. Earlier teacher-only scope is expanded for this sequence. See [distillation baseline and smoke results](distillation_baseline.md). The sequential workflow is running in tmux `fastgen_teacher_to_students`.
