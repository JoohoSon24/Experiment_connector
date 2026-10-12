# Review of the completed 50,000-update teacher

## Frozen baseline and evaluation

Run: `teacher_lr2e4_linear_aug_50k`, random seed 42, 48,883,107 parameters.
Training completed all 50,000 updates with exit code 0 in 11,261.44 seconds
(3h7m41s), using 799,586 training-image exposures (~46.8 epochs).
Peak allocated training VRAM: 2,227,817,472 bytes (2.0748 GiB).
All 17,079 training paths are disjoint from the 3,020 validation paths.

The frozen original EMA / DDIM-50 / CFG-1 / bf16 configuration achieved
**FID 189.93626140473657**, with 20 images per class (3,020 total), seed 42,
evaluation batch size 16, and all 3,020 official validation references.
This is a teacher reporting score at 50 NFE, not an eligible student submission.
Artifacts: `experiments/teacher_lr2e4_linear_aug_50k/fid/{metrics.json,console.log,generated/}`.

A second frozen setting, selected from the **training-only sampler comparisons**,
was then evaluated once: EMA / respaced DDPM-50 / intermediate x0 clipping /
FP32 scheduler arithmetic / CFG-1 / bf16 backbone. It achieved
**FID 82.98979405743563** with the same checkpoint, seed, batch size, sample count,
reference set, and **50 NFE**. No retraining was performed. This is a combined
sampler/clipping/numerics change, not a numeric FID ablation of those components
individually. It demonstrates that the original DDIM score was strongly affected
by the inference configuration, not just the training hyperparameters.
The remaining score and visual defects do not establish a sufficiently strong
teacher for distillation. Both results are retained; no validation sweep or
best-checkpoint selection was performed.

Artifacts: `experiments/teacher_lr2e4_linear_aug_50k/fid_ddpm50/` includes metrics,
generated PNGs, console log, and the evaluation teacher source snapshot.

```bash
.venv/bin/python -m teacher.sample \
  --checkpoint experiments/teacher_lr2e4_linear_aug_50k/final.pt \
  --labels 0 13 103 133 --steps 50 --guidance-scale 1 --seed 42 \
  --sampler ddpm --clip-denoised --output experiments/teacher_lr2e4_linear_aug_50k/ddpm_example.png

# Already generated all 3,020 images: recompute score without regenerating.
.venv/bin/python -m teacher.evaluate \
  --checkpoint experiments/teacher_lr2e4_linear_aug_50k/final.pt \
  --output-dir experiments/teacher_lr2e4_linear_aug_50k/fid_ddpm50 \
  --sampler ddpm --clip-denoised --steps 50 --guidance-scale 1 \
  --seed 42 --batch-size 16 --precision bf16 --score-only
```

The earlier 100,000-update constant-LR/no-augmentation run had FID 111.68.
Its artifacts were deleted at the user's request; the number is a historical
record. Duration, LR decay, and augmentation changed together, so this is not
an ablation establishing which change caused the regression. Official validation
scores are reporting-only, not hyperparameter-selection criteria.

Mean training epsilon loss:

| Updates | Mean loss |
| --- | ---: |
| 1–1,000 | 0.06128 |
| 9,001–10,000 | 0.02160 |
| 24,001–25,000 | 0.01904 |
| 49,001–50,000 | 0.01819 |

The lower training loss does not establish better generation. The distribution
of augmented images also differs from the former run's objective inputs.

## Visual evidence and sampler diagnosis

The final automatic grid contains some class-like silhouettes, but also strongly
saturated high-frequency artifacts and incoherent anatomy. Real 64×64 training
images retain substantially better outlines and detail; varied image styles and
backgrounds are present. This is more than ordinary softness at low resolution.

`scripts/diagnose_teacher.py` compared EMA/raw, 50/250/1,000 DDIM steps, CFG 1/2,
and bf16/FP32 on the same four classes and seed. More deterministic DDIM steps
and FP32 did not visibly repair the invalid images. Raw weights were often softer
than EMA and did not solve the problem globally. CFG 2 sharpened some class cues
but retained broken images. These are small qualitative observations, not global
quality measurements.

`scripts/audit_teacher.py` then compared four samples per class (Abra, Charizard,
Pikachu, Tauros), with identical initial noise and fixed seed:

- Intermediate x0 clipping bounds DDIM trajectories but does not reliably repair
  the artifacts, even at 1,000 steps.
- Ancestral DDPM sampling produces more coherent forms with much less pixel
  corruption, including at 50 steps. DDPM-50 is a respaced chain using effective
  beta `1-alpha_t/alpha_s`, not a newly trained 50-step noise schedule.
- DDPM-250/1,000 still contain malformed anatomy and inconsistent detail. The
  comparison does not establish teacher quality sufficient for distillation.
- Both raw and EMA DDPM samples retain class cues. EMA is not the only cause.

DDPM and DDIM are different sampling trajectories; stochastic DDPM draws extra
noise after the matched initial latent. A seed match does not make their entire
noise histories identical. Sixteen images from four classes do not establish an
overall FID ordering. Images and timestep traces are under this run's `audit/`.

## Code/math checks and numerical risks

- Dataset labels, null token 151, train-only loading, normalization, epsilon loss,
  class dropout, EMA update, gradient accumulation and LR decay were inspected.
  No label-map mismatch or obvious sign/indexing error was found.
- A perfect epsilon oracle reconstructs a nonconstant real-image target through
  the original sampler with max absolute error 2.33e-9. This checks DDIM endpoint
  and update arithmetic; it cannot prove correctness of the learned score.
- Conditional/wrong-class/null checks on 64 *seen* training images show label
  effects. At t=500, epsilon MSE is 0.006245 with the correct class and 0.006911
  with a wrong class (~10.7% higher). This is not a generalization evaluation.
- At t=999, alpha_bar is ~4.04e-5. Epsilon error is amplified in x0 by
  `(1-alpha_bar)/alpha_bar` in MSE. EMA epsilon MSE 0.000125 corresponds to x0
  MSE 3.107 on these images. Raw epsilon MSE is 0.0000781, x0 MSE 1.936.
  The score's high-noise accuracy and trajectory stability merit investigation;
  these seen-image reconstruction numbers alone do not prove an EMA root cause.
- EMA 0.9999 averages over roughly 10,000 updates and retains initialization
  weight `(0.9999)^50000 = 0.006736`. A shorter-run EMA ablation is justified,
  but this scalar retention does not directly predict network output error.
- bf16 CFG subtraction/addition added rounding. `apply_cfg` now promotes low
  precision predictions to FP32 before the single centralized CFG formula.
  Scale 1 bypasses CFG, so this fix does not change the frozen baseline FID.
- Float32 scalar coefficients multiplied by bf16 predictions can still produce
  bf16 results. The new DDPM/clipped-DDIM options explicitly use FP32 predictions
  for scheduler arithmetic. Legacy default DDIM arithmetic is preserved.
- Attention's `.float()` operands inside autocast do not force bmm to FP32;
  a probe returned bf16. This is a precision limitation to test, not a confirmed
  cause of the observed failure. The trained model architecture was not changed.
- The model's per-forward label validity check forces host synchronization;
  this is a throughput issue, not an explanation for invalid image content.

The new diagnostic initially kept autocast open while swapping raw/EMA weights,
which can retain cached casts. This diagnostic error was corrected by exiting
autocast before weight swaps (or disabling its cache), and affected comparisons
were rerun. The trainer and original FID evaluator already exit autocast at the
appropriate boundaries; their measurements were not affected by that mistake.

Production `teacher.diffusion.sample`, `teacher.sample`, and `teacher.evaluate`
now accept `sampler=ddpm` and `clip_denoised=True` while keeping old defaults.
The new implementation was checked against an independent sampler, with
CFG-0/null equality, deterministic repeatability at a fixed seed, NFE accounting,
and centralized FP32 CFG checks. Protected challenge files remain unchanged.
Independent-reference comparison had max error 0 for original DDIM in FP32
and for both new clipped modes with a bf16 backbone and FP32 scheduler.
Checks are saved in `audit/sampler_checks.json`.

Sources: [OpenAI improved-diffusion sampling implementation](https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py),
[DDIM](https://arxiv.org/abs/2010.02502),
[Improved DDPM, Section 3.2](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf).
The latter motivates a cosine *training noise* schedule at 64×64; this is different
from the linear *learning-rate* schedule. Do not change the noise schedule of a
checkpoint trained with linear betas and call that an ablation of the same model.

## GPU memory and throughput

`scripts/benchmark_teacher.py` uses synthetic resident images, bf16, AdamW,
gradient clipping and EMA, with three warmup and three timed updates per setting.
No trained weights are saved; data loading, augmentation and checkpoint I/O are
excluded. These are short capacity/throughput measurements, not full-run timings.

| Microbatch × accumulation | Effective batch | Peak allocated GiB | Images/sec |
| --- | ---: | ---: | ---: |
| 8 × 2 | 16 | 2.07 | 91.7 |
| 16 × 1 | 16 | 2.88 | 137.4 |
| 32 × 1 | 32 | 4.89 | 262.9 |
| 64 × 1 | 64 | 8.91 | 433.4 |
| 96 × 1 | 96 | 12.95 | 474.9 |

Batch 128 was skipped because a conservative memory estimate exceeded the margin.
Allocated VRAM is not GPU compute utilization. Filling memory is not itself a
quality objective. Batch 16 × 1 keeps effective batch 16 and yields ~1.5× greater
isolated throughput than 8 × 2. Batch 64 offers a useful capacity/throughput
starting point for a larger-batch experiment; batch 96 uses ~45% more allocated
memory than 64 for only ~10% higher throughput here.

Increasing effective batch changes images seen per update. A fair batch-16 vs
batch-64 comparison can use 50,000 vs 12,500 updates (~800,000 exposures), with
LR decay indexed by exposures. Matching EMA's image-space smoothing would use
`0.9999**4 ≈ 0.99960006` at batch 64; retaining 0.9999 would quadruple its effective
image history. No automatic linear LR scaling is assumed. Validate real augmented
data throughput before a sustained run. Extra memory can also cheaply hold
parallel EMA tracks (about 0.182 GiB per FP32 model) without changing gradients.

## Proposed experiment order

1. **Frozen-checkpoint sampler diagnosis first:** paired DDIM/DDPM, 50/250 steps,
   clipping on/off, raw/EMA, and CFG 0/1/2/3 on training-only qualitative examples.
   The current audit already supplies much of this matrix. Confirm across more
   classes/seeds before treating a visible improvement as global.
2. **EMA and noise-level diagnostics:** future training tracks raw, EMA 0.999,
   and EMA 0.9999 from the same optimizer trajectory. Record epsilon/x0 errors
   by timestep; inspect high-noise score calibration and generated saturation.
3. **Small-set memorization check:** train on 16–64 known training images with
   no augmentation, then test conditional DDPM/DDIM reconstruction and sampling.
   If even this fails, investigate implementation/optimization before full runs.
4. **Controlled training ablations:** fix sampler, architecture, seed and image
   exposure budget. Change only one of (a) affine vs flip-only vs none,
   (b) constant LR vs linear decay, or (c) linear vs cosine noise schedule.
   Changing the noise schedule requires a fresh model, with the selected schedule
   used consistently for training and sampling. Avoid architecture search.
5. **Larger-batch comparison:** start at 64 after 16×1 execution optimization;
   match image exposures and EMA smoothing, then assess class coverage and quality.

Development quantitative selection requires a predeclared training-only internal
split for future fresh runs, omitted from optimizer updates in all compared runs.
Current checkpoint saw the whole training split: its train-image diagnostics
cannot establish overfitting/generalization. Official validation cannot select
guidance, LR, augmentation, schedule or checkpoints. Report it only for frozen
settings. No long retraining or distillation was started during this review.
