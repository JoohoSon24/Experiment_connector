# Sampling defaults and missing guided-DDIM comparisons

The completed configs' `sample_sampler=ddpm` and `guidance_scale=1` control
training-time preview generation, not the epsilon-MSE training objective or
which checkpoint weights are optimized. Evaluation CLI settings override preview
settings. `guidance_scale=1` is plain conditional prediction; zero is unconditional.

DDPM became the preview sampler after the historical 50K checkpoint produced
FID 189.94 with original DDIM-50 and 82.99 with clipped DDPM-50. Those differed
in clipping/precision as well as sampler, so they were operational evidence to
use that DDPM setup, not an isolated proof that DDPM is intrinsically better.
CFG 1 remained the baseline preview to observe learned conditional behavior.
Subsequent CFG 2 results supported more useful guided quality previews; the
original preview alone does not represent the model's strongest available output.

The diagnostics originally swept CFG 0/2/3 only for DDPM. DDIM CFG 2/3 being
absent was a comparison omission, not a model or algorithm restriction.

Correction:
- Future `training_diagnostics` generates primary-EMA clipped DDIM and DDPM
  at CFG 0/1/2/3, retaining raw and alternate-EMA CFG 1 comparisons.
- `scripts/review_teacher_run.py` adds primary-EMA clipped DDIM-50 CFG 2/3
  to the existing holdout matrix (15 settings in total).
- `scripts/supplement_ddim_diagnostics.py` appended the missing DDIM CFG 2/3
  grids and metadata to both completed constant-LR runs at their final step;
  existing images, checkpoint configs and source snapshots were preserved.
- Four actual grid generations passed shape/finite-value/NFE assertions;
  at 50 steps and CFG 2/3 each sampler costs 100 backbone evaluations.
- The remaining full holdout comparisons run sequentially in tmux
  `fastgen_ddim_cfg_review`; stdout/stderr is recorded in
  `experiments/sampler_cfg_review/console.log`, final status in `exit_code`.
  Completed settings are reused, so no retraining or existing PNG replacement.

Before the guided-DDIM results are available, the matched CFG 1 primary-EMA
clipped holdout scores are:

| Checkpoint | DDIM-50 | DDPM-50 |
| --- | ---: | ---: |
| Constant LR with affine | 59.20 | 61.87 |
| Constant LR, flip only | 55.06 | 52.69 |

The winner differs by checkpoint. DDIM's fast-sampling advantages do not
establish that it wins every equal-step comparison against our respaced DDPM.
Guided scales should be compared with identical weights, clipping, seed,
batch size and sample counts; deterministic DDIM and stochastic DDPM cannot
share every subsequent noise draw. Selection stays on the 755-image training
holdout. No additional official validation sweep is used for tuning.

References: [DDIM paper](https://arxiv.org/abs/2010.02502),
[CFG paper](https://arxiv.org/abs/2207.12598).
CFG trades fidelity against diversity; stronger guidance is not monotonically
better. Our holdout DDPM scores favor CFG 2 over CFG 3 for both completed runs.
