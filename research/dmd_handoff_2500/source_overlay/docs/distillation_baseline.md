## Completed result

All stages and protected official evaluation finished with exit code 0. Four-NFE official FID **33.0517**; one-NFE official FID **61.4540**. See [final results, image sheets and diagnosis](distillation_results_review.md).

## Live result update (2026-09-30)

The corrected 42/1042/2042 sampler experiment completed (exit 0): development
FID means 46.05883 at 20 steps, 44.08437 at 50, 44.05665 at 100.

The no-augmentation teacher completed 100,000 updates (exit 0), with 16,613.09
seconds training and 9,777,302,016 bytes peak allocated VRAM. The automatic
holdout comparison also completed (exit 0). On the fixed 755-image holdout,
its primary-EMA clipped DDIM50 CFG2 FID is **43.57209**, compared with **44.36898**
for the flip-only model. This is why the selection policy chose no-augmentation
for distillation. It does not imply no augmentation wins at all scales: DDIM
CFG1 worsened (62.97505 vs 55.05821).

Selected no-augmentation EMA DDIM32 CFG2: development FID **44.11891** and
frozen official FID **26.65365** (64 teacher NFE). Its single-forward fixed-CFG
student finished 50,000 guidance updates and has development FID **43.25620** at
32 NFE; it passed the predeclared quality gate. Stage n16 is training. No
n16/4/1 full-run FIDs are available yet. The unchanged official 1-/4-NFE
submission evaluator is queued only after corresponding trained checkpoints.

## Seed-repeat correction

The completed original seed-42/43/44 comparison had overlapping batch RNG streams. Its scores remain valid but are not independent repeats. The script now uses disjoint ranges from seeds 42/1042/2042; corrected sampling repeats are queued after the current no-augmentation teacher and before distillation. See [analysis and preserved original results](sampling_steps_seeds_analysis.md).

# Fixed-CFG and progressive distillation baseline

## Authorized scope and execution order

The user expanded the earlier teacher-only scope to: sampling robustness
experiment 2, then no-augmentation experiment 3, then 1-/few-NFE distillation
FID. CODEX.md records this authorization and the constant-LR policy.

1. `scripts/compare_teacher_steps_seeds.py`: frozen flip-only primary EMA,
   clipped DDIM CFG 2, steps 20/50/100, seeds 42/43/44, 3,020 generated images
   per setting against 755 held-out official-training images. Existing 50-step
   seed-42 results are reused. A separate 32-step seed-42 result anchors the
   power-of-two progressive grid. Report mean and sample std across seeds;
   these measure sampling variability, not training-seed variability.
2. Fresh `teacher_b64_noaug_constant_100k`: exactly the flip-only config with
   horizontal flip probability 0.5 -> 0; no affine transforms; batch 64,
   accumulation 1, 100K updates, constant LR 2e-4, no warmup, same manifest,
   optimizer/architecture/noise schedule/EMA/seed. The existing 15-setting
   holdout comparison runs after training.
3. Select flip-only or no-augmentation teacher using the final primary-EMA
   DDIM-50 CFG 2 holdout FID. Selection is recorded and frozen before any
   official reporting. The selected teacher's 32-step holdout and official
   scores are recorded separately to match the student starting grid.
4. Fixed-CFG transfer, then progressive halving, with 4- and 1-NFE exports
   and official evaluation through unchanged evaluate.py.

Launch/restart: `bash scripts/run_teacher_to_students.sh` in tmux.
The currently running session is `fastgen_teacher_to_students`; its active
first phase was started before student implementation and its queued phases
use the same scripts. The implementation-ready marker is written only after
smoke verification. Failure in teacher execution or guidance quality stops
later dependent stages and leaves logs/results for inspection.

## Method and sources

Baseline follows fixed-guidance transfer and step halving, motivated by
[Meng et al., CVPR 2023](https://arxiv.org/abs/2210.03142) and
[Salimans & Ho, ICLR 2022](https://arxiv.org/abs/2202.00512).
It is a discrete-schedule adaptation, not a claim to reproduce all paper
hyperparameters. No consistency model, variable guidance embedding, LoRA,
curriculum, CTM, DMD2 or architecture search is included.

The student's U-Net is the same 48,883,107-parameter architecture as the teacher.
Teacher backbone features initialize the first student; the epsilon output head
is reset to zero because its output is now interpreted as v. Subsequent students
initialize all weights from the previous stage's EMA. No external pretrained
weights are used. Class dropout is not used for students: CFG 2 is fixed and
absorbed into a single conditional forward; the null embedding remains counted
as part of the inherited architecture.

For `a=sqrt(alpha_bar)` and `s=sqrt(1-alpha_bar)`:

- Teacher: epsilon -> clipped clean estimate, using the shared `apply_cfg`.
- Student: `v=a*epsilon-s*x0`; clean estimate `x0=a*x_t-s*v`.
- Guidance transfer supervises the clipped teacher clean prediction at sampled
  grid nodes. Loss is truncated-SNR clean-image error, implemented equivalently
  as `max(a²,s²)*MSE(v_prediction,v_target)` for stable gradients.
- Progressive training takes two frozen teacher DDIM transitions and inverts
  the one-transition equation to form the effective target clean prediction.
  Every train/sample stage uses exactly the same nested nodes.
- The full starting grid is the existing rounded 32-node sequence from 999 to 0,
  followed by the exact clean endpoint represented by -1. Every stage uses
  every second preceding node: 32 -> 16 -> 8 -> 4 -> 2 -> 1.
- Guidance-stage predictions clip to [-1,1] to match the selected teacher.
  Progressive effective targets and intermediate student clean predictions
  are *not* clipped: clamping would invalidate the two-to-one endpoint mapping.
  Final exported RGB images clip to [-1,1].
- For the highest node, training input is pure noise as at inference. The
  inherited alpha at t=999 remains nonzero (~0.00635); unlike the paper we do
  not retrofit a zero-terminal-SNR schedule onto the existing trained teacher.
  This approximation and changing the output head may limit final quality.

## Training configuration

`configs/distillation/fixed_cfg2_progressive.json`:

| Setting | Value |
| --- | --- |
| Batch / accumulation | 64 / 1 |
| Guidance transfer updates | 50,000 |
| Updates for each of 16/8/4/2/1 stages | 25,000 |
| Total student updates | 175,000 |
| LR | 2e-4 constant, no warmup/decay |
| Optimizer | AdamW, betas (0.9,0.999), eps 1e-8, weight decay 0 |
| Student EMA | 0.999 |
| Grad clipping | 1 |
| Prediction / loss | v / truncated SNR |
| Precision | bf16 |
| Seed | 42 |
| Checkpoint / preview interval | 500 / 1,000 |

EMA 0.999 reduces retention of the initial reset head during shorter stages.
The selected teacher's training split, augmentation and diffusion noise schedule
are inherited. Full source/config snapshots, per-update loss/LR/grad/VRAM and
optimizer/RNG/epoch/cursor states are saved. Resume validates settings, teacher
path, image manifest hash and update budget. The first guidance-transfer holdout
FID must be no more than teacher32 FID + 15 before step halving is allowed.
This is a declared prototype quality guard, not a benchmark success threshold.
If it fails, stop and investigate transfer rather than distill its poor target.

## Submission interface

Only ModelOneNFE/ModelFewNFE bodies in model.py were implemented. The protected
base Model is byte-for-byte unchanged. One-NFE uses the trained n1 EMA; Few-NFE
uses the trained n4 EMA. Each loads one backbone and fixed schedule buffers;
there is no teacher, auxiliary learned network or runtime CFG. The user-facing
sample signature and base checkpoint-loading/parameter counting are preserved.
Exports are `checkpoints/one_nfe.ckpt` and `checkpoints/few_nfe.ckpt`, with the
required state_dict field and provenance. The pipeline refuses to replace an
unrelated existing submission checkpoint.

Each final checkpoint is evaluated once through protected `evaluate.py`, 20
images/class, 3,020 generated/reference images. The pipeline captures stdout
and writes a structured FID report. Sampling seeds are implemented inside the
student wrappers as seed 42 plus batch index, matching development batch 16.
Few-NFE here means exactly 4; the intermediate n2 checkpoint is also retained
and development-scored, but arbitrary 3-step sampling is not advertised.

## Actual smoke results

`experiments/distillation_smoke/results.json` and
`experiments/distillation_smoke_console.log`:

- Real dataset batch 64; guidance and all five halving stages each trained
  3 updates with finite loss/backward, bf16 and AdamW.
- Parameter count 48,883,107 for both exported models (<60M).
- Peak allocated training VRAM 9,791,668,224 bytes (~9.12 GiB), below 20GB.
- Two-to-one endpoint inversion: float64 max error 2.22e-16; actual nested
  FP32 grid max error 4.77e-7. Velocity/loss identities passed.
- Checkpoint save/load and completed-checkpoint resume preserved weights/step.
  Interrupted mid-epoch continuation has not yet been exercised separately.
- Export through base Model.load_checkpoint passed for both wrappers.
- Actual backbone forward hooks measured exactly 1 and 4 calls per sample.
- Same explicit seed returned identical finite RGB64 outputs in [-1,1].
- Base Model prefix unchanged; protected evaluator/data/utils/requirements diff clean.

These short smoke checkpoints are kept in the experiment directory and are
not submitted or used as production students. They establish execution and
arithmetic, not image quality or final FID.

## Logs and commands

```bash
cd /root/workspace_distillation
tmux attach -t fastgen_teacher_to_students
# Phase 2:
tail -f experiments/teacher_steps_seed_console.log
# Phase 3:
tail -f experiments/teacher_b64_noaug_constant_100k/console.log
# Distillation:
tail -f experiments/fixed_cfg2_progressive/console.log
```

Final student comparison: `experiments/fixed_cfg2_progressive/comparison.json`.
Each stage has train.log/metrics.json, latest.pt/final.pt, samples and its
holdout evaluation; n4/n1 also have official_evaluation/metrics.json.
Training/evaluation is queued on the single GPU, and may take many hours.
No final student FID is available at implementation time.
