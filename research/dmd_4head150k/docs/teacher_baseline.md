## Authorized next stages

Sampling step/seed robustness -> fresh no-augmentation teacher (B64/100K, constant LR) -> fixed-CFG transfer and progressive 1-/4-NFE distillation. The sequential tmux workflow is running. Student implementation passed real B64 backward, export/load and empirical NFE checks; final student FID awaits full training. See [configuration, method and smoke results](distillation_baseline.md).

## Latest official report: flip-only DDIM CFG 2

Official FID **25.12628**, EMA 0.9999, clipped DDIM 50 steps, CFG 2, 100 NFE, seed 42, 3,020 generated/reference images. See [remaining experiments and augmentation discussion](teacher_remaining_experiments.md). This remains a teacher result, outside 1/4-NFE submission constraints.

## Latest teacher review (2026-09-29)

Constant-LR B64/100K completed normally. Official DDPM-50 FID: **52.63 at CFG 1 (50 NFE)** and **31.50 at CFG 2 (100 NFE)**. The next controlled ablation removes affine transforms while retaining horizontal flips, batch 64, 100K updates and fixed LR 2e-4. See [results and experiment rationale](teacher_constant_results_and_fliponly.md).

# Class-conditional diffusion teacher baseline (Phase 1)

Latest completed B64/100K teacher: official FID **60.61 (CFG 1, 50 NFE)** and
**32.98 (training-holdout-selected CFG 2, 100 NFE)**. Future experiments use
constant LR without warmup. The memorization sanity test and subsequent
full-data constant-LR comparison are described in
[teacher_b64_results_and_next.md](/root/workspace_distillation/docs/teacher_b64_results_and_next.md).

Active fresh baseline: [batch 64 / 100,000 updates and ordered development comparisons](/root/workspace_distillation/docs/teacher_b64_100k.md).
It tracks two EMAs on one optimizer trajectory and excludes a declared
training-only development holdout. The brief B128 run was stopped at the user's
request; activation checkpointing is disabled in the active B64 config.

The 50,000-update run completed. Its original DDIM-50 / CFG-1 FID is **189.94**;
the same checkpoint with clipped DDPM-50 / CFG-1 achieved **82.99**, also at 50 NFE.
Sampler comparisons, code checks, experiment plans, and measured larger-batch
VRAM/throughput are recorded in [teacher_review_50k.md](/root/workspace_distillation/docs/teacher_review_50k.md).
Teacher sampling now also supports optional respaced DDPM and intermediate x0
clipping; existing DDIM defaults remain available. CFG arithmetic promotes low
precision predictions to FP32.

Current run: [LR 2e-4 with linear decay, augmentation, fresh 50,000 updates](/root/workspace_distillation/docs/teacher_lr2e4_linear_aug_run.md).
Previous experiment artifacts (including smoke/debug and baseline checkpoints)
were deleted at the user's request; measurements below remain historical records.

The user requested a fresh 50,000-step run with LR `5e-5` and mild augmentation.
Settings and logging are documented in [teacher_lr5e5_aug_run.md](/root/workspace_distillation/docs/teacher_lr5e5_aug_run.md).

Status: implementation and real-data smoke checks complete. A sustained 100,000-step teacher run has finished, but sample quality remains insufficient. Current diagnosis and FID reporting are documented in [teacher_diagnosis.md](/root/workspace_distillation/docs/teacher_diagnosis.md).

## Architecture and size

`teacher.model.ClassConditionalUNet` is a pixel-space U-Net with residual down/up blocks, GroupNorm, SiLU, learned class embeddings, sinusoidal timestep embeddings, FiLM-style scale/shift in each residual block, and attention at 16×16 plus the middle block. No upstream model architecture existed to reuse. `configs/teacher/initial.json` sets 96 base channels, channel multipliers `[1,2,3,4]`, two residual blocks per level, and 64×64 RGB input. `src.utils.parameter_summary` reports **48,883,107 trainable parameters**. The tiny 32×32 debug model has **775,619 parameters**. Teacher size is outside the eventual submission parameter count because it will not run in student inference.

## Data and conditioning

`teacher.data.TrainImages` uses upstream `PokemonDataset` with the provided `train_split.txt` only, the immutable `category_to_id.json` mapping, RGB conversion, resize, and normalization to `[-1,1]`. The real split has 17,079 images across 151 classes with integer labels 0–150. Class index **151** is a learned null/unconditional embedding. During training, `teacher.train.drop_classes` independently replaces each label with null at probability `class_dropout` (initial config: 0.1). No pretrained model or validation image enters training.

## Diffusion formulation and objective

This baseline predicts **epsilon**. It is the simplest direct match for standard DDPM training and deterministic DDIM sampling; the upstream repository does not favor another parameterization. For timestep `t` sampled uniformly from 0…999, a linear beta schedule (`0.0001→0.02`) defines `alpha_bar_t`. Training forms `x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon` and minimizes mean squared error between network output and sampled `epsilon`. The same `teacher.diffusion.Diffusion` owns the schedule for training and sampling. DDIM uses `eta=0`; only the final image is clipped to `[-1,1]`.

## CFG and sampling NFE

`teacher.cfg.apply_cfg(uncond, cond, guidance_scale)` is the single definition of `uncond + scale × (cond − uncond)`. The sampler calls the null and class branches at general scales. At scale 0 it calls only null; at scale 1 only class. Thus a 50-step teacher sample costs 100 backbone evaluations at scale 2, and 50 at scale 0 or 1. This teacher sampler is for training diagnostics, not `ModelOneNFE` or `ModelFewNFE` submission. A CFG scale sweep must follow a meaningful teacher training run. Any guidance-scale choice must use a training-only held-out set; the fixed validation split is reserved for final FID reporting under the upstream README rules.

## Training and reproducibility

The initial config uses batch size 8, gradient accumulation 2, AdamW at `2e-4`, 2,000-step warmup, gradient clip 1, EMA decay `0.9999`, bf16 autocast on A100, 200,000 target optimizer steps, and seed 42. The first sustained training run instead uses the literature-informed [teacher_ddpm_cfg_v1.json](/root/workspace_distillation/configs/teacher/teacher_ddpm_cfg_v1.json), with its exact choices and live status in [teacher_training_run.md](/root/workspace_distillation/docs/teacher_training_run.md). Configs use JSON from Python's standard library, so the teacher adds no TA-environment dependency. Training writes a config copy, JSON metrics, log, `latest.pt`, and `final.pt` under `experiments/<experiment_name>/`. Checkpoints include raw model, EMA, optimizer, AMP scaler, epoch/batch cursor, RNG states, config, and class mapping. Resume rejects mapping and formulation/architecture mismatches. There is no `best_fid` checkpoint selection; the frozen final-teacher validation FID is 111.68 (50 NFE, scale 1), used for reporting only.

### Commands

Run at repository root. Install PyTorch appropriate to the GPU first, then `pip install -r requirements.txt`. The supplied downloader is `python download_dataset.py`.

```bash
python -m teacher.smoke --config configs/teacher/debug.json --output experiments/teacher_debug/smoke.json
python -m teacher.train --config configs/teacher/debug.json
python -m teacher.train --config configs/teacher/debug_resume.json --resume experiments/teacher_debug/final.pt
python -m teacher.train --config configs/teacher/debug_fresh_five.json
python -m teacher.train --config configs/teacher/initial_probe.json
python -m teacher.train --config configs/teacher/initial.json
python -m teacher.sample --checkpoint experiments/teacher_debug/final.pt --labels 0 103 --guidance-scale 2 --steps 4 --seed 42 --output experiments/teacher_debug/manual_sample.png
```

`initial_probe.json` performs exactly one optimizer step with the full initial architecture and batch/accumulation settings. It is a memory probe, not a trained checkpoint. Do not use the debug or probe checkpoint to assess sample quality.

## Executed validation

Environment: local Python 3.12, PyTorch 2.6.0+cu124, NVIDIA GRID A100D-20C (20,480 MiB), real downloaded Kaggle images. The upstream downloader composited 1,476 transparent files to white. All 17,079 training manifest paths exist. Smoke and full-size probe were rerun successfully after removing PyYAML from the local environment; TA Python 3.10 compatibility remains untested here.

| Check | Result |
| --- | --- |
| Real train batch | Pass: `(2,3,32,32)`, normalized `[-1,1]`, 151-class mapping |
| Forward shape; class and null labels | Pass |
| Class dropout | Pass at probabilities 0 and 1 |
| Loss backward and fixed-batch sanity | Pass; MSE `0.997811 → 0.985018 → 0.960433` in three updates |
| EMA and checkpoint state round-trip | Pass |
| Sampling / PNG output | Pass; 4 steps at CFG 2 reported 8 NFE |
| CFG scale 0 | Pass: same output as direct null-condition sampling, 4 NFE |
| Same-seed sample repeat | Pass |
| Debug trainer | Pass: 3 optimizer steps, `latest.pt`/`final.pt` and PNG saved |
| Resume trainer | Pass: checkpoint at step 3 continued to steps 4 and 5 |
| Resume determinism | Pass: resumed 5-step model and EMA tensors match a fresh 5-step run exactly (maximum absolute difference 0) |
| Full-size training probe | Pass: one optimizer step at initial model size, batch 8 × accumulation 2 |
| Full-size checkpoint sampling | Pass: 4 DDIM steps, CFG 2, 8 NFE, PNG saved |

**Peak allocated VRAM** from `torch.cuda.max_memory_allocated()`: 167,523,840 bytes for the debug smoke process; 49,648,128 bytes for the three-step debug trainer; **1,814,135,808 bytes (about 1.69 GiB)** for the full-size one-step training probe. These are allocated peaks for the measured short runs, not an estimate for all 200,000 steps. Both trainer metrics and probe artifacts are in `experiments/` locally.

## Still outstanding

- Further teacher quality improvements and training-only CFG selection. The new `teacher.evaluate` adapter reuses the unchanged upstream directory evaluation functions for teacher FID reporting without student wrappers.
- `best_fid` checkpoint selection remains unimplemented; validation FID cannot select checkpoints or hyperparameters. Frozen final-teacher FID reporting is now available through `teacher.evaluate` (111.68 at 50 NFE, scale 1).
- Student distillation and consistency methods, guided student, LoRA, guidance curriculum, CTM, DMD2, and architecture search are outside this phase.
- Final FastGen wrappers and 1/4-NFE submission checkpoints are outside this phase.
