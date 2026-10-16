# Teacher quality diagnosis, 2026-09-28

## Observed run and limitations

`teacher_ddpm_cfg_v1` completed 100,000 optimizer steps in 22,842.83 seconds (~6h21m), with no non-finite-loss or gradient failure. Mean training epsilon MSE was `0.06463` in steps 1–1,000, `0.02133` in 49,001–50,000, and `0.02006` in 99,001–100,000. This curve does not establish generation quality or rule out suboptimal learning rate. Its small late improvement motivates a controlled continuation test rather than assuming more steps alone will solve the problem.

The dataset has 46–183 training images per class (median 111), with varied poses, backgrounds and styles. Provided 64×64 training examples retain more detail than generated samples. Current class identity and detail learning are incomplete; this is not simply an image viewer scaling issue.

## Controlled qualitative comparisons

`scripts/diagnose_teacher.py` sampled the final checkpoint with seed 42 and the same four labels (Abra, Charizard, Pikachu, Tauros). No validation images are accessed. Images and metadata are in `experiments/teacher_diagnostics/`.

| Change from EMA / 50 DDIM steps / scale 1 / bf16 | Observation on four samples |
| --- | --- |
| 250 steps | Very similar shapes and softness |
| 1,000 steps | Very similar; step count alone did not restore details |
| float32 at 250 steps | Very similar to bf16; precision does not appear dominant here |
| CFG scale 2 at 250 steps | Sharper outlines and stronger class cues; also stronger colors, still imperfect |
| Raw weights at 250 steps | More unstable/distorted details than EMA |

This is a small qualitative check, not proof of global FID improvement. The likely issue is the learned denoising/conditioning quality and its training setup. Current automatic sample grids used scale 1, which is conditional generation without extra CFG. LR `2e-4` did not visibly diverge, but a lower-LR continuation could reduce late-stage fluctuations and must be tested separately. No augmentation ablation was run, so insufficient augmentation is a hypothesis, not an established cause. Neither limited steps nor bf16 explained the observed softness in this check.

The [Improved DDPM paper, Section 3.2](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf) reports that the original linear schedule is suboptimal at 32×32 and 64×64, and introduces a cosine schedule. This makes a cosine training baseline a defensible future ablation; it does not demonstrate that this run's blur has one known cause. Do not silently swap the schedule of the current checkpoint. Training-only controlled experiments should change one variable at a time. Mild horizontal flips can be another ablation; aggressive crops may remove class-defining parts.

## Teacher FID without submission wrappers

`teacher.evaluate` adds a teacher adapter with the required sample signature and reuses the unchanged `evaluate.generate_samples`, `prepare_reference_set`, and `compute_fid`. FID operates on image directories and imposes no 1/4-NFE requirement itself. The teacher's measured NFE is recorded explicitly and the result is not an eligible FastGen student submission score.

```bash
.venv/bin/python -m teacher.evaluate \
  --checkpoint experiments/teacher_ddpm_cfg_v1/final.pt \
  --output-dir experiments/teacher_fid_baseline \
  --steps 50 --guidance-scale 1 --seed 42 --batch-size 16 --precision bf16
```

This frozen baseline generates 20 images per class, 3,020 images total, and compares with all 3,020 validation references via the provided clean-FID function. `clean-fid` is already an upstream dependency; it was installed locally without changing requirements. Inception weights are used for evaluation only. The fixed validation FID is reporting-only and must not choose LR, augmentation, CFG, schedule or training duration. The four-image generation smoke passed before the full run was launched in tmux `fastgen_teacher_fid`.

The first full score calculation encountered an environment issue: clean-fid 0.1.35 calls `scipy.linalg.sqrtm(..., disp=False)`, which SciPy 1.18 removed. Local SciPy was downgraded to 1.16.3; no provided evaluator or clean-FID implementation was modified. `constraints/evaluation.txt` constrains versions of existing upstream dependencies for this compatibility. Add `--score-only` to the command above to reuse the 3,020 generated images; the script verifies matching metadata and exact per-class filenames before scoring.

The recovered calculation completed successfully: **FID = 111.67612714633668**. Settings: final 100,000-step checkpoint, EMA, 50 DDIM steps / 50 NFE, guidance scale 1, seed 42, bf16, batch size 16, 3,020 generated images and 3,020 validation references. This quantifies the current frozen baseline; it does not identify an optimal setting. Metrics and the complete log are saved in `experiments/teacher_fid_baseline/metrics.json` and `console.log`. Environment: torch 2.6.0+cu124, clean-fid 0.1.35, SciPy 1.16.3, NumPy 2.5.2. No CFG-2 FID comparison was made against validation.

## Disk space

After FID evaluation, the shared workspace filesystem still had **75.62GiB available** out of 95.82GiB, so the requested 10GB threshold was not met and no files were deleted. The workspace occupied ~11GiB: `.venv` ~5.2GiB, `experiments` ~4.5GiB, and data ~1.4GiB. Outside the workspace, pip and Kaggle caches occupied ~3GiB. The full-size probe runs each store duplicate `latest.pt`/`final.pt` files (~1.5GiB per run); those and download caches are possible future cleanup candidates, not current bottlenecks.
