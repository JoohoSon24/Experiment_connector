# Experiment storage cleanup (2026-09-30)

This is the compact record for completed runs whose bulky artifacts were removed.
`experiments/` fell from 33,161,510,912 to 3,383,357,440 bytes (27.74 GiB
reclaimed). Filesystem available space is now about 77 GiB.
All runs used the local training data; development FID uses 755 held-out images
from the training split, while official FID uses 3,020 validation references.
These two FID columns are not numerically interchangeable. Model size was
48,883,107 parameters. Exact historical hyperparameters remain in `configs/`;
method and interpretation remain in the related `docs/` reports.

## Retired teacher experiments

| Deleted run under `experiments/` | Setup and result | Reason for removal |
| --- | --- | --- |
| `teacher_b128_100k` | Batch-128 feasibility probe, stopped at 102 updates after OOM; batch 64 selected. | Incomplete probe. |
| `teacher_lr2e4_linear_aug_50k` | 50K updates, batch 8 with accumulation, linear LR from 2e-4, affine plus flip. Official DDPM50 CFG1 FID 82.98979; DDIM50 CFG1 FID 189.93626. | Superseded by 100K teachers. |
| `teacher_b64_100k` | 100K updates, batch 64, linear LR, affine plus flip. Official DDPM50 CFG1/CFG2 FID 60.60930/32.98045; peak allocated VRAM 9,777,302,016 bytes. | Superseded by constant-LR experiments. |
| `teacher_b64_constant_100k` | 100K updates, batch 64, constant LR 2e-4, affine plus flip. Official DDPM50 CFG1/CFG2 FID 52.62624/31.50218; development DDIM50 CFG2 FID 48.40064. | Augmentation ablation completed. |
| `teacher_b64_fliponly_constant_100k` | 100K updates, batch 64, constant LR 2e-4, horizontal flip p=0.5, no affine. Development DDIM50 CFG2 FID 44.36898; official DDIM50 CFG2 FID 25.12628. | No-augmentation teacher won the prespecified training-holdout selection for distillation (43.57209 vs 44.36898). This official score is still useful context, but its weights are no longer needed for the selected pipeline. |
| `teacher_memorization64_constant` | 2K updates on 64 training images; final loss 0.00424. Conditional denoising and sampler oracle checks passed. | Diagnostic complete. |
| `teacher_sampling_steps_seeds` | Flip-only EMA/DDIM CFG2, disjoint seeds 42/1042/2042. Development mean FID: 20 steps 46.05883, 50 steps 44.08437, 100 steps 44.05665; DDIM32 seed-42 anchor 44.84112. | Step/seed sensitivity answered; detailed analysis in `docs/sampling_steps_seeds_analysis.md`. |
| `sampler_cfg_review` | Sampler/CFG implementation review completed. | Logs already interpreted in `docs/sampling_defaults_and_ddim_cfg.md`. |

The retained teacher is `experiments/teacher_b64_noaug_constant_100k/final.pt`:
100K updates, batch 64, constant LR 2e-4, no augmentation, development DDIM50
CFG2 FID 43.57209. This selected run is the source of the present students.
Its DDIM32 CFG2 official teacher reference had FID 26.65365 at 64 NFE.

## Retired student artifacts

| Deleted artifact | Recorded outcome |
| --- | --- |
| `distillation_smoke/` | All six 3-update stages, real-batch backward, checkpoint reload, export, deterministic samples and empirical 1/4 NFE passed. Peak allocated training VRAM 9,791,668,224 bytes. |
| `fixed_cfg2_progressive/guidance/` | Fixed CFG2 transfer, 50K updates, development FID 43.25620, 32 NFE. |
| `fixed_cfg2_progressive/n16/` | 25K updates, development FID 45.26116. |
| `fixed_cfg2_progressive/n8/` | 25K updates, development FID 46.38606. |
| Development/official generated PNG directories and `teacher32_*` inside `fixed_cfg2_progressive/` | All 3,020-image evaluations completed; teacher DDIM32 CFG2 development/official FID 44.11891/26.65365. Student n4 development/official FID 47.62077/33.0517; n2 development FID 51.69442; n1 development/official FID 68.16127/61.4540. Detailed interpretation in `docs/distillation_results_review.md`. |

The `n4`, `n2`, and `n1` **final full training checkpoints** remain under
`experiments/fixed_cfg2_progressive/`; they include model, EMA, optimizer and RNG
state. They support the proposed low-NFE budget ablations. Redundant `latest.pt`
copies were removed. The export checkpoints remain at `checkpoints/few_nfe.ckpt`
and `checkpoints/one_nfe.ckpt`. Two combined 151-class comparison sheets remain
under `experiments/fixed_cfg2_progressive/quality_review/`.

The 755-image development reference was moved out of the retired flip-only run
to `data/development/seed42_holdout5/reference/`. Training and holdout manifests
remain under the same `data/development/seed42_holdout5/` directory.

Previously generated PNG sets and intermediate stage checkpoints cannot be
recovered from this summary; rerunning their config is needed to regenerate
them. The current teacher and n1/n2/n4 checkpoints remain available for the
next experiment. The class-sheet metadata lists original PNG source paths that
were removed; the contact sheets themselves remain viewable. Old documentation
may refer to deleted artifact paths as a historical record.
