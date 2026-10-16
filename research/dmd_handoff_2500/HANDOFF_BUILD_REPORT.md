# Handoff build report

Destination: `/root/dmd_handoff_2500`. User-approved lightweight handoff for the development-selected current DMD 2500 checkpoint. Source `/root/DMD` was read only. No training, FID evaluation, source rename/delete/overwrite or source experiment output was performed.

## Final verification

- Source checkpoint and copied checkpoint SHA256 match: `acc99889d73d905c1c31da57e1f2434b396a6640dd854a96cfdf1b9cc07d708c`.
- Source reproduction script and copied script SHA256 match: `27f44ddccb02cbd9ebca69bfbe94863908efebc8b2ac6348968edb4246248769`.
- Final-location GPU check loads one checkpoint and packaged code: 48,883,107 parameters, sampling times [999], exactly one backbone forward for each batch size 1, 2 and 16, all outputs finite. No teacher/fake/endpoint checkpoint opened; no inference CFG.
- Source comparison PASS: 36145 regular files, 5092174578 bytes hashed; no real source additions/removals or file content/recorded metadata changes. Git commit and git status unchanged. Three empty sandbox-only virtual directory entries (.agents, .aws, .codex) are excluded from directory-metadata comparison and documented in the verification record; no regular file is excluded.
- Exact overlay hashes, protected official hashes, split audit, 755 reference hashes and in-memory PNG reconstruction passed. Endpoint input measured four backbone calls.
- Authored README/METHOD/PROVENANCE/INPUTS/retraining guide reviewed for consistent lineage, selection, objective and evidence status. Historical originals retain their text and have explicit caveats.
- Package has no symlinks or broken symlinks, and no unintended absolute inference dependency. Historical absolute paths are classified in `verification/portability_check.json`; archived scripts are not direct entrypoints.
- SHA256SUMS covers all 182 other regular files, including this report. The manifest does not hash itself. The build procedure runs `sha256sum -c SHA256SUMS` on the final sealed bytes and requires every entry to pass before reporting completion.

## Size

Regular files: 183. Total regular-file payload: **203033584 bytes** (203.033584 MB; 193.627914 MiB). Allocated disk usage from `du -s -B1`: **203575296 bytes**. The complete per-file tree with byte sizes is below. No compressed archive was created.

## Excluded large inputs and state

| Source path | Bytes | Reason |
|---|---:|---|
| /root/DMD/checkpoints/one_nfe.ckpt | 195649098 | Required external retraining G input |
| /root/DMD/checkpoints/few_nfe.ckpt | 195646254 | Required external endpoint input |
| /root/DMD/experiments/night_run/fresh_teacher/final.pt | 978259214 | Required external real/fake initialization input |
| /root/DMD/experiments/night_run/dmd_preserve_team/latest.pt | 1369590460 | Optional 10000-step training state, not a 2500 resume state |

Other milestone weights, duplicate selected export, training/validation images, generated images and reference PNGs are excluded. All three retraining inputs have identities in INPUTS.md and provenance/artifacts.json. Small code/manifests and byte-verified reference reconstruction instructions are included.

## Evidence limitations retained

This is DMD-inspired post-distillation fine-tuning, not an exact original-DMD reproduction. Team holdout exclusion across all ancestors is not independently proven. Current DMD official 2500/5000 values are user-recovered, with command history but no located numeric artifact; 10000 is not recorded. Historical older DMD results are identified as a different checkpoint. Latent-correspondence rationale is retrospective. The archive has no exact step2500 optimizer/fake-score resume state.

## Recheck after transfer

```bash
cd /path/to/dmd_handoff_2500
sha256sum -c SHA256SUMS
python -B reproduction/verify_model.py --device cuda
```

## Full tree and file sizes

```text
HANDOFF_BUILD_REPORT.md	13605 bytes
INPUTS.md	3810 bytes
METHOD.md	14632 bytes
PROVENANCE.md	33727 bytes
README.md	4768 bytes
SHA256SUMS	19752 bytes
checkpoints/dmd_step_002500.ckpt	195653282 bytes
configs/dmd_config.json	1410 bytes
configs/fresh_teacher_config.json	1640 bytes
provenance/artifacts.json	84237 bytes
provenance/audit_results.json	1230 bytes
provenance/evidence_extraction.json	935 bytes
provenance/fresh_teacher_metadata.json	4340 bytes
provenance/git_status_before.txt	216 bytes
provenance/historical_caveats.md	2692 bytes
provenance/historical_lineage_evidence/COURSE_RULES_OVERRIDE.md	1920 bytes
provenance/historical_lineage_evidence/LINEAGE_REAUDIT.md	4679 bytes
provenance/historical_lineage_evidence/USER_CHECKPOINT_ATTESTATION.json	915 bytes
provenance/historical_lineage_evidence/fresh_teacher_config.json	1640 bytes
provenance/historical_lineage_evidence/fresh_teacher_metrics.json	3464 bytes
provenance/historical_lineage_evidence/lineage_reaudit.json	1946 bytes
provenance/historical_lineage_evidence/official_baseline_metrics.json	772 bytes
provenance/historical_lineage_evidence/official_older_different_dmd_metrics.json	813 bytes
provenance/historical_lineage_evidence/overnight.py	13151 bytes
provenance/historical_lineage_evidence/previous_run_reproduction_manifest.json	3121 bytes
provenance/historical_lineage_evidence/teacher_preflight_config.json	1638 bytes
provenance/historical_lineage_evidence/teacher_preflight_metrics.json	3333 bytes
provenance/input_checkpoint_metadata.json	5219 bytes
provenance/model_vs_upstream.patch	2678 bytes
provenance/official_validation_records.md	2230 bytes
provenance/reference_manifest.json	141955 bytes
provenance/script_diff.patch	1618 bytes
provenance/shell_history_excerpt.txt	23596 bytes
provenance/tmux_preflight_launch_excerpt.txt	2568 bytes
provenance/validation_policy.md	2123 bytes
reports/REPORT.md	4435 bytes
reports/dmd_reproduce_console.log	181976 bytes
reports/fake_score_warmup.json	197 bytes
reports/input_checkpoint_hashes.json	165 bytes
reports/metrics/dmd_002500.json	404 bytes
reports/metrics/dmd_005000.json	404 bytes
reports/metrics/dmd_010000.json	406 bytes
reports/metrics/supplied_baseline.json	383 bytes
reports/results.json	1732 bytes
reports/selected_model.json	701 bytes
reports/train.log	2564836 bytes
reproduction/RETRAIN_GUIDE.md	9010 bytes
reproduction/prepare_development_reference.py	2809 bytes
reproduction/verify_inputs.py	5267 bytes
reproduction/verify_model.py	3388 bytes
scripts/dmd_preserve_team.py	15695 bytes
scripts/dmd_reproduce.py	15509 bytes
source_overlay/.gitignore	195 bytes
source_overlay/CODEX.md	15255 bytes
source_overlay/OVERLAY_MANIFEST.json	15646 bytes
source_overlay/PROJECT_SPEC.md	12142 bytes
source_overlay/README.md	19357 bytes
source_overlay/TEAM_HANDOFF.md	5599 bytes
source_overlay/configs/consistency/fixed_cfg2.json	543 bytes
source_overlay/configs/consistency/no_extra_guidance.json	559 bytes
source_overlay/configs/distillation/fixed_cfg2_progressive.json	789 bytes
source_overlay/configs/distillation/low_nfe_budget_100k.json	846 bytes
source_overlay/configs/teacher/debug.json	886 bytes
source_overlay/configs/teacher/debug_fresh_five.json	897 bytes
source_overlay/configs/teacher/debug_resume.json	893 bytes
source_overlay/configs/teacher/initial.json	925 bytes
source_overlay/configs/teacher/initial_probe.json	918 bytes
source_overlay/configs/teacher/teacher_b128_100k.json	1496 bytes
source_overlay/configs/teacher/teacher_b64_100k.json	1484 bytes
source_overlay/configs/teacher/teacher_b64_constant_100k.json	1495 bytes
source_overlay/configs/teacher/teacher_b64_fliponly_constant_100k.json	1502 bytes
source_overlay/configs/teacher/teacher_b64_noaug_constant_100k.json	1499 bytes
source_overlay/configs/teacher/teacher_ddpm_cfg_v1.json	924 bytes
source_overlay/configs/teacher/teacher_ddpm_cfg_v1_probe.json	919 bytes
source_overlay/configs/teacher/teacher_lr2e4_linear_aug_50k.json	1109 bytes
source_overlay/configs/teacher/teacher_lr5e5_aug_50k.json	1063 bytes
source_overlay/configs/teacher/teacher_lr5e5_aug_probe.json	1054 bytes
source_overlay/configs/teacher/teacher_memorization64_constant.json	1484 bytes
source_overlay/consistency/__init__.py	80 bytes
source_overlay/consistency/check_submission.py	1600 bytes
source_overlay/consistency/config.py	2459 bytes
source_overlay/consistency/core.py	5734 bytes
source_overlay/consistency/evaluate.py	6903 bytes
source_overlay/consistency/export.py	2799 bytes
source_overlay/consistency/submission.py	558 bytes
source_overlay/consistency/tests.py	7481 bytes
source_overlay/consistency/train.py	16043 bytes
source_overlay/constraints/evaluation.txt	113 bytes
source_overlay/data/development/seed42_holdout5/holdout.txt	31510 bytes
source_overlay/data/development/seed42_holdout5/metadata.json	349 bytes
source_overlay/data/development/seed42_holdout5/train.txt	671861 bytes
source_overlay/data/pokemon-generation-one-22k/category_to_id.json	2656 bytes
source_overlay/data/pokemon-generation-one-22k/train_split.txt	703371 bytes
source_overlay/data/pokemon-generation-one-22k/val_split.txt	153787 bytes
source_overlay/dataset.py	5923 bytes
source_overlay/distillation/__init__.py	78 bytes
source_overlay/distillation/core.py	2646 bytes
source_overlay/distillation/evaluate.py	4591 bytes
source_overlay/distillation/export.py	2014 bytes
source_overlay/distillation/pipeline.py	6860 bytes
source_overlay/distillation/smoke.py	4291 bytes
source_overlay/distillation/submission.py	2068 bytes
source_overlay/distillation/train.py	12048 bytes
source_overlay/docs/consistency_distillation.md	12456 bytes
source_overlay/docs/distillation_baseline.md	10286 bytes
source_overlay/docs/distillation_results_review.md	8350 bytes
source_overlay/docs/experiment_cleanup.md	4947 bytes
source_overlay/docs/low_nfe_budget_experiment.md	5666 bytes
source_overlay/docs/model_architecture.md	10937 bytes
source_overlay/docs/repository_analysis.md	9020 bytes
source_overlay/docs/sampling_defaults_and_ddim_cfg.md	2981 bytes
source_overlay/docs/sampling_steps_seeds_analysis.md	4915 bytes
source_overlay/docs/teacher_b64_100k.md	6386 bytes
source_overlay/docs/teacher_b64_results_and_next.md	6522 bytes
source_overlay/docs/teacher_baseline.md	10621 bytes
source_overlay/docs/teacher_constant_results_and_fliponly.md	7648 bytes
source_overlay/docs/teacher_diagnosis.md	5569 bytes
source_overlay/docs/teacher_fliponly_preflight.json	521 bytes
source_overlay/docs/teacher_lr2e4_linear_aug_run.md	2980 bytes
source_overlay/docs/teacher_lr5e5_aug_run.md	3374 bytes
source_overlay/docs/teacher_remaining_experiments.md	5020 bytes
source_overlay/docs/teacher_review_50k.md	12246 bytes
source_overlay/docs/teacher_training_run.md	5013 bytes
source_overlay/download_dataset.py	3553 bytes
source_overlay/evaluate.py	5578 bytes
source_overlay/model.py	3742 bytes
source_overlay/requirements.txt	50 bytes
source_overlay/scripts/audit_teacher.py	8545 bytes
source_overlay/scripts/benchmark_teacher.py	3714 bytes
source_overlay/scripts/compare_teacher_steps_seeds.py	2385 bytes
source_overlay/scripts/create_handoff.py	3686 bytes
source_overlay/scripts/diagnose_teacher.py	3283 bytes
source_overlay/scripts/make_student_class_grids.py	3570 bytes
source_overlay/scripts/report_teacher_official.py	2842 bytes
source_overlay/scripts/review_ddim_cfg.sh	395 bytes
source_overlay/scripts/review_teacher_run.py	2868 bytes
source_overlay/scripts/run_consistency.sh	1109 bytes
source_overlay/scripts/run_fixed_guidance_distillation.sh	670 bytes
source_overlay/scripts/run_low_nfe_budget.py	7016 bytes
source_overlay/scripts/run_low_nfe_budget.sh	380 bytes
source_overlay/scripts/run_teacher_next_experiments.sh	464 bytes
source_overlay/scripts/run_teacher_to_students.sh	693 bytes
source_overlay/scripts/supplement_ddim_diagnostics.py	2531 bytes
source_overlay/scripts/train_teacher_b128.sh	1104 bytes
source_overlay/scripts/train_teacher_b64.sh	1102 bytes
source_overlay/scripts/train_teacher_b64_constant.sh	1120 bytes
source_overlay/scripts/train_teacher_b64_fliponly_constant.sh	1138 bytes
source_overlay/scripts/train_teacher_b64_noaug_constant.sh	1132 bytes
source_overlay/scripts/train_teacher_lr2e4_linear_aug.sh	797 bytes
source_overlay/scripts/train_teacher_lr5e5_aug.sh	783 bytes
source_overlay/scripts/train_teacher_memorization64.sh	1152 bytes
source_overlay/scripts/train_teacher_v1.sh	784 bytes
source_overlay/src/__init__.py	0 bytes
source_overlay/src/utils.py	1027 bytes
source_overlay/teacher/__init__.py	52 bytes
source_overlay/teacher/cfg.py	509 bytes
source_overlay/teacher/config.py	3907 bytes
source_overlay/teacher/data.py	3009 bytes
source_overlay/teacher/diffusion.py	3941 bytes
source_overlay/teacher/evaluate.py	8679 bytes
source_overlay/teacher/model.py	7323 bytes
source_overlay/teacher/sample.py	2594 bytes
source_overlay/teacher/smoke.py	5668 bytes
source_overlay/teacher/splits.py	2393 bytes
source_overlay/teacher/train.py	16107 bytes
source_overlay/teacher/utils.py	2526 bytes
verification/architecture_check.json	33369 bytes
verification/document_consistency.json	952 bytes
verification/endpoint_nfe_check.json	423 bytes
verification/environment.txt	1237 bytes
verification/nfe_check.json	1223 bytes
verification/optimizer_steps.json	898 bytes
verification/parameter_counts.json	341 bytes
verification/portability_check.json	96883 bytes
verification/preflight_current.json	1117 bytes
verification/preflight_summary.md	820 bytes
verification/source_before.json.gz	2022813 bytes
verification/source_execution_view_differences.json	2914 bytes
verification/source_hashes.json	2668 bytes
verification/source_unchanged.json	1542 bytes
verification/split_reference_check.json	986 bytes
verification/staging_nfe_check.json	1274 bytes
verification/vram_check.json	471 bytes
```
