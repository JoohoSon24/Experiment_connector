# Night report — preserve supplied team models
Updated UTC: 2026-10-08T08:35:48Z
Status: complete DMD existing-team experiment

User explicitly superseded the extra fresh-lineage/755-image exclusion requirements with the supplied official course rules. Existing checkpoints are accepted as course-compliant on user attestation. Their exact training manifests are not independently embedded/verified; These scores use the same755-image development reference as the documented team holdout. Team configs/docs specify exclusion from training, accepted on user confirmation; original full teacher/student training artifacts are absent, so exclusion is not independently proven from the inference files. Scores are not official validation FID. Reference755 original-training images, generated3020 class-balanced images, batch16, seed42+batch index, bf16, clean-fid default. No official validation images/statistics/evaluation used.

Team architecture, velocity output and one/four-NFE sampler preserved. Initialization is exactly checkpoints/one_nfe.ckpt; frozen distillation teacher exactly checkpoints/few_nfe.ckpt. All models individually48,883,107 parameters. No external generator or new packages; original files/checkpoints remain unchanged.

New user-authorized substantial objective change: transfer the existing four-NFE teacher endpoint into the existing one-NFE student using matched initial noise and classes. DMD adds an estimated distribution gradient from a frozen real diffusion score and a trainable fake score. Weak endpoint regression stabilizes the change. This is a VP/v-generator adaptation, not an exact reproduction of the paper. DMD normalized score difference plus0.1-weight endpoint MSE, generator AdamW LR2e-5 constant; fake score AdamW LR2e-4 constant, no warmup, batch64/microbatch16, EMA0.999, clip grad1, seed42. Optimizer freshly initialized because supplied inference exports do not include optimizer state; this is fine-tuning, not exact historical optimizer continuation. Synthetic training pairs derive only from the course-compliant team teacher; neither reference nor validation images enter the loss.

Candidate screening2.5K updates; extend5K only if FID below supplied baseline; extend10K only if5K FID improves its2.5K score. No further extension. This follow-up was explicitly authorized after the prior three-candidate phase ended. Only one new DMD candidate is run; all failed artifacts remain preserved. No fresh recreation stages continue.

| Model | Added updates | Training-reference FID | Parameters | NFE | Peak allocated VRAM bytes | Training seconds |
|---|---:|---:|---:|---:|---:|---:|
| supplied_baseline | 0 | 65.386270 | 48883107 | 1 | 0 | 0.00 |
| dmd_002500 | 2500 | 55.725149 | 48883107 | 1 | 4522762752 | 3643.40 |
| dmd_005000 | 5000 | 58.622004 | 48883107 | 1 | 4522762752 | 7264.99 |
| dmd_010000 | 10000 | 60.444141 | 48883107 | 1 | 4522762752 | 14320.56 |

Selected model: {"name": "dmd_002500", "step": 2500, "fid": 55.72514913187206, "reference_kind": "development_reference_team_documented_holdout", "reference_count": 755, "sample_count": 3020, "seed": 42, "batch_size": 16, "precision": "bf16", "parameter_count": 48883107, "nfe": 1, "measured_batch_backbone_calls": 189, "peak_vram_bytes": 4522762752, "training_seconds": 3643.402763916005, "selected_source": "/root/DMD/experiments/night_run/dmd_preserve_team/dmd_step_002500.ckpt", "export": "/root/DMD/experiments/night_run/dmd_preserve_team/selected_one_nfe.ckpt", "export_sha256": "acc99889d73d905c1c31da57e1f2434b396a6640dd854a96cfdf1b9cc07d708c", "export_measured_nfe": 1}

Selection: minimum fixed training-reference FID, with the supplied baseline retained if the candidate degrades. Interpret holdout cleanliness under the documented/user-attested lineage, not as independently checkpoint-verified provenance. Official evaluation can only report a frozen model and never choose hyperparameters/candidates.

Next experiments at mosttwo: compare an alternative teacher endpoint target under the same budget; replicate a winning candidate with another training seed.

Historical failure: new VP consistency sweep collapsed near uniform gray, FID515–578. It was not an improvement over the supplied progressive models and is excluded from this selection. Fresh teacher/recreation and all failure logs preserved. See QUALITY_DIAGNOSIS.md, CONSISTENCY_REPORT_BEFORE_RECOVERY.md and COURSE_RULES_OVERRIDE.md.
