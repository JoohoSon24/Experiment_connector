# User clarification: follow official course requirements

The user provided the full course README/slides and stated: 여기 나온대로먼 지키면 돼 과제 조건은.
This supersedes the earlier extra requirement that supplied checkpoints and every ancestor exclude the internally selected755 images. The supplied checkpoints are accepted as course-compliant on the user's confirmation; no claim of independent artifact-verified holdout exclusion is made.

Preserve both supplied checkpoint files and the team's existing progressive architecture/sampler. Use only official training-derived data or synthetic targets from these allowed models for training and model selection. All learned models individually<100M, peak allocated training VRAM<20GB, final one_nfe exactlyone backbone forward and few_nfe at mostfour; no external pretrained generator, extra packages or protected-file/split edits.

The provided README explicitly disallows using validation images for training, distillation or hyperparameter selection. Validation evaluation is permitted for reporting a frozen selected model, but it is not a model-selection metric. No validation evaluation will be used in development.

The755-image manifest may be used as a fixed training-distribution diagnostic reference; it is not claimed to be an unseen holdout for the existing checkpoints. Generated3020 images, seed42, class-balanced20/class, batch16, bf16, clean-fid remain fixed. Any new results using this reference will be labeled training-reference FID. Do not compare these scores directly with validation-reference FIDs.

Fresh diffusion teacher and consistency/progressive recreation artifacts are preserved but excluded from this restoration of the supplied baseline. All development updates and exports go under experiments/night_run with distinct filenames. Prior lineage exclusion reports are historical and superseded by this clarification.
