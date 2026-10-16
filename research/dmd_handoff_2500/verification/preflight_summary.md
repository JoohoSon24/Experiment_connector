# Preflight verification

The archived tmux excerpt records ALL PREFLIGHT CHECKS PASS before the original launch, including an absent run directory. New read-only checks independently validate input hashes, source files, model counts, development reference bytes and GPU bf16 support. The current-run directory now exists, so packaging uses explicit audit mode, not launch mode.

The selected generator is verified from packaged code and a single checkpoint in a process whose working directory is outside /root/DMD. Final-location verification is in nfe_check.json. Original input model counts, endpoint NFE and state-schema equality are recorded separately. Training memory comes from historical logs, not a new training run. No official FID is recomputed. The source hash tree is compared after all read-only checks.
