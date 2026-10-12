# Official validation evidence

Official validation metrics are documented separately as evaluation/reporting results and are not the stated basis for selecting the 2500-step checkpoint. Development `results.json` and the executable selection logic select 2500 independently.

| Model | FID | Evidence status |
|---|---:|---|
| Supplied team 1-NFE baseline | 58.15948330602913 | Stored previous-workspace metrics with the exact current baseline hash `8173c059...` |
| Current reproduced DMD 2500 | 38.605112 | User-recovered/manual result; evaluation command found in shell history, numerical output artifact not found |
| Current reproduced DMD 5000 | 41.246506 | User-recovered/manual result; evaluation command found in shell history, numerical output artifact not found |
| Current reproduced DMD 10000 | not recorded | No result found in inspected current files, related historical evaluation records or retained tmux history |

The baseline artifact is copied to `historical_lineage_evidence/official_baseline_metrics.json`. Its original absolute path is `/root/AIC313-Project-FastGen/experiments/night_run/three_frozen_models_official_comparison/team_one_nfe/metrics.json`. It records 3020 reference/generated images, batch 16, seed 42+batch index, and `used_for_selection:false`.

The history commands evaluated existing generated folders against `results/reference` with `compute_fid(...,device='cuda',batch_size=16)`. Commands establish the requested procedure, not their printed numeric results or precise execution timestamp. `results/reference` currently contains 3020 PNG. Neither references nor generated images are copied into this bundle.

A separate previous DMD run records official **38.43229103196154**, checkpoint SHA256 `5af78d5362b0b0fca3a77de0dae1509dd46d95db86fd2a9de9397fb564f172a7`. That artifact is included only as `official_older_different_dmd_metrics.json` to prevent misattribution; the current selected checkpoint SHA256 is `acc99889d73d905c1c31da57e1f2434b396a6640dd854a96cfdf1b9cc07d708c`.

No new official validation evaluation was run during inspection or packaging. Future reports should preserve the distinction between artifact-backed values, user-recovered values and unrecorded results.
