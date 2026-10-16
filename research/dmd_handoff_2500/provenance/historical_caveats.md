# Historical evidence and corrections

Files in `reports/`, `source_overlay/` and `provenance/historical_lineage_evidence/` are preserved verbatim. Their historical text is not silently updated. Current handoff conclusions are in the four top-level documents and the retraining guide.

| Historical statement or field | Interpretation for the current run |
|---|---|
| `REPORT.md` conditional 2500/5000 extension | Stale template; reproduction removed both early stops and the timeout and reached 10000 |
| `no warmup` | No LR warmup; fake-score warmup is 200 updates |
| `No official validation ... evaluation used` | The DMD training script uses development evaluation; separate official commands exist in shell history |
| `config.json` references attestation/override files | Originals are in the previous workspace and copied here as historical evidence; they are not runtime trainer dependencies |
| `reference_manifest.json` old absolute source path | Origin metadata; image list/hash and local holdout content are verified; generation does not require that old path |
| `selected_model.json` absolute source/export paths | Historical identities; use the relative packaged checkpoint for inference |
| old DMD snapshot dev FID 55.557974634094535 | A different previous run; current reproduced 2500 is 55.72514913187206 |
| old DMD official FID 38.43229103196154 | Different checkpoint hash `5af78d...`; not current `acc998...` |
| `CODEX.md` 60M budget | Older project instruction; current official README and evaluator use 100M |
| entire ancestry excludes the 755 images | Team documents/user attestation are not independent ancestor-checkpoint proof; override expressly limits that claim |
| `source_overlay/TEAM_HANDOFF.md` mentions ZIPs | Historical delivery layout; those ZIPs are not runtime dependencies of this package |
| archived reports refer to absent failure reports | Historical background, not required to load or reproduce the current DMD trainer |

The source-of-truth order is executable run code, checkpoint metadata, run config, results/logs, shell/tmux history, snapshots, old reports, user-recovered values, retrospective interpretation. Shell history has no reliable per-command timestamp; line order may combine multiple shells. The terminal excerpt documents launch/preflight, not output that was never located.

No numerical official evaluation was performed for this handoff. The new verification JSON and Python helpers are explicitly build-time artifacts, not contemporaneous training logs. Legacy scripts retain their original hardcoded paths and behavior for provenance; only the documented reconstruction path and new helpers are intended entrypoints.
