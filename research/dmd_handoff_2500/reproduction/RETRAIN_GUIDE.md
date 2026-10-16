# Reproduce the FastGen DMD run in a new workspace

This guide reruns DMD from the three original input checkpoints. It does not recreate the team ancestors or retrain the fresh diffusion from scratch. Obtain byte-identical external inputs from `INPUTS.md`. Training is expected to run through 10000 generator updates and select an EMA export using development FID. A bitwise-identical checkpoint/FID across hardware or library changes is not guaranteed.

The archived script creates files at import and has no CLI/config reader or resume mode. **Never run it in `/root/DMD` or directly from this package's `scripts/` directory.** Use a fresh checkout, restore its original relative location, and retain the package as an archive. Do not resolve an existing-output failure by deleting an old experiment.

## 1. Environment and checkout

Recorded Python is 3.10.21, torch 2.6.0+cu124, torchvision 0.21.0+cu124, clean-fid 0.1.35, scipy 1.15.3, numpy 2.2.6, Pillow 12.3.0; GPU GRID A100D-20C, bf16 capable. See `verification/environment.txt`. Follow the included official README for environment installation; no new package was installed for this handoff.

Set these variables for your machine. Use an empty destination; do not reuse the inspected source:

```bash
export HANDOFF_DIR=/path/to/dmd_handoff_2500
export REPLAY_PARENT=/path/to/new/work
export FASTGEN_PY=/path/to/fastgen/environment/bin/python

cd "$HANDOFF_DIR"
sha256sum -c SHA256SUMS
cd "$REPLAY_PARENT"
git clone https://github.com/KAIST-Visual-AI-Group/AIC313-Project-FastGen.git dmd_replay_2500
cd dmd_replay_2500
git checkout e3f5120000cd91037017d86f23993492195101a5
git status --short
export REPLAY_ROOT="$PWD"
```

The checkout must be clean before applying the overlay. `source_overlay/OVERLAY_MANIFEST.json` is the original 103-file team overlay manifest. The package additionally contains unchanged official runtime files, README, requirements and fixed splits so inference can load without another checkout.

## 2. Apply the verified overlay

```bash
cp -a "$HANDOFF_DIR/source_overlay/." "$REPLAY_ROOT/"
git diff --exit-code -- evaluate.py dataset.py download_dataset.py src/utils.py \
  data/pokemon-generation-one-22k/train_split.txt \
  data/pokemon-generation-one-22k/val_split.txt
mkdir -p experiments/night_run/fresh_teacher checkpoints
cp -n "$HANDOFF_DIR/scripts/dmd_reproduce.py" experiments/night_run/dmd_reproduce.py
cp -n "$HANDOFF_DIR/scripts/dmd_preserve_team.py" experiments/night_run/dmd_preserve_team.py
cp -n "$HANDOFF_DIR/provenance/reference_manifest.json" experiments/night_run/reference_manifest.json
cp -n "$HANDOFF_DIR/provenance/audit_results.json" experiments/night_run/audit_results.json
```

Do not pre-create `experiments/night_run/dmd_preserve_team/`: the script requires it not to exist. The packaged `configs/dmd_config.json` is an archival output snapshot, not an input consumed by the script.

## 3. Dataset, development split and references

Only in the new workspace:

```bash
"$FASTGEN_PY" -B download_dataset.py
"$FASTGEN_PY" -B "$HANDOFF_DIR/reproduction/prepare_development_reference.py" \
  --workspace "$REPLAY_ROOT"
```

The official downloader performs the provided white-background transparency preprocessing. The included split manifests are deterministic seed 42, five/class, with 16324 train and 755 holdout paths. `teacher/splits.py` records the algorithm; the build independently reconstructed the exact order. Use the delivered manifests to avoid accidental split changes.

The new reference helper uses RGB, torchvision Resize(64,64), ToTensor, Normalize(0.5), then inverse normalization and torchvision PNG export, reproducing the historical reference bytes. It checks all 755 expected hashes before writing and refuses any existing target directory. `--check-only` regenerates and verifies in memory without writing.

## 4. Place the three original checkpoints

Copy or transfer the exact files from their original sources into these paths in the new checkout:

```text
checkpoints/one_nfe.ckpt
checkpoints/few_nfe.ckpt
experiments/night_run/fresh_teacher/final.pt
```

When the inspected server files are still available, the concrete copy commands are:

```bash
cp -n /root/DMD/checkpoints/one_nfe.ckpt checkpoints/one_nfe.ckpt
cp -n /root/DMD/checkpoints/few_nfe.ckpt checkpoints/few_nfe.ckpt
cp -n /root/DMD/experiments/night_run/fresh_teacher/final.pt experiments/night_run/fresh_teacher/final.pt
```

These source paths are examples for the original server only. On another machine obtain the same files and verify their hashes; the copied G, endpoint and score models must not be interchanged.

## 5. Preflight

```bash
"$FASTGEN_PY" -B "$HANDOFF_DIR/reproduction/verify_inputs.py" \
  --workspace "$REPLAY_ROOT" --device cuda
```

This helper checks the exact input hashes and sizes, protected files and original overlay, required script/config identities, development split audit, 755 reference hashes, fresh checkpoint train-path digest, model counts and GPU bf16/VRAM metadata. It rejects an existing run directory. The train-path digest excludes the terminal newline; do not compare it to raw `sha256sum train.txt`.

The old preflight's literal `ALL PREFLIGHT CHECKS PASS` is preserved in `provenance/tmux_preflight_launch_excerpt.txt`. The new helper's JSON is a current check, not a historical log. It cannot measure future peak training memory. Existing DMD logs show 4,522,762,752 peak allocated bytes; the trainer asserts `<20_000_000_000` throughout.

## 6. Exact historical launch

The original working directory was `/root/DMD`; for your rerun remain in the new checkout. This is the exact recovered command, executable path included:

```bash
export OMP_NUM_THREADS=4
export PYTHONUTF8=1

/root/miniconda3/envs/fastgen/bin/python \
  experiments/night_run/dmd_reproduce.py \
  2>&1 | tee experiments/night_run/dmd_reproduce_console.log
```

On another machine substitute `"$FASTGEN_PY"` for the executable path. Shell `set -o pipefail` is a useful additional operational safeguard for a new run; it is not part of the recovered command. The archived command's `tee` captures output, but by itself its exit code is not proof of Python success.

## 7. Expected behavior and output

Baseline generation/evaluation precedes fake warmup. Warmup performs 200 fake optimizer steps, logging every 20. Main training accumulates four G microbatches of 16 per update; fake steps each microbatch. G/fake learning rates remain 2e-5/2e-4, gradient clip 1, no LR warmup, G EMA 0.999. Primary fresh EMA supplies real score; CFG=2 is applied only there.

The run writes `config.json`, input hashes, warmup JSON, 10000 JSON-lines in `train.log`, `results.json`, EMA inference checkpoints at 2500/5000/10000, metric JSON and generated PNGs, `selected_model.json`, `selected_one_nfe.ckpt`, and `latest.pt`. It also writes top-level `NIGHT_REPORT.md` and `run_status.json`. No early stopping remains. See `provenance/script_diff.patch`.

The script's old report template still describes conditional early stopping and references historical files. Treat executable control flow and current metrics as authoritative; see `historical_caveats.md`.

## 8. Evaluation and checkpoint selection

Use the script's fixed development evaluation: 755 references, 3020 samples, 151×20 labels, batch 16, bf16, seed 42+batch index. A changed batch size changes the RNG partition and can change FID. Selection is minimum FID among baseline/2500/5000/10000; baseline is retained if candidates are worse. This recorded run selected 2500. A new run should use its own development results, not assume its optimum is 2500.

Official validation rules prohibit training/distillation/hyperparameter selection use and designate validation as FID reference. Merely viewing scores is not explicitly prohibited. Keep reporting separate from development selection. This guide does not run official evaluation. Existing official metrics and their evidence limitations are in `provenance/official_validation_records.md`.

## 9. Final inference verification

```bash
"$FASTGEN_PY" -B "$HANDOFF_DIR/reproduction/verify_model.py" \
  --checkpoint "$REPLAY_ROOT/experiments/night_run/dmd_preserve_team/selected_one_nfe.ckpt" \
  --device cuda
```

The helper loads only that generator checkpoint and packaged architecture code, counts all parameters and hooks the backbone for batches 1, 2 and 16. Expected count is 48,883,107, sampling times `[999]`, one forward/batch, finite RGB output. A new training result need not have the archival hash.

## 10. Scope and continuation

This package does not contain an exact 2500-step optimizer/fake/RNG resume state. Original `latest.pt` is 10000. Fine-tuning the exported EMA with a new optimizer is a new experiment, not an exact continuation. Rebuilding fresh diffusion or the team distillation pipeline is separate work requiring its own inputs/data-provenance checks. Historical absolute paths in metadata identify origins and do not authorize writing those locations.
