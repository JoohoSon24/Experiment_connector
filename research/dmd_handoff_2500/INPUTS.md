# Training inputs and inference requirements

**IMPORTANT: The fresh teacher is NOT the parent diffusion of the provided team 1-NFE / 4-NFE models.** Three input checkpoints have different roles:

1. `one_nfe.ckpt` is the **trainable generator starting point**.
2. `few_nfe.ckpt` is the **frozen, exactly 4-NFE endpoint teacher**.
3. `fresh_teacher/final.pt` supplies **frozen real-score weights and fake-score initialization**. Fake subsequently trains on current G images.

All three are required to rerun the archived DMD training. None is copied into the lightweight handoff. All have 48,883,107 backbone parameters. Full metadata and source identities are in `provenance/artifacts.json`.

| Input | Original absolute path | Bytes | Required for DMD retraining |
|---|---|---:|---|
| Team 1-NFE | `/root/DMD/checkpoints/one_nfe.ckpt` | 195649098 | Yes |
| Team 4-NFE | `/root/DMD/checkpoints/few_nfe.ckpt` | 195646254 | Yes |
| Fresh diffusion | `/root/DMD/experiments/night_run/fresh_teacher/final.pt` | 978259214 | Yes |

```text
8173c059ca3dda5d63c2c3a9d463b100218867c5fc08b995f5c93162844f7266  checkpoints/one_nfe.ckpt
d3c6d0cccc1b81d1709ed090abd68961197ef7417a7c6ae67724835bf7bfa596  checkpoints/few_nfe.ckpt
b5c09c58954352f71291a2c4421c5f542ac6abb392acda4074c1f118ab9c3a27  experiments/night_run/fresh_teacher/final.pt
```

These absolute paths identify the inspected server inputs; after relocation, place byte-identical copies at the corresponding **relative paths in a new training workspace**. They are not runtime paths for packaged inference. The fresh checkpoint has historical absolute dataset paths in config, but the archived DMD script verifies the local train manifest and uses only the architecture/config and `ema` weights. It does not read the old dataset root.

The selected model is `checkpoints/dmd_step_002500.ckpt` inside this package (195653282 bytes). Its state contains the EMA generator, schedule and sampler buffers, plus metadata. Inference needs that file, the included `source_overlay` Python modules and the recorded dependencies. **It does not need fresh diffusion, fake-score weights or the team 4-NFE checkpoint, and does not use inference CFG.** The `teacher/` code folder defines the backbone class; this code dependency is distinct from a teacher-weight dependency.

The original `/root/DMD/experiments/night_run/dmd_preserve_team/latest.pt` is 1369590460 bytes, SHA256 `a568884c06e0c0aa1bda6952c7b683fb6e06fd88d974b208416c474185cf51e4`. It is excluded. It contains **10000-step**, not 2500-step, optimizer/fake-score/RNG state. It is not required for a fresh rerun; using it for continuation requires a separately reviewed resume implementation. The archived script has no resume CLI.

Other prerequisites are the course dataset, official fixed splits/class mapping, development manifests and reference preparation. The overlay includes 16324 training and 755 holdout paths, seed 42, five/class. No training/validation images, generated image folders or development reference PNGs are included. `prepare_development_reference.py` rebuilds the 755 references and verifies their exact recorded bytes. It never creates an official validation reference.

`train.txt` raw-file SHA256 is `99e2aef7b68f4792ce1440c1196dd948310fdd7fc6479068595c96a30558b24e`. The checkpoint's train-path digest is **SHA256 of newline-joined paths without a terminal newline**, `b4020d6dc65948c7cd2a0de048382284d0fc7a1dd19237583e4f67cc2d353f51`. These are intentionally different encodings.

Fresh input primary EMA decay is 0.9999; generator EMA decay during DMD is 0.999. Fake initialization uses a deepcopy of the primary real-score model, not `ema_tracks['0.999']`. Holdout exclusion for every ancestor of the supplied team models is not independently proven by their inference exports.
