# FastGen DMD 2500 handoff

The selected model is **[checkpoints/dmd_step_002500.ckpt](checkpoints/dmd_step_002500.ckpt)**: a 48,883,107-parameter EMA generator using **exactly one backbone forward per sampling batch**. It is ready for inference with the included code; no auxiliary checkpoint is needed.

This experiment is **DMD-inspired post-distillation fine-tuning with 4-NFE endpoint regularization**, not an exact reproduction of original DMD. It starts with an already-distilled team 1-NFE model.

| Training input | Role |
|---|---|
| Team `one_nfe.ckpt` | Trainable generator starting point |
| Team `few_nfe.ckpt` | Frozen, exactly 4-NFE endpoint regression teacher |
| Independent fresh diffusion `final.pt` | Frozen real-score weight source and trainable fake-score initialization source |

**The fresh diffusion teacher is NOT the parent teacher of the provided team 1-NFE or team 4-NFE checkpoints.** Fake-score training tracks current generator outputs. The 4-NFE teacher is not a real-score or fake-score model. Final inference loads only the selected generator weights and uses no inference CFG; the three original training input checkpoints are not opened. Architecture code in `source_overlay/teacher/` remains necessary.

The **2500-step checkpoint is selected based on the development holdout results**:

| Checkpoint | Development FID |
|---|---:|
| Supplied baseline | 65.38627014891784 |
| **DMD 2500** | **55.72514913187206** |
| DMD 5000 | 58.62200371627705 |
| DMD 10000 | 60.44414080173647 |

All use 755 development references, 3020 generated images, 151 classes, batch 16, seed `42 + batch index`, bf16. The split is inside official training. Fresh-teacher exclusion of the holdout is supported by its embedded manifest digest; exclusion throughout the team models' entire ancestry is **not independently checkpoint-verified**. Treat that limitation as part of these results.

Official validation metrics are documented separately as evaluation/reporting results and are not the stated basis for selecting the 2500-step checkpoint. Baseline 58.15948330602913 is artifact-backed for the identical input hash. The user-recovered current-run values 38.605112 (2500) and 41.246506 (5000) have evaluation commands in shell history but no numerical output artifact located during inspection. Current-run 10000 official FID is not recorded in the inspected evidence. See [records](provenance/official_validation_records.md).

## Verify and load

From this directory, using the recorded environment or compatible dependencies:

```bash
sha256sum -c SHA256SUMS
python -B reproduction/verify_model.py --device cuda
```

For Python inference, set `HANDOFF_DIR` to this package's location. This works after relocation and does not open any training input checkpoint:

```python
import os, sys
from pathlib import Path
import torch

handoff = Path(os.environ["HANDOFF_DIR"]).resolve()
sys.path.insert(0, str(handoff / "source_overlay"))
from model import Model

model = Model.load_checkpoint(
    str(handoff / "checkpoints/dmd_step_002500.ckpt"),
    evaluate_mode="one_nfe", device="cuda",
)
with torch.inference_mode():
    images = model.sample(
        (1, 3, 64, 64), device="cuda",
        category=torch.tensor([0], device="cuda"), seed=42,
    )
```

The checkpoint SHA256 is:

```text
acc99889d73d905c1c31da57e1f2434b396a6640dd854a96cfdf1b9cc07d708c
```

## Retraining and documentation

Full DMD retraining needs the **three external input checkpoints** above, the course dataset, and a new workspace. They are deliberately excluded from this lightweight package; exact sizes, hashes and source paths are in [INPUTS.md](INPUTS.md). The selected checkpoint is an inference export, not a 2500-step optimizer/fake-score resume state. The archived training script has no resume CLI and runs unconditionally through 10000 steps.

- [METHOD.md](METHOD.md): actual objective, lineage, original-DMD comparison and interpretation limits.
- [PROVENANCE.md](PROVENANCE.md): workspace reconstruction, artifact identities and historical/current-run distinctions.
- [INPUTS.md](INPUTS.md): the three training models and external prerequisites.
- [RETRAIN_GUIDE.md](reproduction/RETRAIN_GUIDE.md): reconstruct a fresh workspace and launch the archived implementation.
- [HANDOFF_BUILD_REPORT.md](HANDOFF_BUILD_REPORT.md): final tree, byte sizes and verification results.

`reports/REPORT.md` and `source_overlay/docs/` are unchanged historical documents. Read [historical caveats](provenance/historical_caveats.md) before applying their claims to this run. The two scripts under `scripts/` are archival copies: **do not execute the training script there**; restore it to `experiments/night_run/dmd_reproduce.py` in a new workspace as the guide explains.
