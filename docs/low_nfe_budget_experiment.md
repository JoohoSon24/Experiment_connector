# Low-NFE training-budget ablation

## Question and controls

Does increasing the last-stage training budget from 25,000 to 100,000 optimizer
updates improve 1- and 2-NFE quality? The runs continue the **exact saved
student, EMA, AdamW optimizer, batch cursor and RNG state**, not a fresh model.
The n1 continuation uses the frozen original n2 teacher; the subsequent n2
continuation uses the frozen original n4 teacher. Thus n2's later improvement,
if any, does not change the n1 target in this experiment.

Both runs use batch 64, constant learning rate 2e-4 without warmup, seed 42,
bf16, EMA 0.999, the same 16,324 training images, model architecture,
v-prediction objective and truncated-SNR weighting. The only intended change
from the 25K baseline is **total training updates**. Config:
`configs/distillation/low_nfe_budget_100k.json`.

| Stage | Frozen teacher | Original development FID at 25K | Milestones |
| --- | --- | ---: | --- |
| n1 | `experiments/fixed_cfg2_progressive/n2/final.pt` | 68.16127 | 50K, 75K, 100K |
| n2 | `experiments/fixed_cfg2_progressive/n4/final.pt` | 51.69442 | 50K, 75K, 100K |

Each milestone is scored against the same 755-image training-only holdout,
with 3,020 generated images and seed 42. The best of 25K/50K/75K/100K is
selected **on development FID only**. The selected n1 checkpoint is then
reported through the protected 1-NFE evaluator; the selected n2 checkpoint is
reported with the two-step internal evaluator against the official references.
Official scores are reports, not a tuning criterion. The existing submission
checkpoints are not overwritten by this ablation.

## Execution and validation

The resume implementation accepts the previous full `final.pt` in a new output
directory and permits only the budget/output bookkeeping fields to differ.
Other settings, teacher path, stage and data-manifest digest must match. A
25,000→25,001 single-update GPU smoke passed: the next update processed 64
images, changed student weights, retained optimizer state and reached 9.77 GB
peak allocated VRAM. Its temporary artifacts were removed.

The tmux runner performs n1 training and evaluation first, then n2 training and
evaluation. It writes one console log per stage plus a top-level log and exit
code. Interrupted training resumes from that run's `latest.pt`; baseline
checkpoints are never modified. Output: `experiments/low_nfe_budget_100k/`.
Launched in tmux `fastgen_low_nfe_budget` on 2026-09-30. The n1 run passed
25,500 updates and wrote its first new resumable `latest.pt`; its 25,001st
update loss matched the independent resume smoke exactly. Observed peak
allocated VRAM was 9,796,354,048 bytes while training continued. The first
attempt to invoke the runner as a file failed at import before training; the
module invocation was corrected, and the active run started from the untouched
25K checkpoint.

```bash
cd /root/workspace_distillation
tmux attach -t fastgen_low_nfe_budget
tail -f experiments/low_nfe_budget_100k/console.log
```

Machine-readable results will be in
`experiments/low_nfe_budget_100k/summary.json`. A later experiment can use a
better n2 to train a **new** n1, but that changes its teacher target and is
separate from this one-variable budget comparison.

## Completed results (2026-09-30)

Both continuations reached 100,000 updates. Each added exactly 75,000 updates
from the preserved 25K checkpoint. The final checkpoints still contain EMA and
AdamW optimizer state. An independent post-run audit checked every development
metric, selected checkpoint and official score; machine-readable evidence is
`experiments/low_nfe_budget_100k/completion_verified.json`.

| Stage | 25K development FID | 50K | 75K | 100K | Selected |
| --- | ---: | ---: | ---: | ---: | ---: |
| n1 | 68.16127 | 66.77484 | 66.15214 | **65.38627** | 100K |
| n2 | 51.69442 | 51.31244 | **50.89945** | 50.94633 | 75K |

All development scores use the same 755 training-only references, 3,020
generated images, seed 42 and EMA weights. The n1 best checkpoint scored
**58.1595 official FID** through `evaluate.py`, versus **61.4540** for the
original 25K 1-NFE checkpoint (improvement 3.2945). The n2 selected 75K
checkpoint scored **39.54904 official FID** through the internal two-step
evaluator. There is no 25K n2 official FID, so its official improvement cannot
be quantified. The existing 4-NFE submission score remains 33.0517.

The n1 development improvement was 2.77500 FID points; the n2 improvement was
0.79497. The n2 75K-to-100K difference is 0.04688 and should not be read as
evidence of overfitting from one seed. The final 1,000 added training updates
averaged loss 0.01965 for n1 and 0.00668 for n2, down from 0.02194 and 0.00820
in the first 1,000 added updates. Both measured peak allocated VRAM at
9,796,747,776 bytes, below the 20GB constraint. Model size stayed 48,883,107.

The raw tmux wrapper wrote exit code 127 because its shell script was edited
while running; after the Python runner printed “Both low-NFE budget ablations
completed”, the shell attempted a stray `dget` command. This happened after
both official scores and `summary.json` were written. The separate completed
checkpoint/metric audit passed. The initial import failure before training was
also corrected at launch; neither event invalidated the measured scores.

More updates helped n1, but its official FID is still much worse than the 4-NFE
baseline. Longer training of n2 gave a smaller holdout benefit. This experiment
does not test whether a *new* n1 trained against the stronger 75K n2 would
improve, because the n1 target was deliberately kept fixed. No subsequent
experiment has been started.
