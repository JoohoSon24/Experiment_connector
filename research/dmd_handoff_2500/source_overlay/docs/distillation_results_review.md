# Fixed CFG 2 distillation results and 1-NFE diagnosis

## Completion and scores (2026-09-30)

The full sequential experiment exited with code 0. Both FastGen checkpoints
exist and were scored through the protected `evaluate.py` entry point.
All learned sampling models have 48,883,107 parameters (<60M); smoke hooks
measured exactly one and four backbone evaluations. Training peak allocated
VRAM was 9.12 GiB (n4/n2/n1 9,796,461,056 bytes). The full teacher used
64 NFE for the matched DDIM32 CFG2 reference; it is not a submission model.

| Stage | Training updates | Development FID (755 training holdout refs) | Official FID (3020 validation refs) | NFE |
| --- | ---: | ---: | ---: | ---: |
| Selected noaug teacher, DDIM32 CFG2 | 100,000 teacher | 44.11891 | 26.65365 | 64 |
| Fixed guidance transfer | 50,000 | 43.25620 | — | 32 |
| Progressive n16 | 25,000 | 45.26116 | — | 16 |
| Progressive n8 | 25,000 | 46.38606 | — | 8 |
| Progressive n4 / `ModelFewNFE` | 25,000 | 47.62077 | **33.0517** | **4** |
| Progressive n2 | 25,000 | 51.69442 | — | 2 |
| Progressive n1 / `ModelOneNFE` | 25,000 | 68.16127 | **61.4540** | **1** |

Official numbers are from `experiments/fixed_cfg2_progressive/n4/official_evaluation`
and `n1/official_evaluation`, with the submission checkpoints
`checkpoints/few_nfe.ckpt` and `checkpoints/one_nfe.ckpt`. Both use 151 classes,
20 images/class, the same seed and exact same protected reference set.
**All 3,020 generated PNGs for each stage exactly match byte-for-byte between
its development and official evaluation directories**. Their two FIDs differ
only because the reference sets differ, not because a different student sampled.
The development and official scores must not be compared numerically to each
other.

The 2→1 internal holdout increase is **16.46686** FID points. Earlier halvings
increase FID by 2.00496, 1.12489, 1.23471, and 4.07365 points respectively.
The final 4→1 official FID gap is 28.4023. Thus the largest *within-holdout*
deterioration is specifically the n2→n1 transition. The one-step result is not
simply a different sampling setting applied to the 4-step checkpoint; it is a
separately trained student initialized from n2 and trained for only 25K updates.

## Complete class sheets

`scripts/make_student_class_grids.py` generated, from **existing official
sampler PNGs**, two sample-index-aligned contact sheets: indices 0 and 10.
Every sheet includes all 151 class names/IDs and one image/class/model. In the
combined sheet, each cell is 4-NFE above 1-NFE. Each model also has a separate
one-PNG sheet per index. The 64x64 originals are enlarged 2x with nearest-neighbor
only for inspection; no generation, FID file or learned image was modified.
`quality_review/metadata.json` and `metadata_sample10.json` record source paths
and SHA256 digests for all class images.

Observed: both outputs generally preserve recognizable class palettes and
silhouettes, but 1-NFE has softer boundaries, weaker small features and
occasional warped details/backgrounds. The paired images share class, index and
starting RNG seed, making the difference inspectable directly. This is a
qualitative observation, not proof that semantic diversity collapsed.

Across all 3,020 official generated PNGs per model:

| Simple pixel statistic | 1 NFE | 4 NFE |
| --- | ---: | ---: |
| Mean within-class per-pixel variance | 0.08679 | 0.09254 |
| Mean adjacent-pixel absolute difference | 0.03881 | 0.04538 |
| Fraction of channel values near 0 or 1 | 0.14717 | 0.17390 |

These diagnostics are computed on RGB values mapped to [0,1] from saved PNGs.
The 1-NFE set has lower pixel contrast/high-frequency detail and slightly less
pixel-level variation. These **are not precision, recall, or perceptual diversity
metrics**, so do not by themselves determine why FID worsened.

## Why one step is weak: evidence, hypotheses and code audit

1. **The final compression is harder:** n1 must approximate the entire n2
   two-transition result from pure Gaussian noise and the class in one pass.
   Earlier stages split this mapping. Progressive distillation errors accumulate
   across stages and the final internal FID jump is much larger than preceding
   jumps. This is an evidence-backed difficulty diagnosis, not proof that one
   pass is mathematically impossible or that the loss inherently blurs targets.
   The progressive target is deterministic for a fixed start, class and teacher.
2. **A short final-stage budget:** current n2 and n1 each received 25,000 updates.
   The n1 first-1000 mean training loss was 0.02498 and last-1000 mean 0.02235,
   so it was still falling near the stop; this is consistent with undertraining,
   although lower training loss need not reduce FID. Meng et al.'s supplemental
   implementation description uses 100,000 updates for one-/two-step stages,
   compared with 50,000 for other stages. Their settings differ from ours and
   are not a guaranteed recipe; our user policy retains constant LR.
   [Primary supplemental](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Meng_On_Distillation_of_CVPR_2023_supplemental.pdf).
3. **The current teacher and time-grid adaptation:** our high-noise endpoint is
   t=999 with a small nonzero alpha, while the highest-node training input is
   pure noise. The original progressive-distillation paper uses a zero terminal
   signal-to-noise ratio for that node. We kept the trained teacher's existing
   schedule, so changing this now would require a new controlled teacher.
   This is a plausible second-order mismatch, not demonstrated as the cause.
   [Progressive Distillation](https://arxiv.org/abs/2202.00512).
4. **No obvious arithmetic or NFE implementation failure found:** numerical
   two-to-one endpoint inversion was accurate to 4.77e-7 on the actual FP32
   nested grid; v target/loss identities, checkpoint loading, deterministic
   sampling and empirical backbone call counts passed the smoke test. The base
   Model/evaluator remain unchanged. This cannot exclude subtle learning or
   distribution-shift errors. Training uses noisy real data for nonterminal
   nodes while inference follows generated trajectories, which is an additional
   hypothesis to examine before a larger algorithm change.

## Controlled improvement order

1. **One-variable budget ablation:** preserve the n2 teacher, architecture,
   objective, data split, batch 64 and constant LR 2e-4; continue n1 from its
   exact 25K optimizer/RNG/EMA checkpoint toward 50K/75K/100K updates in a new
   run directory. Select using the 755-image training-only holdout. Preserve the
   present 25K checkpoint and official report. A same-teacher continuation is
   the cleanest first test of undertraining; do not promise improvement.
2. If n1 alone plateaus, test **n2 25K→100K** with its fixed n4 teacher, then
   derive a fresh n1. This changes the target quality as a separate experiment.
   It is motivated by the paper's longer very-low-NFE training, not by an
   unmeasured claim about the current n2 optimum.
3. On frozen checkpoints, compare teacher two-step targets with student
   one-step outputs for matched seeds/classes; stratify by class, pixel
   high-frequency error, saturation and holdout-vs-training examples. Add a
   proper quality/coverage metric such as improved precision/recall on the
   training-only development split if the chosen implementation is compatible
   with the existing dependency policy. This separates blur from diversity.
   [Improved Precision and Recall](https://arxiv.org/abs/1904.06991).
4. If budget increases do not address the large n1 gap, implement and test
   **consistency distillation** as the next independent project baseline rather
   than tuning architecture or several losses at once. Consistency models are
   specifically designed for one-/few-step generation; compare fixed teacher,
   fixed split and exact 1-/4-NFE FID. This is a new method, not an outcome of
   the present progressive experiment.
   [Consistency Models](https://arxiv.org/abs/2303.01469).

The TA README reference scores (39.15 one-NFE / 30.90 few-NFE) show the current
submission gap: +22.3040 and +2.1517 FID respectively. We should preserve the
functional 4-NFE baseline while investigating the weaker one-step result.
No new training was started as part of this results review.
