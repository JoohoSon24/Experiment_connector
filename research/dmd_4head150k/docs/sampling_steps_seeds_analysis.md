## Corrected disjoint-seed result (2026-09-30)

The rerun with base seeds 42/1042/2042 completed with exit code 0.
The evaluation code's 189 per-batch RNG seed ranges are disjoint. The selected
flip-only EMA model, DDIM CFG 2, same 755-image development reference and
3,020 generated images per setting were unchanged.

| Steps | NFE | Mean FID | Sample std |
| ---: | ---: | ---: | ---: |
| 20 | 40 | 46.05883 | 0.31300 |
| 50 | 100 | 44.08437 | 0.37852 |
| 100 | 200 | 44.05665 | 0.34403 |

20 -> 50 improves the mean by 1.97446. 50 -> 100 improves by only 0.02773
while doubling NFE. This reinforces a practical 50-step stopping point for the
current teacher, with no claim that the small difference is statistically
resolved. The 32-step seed-42 anchor is unchanged: 44.84112 at 64 NFE.
`experiments/teacher_sampling_steps_seeds/comparison.json` is the corrected
report; the adjacent-seed result is archived in `comparison_adjacent_seeds.json`.

# DDIM CFG2 step/seed comparison analysis

## Completed original run

Exit code 0; all 9 main settings and the 32-step anchor completed. Same flip-only
100K checkpoint, EMA 0.9999, clipped DDIM, CFG 2, bf16, batch 16, 3,020 generated
images and 755 held-out official-training reference images. No official validation
was used to select these settings. The original full report is preserved as
`experiments/teacher_sampling_steps_seeds/comparison_adjacent_seeds.json`.

| Steps | NFE | Seed 42 | Seed 43 | Seed 44 | Mean | Sample std |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 40 | 46.38084 | 46.58908 | 46.03201 | 46.33398 | 0.28147 |
| 50 | 100 | 44.36898 | 44.33235 | 44.04426 | 44.24853 | 0.17785 |
| 100 | 200 | 44.28989 | 44.21722 | 43.94968 | 44.15226 | 0.17917 |

32-step seed-42 anchor: FID 44.84112, 64 NFE. This is 0.47214 above the same-seed
50-step value, at 36% fewer evaluations; only one anchor seed was measured.

## Interpretation

- 20 -> 50 steps improves mean FID by 2.08545 (~4.50%), at 2.5x NFE.
  All three individual seeds improve by approximately 1.99–2.26 points.
- 50 -> 100 steps improves mean by 0.09627 (~0.22%), at 2x NFE.
  All three paired improvements are positive (0.07909, 0.11513, 0.09458).
  Thus a small consistent observed benefit exists; it should not be described
  as a proof of equality or significance. 50 steps is a practical cost/quality
  choice among these measured settings. 100 steps is numerically best here.
- The plateau makes insufficient sampling steps above 50 an unlikely major
  explanation for remaining defects in this specific sampler/CFG/checkpoint.
  It does not isolate data limitations from score-learning error or exclude
  benefits from other samplers/schedules. It says nothing yet about 1/4-step
  student quality: distillation must learn a new coarse transition mapping.
- Small-reference FID is not comparable numerically to official FID 25.12628.
  The reported std is sampling-seed variation, not a training-seed estimate,
  confidence interval or test for statistical significance.

## Seed overlap discovered and corrected

Teacher evaluation seeds each batch with `base_seed + batch_index`, using 189
batches. Adjacent base seeds 42/43/44 therefore reuse random streams in shifted
batches. CPU SHA256 checks found **600 identical same-class PNGs** between
42/43 and between 43/44 for each of the 20/50/100-step sets (~19.9% of images).
42/44 had no exact same-class image overlap, but shares batch RNG seeds with
shifted categories. These are not three independently drawn sample sets.
Scores and paired step comparisons remain valid for those artifacts; the small
std must not be overinterpreted as independent-seed robustness.

`scripts/compare_teacher_steps_seeds.py` now uses base seeds **42/1042/2042**:
its 189-batch RNG seed ranges are disjoint. Seed 42 is reused, the other two
seeds generate fresh balanced samples for all three step counts, and the old
report remains preserved. The new report will be in comparison.json. No trained
weights, PNGs or protected evaluation files were changed. This corrects our
experimental design rather than changing the challenge metric.

The current no-augmentation training already started successfully; it is not
interrupted. The queued distillation runner executes the corrected seed repeat
first, after the teacher training/holdout evaluation finishes. Logs:
`experiments/teacher_disjoint_seeds_console.log`, status:
`experiments/teacher_disjoint_seeds_exit_code`.

## Current pipeline

Original sampling comparison completed -> fresh no-augmentation teacher is
running (B64, 100K, constant LR 2e-4; finite loss, ~9.11 GiB peak at startup) ->
corrected disjoint-seed comparison -> teacher selection and fixed-CFG/progressive
student training -> actual one-/four-NFE official FID.
No student FID is available yet. This analysis does not launch another teacher
or change the previously authorized training/update budgets.
