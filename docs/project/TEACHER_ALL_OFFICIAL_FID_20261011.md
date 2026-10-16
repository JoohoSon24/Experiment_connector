# Teacher Official FID 전체 비교

2026-10-11. 후속실험 및 officialreport 모두 완료. clippedDDIM32 CFG2,batch16,bf16,생성3020장,공식reference3020장. 낮을수록 좋다. 아래 seed42 비교와 samplingseed 반복 평균을 구분한다.

## 기존·구조·fine-tuning 비교: seed42

| 모델 | EMA | OfficialFID |
|---|---|---:|
| 이전noaug100K (과거기록) |0.9999|26.65365|
| 원single-head200K |0.9999|26.28631|
| attention4_100k |0.9999|26.02626|
| attention4_150k |0.9999|25.49841|
| attention4_200k |0.9999|26.77276|
| dropout005_210k |0.9999|27.22299|
| attention32_100k |0.9999|28.66605|
| 4-head150K에서 control_lr2e4_155000 |0.9999|25.89570|
| 4-head150K에서 control_lr2e4_160000 |0.9999|26.09855|
| 4-head150K에서 finetune_lr1e4_155000 |0.9999|26.17752|
| 4-head150K에서 finetune_lr1e4_160000 |0.9999|27.36546|

## EMA 및samplingseed 반복

| 모델 | EMA | seed42 |seed1042|seed2042|평균±표본표준편차|
|---|---|---:|---:|---:|---:|
|attention4_150k|ema09999|25.49841|24.90893|25.32272|25.24336 ± 0.30265|
|attention4_150k|ema0999|27.99855|27.29224|28.02494|27.77191 ± 0.41562|
|singlehead_200k|ema09999|26.28631|26.23754|26.25944|26.26110 ± 0.02442|
|singlehead_200k|ema0999|28.80471|28.52178|28.42313|28.58321 ± 0.19807|

## 결론

**고정 seed42의 가장 좋은 후보는4-head150K / primaryEMA0.9999, OfficialFID25.49841이다.**3 samplingseed 평균은25.24336으로 원single-head200K의26.26110보다 약1.01774 낮고, 세 seed 모두에서 개선됐다. 이는trainingseed 반복이나reference 불확실성 평가가 아니다. 단일 최저점24.90893은seed1042 결과이며, seed를 골라 공식 기준seed42 결과를 대체하지 않는다.

EMA0.999는 두 모델 모두 나빴다.4-head150K에서 추가5K/10K 학습도LR2e-4와1e-4 모두 기준25.49841보다 나빴다. 현재는4-head150K / EMA0.9999를 유지한다. 공식평가로hyperparameter/checkpoint를 다시 선택하지 않고development에서 선택한 후보를 보고한다.

[후속16조건 rawmetrics](workspace_distillation/experiments/teacher_4head150k_followup/official_report/results.json) · [기존variant metrics](workspace_distillation/experiments/teacher_variant_sequence/official_comparison/results.json)

원single-head200K에서 수행한 과거LR2e-4/5e-5 추가학습은 별도실험이며 공식평가는 아직 없다. 이번에 측정한 공식값과 구분한다.
