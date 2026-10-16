# Teacher 공식 FID 비교

2026-10-10. primaryEMA, clipped DDIM32 CFG2, seed42, batch16, bf16, 생성3020장 / 공식validation reference3020장으로 비교했다. 개발holdout 평가 때 만든 동일 PNG를 재사용하고 reference만 변경했다. 체크포인트·sampler·step·seed·batch·precision·파일 개수의 일치를 evaluator가 검증했다.

| 모델 | 누적 step | Official FID ↓ |
|---|---:|---:|
| 이전 선택 noaug teacher |100K|26.65365 (과거 기록)|
| 이번 원본 single-head |200K|26.28631|
| attention4_100k | 100000 | 26.02626 |
| attention4_150k | 150000 | 25.49841 |
| attention4_200k | 200000 | 26.77276 |
| dropout005_210k | 210000 | 27.22299 |
| attention32_100k | 100000 | 28.66605 |

현재 보고된 가장 낮은 공식FID는 **4-head150K:25.498414939550457**이다. 개발holdout에서 이미 선택한 후보로 공식평가에서도 좋은 결과를 보였다. 기존200K26.286307832723594 대비0.787892893173137 개선(약3.00%)이며, 이전100K기록26.65365 대비1.155235060449543 개선(약4.33%)이다. 반복seed나통계적유의성은아직검증하지않았다. 공식FID가가장높은조건은32×32attention100K:28.666051090037456이다. FID는낮을수록좋다.

공식점수는다른reference의holdout41.73751과직접비교하지않는다. 신규학습/gate에이공식결과를사용하지않고개발holdout으로후속실험을선택한다.

[원본 결과 JSON](workspace_distillation/experiments/teacher_variant_sequence/official_comparison/results.json)
