# Teacher 실험 결과와 다음 실험

2026-10-10. 세 variant 실험 완료. 기준 지표는 primaryEMA, clippedDDIM32 CFG2,seed42,batch16,bf16, 생성3020장 / development holdout755장의 FID다. FID는 낮을수록 좋다. 공식 validation FID와 직접 비교하지 않는다.

| 모델 | 누적 step | DDIM32 holdout FID | DDIM50 holdout FID |
|---|---:|---:|---:|
| 원본 single-head | 100K |44.11891|43.57209|
| 원본 single-head |150K|42.25415|41.94749|
| 원본 single-head |200K|41.94997|41.69311|
| 4-head |100K|42.92468|42.36305|
| **4-head** |**150K**|**41.73751**|**41.37549**|
| 4-head |200K|41.93594|41.60070|
| 원본200K+LR2e-4追加10K |210K|42.50173|42.01594|
| 원본200K+LR5e-5追加10K |210K|42.87076|42.51870|
| 원본200K+dropout0.05追加10K |210K|42.50246|42.20021|
| 32×32 attention 추가 |100K|46.62329|44.11634|

## 선택과 해석

현재 측정된 teacher의 최고 품질 후보는 **4-head150K**다. 원본150K보다DDIM32 FID0.51664, 원본200K보다0.21246 낮다. 그러나 sampling seed1개와755장reference 기준이므로 원본200K보다 확실히 우월하다고 단정하지 않는다.4-head200K는150K보다DDIM32가0.19843 나빠졌고 원본200K 대비 이득은0.01402에 불과했다. 따라서200K보다150K를 후보로 유지한다.

dropout0.05 추가10K의DDIM32 FID42.50246은 같은 budget의dropout0 대조군42.50173과 사실상 같은 수준이다. DDIM50도 개선되지 않았다. 이번 짧은 warm-start 실험에서 효과가 없었으며, 처음부터dropout을 적용한 모든 실험을 배제하는 결론은 아니다.

32×32 attention은100K에서 원본100K보다DDIM32 FID2.50439 나빴다. 설정된 연장 기준을 만족하지 않아100K에서 종료했다. 메모리도약12.68GB로4-head의약10.16GB보다 높다. 지금 우선순위에서는 제외한다.

수치가 가장 높은(가장 나쁜) 이번teacher 조건은 **32×32 attention100K, DDIM32 FID46.62329**다. 최고 품질과 최대FID는 반대 의미다. student FID는 다른 모델/계산budget이므로 이 teacher 순위에 섞지 않는다.

공식 validation으로 확인된 후보는 여전히원본200K, **FID26.28631**이다.후속 공식평가에서4-head150K는25.49841로측정됐다. 자세한비교는[공식FID비교](TEACHER_OFFICIAL_FID_COMPARISON_20261010.md)를참고한다. 위holdout41.73751과공식점수는직접비교하지않는다.

## 다음 실험 우선순위

1. **후보 검증과EMA 비교:** 4-head150K와원본200K를 primaryEMA0.9999 / extraEMA0.999 및 samplingseed42/1042/2042로 평가한다. 학습 없이 가능한 비교이며0.21 차이가 seed에 걸쳐 유지되는지 확인한다. extraEMA 결과는 아직 없다. sampling repeat는 training-seed 반복과 다르다. 개발holdout으로 후보를 고정한 다음4-head150K를 공식validation에서 한 번 평가한다.
2. **4-head 주변head 수 비교:** 2-head를freshseed42로100K까지 학습하고 유망하면150K까지 진행한다. 폭·attention위치·dropout·recipe는 그대로 유지한다.1/2/4-head를동일budget으로 비교한다.4-head 결과가 좋아서 인접한 작은변경을 먼저 보는 것이며2-head가 더좋다는 증거는없다.
3. **noise 구간 진단후 Min-SNR weighting 비교:** 고정timestep/noise의train·holdoutepsilonMSE를bin별로 측정해 병목을확인한다. 이후4-head 구조에서 Min-SNR gamma5를 단독변경한 fresh100K 실험을 무가중4-head100K와 비교할 수 있다. gamma5는 첫 후보이며 보장된 최적값이 아니다. weighting을 바꾸면 trainloss 척도도변해 기존loss와직접비교하지 않는다. [원논문](https://arxiv.org/abs/2303.09556)은 timestep간gradient충돌완화를 위한lossweighting을제안한다.현재데이터에서효과는검증되지않았다.

추가budget만늘리거나32×32attention·dropout·LR를동시에바꾸는실험은 우선하지않는다. 위제안은아직실행하지않았다.

## 결과와 checkpoint

- 4-head150K: `workspace_distillation/experiments/teacher_variant_sequence/attention4/attention4_150k/final.pt`
- [variant 결과 JSON](workspace_distillation/experiments/teacher_variant_sequence/results.json)
- [LR 비교 결과 JSON](workspace_distillation/experiments/teacher_lr_comparison_200k/results.json)
- [4-head W&B](https://wandb.ai/fv2z6n929c-kaist-digital-humanities-and-social-sciences-/pokemon-diffusion/runs/4b5499eb)
- [dropout W&B](https://wandb.ai/fv2z6n929c-kaist-digital-humanities-and-social-sciences-/pokemon-diffusion/runs/77fe2f05)
- [32×32 attention W&B](https://wandb.ai/fv2z6n929c-kaist-digital-humanities-and-social-sciences-/pokemon-diffusion/runs/2537bf54)
