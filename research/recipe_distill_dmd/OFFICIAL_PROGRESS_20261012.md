# Distillation 진행 상황 및 Official FID

확인 시각: 2026-10-12T17:07:25.150145+00:00

현재 단계: n2, 400/25,000 step. 상태: distillation_training.

## 완료 결과

동일한 reference 3,020 PNG와 generated 3,020 PNG, seed42, batch16, EMA, bf16 생성 결과를 보호된 evaluate.compute_fid로 평가했다. Development reference는755장이다. 기존 생성 PNG를 재사용했으며 학습은 계속 진행했다.

| 모델 | 학습량 | 실제 NFE | Official FID | Development FID |
|---|---:|---:|---:|---:|
| 선택 teacher (기측정) | 150K | 64 (DDIM32 CFG2) | 25.4984 | 41.7375 |
| CFG transfer student | 50K | 32 | 24.4645 | 42.6139 |
| Progressive 16-step | 25K | 16 | 26.8949 | 44.2538 |
| Progressive 8-step | 25K | 8 | 29.8472 | 45.8140 |
| Progressive 4-step | 25K | 4 | 36.0318 | 48.6647 |

현재 완료된 student 중 Official FID 최저는32-step guidance student **24.4645**다. 다만4-step 최종 결과는 **36.0318**로 기존4-NFE baseline33.0517보다 **2.9801** 높다. 좋은 teacher가 저NFE student의 개선을 자동으로 보장하지는 않는다. 이번 결과를 따라 기존 대기열은 그대로 진행한다.

4-step20K의 중간 Official FID는35.7365였다. 해당 중간 step의 optimizer checkpoint는 보존하지 않아 최종25K 모델과 구분한다.

## 남은 학습: 현재 단계를 포함해8개

| 순서 | 학습 | 상태 / 예산 |
|---|---|---|
| 1 | Progressive2-step | 진행 중: 400/25K |
| 2 | Progressive1-step | 대기:25K |
| 3 | n1 budget extension | 대기:25K→100K (+75K) |
| 4 | n2 budget extension | 대기:25K→100K (+75K) |
| 5 | DMD A:reg0.1, G LR2e-5 | 대기:2.5K |
| 6 | DMD B:reg0, G LR2e-5 | 대기:2.5K |
| 7 | DMD C:reg0.3, G LR2e-5 | 대기:2.5K |
| 8 | DMD D:reg0.1, G LR1e-5 | 대기:2.5K |

요청한 DMD 실험4개는 아직 모두 대기 중이다. 앞의4개는 DMD 출발 student를 만드는 학습 단계다. 학습 단계 사이에 정해진 FID/qual/checkpoint 검증을 실행하며 supervisor가 자동으로 다음 단계로 넘어간다.

## 파일과 W&B

- [Official FID 측정 W&B](https://wandb.ai/fv2z6n929c-kaist-digital-humanities-and-social-sciences-/pokemon-diffusion/runs/07hhndrr)
- [현재 학습 W&B](https://wandb.ai/fv2z6n929c-kaist-digital-humanities-and-social-sciences-/pokemon-diffusion/runs/9c1e5086)
- 원시 Official 측정 결과: experiments/recipe/official_progress/summary.json
- 실시간 상태: experiments/recipe/status.json
- 전체 로그: experiments/recipe/console.log
- Teacher와 student의 전체 계보 및 통제 조건: EXPERIMENT_PLAN.md
