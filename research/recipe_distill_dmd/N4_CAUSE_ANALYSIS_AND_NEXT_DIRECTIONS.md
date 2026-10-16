# 4-step 악화 원인 분석과 후속 방향

현재 학습 supervisor·설정·체크포인트·DMD 대기열은 변경하지 않았다. 고정된 n8/n4 EMA에 대한 별도 inference-only 진단을 수행했다. 새 학습 실험은 시작하지 않았다.

## 결론의 범위

현재 가장 직접적인 증거는 **8→4 압축의 target을 현 student/optimizer recipe가 충분히 재현하지 못한다**는 것이다. 실패가 발생하는 단계는 좁혔지만, 기존보다 나빠진 원인이 attention heads인지, teacher trajectory/target 특성인지, 학습 설정인지 확정하려면 통제 실험이 필요하다. 기존 n8 full checkpoint는 없어서 옛/새 n8 target의 난이도를 같은 입력에서 직접 비교하지 못했다.

## 확인된 결과

- 기존4-NFE checkpoint의 Official FID를 현재 환경에서 **33.05174275**로 재현했다. 새4-step25K는 **36.03177398**, 차이+2.98003123이다. 생성3020장/reference3020장, seed42, batch16, EMA, bf16, 동일한 compute_fid를 사용했다.
- 새32-step guidance student Official FID **24.46447**은 새 teacher25.49841보다 낮다. 전체 CFG transfer의 실패로 설명하기 어렵다.
- Development FID의8→4 증가량은 기존 **+1.23471**, 새 **+2.85070**이다. 새16/8-step은 같은 holdout에서 기존보다 낮지만4-step에서 순위가 뒤집힌다.
- 새 n4 development FID는10K **48.41614** →20K **48.42895** →25K **48.66473**이다. 25K까지 더 학습한 것이 해당 holdout을 개선하지 않았다. 단일seed의 작은 차이만으로 overfitting을 확정하지 않는다.
- n4 loss의10K/20K/25K 직전1000-update 평균은 **0.008800 / 0.008695 / 0.008643**이다. 후반부는 plateau에 가깝다. 이 결과만으로 무조건 학습 budget을 늘리면 개선된다고 말할 수 없다.

## 구현 및 평가 audit

원본과 새 distillation/core.py의 SHA256은 모두 `45c4cff601e442f01a42b7af5b19fd52474901c720c9cc76e60957d55441357b`이다. DDIM 두 전이의 merge target, epsilon/v 변환, truncated-SNR loss 수식을 바꾸지 않았다. 원본 trainer와의 diff는 W&B/평가 callback 및 checkpoint 저장·보존 방식이다. 평가 callback은 CPU/CUDA/Python RNG를 복원한다.

4-head 모델의1/4-NFE export와 직접 sampler 결과는 preflight에서 exact parity를 통과했다. 실제 batch64 학습과 AdamW/RNG resume도 통과했다. 이는 명백한 weight loading, head 의미 변경, NFE 카운트 오류의 가능성을 낮추지만 모든 미세 구현 차이를 배제하는 증명은 아니다.

기존은100K1-head teacher/student, 새 모델은150K4-head teacher/student다. Head count와 teacher budget을 함께 바꾸었으므로 FID 차이의 원인을 어느 하나에만 귀속할 수 없다. 파라미터 수가 같아도 attention 연산은 다르다.

## 실제 frozen-model 진단

별도 script로 각 분포 64 sample, microbatch8, seed123456을 조사했다. 학습 이미지와holdout 이미지를 noising한 입력, n8 teacher가 생성한 중간 상태, n4 student가 생성한 중간 상태를 비교했다. Official FID나 인과효과 추정이 아닌 작은 탐색적 측정이다.

n4 sampling times는 `[999,741,483,226]`이다. Student가 예측한 x0와 n8 teacher의 두 전이를 하나로 merge한 target x0 간 MSE를 측정했다.

| 시간점 | Holdout noised-real x0 MSE | Student 생성 경로 x0 MSE | 생성 경로의 한 전이 endpoint MSE |
|---:|---:|---:|---:|
| 999 | 0.0008121 | 0.0008615 | 0.00000262 |
| 741 | 0.0098888 | 0.0106894 | 0.00064080 |
| 483 | 0.0085391 | 0.0094618 | 0.00297263 |
| 226 | 0.0017776 | 0.0018306 | 0.00183060 |

**중간 시간점741/483에서 x0 target fitting 오차가 크고, 전이 이후의 오차는483 구간에서 가장 크다.** 따라서 전체 모델 재학습보다 해당4-step 압축을 우선 다룰 근거가 있다. 서로 다른시간의 target scale과 전이 계수가 다르므로 MSE만으로 FID 영향도를 확정할 수 없다.

생성 경로의 x0 오차는 holdout noised-real 대비741에서 **+8.10%**, 483에서 **+10.80%**였다. 학습 입력/추론 입력 mismatch가 있을 수 있지만, 이 측정에서는 압도적인 차이가 아니었다. 지금 rollout mismatch를 단일 주원인으로 지목하는 것은 이르다.

같은입력에서 bf16/FP32 teacher target을 비교한 MSE는741/483에서 student fitting MSE의 대략 **0.14–0.18%**, 최고 noise 999에서 **3.1–3.3%** 수준이었다. 이 진단에서는 bf16 수치오차가 주된 fitting 오차를 설명하지 못했다. FP32 전면전환은 현재 우선순위가 낮다.

## 이미지 변화

동일한3020 PNG를 비교했다.

| 단순 pixel 통계 | 기존4-step | 새4-step |
|---|---:|---:|
| 인접pixel 절대차 평균 | 0.0453845 | 0.0421603 (-7.10%) |
| Class 내부pixel variance | 0.0925379 | 0.0837526 (-9.49%) |
| Pixel 값이0/1 근처인 비율 (≤1/255 또는≥254/255) | 0.0732100 | 0.0467144 |

새 모델에서 pixel 대비/변동이 작아졌다. 디테일이나 다양성 감소와 양립하는 관찰이지만, perceptual diversity·precision/recall·mode collapse의 증거로 해석하지 않는다. 클래스와배경/스타일 차이도 영향을 준다.

## 원인 가설 우선순위

1. **현재 8→4 target에 대한 fitting/optimization 병목**: 직접적인 단계별FID, loss plateau, frozen-target MSE가 지지한다. 기존 teacher보다 target 자체가 더 어려워졌는지는 기존 n8가 없어 확인하지 못했다.
2. **중간student 오차와 새로운trajectory 특성**: 32-step은좋지만 압축한8-step Official29.84723은32-step24.46447보다 높다.4-step이 배우는 직접teacher는 원래좋은epsilon teacher가 아니라 이미압축된n8이다. 이런target계보가 추가손실을 전달할 가능성이 있다. 이번데이터의해석이며 인과 증명는 아니다.
3. **4-head student의 few-step 적합성**: 학습조건의실제변수이지만 현재 직접 근거는 없다. 좋은teacher attention구조가 큰step을예측하는student에도최적이라는보장은이번결과에서확인되지않았다.
4. **학습/추론입력분포차이**: 작은측정에서8–11% 추가오차로관찰됐다. 보조개선후보이다.
5. **정밀도·평가·명백한merge수식오류**: 기존FID재현,수식동일,preflight,FP32진단을고려하면우선순위가낮다. 완전히배제한것은아니다.

## 후속 실험 제안 — 현재 대기열과 별도, 미실행

### 1. 먼저: 같은 n4 EMA에서 낮은 LR refinement

- Frozen parent: 새 n8 EMA 25K.
- 공통 student 시작점: 새 n4 EMA 25K.
- 동일한 fresh AdamW로 **LR 2e-4 대조군 / 5e-5 실험군**을 시작한다.
- 각각 추가 10K. Development FID를 2.5K/5K/10K에서 측정한다.
- Batch64, noaug, v loss, seed42, EMA0.999, clip1을 고정한다.

전체 progressive를 다시 돌리지 않고 후반 plateau와 FID 악화를 낮은 LR로 개선할 수 있는지 확인한다. 무조건 budget을 늘리는 것보다 비교할 변수가 명확하다. 개선은 보장하지 않는다.

현재 n4 parent는 완료 후 EMA로 축소되어 **원래 25K AdamW/RNG가 없다**. 두 분기 모두 동일한 EMA와 fresh optimizer에서 시작해야 한다. 이를 원래 full optimizer의 exact continuation이라고 부르면 안 된다.

### 2. 새로운 방향: 좋은 32-step guidance 모델에서 직접 4-step target 생성

현재는 **n8 teacher의 두 전이 → n4 한 전이**를 학습한다. 비교군에서는 Official FID 24.46447인 **32-step guidance EMA의 여덟 전이 → n4 한 전이**를 target으로 사용한다. 같은 n4 시작 weights, LR, batch, 시간 grid로 두 target을 비교한다.

기존 merged_target inversion을 사용하되, teacher의 32-step grid에서 여덟 번 전이하고 guidance parent의 clip 규칙을 유지해야 한다. Merge된 target을 clamp하면 안 된다.

이 실험은 intermediate n16/n8의 근사 오차가 target으로 전달되는 경로를 줄일 수 있는지 검증한다. Teacher의 원래 학습 step이나 weight는 바꾸지 않는다. Target당 teacher forward가 2회에서 8회로 늘며, 더 정확한 target이 student에 더 어려운 mapping일 수도 있다. 기존 논문의 표준 factor-two baseline과 구분하는 후속 가설 실험이다.

### 3. 보조 방향: inference 경로의 입력을 일부 섞기

같은 n8 target을 유지하고, **기존 noised-real 75% + 고정된 n4 EMA rollout의 중간 상태 25%**로 학습한다. Nonterminal 시간점 741/483/226에 적용하고 999의 pure-noise 조건은 유지한다. 새 noise를 계속 생성해 작은 고정 pair pool에 반복적으로 맞추는 문제를 피한다.

관찰된 생성 경로의 추가 오차를 겨냥한다. 현재 차이가 압도적이지 않으므로 낮은 LR와 target 변경 다음 순위다. 약한 student의 rollout을 너무 많이 학습하면 기존 문제를 고착시킬 수 있어 25%부터 비교한다.

### Architecture 원인 규명은 그다음

위 실험으로 개선되지 않으면 동일한 4-head 150K teacher를 고정하고, **student만 1-head / 4-head**로 나누어 CFG transfer부터 progressive까지 비교한다. Teacher head 수와 student head 수를 분리하는 실험이다. 4-head checkpoint를 1-head로 바꿔 그대로 추론하면 함수가 달라지므로 공정한 구조 비교가 아니다. 비용이 큰 별도 실험이며 이번 분석에서 시작하지 않았다.

## 평가와 체크포인트 보존

후속 실험의 선택 기준은 755-reference development FID다. 선택한 모델의 Official FID를 동일한 3,020 reference, seed42, batch16으로 보고하고, 기존 4-step 33.05174275와 새 baseline 36.03177398를 함께 남긴다. Loss 감소만으로 성공을 판정하지 않는다.

현재 10K/20K의 FID용 PNG는 있지만 당시 n4 weights는 보존되어 있지 않아 그 checkpoint로 이어 학습할 수 없다. 후속 분기는 milestone EMA를 저장해야 한다. 현재 예약된 student 학습과 DMD 4개는 유지한다.

## 자료

- [공식progressive distillation논문](https://arxiv.org/abs/2202.00512): deterministic sampler의factor-two compression을반복하는방법.
- [Guided diffusion distillation논문](https://arxiv.org/abs/2210.03142): CFG transfer후progressive distillation의두단계방법.
- 실제진단script: scripts/diagnose_n4.py
- 원시target진단: experiments/n4_cause_analysis/target_audit.json
- 원시pixel통계: experiments/n4_cause_analysis/image_statistics.json
- 기존/새4step재평가: OLD_VS_NEW_4STEP.md
