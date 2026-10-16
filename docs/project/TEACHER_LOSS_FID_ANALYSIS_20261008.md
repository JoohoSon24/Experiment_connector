# Noaug teacher: loss와 FID 분석

2026-10-08, 로컬 완료 로그와 milestone metrics 기준. Run: [W&B 996c9a40](https://wandb.ai/fv2z6n929c-kaist-digital-humanities-and-social-sciences-/pokemon-diffusion/runs/996c9a40). 학습은 fresh seed42로 시작해 200K까지 완료했다. 원래 100K checkpoint에서 재개한 실험은 아니다.

## 측정 결과

| 누적 update | 마지막 10K 구간 평균 train loss | DDIM32 CFG2 holdout FID | DDIM50 CFG2 holdout FID |
|---:|---:|---:|---:|
| 100K | 0.016439 | 44.11891 | 43.57209 |
| 150K | 0.014973 | 42.25415 | 41.94749 |
| 200K | 0.013731 | 41.94997 | 41.69311 |

loss는 stdout에 기록된 20-update 간격 값 중 `milestone-10000 < step <= milestone`인 500개를 평균했다. 모든 update의 평균이 아니다. 마지막 단일 batch loss는 0.010164이며 대표값으로 사용하지 않았다. 각 구간 loss 표준편차는 약 0.00497, 0.00487, 0.00502이다. batch 이미지·noise·timestep이 달라져 변동하므로 단일 step끼리 비교하지 않는다.

평가는 primary EMA, development reference755장 / 생성3020장, seed42, batch16, bf16, clipped DDIM을 유지했다. reference가 적으므로 작은 FID 차이의 일반화를 확정하지 않는다. 공식 validation은 후속 측정에서 200K DDIM32 CFG2 FID 26.28631로 확인했다. 과거 official FID 26.65365보다 0.36734 낮다. 위 holdout 표와 official 수치는 reference가 달라 직접 비교하지 않는다.

## 해석

- 100K → 150K: 평균 loss 약 8.9% 감소, DDIM32 FID 1.86476 개선, DDIM50 1.62460 개선.
- 150K → 200K: 평균 loss 약 8.3% 감소, DDIM32 FID 0.30418 개선, DDIM50 0.25438 개선.
- 100K → 200K: 평균 loss 약 16.5% 감소, DDIM32 FID 2.16894 개선(약 4.9%).

학습 budget 확대는 두 sampler 조건 모두에 이득이 있었다. 그러나 마지막 50K의 DDIM32 개선 폭은 앞선 50K의 약 16%에 그친다. 현재 기록은 **FID 개선의 둔화**를 보여준다. 세 지점만으로 완전 수렴, overfitting, class fidelity 개선을 판정할 수 없다. holdout denoising loss와 timestep별 loss가 여기에는 없어 train/holdout loss gap도 알 수 없다.

훈련 loss는 raw model의 noise 예측 목적이고, FID는 EMA 모델을 CFG2·유한 DDIM step으로 샘플링한 결과다. 평가 대상과 목적이 다르므로 loss의 동일 비율 감소가 FID의 동일 비율 개선을 뜻하지 않는다. grad norm의 마지막 값은 0.01251, peak allocated GPU memory는 약 9.78GB이며 로그상 loss 폭주를 보이지 않았다. 낮은 grad norm만으로 학습 정체를 확정하지 않는다.

DDIM50이 DDIM32보다 좋은 폭은 100K 0.54681 → 150K 0.30666 → 200K 0.25686으로 줄었다. 200K에서는 32에서 50으로 step을 늘리면 CFG의 실제 forward 수가 64에서100으로 늘지만 FID 이득은 약0.257이다. 속도를 고려하면 DDIM32를 주 선택 조건으로 유지하는 것이 합리적이다.

현재 선택 checkpoint: `workspace_distillation/experiments/teacher_budget_noaug_fresh_200k/teacher_b64_noaug_fresh_200k/final.pt`.

## 다음 실험 우선순위

### 1. 200K에서 낮은 constant LR fine-tuning

가장 먼저 LR 2e-4와 5e-5 두 분기를 동일 200K full checkpoint에서 시작해 추가10K, 필요하면25K까지 비교한다. 두 분기 모두 raw model·optimizer·EMA·RNG를 그대로 복원하고 noaug, constant LR, warmup0, batch64를 유지한다. DDIM32/50 CFG2 평가도 유지한다.

현재 큰 LR에서 loss는 계속 내려가지만 FID 이득이 작아져 작은 update로 추가 개선할 수 있는지 확인할 가치가 있다. 이는 가설이며 개선을 보장하지 않는다. 같은 budget의 2e-4 대조군 없이 낮은 LR만 실행하면 단순 추가 학습 효과와 구분할 수 없다. 현 budget-only 실행기의 recipe guard는 LR 변경을 거부하므로 별도 실험 설정이 필요하다. 제안만 했으며 아직 추가 학습을 실행하지 않았다.

### 2. 평가 안정성과 noise 구간 진단

200K 후보를 seed42 /1042 /2042 등 추가 sampling seed에서도 같은 batch16·sample count로 평가한다. 이전 milestone도 같은 seed로 비교해 마지막0.30 FID 차이가 안정적인지 확인한다. 이것은 generation seed 변화의 확인이며 training seed 재현성과는 다르다. 작은 reference set 자체의 한계도 제거하지 않는다.

별도 진단으로 augmentation 없는 train subset과 holdout에서 고정 이미지·timestep·noise의 epsilon MSE를 계산하고 timestep 또는 log-SNR bin별로 기록한다. 현재 stochastic train loss만으로 overfitting 여부와 어떤 noise 구간이 병목인지 알 수 없기 때문이다. official validation으로 hyperparameter를 선택하지 않는다.

### 3. Weak flip과 architecture는 다음 단계

augmentation은 flip0.25만 별도 분기로 시험한다. 기존 100K DDIM50 CFG2에서 flip-only44.36898은 noaug43.57209보다 나빴고 affine+flip48.40064는 더 나빴으므로 강한 변형을 먼저 넣을 근거가 없다. matched-budget noaug 대조군을 두고 형태·방향성을 함께 확인한다.

architecture는 현재 위치에서 single-head→4-head attention을 첫 후보로 유지하되, 같은 checkpoint의 단순 continuation으로 취급하지 않는다. weight shape가 같아도 attention 함수가 바뀌며 별도 구조·학습 실험으로 비교해야 한다. 현재 결과만으로 model capacity가 병목이라는 증거는 없다.

### 4. Noise loss weighting은 진단 후 후보

특정 noise 구간의 충돌이나 불균형이 확인되면 Min-SNR weighting을 별도 recipe로 비교할 수 있다. [원 논문](https://arxiv.org/abs/2303.09556)은 timestep을 여러 학습 과제로 보고 weighting으로 충돌을 완화하는 방법을 제안한다. 해당 논문의 효과가 현재 Pokemon teacher에서도 그대로 발생한다고 가정하지 않는다. weighting을 바꾸면 loss의 척도도 달라져 기존0.01373과 직접 비교할 수 없다. 지금 LR·augmentation·architecture와 동시에 바꾸지 않는다.

## 근거 파일

- `workspace_distillation/experiments/teacher_budget_noaug_fresh_200k/status.json`
- 같은 디렉토리의 `milestone_100000.json`, `milestone_150000.json`, `milestone_200000.json`
- 각 `teacher_b64_noaug_fresh_{100,150,200}k/console.log`
- [앞선 개선 우선순위](TEACHER_IMPROVEMENT_PRIORITIES.md)

## 후속 공식 validation 평가

200K primary EMA / clipped DDIM32 CFG2 / seed42 / batch16 / bf16 / 64 NFE, 생성3020장 및 공식 reference3020장으로 측정했다. FID는 **26.286307832723594**다. 과거100K 공식 기록26.65365 대비 **0.367342167276406 개선(약1.38%)**이다. 같은 sampling seed 한 번의 비교이며 training/sampling seed 반복에 대한 유의성을 검증한 것은 아니다. 평가 이후 checkpoint나 recipe를 공식 validation에 맞춰 다시 선택하지 않았다.

원본 metrics: [metrics.json](workspace_distillation/experiments/teacher_budget_noaug_fresh_200k/evaluation/official_step_200000/ddim32_cfg2/metrics.json).
