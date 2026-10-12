# Teacher 개선 우선순위와 비교 실험

작성일: 2026-10-07.

**후속 실행 변경:** 원래 100K checkpoint가 없으므로 사용자가 처음부터 학습하도록 승인했다. 새 실행은 seed42의 fresh initialization에서 100K → 150K → 200K를 수행한다. 아래 원본 checkpoint 대기는 이전 계획이며 새 실행에 적용되지 않는다. 새 결과 경로는 `workspace_distillation/experiments/teacher_budget_noaug_fresh_200k`, 실행기는 `scripts/run_teacher_budget_fresh.py`이다. 과거 100K 결과와 새 100K 결과를 구분한다.

 아래 우선순위는 현재 코드와 handoff 결과를 바탕으로 한 제안이며, 개선 효과가 검증된 결과가 아니다. 모델 상세는 [TEACHER_MODEL_DETAILS.md](TEACHER_MODEL_DETAILS.md), 실험 기록은 [DIFFUSION_STATUS_AND_EXPERIMENTS.md](DIFFUSION_STATUS_AND_EXPERIMENTS.md)를 참고한다.

## 먼저 수행할 실험: noaug 학습 budget 확대

선택 teacher의 **원래 100K full training checkpoint**에서 150K, 이어 200K까지 계속 학습한다. 새로 초기화한 teacher나 student weight는 대체재가 아니다. 현재 전달 자료에는 이 checkpoint가 없어 실행기는 파일을 기다린다. W&B에 기다리는 상태를 등록하며, 실제 학습 시작·완료 및 FID는 실행 후에만 기록한다.

배치 64, seed 42, AdamW LR 2e-4 constant / warmup 0 / weight decay 0, 클래스 dropout 0.1, primary EMA 0.9999와 추가 EMA 0.999, bf16, epsilon prediction, linear 1000-step noise, augmentation 0을 유지한다. train manifest, 클래스 순서, UNet 구조와 학습 코드는 그대로 유지한다. optimizer·EMA·RNG·epoch·batch cursor까지 복원한다. 변경되는 것은 총 update 수와 결과 저장 위치뿐이다.

| 누적 update | 직전 milestone 대비 추가 update | 동일 평가 |
|---|---:|---|
| 100K | 기준 checkpoint 재평가 | DDIM32 / DDIM50, CFG2 |
| 150K | 50K | DDIM32 / DDIM50, CFG2 |
| 200K | 50K | DDIM32 / DDIM50, CFG2 |

평가는 primary EMA, development holdout 755장, 생성 3,020장, seed 42, batch 16, bf16, clipped DDIM을 고정한다. 이 코드의 기존 평가 방식을 유지하기 위해 생성 수가 holdout 수보다 크다. 절대 FID에는 표본 규모의 한계가 있으므로 같은 조건의 상대 비교에 사용한다. DDIM32 CFG2 holdout FID를 선택 기준으로, DDIM50을 보조 지표로 사용한다. official validation으로 선택하지 않는다. 기존 100K의 DDIM32 holdout FID 44.11891은 과거 기록이며, 이번 실행의 측정값으로 대신 기록하지 않는다.

## 각 축에서 가장 먼저 바꿔볼 한 가지

| 축 | 첫 변경 | 이유 | 주요 확인 사항 |
|---|---|---|---|
| Architecture | 기존 attention 위치에서 single-head → 4-head | convolution·폭·깊이를 유지하면서 공간 관계 표현 방식을 비교할 수 있다 | 동일 budget FID, peak GPU memory, update 시간 |
| Data augmentation | horizontal flip 확률 0 → 0.25; affine는 0 | 강한 변형을 피하면서 제한된 데이터의 좌우 다양성을 늘리는 작은 실험이다 | noaug 대조군 대비 CFG2 FID, 방향·형태 보존 |
| Fine-tuning | 선택 checkpoint에서 전체 UNet을 LR 5e-5 constant로 짧게 추가 학습 | 구조와 데이터가 같아 결과의 원인을 학습률 변화로 좁힐 수 있다 | 같은 추가 budget의 LR 2e-4 대조군, EMA 결과 |

전체 실행 순서는 **현재 noaug budget 확대 → 낮은 LR fine-tuning → 약한 flip 비교 → attention 구조 비교**를 권한다. 앞선 두 실험은 기존 teacher를 활용하기 쉽고, architecture 변경은 재학습 및 student와의 정합성 확인 비용이 더 크다. 한 실험에서 세 축을 동시에 바꾸지 않는다.

## 1. Architecture: 4-head attention 비교

현재 teacher는 약 48.88M parameter의 클래스 조건부 UNet이며, base channel 96, 배율 (1, 2, 3, 4), 기존 attention은 16×16과 middle 8×8에 있다. 그 위치와 QKV/projection을 유지하고 head 수만 4로 바꾼다. attention channel 288과 384는 4로 나누어 떨어져 head당 72·96 channel이 된다. 같은 projection을 쓰면 parameter 수는 유지된다.

head별로 다른 공간 관계를 표현할 여지가 생긴다는 것이 가설이다. **4-head가 이 데이터에서 더 좋다는 증거는 아직 없다.** 기존 single-head 대비 matched-budget 실험으로 판단한다. 공식 guided-diffusion 구현에도 configurable multi-head attention이 있다([원본 UNet 코드](https://raw.githubusercontent.com/openai/guided-diffusion/main/guided_diffusion/unet.py)). 구현의 존재는 성능 향상의 증거가 아니다.

변경 범위는 모델 attention, config의 head 수, 모델 생성 경로, checkpoint metadata, student backbone/export 정합성이다. head 수가 달라도 weight tensor shape가 같을 수 있으므로 shape 검증만으로 구조 일치를 판단하면 안 된다. **같은 크기의 weight를 읽었다고 원래 teacher의 정확한 continuation인 것은 아니다.** 구조 실험은 별도 이름과 설정으로 관리하고, 우선 동일 seed·동일 training budget의 새 학습과 대조한다. 100K weight로 warm-start한다면 그 사실과 변경 시점을 명시하고 별도의 대조군을 둔다.

width/depth 확대, attention 위치 확대, kernel 변경을 동시에 적용하지 않는다. 이미 약 48.88M이며 기존 제약의 60M parameter·20GB peak memory를 지켜야 한다. VRAM과 실행 시간을 실제 측정한다.

## 2. Augmentation: 약한 horizontal flip부터

첫 후보는 flip probability 0.25, rotation/translation/scale 변화는 모두 0이다. 0.25는 검증된 최적값이 아니라 보수적인 실험 시작점이다.

과거 100K DDIM50 CFG2 holdout 기록은 noaug 43.57209, flip-only 44.36898, affine+flip 48.40064였다. 따라서 **기존 기록에서 augmentation은 선택 조건의 품질을 개선하지 못했다.** 반면 CFG1에서는 flip-only가 더 좋았으므로 효과는 sampling guidance에도 의존한다. CFG2를 고정한 상태에서 낮은 확률의 flip만 분리해 보는 이유다. 64×64 이미지의 강한 affine는 형태와 작은 세부를 바꾸고 재보간 영향을 줄 수 있다.

가장 명확한 비교는 같은 초기화·split·seed·budget으로 noaug와 weak-flip를 각각 학습하는 것이다. 비용 때문에 100K teacher에서 분기한다면 두 실험 모두 동일 checkpoint에서 시작하고, 추가 50K noaug 대조군과 추가 50K weak-flip를 비교한다. 이것은 처음부터 augmentation을 적용한 실험과는 다른 질문이다. holdout/reference에는 augmentation을 적용하지 않는다.

flip이 캐릭터의 방향성을 훼손하는 사례와 class fidelity를 확인한다. 초기 비교에 random crop, hue 변경, MixUp, 강한 affine를 함께 넣지 않는다. augmentation conditioning은 후속 후보이며 loss/model 변경이 추가된다. 예를 들어 EDM 구현은 augmentation label을 모델에 전달하는 방식을 제공한다([원본 loss 코드](https://raw.githubusercontent.com/NVlabs/edm/main/training/loss.py)). 현재 teacher에 자동으로 적용되는 기능은 아니다.

## 3. Fine-tuning: 전체 모델의 낮은 constant LR

150K/200K 평가까지 수행한 뒤 DDIM32 CFG2 holdout으로 선택된 milestone을 기준으로 별도 분기한다. LR만 2e-4에서 5e-5로 낮추고, 추가 10K를 먼저 비교한다. 유망하면 추가 25K까지 확인한다. 두 경우 모두 **constant LR / warmup 0**을 유지한다. 현재 noaug budget continuation의 LR를 중간에 바꾸는 것은 아니다.

대조군도 같은 checkpoint에서 같은 update 수만큼 LR 2e-4로 학습한다. 전체 backbone을 학습하며 데이터·noise·batch·class dropout·EMA를 유지한다. 충분히 학습된 teacher에서 작은 update로 세부를 조정할 수 있다는 가설이지만, 낮은 LR가 개선을 보장하지는 않는다. 이전보다 FID가 나쁘면 원래 milestone을 유지한다.

두 분기는 같은 raw model·optimizer·EMA·RNG 상태에서 시작한다. 한쪽은 EMA를 raw model로 대체하고 다른 쪽은 raw weight로 시작하는 비교를 피한다. optimizer를 로드한 뒤 실제 적용 LR를 기록해야 한다. 현재 trainer는 update마다 config LR를 적용한다. 이번 budget-only 실행기는 LR 변경을 의도적으로 거부하므로 fine-tuning에는 별도 config/실험 이름과 검증 정책을 사용한다.

LoRA나 layer freezing은 초기 우선순위가 아니다. 현재 목적은 전체 모델이 가능한 환경에서 teacher 품질을 개선하는 것이고, 낮은 LR 비교가 구조적 변경 없이 원인을 평가하기 쉽다. 외부 pretrained 모델을 도입하지 않는다.

## 판단 기준과 후속 distillation

학습 loss 감소만으로 개선을 판단하지 않는다. 고정 DDIM32/50 CFG2 holdout FID, 클래스별 샘플의 형태·배경·다양성, GPU memory·시간을 함께 본다. 작은 FID 차이를 확정적 개선이라고 말하지 않고, 후보가 정해지면 동일 protocol의 추가 seed로 안정성을 확인한다. 반복 비교로 holdout에 과도하게 맞출 가능성도 기록한다.

teacher 변경 후 distillation은 별도 실험이다. teacher FID 개선이 1-NFE student 개선을 보장하지 않는다. 선택 teacher에서 fixed-guidance와 progressive stage를 다시 생성하고 teacher checkpoint hash·sampling schedule·CFG를 student 결과와 연결한다. 기존 n1 budget extension 결과를 teacher budget 증가 결과로 혼동하지 않는다.

## 현재 구현 범위

이번에 구현한 것은 **원래 noaug recipe의 100K → 150K → 200K continuation, milestone 평가, 외부 W&B observer**이다. 위 architecture·augmentation·fine-tuning 변경은 제안이며 아직 적용하지 않았다. 실행 설명은 [teacher_budget_continuation.md](workspace_distillation/docs/teacher_budget_continuation.md)를 참고한다.
