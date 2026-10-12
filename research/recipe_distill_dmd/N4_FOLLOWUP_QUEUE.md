# 추가 n4 원인 분석 실험 — 기존 대기열 뒤에 예약

기존 distillation과 DMD4개를 유지하고, 해당 supervisor의 status가 `completed`가 되면 다음3개를 자동으로 순차 실행한다. 같은GPU에서 기존 학습과 추가 학습을 겹쳐 실행하지 않는다. 기존 대기열이 실패하면 복구를 기다리며 후속 실험을 먼저 시작하지 않는다.

| 순서 | Run | Student 시작점 | Frozen target 모델 | LR | 추가 학습 |
|---|---|---|---|---:|---:|
| 1 | n4_lr2e4 | 같은 n4 EMA25K | n8 EMA25K, 두 전이 merge | 2e-4 | 10K |
| 2 | n4_lr5e5 | 같은 n4 EMA25K | n8 EMA25K, 두 전이 merge | 5e-5 | 10K |
| 3 | n4_direct32_lr5e5 | 같은 n4 EMA25K | guidance EMA50K, 여덟 전이 merge | 5e-5 | 10K |

**우선순위1은1번vs2번, 우선순위2는2번vs3번**으로 비교한다.2번을 공통 대조군으로 재사용하므로 중복 학습 없이 총3개의 추가 실험이다.3번은2번 결과에 따라 초기 weights나LR을 변경하지 않아 target 효과를 직접 비교할 수 있다.

모든 분기는 같은25K n4 EMA에서 **fresh AdamW**로 시작한다. 기존n4 optimizer/RNG는 완료parent 축소로 남아 있지 않아 exact optimizer continuation이 아니다. Step0은 원래n4 25K 모델이고, 추가10K 완료는 student 학습량 관점에서25K→35K에 해당한다. 로그step은 추가 학습 횟수0..10000이다.

## 공통 조건

- 4-head student/teacher 구조 유지,48,883,107 params.
- Noaug, 실제train16,324장, batch64, constantLR, warmup0.
- v prediction + 기존truncated-SNR loss, EMA0.999, gradclip1, bf16, seed42.
- Student grid `[999,741,483,226]` 유지. 가장높은시간은pure-noise 입력, 나머지는실제이미지를noising한입력.
- Parent n8은clip=False, guidance32는원래clip=True 의미를유지한다. Merge된x0 target은clamp하지않는다.
- Production에서teacher의여덟전이는no_grad로순차실행해backprop graph를만들지않는다.

## 평가 및 W&B

Development FID: baseline, 추가2.5K/5K/10K.755reference와3020generated, seed42,batch16. Baseline을포함해최저development FID를선택하고동일PNG의Official FID를3020reference로보고한다.OfficialFID를모델선택에사용하지않는다.

기존4step **33.05174275**, 새4step 출발점 **36.03177398**를비교기준으로둔다.

W&B entity `fv2z6n929c-kaist-digital-humanities-and-social-sciences-`, project `pokemon-diffusion`, group `recipe4head150k_progressive_DMD`. Loss20step, qual1000step, developmentFID각milestone, 선택checkpoint OfficialFID를기록한다. 각run이시작될때W&B URL을status/identity에기록한다.

## 구현 및 검증

`distillation/refine.py`는현재학습코드를변경하지않는별도trainer다. Student 초기화와teacher target stride만추가했다. 기존두전이target과새generic-loop ratio2 target이bf16에서**exact parity**를통과했다. Ratio2/8 각각GPU batch8로두업데이트를실행해finite loss를확인했다. 기존학습을유지하기위해사전검증만batch8로진행했으며실제예약학습은batch64다. 검증weight는정리하고metrics/log를보존했다.

- Preflight: `experiments/n4_followup_preflight.json`
- 실행계획: `experiments/n4_followup/plan.json`
- 실시간상태: `experiments/n4_followup/status.json`
- Console: `experiments/n4_followup/console.log`
- 결과: `experiments/n4_followup/results.json`
- tmux: `n4_followup_after_DMD4`
- Scheduler: `scripts/run_n4_followup_queue.py`
- Configs: `configs/n4_lr2e4.json`, `configs/n4_lr5e5.json`, `configs/n4_direct32_lr5e5.json`

현재추가대기열상태: **waiting_for_existing_queue**.

Active/latest는fulloptimizer/RNGcheckpoint로resume가능하다. 완료후final은EMA+설정+계보로축소하고가장좋은milestone EMA를별도로보존한다. 이는최종공유·추론및평가용이며완료된분기의optimizer state를그대로이어학습하는용도는아니다.
