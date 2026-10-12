# Diffusion: 실행 환경 복원, 축별 현황, 실험별 설명

작성일: 2026-10-07. 과거 실험 결과는 handoff와 code overlay의 2026-09-30 완료 기록, 현재 환경 확인은 `workspace_distillation/environment/restoration_report.json`을 기준으로 한다.

## 1. 실행 환경 복원

작업 경로는 `/root/Diffusion/workspace_distillation`이다. 지정 upstream commit `e3f5120000cd91037017d86f23993492195101a5`의 깨끗한 checkout에 code overlay와 inference ZIP을 적용했다. 기존 `/root/AIC313-Project-FastGen`, `/root/NFE-1`의 프로젝트 코드는 수정하지 않았다.

| 구성 | 복원 및 검증 결과 |
| --- | --- |
| Python | 독립 `.venv`, Python 3.10.21 |
| PyTorch 계열 | torch 2.6.0+cu124 / torchvision 0.21.0+cu124 / torchaudio 2.6.0+cu124 |
| GPU | GRID A100D-20C, CUDA runtime 12.4; 장치 접근이 허용된 실행에서 확인 |
| 패키지 | 기존 로컬 환경에서 필요한 40개 distribution과 의존성을 독립 복사; lock 생성; `pip check` 통과 |
| 데이터 | 기존 로컬 원본 데이터의 독립 복사; 공식 split 20,099장 전체 파일 존재·이미지 형식·aggregate hash 검증 |
| Split | 공식 train 17,079 / val 3,020; optimizer train 16,324 / holdout 755; 중복·누수 없음 |
| Reference PNG | 공식 3,020장 + 내부 holdout 755장 생성. 보호된 dataset transform과 `evaluate.prepare_reference_set` 사용 |
| 보호 파일 | Dataset/downloader/evaluator/utils/requirements/원본 split·mapping이 지정 commit과 byte 일치 |
| Overlay | Manifest의 89개 파일 hash 확인; `Model` base 부분도 보존 확인 |
| 추론 모델 | 실제 전달 checkpoint 로딩, parameter 48,883,107, hook으로 1/4 NFE 확인 |
| 추론 재현성 | 같은 explicit seed의 동일 출력; RGB shape·유한 값·`[-1,1]` 확인 |
| Distillation 산술 | Merged endpoint 복원 오차 약 1.78e-15; velocity/loss 관계 통과 |
| 임시 학습 smoke | 실제 train batch 64, full-size random-init teacher의 optimizer 1회·EMA·CFG0 확인; peak allocated 9,156,377,600 bytes |

임시 smoke는 실행 가능성 확인이다. 기존 teacher quality나 장시간 학습, exact resume를 재현했다는 의미는 아니다. 제출 weight는 변경하지 않았다. 패키지 버전은 이번 복원의 고정 환경이며, handoff에 원 실험 전체 lock이 없으므로 모든 과거 transitive dependency와 일치한다고 단정하지 않는다.

**공식 FID 재현도 완료했다.** 변경하지 않은 `evaluate.py`, batch 16, 클래스당 20장(총 3,020장), 공식 reference 3,020장으로 평가했다. 두 실행 모두 exit code 0이며 아래 값은 evaluator가 출력하는 소수점 네 자리 기준으로 handoff와 일치한다.

| 전달 모델 | Handoff FID | 이번 복원 FID | 결과 |
| --- | ---: | ---: | --- |
| 개선된 100K 1-NFE | 58.1595 | **58.1595** | 일치 |
| 기존 n4 4-NFE | 33.0517 | **33.0517** | 일치 |

증거는 [official_reproduction.json](workspace_distillation/environment/official_reproduction.json)과 각 evaluator console log에 있다. 원래 실행의 PNG가 전달되지 않았으므로 과거 결과와 byte-for-byte PNG 일치까지 주장하지 않는다.

활성화:

```bash
cd /root/Diffusion/workspace_distillation
source .venv/bin/activate
python --version
```

확인 결과와 재현 도구:

- [restoration_report.json](workspace_distillation/environment/restoration_report.json): 데이터·hash·GPU·parameter·NFE·smoke 증거.
- [requirements.lock.txt](workspace_distillation/environment/requirements.lock.txt): 이번 복원 환경의 고정 package 버전.
- [check_restoration.py](workspace_distillation/environment/check_restoration.py): 복원 검증 재실행. GPU 접근 권한이 필요하다.
- [bootstrap_environment.sh](workspace_distillation/environment/bootstrap_environment.sh): 다른 깨끗한 checkout에서 Python 3.10 기반 dependency 환경 생성.
- [reproduce_official.sh](workspace_distillation/environment/reproduce_official.sh): batch 16의 공식 1/4-NFE 평가; 기존 결과 폴더 보존.

```bash
# 새로운 output 이름으로 공식 평가를 재실행
bash environment/reproduce_official.sh reproduction_02
```

`bootstrap_environment.sh`는 기존 `.venv`를 덮어쓰지 않는다. 다른 머신에서는 `DIFFUSION_PYTHON310=/path/to/python3.10`을 지정할 수 있다. 학습·추론 source는 overlay 원본을 보존했다.

## 2. Teacher model / distillation / student model 축별 정리

| 축 | 완료 지점 | 개선 지점 | 현재 로컬에서 가능한 범위 |
| --- | --- | --- | --- |
| **Teacher model** | 48.88M class-conditional pixel U-Net, epsilon MSE, class dropout·null token·CFG, DDPM/DDIM, EMA, train-only holdout 선택. Noaug 100K teacher 선택, DDIM32 CFG2 공식 FID 26.65365 보고 | High-noise score 오차·terminal SNR mismatch 진단, sampling/training seed 안정성 검증, schedule 변경은 새 통제 실험으로 진행 | 코드·설정·데이터와 full-size 학습 smoke 복원. **선택 teacher full weight는 없음**; 기존 teacher sampling/FID 재현·그 teacher의 정확한 학습 재개는 불가 |
| **Distillation** | Fixed CFG=2를 32-step single-forward student로 전달. 32→16→8→4→2→1 halving 완료. v prediction, truncated-SNR loss, nested grid, endpoint inversion, export 구현. n1/n2 100K budget extension 완료 | **75K 선택 n2를 frozen teacher로 새 n1 학습**; target-vs-output 오류 진단; 독립 fixed-CFG consistency baseline; 이후 variable guidance/curriculum | 산술·코드 복원. **Guidance·n2·stage full checkpoint 없음**; 기존 pipeline 전체를 현재 artifact만으로 재실행/continuation 불가 |
| **Student model** | `ModelOneNFE`/`ModelFewNFE`, teacher 없는 1/4 forward 추론, 각각 48.88M. 전달 1-NFE 100K FID 58.1595, 4-NFE FID 33.0517 | 1-NFE 세부 특징·경계·FID 개선 우선. 4-NFE baseline 보존. 2/3-step은 별도 학습·export·실제 NFE/품질 검증 후 지원 | 두 inference checkpoint 로딩·GPU 생성·NFE 검증 가능. 정확한 optimizer/EMA/RNG continuation은 full training checkpoint 추가 필요 |

“완료 지점”은 과거 실험과 코드 구현의 완료 여부다. “현재 로컬에서 가능한 범위”는 이번에 실제 전달받은 artifact로 할 수 있는 일이다. 코드가 있다고 기존 학습 weight까지 있는 것은 아니다.

## 3. 각 실험을 풀어 설명

`K`는 optimizer update 천 회다. NFE는 생성 중 backbone 평가 횟수다. Development FID는 training-only holdout 755장을 reference로, official FID는 validation 3,020장을 reference로 계산한다. 두 FID column을 서로 직접 비교하지 않는다.

### 3.1 선택 noaug teacher — Teacher 100K / DDIM32 CFG2 / 64 NFE

**질문:** 앞으로 student가 모방할 생성 모델을 어떻게 확보할 것인가?

Noaug는 augmentation이 없다는 뜻이다. 실제 학습 subset 16,324장을 사용해 class-conditional diffusion teacher를 100K 업데이트 학습했다. Network는 noisy image·time·class를 받아 주입된 Gaussian noise인 epsilon을 예측한다. Class label dropout으로 unconditional prediction도 같은 network에서 학습했다.

Teacher는 flip-only와 no-augmentation 후보의 고정 holdout DDIM50 CFG2 비교에서 noaug를 선택했다. 선택된 weight의 **DDIM32 CFG2** 결과를 distillation 시작점으로 별도 보고했다. 32는 sampling transition 수, CFG2는 guidance scale 2다. 각 transition에서 conditional와 unconditional 두 forward를 수행하므로 64 NFE다.

Development FID **44.11891**, official FID **26.65365**다. 품질 기준점이지만 1/4-NFE 제출 모델은 아니다. 필요한 weight는 `experiments/teacher_b64_noaug_constant_100k/final.pt`이며 기본 ZIP에는 없다.

### 3.2 Fixed guidance student — 50K / 32 NFE

**질문:** CFG의 conditional/unconditional 두 계산을 한 계산에 담을 수 있는가?

Teacher의 CFG=2 clean prediction을 student 하나의 conditional forward가 모방하도록 50K 업데이트 학습했다. Backbone 크기는 teacher와 같고, feature weight를 teacher에서 초기화하되 output head는 epsilon에서 v prediction으로 바뀌므로 초기화한다. Student의 guidance scale은 학습에 고정되어 있으며 inference에서 arbitrary scale을 입력하는 모델이 아니다.

Sampling은 아직 32 transitions다. 다만 transition마다 student를 한 번만 호출하므로 **64→32 NFE**다. 이 단계는 guidance의 두 branch를 압축했고 sampling 단계 수를 아직 줄이지 않았다.

Development FID **43.25620**, official FID는 별도로 보고되지 않았다. 선언된 quality gate를 통과해 다음 halving을 진행했다. 이 모델의 full 학습 checkpoint도 전달 ZIP에 없다.

### 3.3 Progressive n4 — 해당 stage 25K / 4 NFE

**질문:** 32-step 생성 경로를 4-step으로 줄여도 품질을 유지할 수 있는가?

Guidance student에서 `32→16→8→4`로 순차 압축했다. 이전 teacher의 두 deterministic DDIM transitions가 도달한 endpoint를 새 student의 한 transition이 재현하도록 effective clean target을 계산한다. 다음 stage는 이전 stage EMA로 초기화한다.

25K는 **n4 단계 자체의 학습량**이다. 전체 비용이 25K라는 의미가 아니며 guidance 50K와 앞선 n16/n8 학습을 먼저 수행했다. 현재 backbone이 transition당 한 번 평가되므로 4 transitions=4 NFE다.

Development FID **47.62077**, official FID **33.0517**다. 제출용 EMA export가 `checkpoints/few_nfe.ckpt`에 들어 있다. 다음 개선 실험에서도 보존할 few-step baseline이다.

### 3.4 기존 progressive n1 — 해당 stage 25K / 1 NFE

**질문:** 생성 경로를 마지막 한 번의 평가까지 압축할 수 있는가?

기존 progressive 경로를 `4→2→1`까지 계속했다. N1은 frozen n2 teacher가 수행하는 두 transitions를 하나로 근사하도록 25K 업데이트 학습한 별도 weight다. N4 checkpoint의 loop 횟수만 1로 줄인 모델이 아니다.

Development FID는 n2 **51.69442**에서 n1 **68.16127**로 약 16.47 악화했다. Official FID는 **61.4540**이다. 여러 stage 중 마지막 2→1이 가장 큰 holdout 손실을 보였다. 경계와 작은 특징이 더 흐리다는 관찰은 있지만 다양성 붕괴의 원인이 입증된 것은 아니다.

이 결과는 이후 budget 실험의 baseline이다. 현재 inference ZIP의 `one_nfe.ckpt`는 이 25K 버전이 아니라 개선된 100K 버전이다.

### 3.5 Budget extension n1 — 누적 100K / 1 NFE

**질문:** 1-NFE의 부진은 마지막 stage 학습량이 부족해서인가?

기존 n1 25K의 weight만 가져온 새 학습이 아니라, **optimizer·EMA·RNG·batch cursor를 포함한 full state에서 continuation**했다. 추가 75K를 수행해 누적 100K까지 학습했다. Architecture, data, loss, batch 64, constant LR 2e-4를 유지했으며 teacher는 **기존 25K n2**로 고정했다.

Development FID는 25K/50K/75K/100K 순서로 **68.16127 / 66.77484 / 66.15214 / 65.38627**이다. Development 기준으로 100K를 선택한 후 공식 FID를 보고했다. Official FID는 **61.4540→58.1595**, 3.2945 개선했다.

Budget 증가가 도움이 됐지만 4-NFE와의 격차는 남았다. 추론 export가 현재 `checkpoints/one_nfe.ckpt`에 포함돼 있다. 기존 학습 continuation을 다시 하려면 `experiments/low_nfe_budget_100k/n1/final.pt`가 필요하다.

### 3.6 Budget extension n2 — 100K까지 완료 / 75K 선택 / 2 NFE

**질문:** 한 단계 앞의 2-NFE student도 학습량을 늘리면 더 좋은 teacher 후보가 되는가?

기존 n2 25K full state에서 이어서 누적 100K까지 학습했다. Teacher는 **기존 n4로 고정**했으며 주된 변경은 update budget이다.

Development FID는 25K/50K/75K/100K 순서로 **51.69442 / 51.31244 / 50.89945 / 50.94633**이다. 75K가 가장 좋아 그 checkpoint를 선택했고, 내부 two-step evaluator로 official reference에 대한 FID **39.54904**를 보고했다. N2 25K의 official FID가 없어 official 개선 폭은 계산할 수 없다. 75K→100K의 약 0.047 차이로 overfitting이라고 단정하지 않는다.

기본 inference ZIP에는 n2가 없다. 선택된 weight는 `experiments/low_nfe_budget_100k/n2/step_075000.pt`가 필요하다. 이 개선된 n2를 teacher로 새 n1을 학습하면 target 품질의 효과를 확인할 수 있지만, **그 실험은 아직 수행되지 않았다.**

### 3.7 계보와 서로 독립인 budget 비교

```text
Noaug teacher 100K, DDIM32 CFG2 (64 NFE)
  └─ Guidance transfer 50K (32 NFE)
      └─ n16 25K → n8 25K → n4 25K (4 NFE, 전달)
                            └─ n2 25K (2 NFE)
                                └─ n1 25K (1 NFE)

n1 budget 비교: 기존 n1 25K → 100K  [teacher = 기존 n2 25K 고정]
n2 budget 비교: 기존 n2 25K → 100K  [teacher = 기존 n4 고정; 75K 선택]

아직 미실행: 개선 n2 75K → 새 n1 학습
```

## 4. 추가 학습 전에 필요한 누락 artifact

| 원하는 작업 | 필요한 파일 |
| --- | --- |
| 선택 teacher의 기존 결과 재현 / distillation 시작 | `experiments/teacher_b64_noaug_constant_100k/final.pt` |
| Guidance/student stage 정확한 재개 | 해당 stage의 full `final.pt` 또는 `latest.pt` |
| 기존 progressive baseline 재개 | `experiments/fixed_cfg2_progressive/{n4,n2,n1}/final.pt` |
| 개선 n1 continuation | `experiments/low_nfe_budget_100k/n1/final.pt` |
| 개선 n2로 새 n1 | `experiments/low_nfe_budget_100k/n2/step_075000.pt` |

서로 다른 GAN 프로젝트의 checkpoint가 같은 `one_nfe.ckpt` 이름을 사용하더라도 이 teacher의 대체 artifact로 사용할 수 없다. 현재 전달 모델의 provenance와 parameter count를 기준으로 구분했다.

Reference 이미지와 데이터는 이번에 복원했으므로 이를 다시 전달받을 필요는 없다. Full 학습 checkpoint만 추가되면 원래 상대 경로를 보존해 재개 검증을 이어갈 수 있다.
