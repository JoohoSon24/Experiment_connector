# Teacher / Distillation / DMD 연구 코드 스냅샷

이 브랜치는 FastGen upstream `e3f5120000cd91037017d86f23993492195101a5` 위에 복원 환경, teacher 실험, 기존 guided/progressive distillation, DMD 실험 및 후속 원인 분석을 보존한다. 실행 중인 학습 디렉토리는 변경하지 않고 별도 작업 트리에서 수집했다.

## 구조

| 경로 | 내용 |
|---|---|
| `teacher/`, `teacher_variants/` | 원본 teacher와 4-head 등 추가 teacher 실험 |
| `distillation/`, `configs/`, `scripts/` | 원본 CFG transfer → progressive → budget extension |
| `environment/` | Python/CUDA package lock과 복원·평가 검증 |
| `research/recipe_distill_dmd/` | 현재 4-head150K teacher 기반 student/DMD 대기열, 원인 진단, 후속3개 대기열 |
| `research/dmd_4head150k/` | 이전 DMD 두 분기 및 실패한 direct-bootstrap 실험 기록/코드 |
| `research/dmd_handoff_2500/` | 제공된 DMD handoff의 코드·설정·문서 및 역사적 평가 기록 |
| `docs/project/` | 아키텍처·개념 LaTeX/PDF·teacher 결과·개선 방향 문서 |
| `reports/` | 저장 시점의 작은 JSON 결과, 실행 계획과 일부 비교 이미지 |
| `reports/SNAPSHOT_MANIFEST.json` | 복사한 파일의 출처·SHA256·스냅샷 시각 |

## 현재 가장 중요한 결과

| 모델 | 실제 NFE | Official FID |
|---|---:|---:|
| 선택 teacher4-head150K, DDIM32 CFG2 | 64 | 25.4984 |
| 새 guidance student | 32 | 24.4645 |
| 새 progressive16 | 16 | 26.8949 |
| 새 progressive8 | 8 | 29.8472 |
| 새 progressive4 | 4 | 36.0318 |
| 기존4-NFE, 현재 환경 재측정 | 4 | 33.0517 |

Official FID는3020-reference/3020-generated, seed42, batch16 기준이다. Development FID는755-reference로 별도 기록한다.

`reports/recipe_distill_dmd/recipe/status.json`은 **저장 시점의 스냅샷**이다. 실행 중인 로그나 앞으로 완료될 실험 결과가 Git에 자동 반영되는 것은 아니다. Teacher→student→DMD4개의 기존 대기열 뒤에 n4 LR 비교2개와32-step direct-target1개를 예약했다. 아직 대기 중인 실험을 완료 결과로 해석하면 안 된다.

## 주요 문서

- [프로젝트 구조](docs/project/DIFFUSION_PROJECT_OVERVIEW.md)
- [Teacher 상세](docs/project/TEACHER_MODEL_DETAILS.md)
- [Teacher 전체 Official FID](docs/project/TEACHER_ALL_OFFICIAL_FID_20261011.md)
- [현재 recipe와DMD4개](research/recipe_distill_dmd/EXPERIMENT_PLAN.md)
- [기존/새4-step 비교](research/recipe_distill_dmd/OLD_VS_NEW_4STEP.md)
- [원인 분석과후속 방향](research/recipe_distill_dmd/N4_CAUSE_ANALYSIS_AND_NEXT_DIRECTIONS.md)
- [추가 n4 실험3개](research/recipe_distill_dmd/N4_FOLLOWUP_QUEUE.md)
- [16클래스 비교 이미지: 위=기존, 아래=새](reports/recipe_distill_dmd/compare_old_new_n4/comparison_16_classes.png)

## 재현에 필요한 외부 파일

Git에는코드·설정·문서·작은결과를보존한다. Dataset, teacher/student full checkpoint, inference weights, Inception evaluation weights, 가상환경, W&B service state는 .gitignore로 제외했다. 관련 SHA256과 경로는기존manifest/결과문서에서확인한다.

핵심 외부teacher: `experiments/teacher_variant_sequence/attention4/attention4_150k/final.pt`. 현재recipe의재시작에는해당parent/full또는EMA체크포인트와관련student체크포인트가필요하다. 기존legacy4-step의weights는 `checkpoints/few_nfe.ckpt`이다. 제공된DMD handoff의21MB checkpoint는truncated 상태였고재현가능한선택모델로사용하지않았다.

현재실행코드는실험호스트의 `/root/Diffusion/workspace_distillation`, `/root/Diffusion/recipe_distill_dmd`, `/root/Diffusion/dmd_4head150k` 경로를사용한다. 다른checkout에서자동실행하기전에Python진입점의ROOT/WORK/teacher경로와config경로를맞춰야한다. `research/`는현재호스트코드의스냅샷이며한명령으로어디서나동작하는새패키지로refactor하지않았다.

깨끗한머신의기본환경복원은 `environment/bootstrap_environment.sh`와package lock을사용한다. 추가W&B 의존성은 `environment/requirements.wandb.lock.txt`에있다. 데이터는원본 `download_dataset.py`와제공된split/label mapping을사용한다. 연구용holdout manifest는 `data/development/seed42_holdout5/`에포함했다.

## 버전 관리 범위

보호된 upstream `dataset.py`, `download_dataset.py`, `evaluate.py`, `src/utils.py`, 원본split/mapping은변경하지않았다. `model.py`는미구현student wrappers를구현했고base Model class는보존했다. 원래명령행평가와비교진단의source snapshot을남겼다.

이브랜치의commit은학습과별도로작성한다. 원격저장소전송은해당저장소와접근권한이확인된뒤명시적으로진행한다.
