# Noaug teacher budget continuation

## 최신 실행: 처음부터 재학습

원본 체크포인트가 없으므로 사용자 승인에 따라 fresh seed42로 0 → 100K → 150K → 200K 학습을 실행한다. noaug recipe와 milestone 평가 조건은 아래와 같다. 시작 스크립트는 `bash scripts/start_teacher_budget_fresh.sh`, tmux session은 `fastgen_noaug_fresh_200k`이다. 결과는 `experiments/teacher_budget_noaug_fresh_200k`에 저장한다. 이 폴더의 `status.json`, `wandb_identity.json`, `orchestration.log`로 상태와 URL을 확인한다. stage 이름은 `teacher_b64_noaug_fresh_100k`, `teacher_b64_noaug_fresh_150k`, `teacher_b64_noaug_fresh_200k`이다. 각 stage의 `latest.pt`로 재개할 수 있다. 새 100K 평가를 직접 수행하고 과거 100K 측정값과 혼합하지 않는다.

기존 checkpoint 대기 프로세스는 종료했다. 아래 내용의 기존 재개 실행기는 파일이 확보된 경우를 위한 별도 절차다.

## 현재 상태와 필요한 파일

100K teacher의 **full training checkpoint가 아직 없다**. student 추론 checkpoint는 재개에 사용할 수 없다. 기본 대기 경로:

`experiments/teacher_b64_noaug_constant_100k/final.pt`

model, optimizer, primary/additional EMA, scaler, RNG, step, epoch, batch cursor, 클래스 순서와 train manifest hash가 있어야 한다. checkpoint의 step은 정확히 100000이고 원래 recipe와 같아야 한다. 실행기가 검사에 실패하면 학습을 시작하지 않는다. 신뢰하는 원본 checkpoint만 제공한다.

## 실행

프로젝트 루트 `/root/Diffusion/workspace_distillation`에서:

```bash
bash scripts/start_teacher_budget_continuation.sh
# 다른 경로를 지정해 새 프로세스를 시작하는 경우:
# bash scripts/start_teacher_budget_continuation.sh /absolute/path/to/final.pt
```

tmux session `fastgen_noaug_budget_200k`가 이미 있으면 보존한다. 기존 프로세스는 처음 지정된 경로를 계속 기다리므로 다른 경로 인자를 전달하는 것만으로 대기 경로가 바뀌지 않는다. 기본 경로에 원본 파일을 배치하거나 그 파일을 가리키는 링크를 제공한다. 파일 크기와 수정 시각이 연속 두 번 안정된 뒤 검증한다.

원래 `teacher.train`을 subprocess로 실행한다. 150K config로 100K에서 재개하고, 150K final에서 200K config로 이어 학습한다. training loss·sampler·model 원본은 변경하지 않았다. 500-update 간격 latest checkpoint와 각 stage final을 저장한다. 재시작하면 latest에서 복원하고 완료된 milestone을 보존한다. 원본 학습 코드와 보호 파일의 hash도 검사한다.

## 평가와 W&B

100K·150K·200K마다 primary EMA, DDIM32/50 CFG2, seed42, batch16, bf16, clipped DDIM을 사용한다. development holdout reference755장에 생성3020장을 비교하는 기존 평가 방식을 고정한다. 가장 낮은 DDIM32 FID의 milestone을 선택하고 DDIM50을 보조 기록한다. official validation은 자동 평가하지 않는다.

W&B online project: `pokemon-diffusion`. 기존 로그인 계정의 entity를 사용한다. trainer는 W&B 의존성을 갖지 않고 외부 observer가 출력된 로그를 관찰한다. stdout의 20-update 간격 train 로그, milestone FID, 기존 CFG1 DDPM50 preview를 올린다. preview는 요청한 DDIM CFG2 holdout 평가와 별도다. 파일 대기 상태에는 `training_started=false`를 기록하며 과거 FID를 새 측정값으로 올리지 않는다.

## 결과와 모니터링

모든 경로는 프로젝트 루트 기준이다.

- `experiments/teacher_budget_noaug_100k_200k/status.json`: 상태·URL
- 같은 폴더 `plan.json`, `wandb_identity.json`, `orchestration.log`
- 같은 폴더 `teacher_b64_noaug_constant_150k/final.pt`와 `teacher_b64_noaug_constant_200k/final.pt`
- 같은 폴더 `evaluation/step_XXXXXX/ddim32_cfg2` 및 `ddim50_cfg2`
- 같은 폴더 `summary.json`: 완료 후 milestone 결과와 선택 step

```bash
cat experiments/teacher_budget_noaug_100k_200k/status.json
tail -n 30 experiments/teacher_budget_noaug_100k_200k/orchestration.log
tmux attach -t fastgen_noaug_budget_200k
```

W&B 추가 dependency lock은 `environment/requirements.wandb.lock.txt`이다. 환경 재구축 시 기본 환경을 만든 뒤 필요하면 `.venv/bin/python -m pip install -r environment/requirements.wandb.lock.txt`로 설치한다.

## 확인한 범위

4개 CPU guard test와 shell/Python syntax 검사를 통과했다. recipe의 부당한 변경, inference-only checkpoint, 로그 step/메모리 제한을 검사했다. 이는 실제 100K checkpoint의 재개 성공을 검증한 것은 아니다. 실제 optimizer·RNG 재개 및 milestone FID는 원본 checkpoint 도착 후 검증된다. 기존 환경의 GPU 학습 smoke/evaluator 재현 결과는 `environment/restoration_report.json`, `environment/official_reproduction.json`에 있으며 이번 continuation과 구분한다.
