# 복원 실행 환경

복원일: 2026-10-07. Upstream commit `e3f5120000cd91037017d86f23993492195101a5` + 원본 code overlay + inference ZIP.

## 실행

```bash
cd /root/Diffusion/workspace_distillation
source .venv/bin/activate
python environment/check_restoration.py
bash environment/reproduce_official.sh reproduction_02
```

검증과 공식 평가는 GPU 장치 접근이 허용된 실행 환경이 필요하다. Codex 샌드박스의 `nvidia-smi` 실패는 호스트의 드라이버 오류가 아니며, 호스트 권한에서 GRID A100D-20C / driver 550.90.07이 확인됐다. 드라이버는 수정하지 않았다.

## 복원 방법과 버전

독립 Python 3.10.21 venv를 만들고 기존 `/root/NFE-1/.venv`의 FastGen 필요 package dependency closure 40개를 새 site-packages로 복사했다. 원본 venv를 가리키는 symlink는 사용하지 않았다. Python interpreter는 로컬 uv-managed Python 3.10을 사용한다.

- torch 2.6.0+cu124 / torchvision 0.21.0+cu124 / torchaudio 2.6.0+cu124
- clean-fid 0.1.35 / numpy 2.2.6 / scipy 1.15.3
- Pillow 12.3.0 / kagglehub 1.0.2
- `pip check`: 통과

`requirements.lock.txt`는 이번 복원의 고정 환경이다. 원 학습의 전체 package lock은 전달 ZIP에 없으므로 과거 모든 transitive dependency와의 동일성을 주장하지 않는다. 공식 FID 재현값으로 결과 호환성을 확인한다. Upstream `requirements.txt`는 변경하지 않았다.

다른 머신의 **깨끗한 checkout**에서 dependency를 설치할 경우:

```bash
DIFFUSION_PYTHON310=/path/to/python3.10 bash environment/bootstrap_environment.sh
python download_dataset.py  # venv 활성화 후 원본 downloader 사용
```

Bootstrap은 이미 있는 `.venv`를 보존하고 종료한다. 위 명령 실행 전 원본 저장소의 정확한 commit에 code overlay와 checkpoint ZIP을 적용하고 이 `environment/`의 lock·스크립트를 함께 복사한다. 공개 package registry와 CUDA wheel index 접근이 필요하다.

## 데이터와 reference

기존 `/root/AIC313-Project-FastGen/data/.../PokemonData`를 독립 복사하고 white-compositing marker를 보존했다. 원본 `download_dataset.py`에서 기존 데이터로 인식하는 것도 확인했다. Official train/val manifest와 class mapping은 해당 commit 원본 그대로다.

20,099개 공식 split image의 존재, PIL verify, 파일 content aggregate SHA-256을 확인했다. Optimizer train 16,324 / holdout 755 / val 3,020이 서로 겹치지 않는다. 전체 official train은 17,079다.

Reference는 보호된 transform과 `evaluate.prepare_reference_set`을 사용해 생성했다:

- `results/reference/`: 공식 validation 3,020 PNG
- `data/development/seed42_holdout5/reference/`: 내부 holdout 755 PNG

## FID 평가 weight

clean-fid 0.1.35의 공식 다운로드 URL에서 NVIDIA TorchScript Inception weight를 받았다. 이 모델은 **평가용 feature extractor**이며 training teacher나 pretrained 생성 모델로 사용하지 않는다.

- URL: `https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt`
- 영구 보관: `environment/inception-2015-12-05.pt`
- Hash: `environment/evaluation_weight.sha256`
- clean-fid가 사용하는 cache: `/tmp/inception-2015-12-05.pt`

`reproduce_official.sh`는 영구 보관 weight가 있으면 cache를 복원한다. 재현용 batch는 **16**이다. Wrapper seed가 기본 42+batch index이므로 batch size를 바꾸면 생성 noise 구성이 달라진다.

## 검증 증거

- `restoration_report.json`: source 보호·overlay hash·packages·data·references·GPU·checkpoint provenance·1/4-NFE·determinism·distillation 수식·임시 training smoke.
- `restoration_console.log`: 검증 stdout/stderr.
- `one_nfe_smoke.png`, `few_nfe_smoke.png`: 각 2개 class의 실제 전달 weight 출력.
- `official_one_console.log`, `official_few_console.log`: 변경하지 않은 `evaluate.py`의 실행 로그.
- `official_reproduction.json`: 완료 후 기록한 재현 FID와 checkpoint/hash/sample count.

Full-size random-init teacher의 실제 batch 64 학습 1회 업데이트·EMA·finite gradients는 통과했다. 해당 smoke의 peak allocated VRAM은 9,156,377,600 bytes이며 장시간 학습 품질이나 exact resume 검증이 아니다. 제출 checkpoint를 수정하지 않았다.

## 복원 범위의 한계

추론, dataset, FID 평가와 fresh training 실행 환경은 복원했다. 기존 teacher 및 중간 stage의 full training checkpoint는 전달되지 않았다. 따라서 선택 teacher quality의 재평가, 기존 전체 distillation pipeline, optimizer/EMA/RNG를 보존한 정확한 학습 continuation은 필요한 `.pt`를 추가로 받아야 한다.

필수 추가 파일은 프로젝트 상위 `DIFFUSION_STATUS_AND_EXPERIMENTS.md`에 작업 목적별로 정리했다. 추론 `.ckpt`는 full training state를 대신하지 않는다.
