# 현재 실험 세팅 및 재현 절차

## 확인된 기존 세팅

근거: teacher/config.py, teacher/splits.py, configs/teacher/variant_attention4_150k.json,
teacher_variants/train.py, distillation/train.py, docs/distillation_results_review.md.

- Pokémon 64×64 RGB, 151 classes, 입력 [-1,1]. 제공 train만 학습에 사용.
- development: train에서 클래스당 5장, seed 42로 755장 분리. 학습 16,324장.
  data/development/seed42_holdout5의 manifest와 metadata를 보존한다.
- official validation reference는 3,020장. 개발 FID(755 reference)와 구분하며 직접 비교하지 않는다.
- attention4_150k 예시: batch 64, accumulation 1, AdamW LR 2e-4 constant,
  weight decay 0, warmup 0, grad clip 1, seed 42, augmentation 없음.
- epsilon teacher, linear beta 1e-4→0.02, 1,000 diffusion steps, class dropout 0.1.
  EMA 0.9999 및 추가 EMA 0.999. bf16. 평가마다 sampler/clip/CFG/seed/NFE를 별도 확인.
- fixed-CFG2 progressive student는 v prediction. 단계별 config/지표를 원본과 연결한다.
- 위 값은 대표 config의 세팅이며 모든 실험에 동일하다고 가정하지 않는다.

## 새 실험

기존 학습 진입점과 JSON config를 유지한다. 새 yaml은 JSON 문법을 사용하는 유효한 YAML 문서로,
추가 YAML 의존성 없이 읽는다. `snapshot_files`는 prepare 시 본문으로 복사하므로 원본을 덮어쓰지 않는다.

1. 저장소 밖(예: /tmp/experiment-spec.json)에 spec을 작성한다.
2. `python3 scripts/run.py prepare --spec /tmp/experiment-spec.json`으로 새 ID/config 생성.
3. 코드와 config를 commit. 이전 결과는 sync해 깨끗한 작업 트리를 유지한다.
4. `.venv/bin/python scripts/run.py run --config configs/exp-XXX.yaml`.
5. 종료 후 sync-notes 절차 수행.

spec 예시(teacher CPU/GPU 자동 선택; 실제 학습 전 별도 허용성 판정 필요):
```json
{
  "parent_id": null,
  "hypothesis": "단일 변수 변경의 목적과 예상 효과",
  "seed": 42,
  "snapshot_files": ["configs/teacher/debug.json"],
  "command": ["{python}", "-m", "teacher.train", "--config", "{run_dir}/ckpt/configs/configs/teacher/debug.json"],
  "metrics_source": "{run_dir}/ckpt/training/teacher_debug/metrics.json"
}
```

스냅샷의 `output_dir`는 **prepare 전에** `{run_dir}/ckpt/training`으로 설정하고 `experiment_name`도 지정한다.
스냅샷 본문의 `{run_dir}`와 `{id}`는 실행 시 실제 값으로 치환된다.
위 metrics_source는 형식 예시이므로 실제 학습 config의 출력 경로로 바꿔야 한다.
기존 결과 경로를 재사용하지 않는다. 재시도도 새 ID를 사용한다.
커스텀 프로그램은 EXP_METRICS_PATH에 최종 JSON object를 쓴다. seed/EXP_RUN_DIR도 환경으로 전달한다.
GPU 정보는 nvidia-smi 장치 목록이며 실제 할당 GPU와 VRAM은 학습 지표에 따로 기록해야 한다.

meta에는 PID/시작시각/config hash도 남긴다. 정상 종료여도 최종 지표가 없으면 failed 처리한다.
SIGINT/SIGTERM은 하위 프로세스 종료와 실패 기록을 시도한다. SIGKILL/정전은 finally를 실행할 수 없으므로
/start가 살아 있지 않은 running을 failed 후보로 표시한다. 자동 확정하지 않는다.

## 소급 기록

`python3 scripts/experiment_notes.py backfill`은 experiments/**/metrics.json을 출처별로 기록한다.
평가 결과는 별도 evaluation ID이며 학습 실행 ID와 동일시하지 않는다. source_path로 중복 이관을 방지한다.
학습 완료 증거는 final.pt 존재, 평가 완료 증거는 FID 측정값이다. 증거가 없으면 status는 null(미확인)로 남긴다.
parent/hypothesis/시간 등 확인되지 않은 값은 null. backfilled=true.
원본 결과·체크포인트·로그는 복사/삭제/이동하지 않는다. 작은 config와 metrics만 복사한다.

## 현재 인프라 검증

GPU 드라이버 통신이 실패했다. CPU 1 step synthetic smoke는 기존 U-Net의 optimizer 갱신과
래퍼/메타데이터 경로만 검증하며 연구 결과나 FID로 취급하지 않는다.

## 고정 FID 프로토콜 v1

정의 파일: `configs/fid_protocol_v1.json` (기존 파일 덮어쓰기 금지).
이 기준은 **teacher 주 비교**에 적용한다. 과거 DDPM50/CFG1, DDIM50, seed1042/2042,
다른 EMA 결과를 이 기준으로 재라벨링하지 않는다. student의 NFE1/4는 별도 새 프로토콜로 명시한다.

| 항목 | 고정 값 |
|---|---|
| 구현체 | clean-fid 0.1.35, mode=clean, Inception-v3 |
| 생성 샘플 | 3,020장 = 151 classes × 20장, 64×64 RGB PNG |
| 생성 파일명 | `0000_00.png` … `0150_19.png`, 클래스별 20장 |
| sampler / clipping | DDIM / clip_denoised=true |
| CFG | 2.0 (conditional/unconditional 두 backbone 평가) |
| steps / NFE | 32 denoising steps / 실제 backbone NFE=64 |
| seed | sampling base seed=42, 생성 batch 순서마다 seed+batch_index |
| batch / precision | batch_size=16, bf16 backbone / FP32 schedule |
| valid reference | train 내부 seed42, 클래스당 5장 holdout, 총 755장 |
| official reference | 제공 validation split 전체 3,020장; TA leaderboard 수치와 구분 |
| valid reference PNG | `data/development/seed42_holdout5/reference/` |
| official reference PNG | `results/reference/` |
| valid reference stats | `results/reference_stats/valid_755_seed42_clean_inception_v3_v1.npz` |
| official reference stats | `results/reference_stats/official_3020_clean_inception_v3_v1.npz` |

`valid_fid`는 기존 `reference_split=development`에 대응한다. 두 reference 간 수치를 직접 비교하지 않는다.
학습/선택에는 valid_fid만 사용하고 official_fid는 frozen 후보 최종 보고에만 사용한다.
EMA track과 checkpoint는 실험별로 명시하며 다른 weight끼리 같은 실험으로 취급하지 않는다.

### stats 및 계산 경로

과거 측정은 보호된 `evaluate.py:compute_fid`가 두 PNG 폴더를 clean-fid에 전달하여 통계를 즉석 계산했다.
**위 .npz 경로는 새 프로토콜의 고정 캐시 위치이며 아직 생성하지 않았다.**
새 `scripts/fid_report.py`는 동일 clean-fid feature extractor와 mean/covariance/Frechet 식을 사용한다.
reference manifest SHA256, 실제 PNG 전체 fingerprint, clean-fid 버전과 mode를 캐시에 함께 기록한다.
캐시가 기존 reference와 다르면 덮어쓰지 않고 중단한다. stats/원본 PNG는 git 제외다.
GPU 문제로 이번 구축에서는 FID 재측정이나 stats 생성을 수행하지 않았다.

```bash
.venv/bin/python scripts/fid_report.py \
  --generated-dir /absolute/path/to/generated \
  --generation-metrics /absolute/path/to/original/metrics.json \
  --protocol configs/fid_protocol_v1.json \
  --output /absolute/path/to/new/final_metrics.json \
  --device cuda
```

프로토콜과 다른 seed/CFG/NFE/샘플 수는 계산 전 거절한다. 기존 결과 경로에는 덮어쓰지 않는다.
출력을 래퍼의 EXP_METRICS_PATH로 쓰거나 metrics_source로 연결한다.
현재 캐시 래퍼의 실데이터 FID 동등성은 미확인이다. 정상 GPU에서 기존 dir-to-dir 점수와 비교 후 사용한다.

### metrics.json 공통 필드

파일명은 최초 구조와 일치하는 복수형 `runs/exp-XXX/metrics.json`을 사용한다.
모든 새 결과와 소급 결과에 두 필드를 둔다. 미측정/대응 불명은 null이며 0이나 다른 split 점수로 채우지 않는다.
```json
{
  "official_fid": null,
  "valid_fid": null
}
```

소급 값은 명시된 reference_split 또는 official/development 원본 JSON으로부터만 채운다.
과거 샘플링 조건과 원본 값은 유지한다. 새 FID 평가 출력에는 fid_protocol과 protocol hash,
generated PNG hash, reference stats 경로/hash도 저장한다. CPU 인프라 smoke의 두 FID는 null이다.
