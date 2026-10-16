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
  "metrics_source": "{run_dir}/ckpt/training/debug/metrics.json"
}
```

스냅샷의 `output_dir`와 `experiment_name`은 **prepare 전에** 새 출력 경로에 맞춘다.
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
