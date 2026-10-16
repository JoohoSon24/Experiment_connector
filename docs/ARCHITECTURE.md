# 모델 및 코드 흐름

- 보호 인터페이스: model.py의 기본 Model, dataset.py, evaluate.py, src/utils.py.
- teacher/model.py: time/class conditioning을 가진 ClassConditionalUNet.
  teacher_variants/model.py는 attention head 변형을 지원한다.
- 대표 teacher: channels 96, channel_mult [1,2,3,4], residual blocks 2,
  attention resolution 16, 4 heads 변형. 기록된 파라미터 수 48,883,107.
- teacher/data.py, splits.py: train 이미지 정규화 및 train 내부 holdout 구성.
- teacher/train.py 및 teacher_variants/train.py: data → noisy image → epsilon prediction
  → MSE/AdamW → EMA → 체크포인트·metrics. teacher/config.py는 JSON config를 읽는다.
- teacher/cfg.py, diffusion.py, sample.py, evaluate.py: CFG와 sampler, 개발 평가.
- distillation/core.py, train.py, pipeline.py: fixed guidance transfer 및 progressive halving,
  v prediction 학습. 새로운 consistency 방법이 검증됐다는 뜻은 아니다.
- distillation/submission.py, export.py: 제출 backbone/버퍼/샘플링과 export.
  model.py의 ModelOneNFE/ModelFewNFE는 각각 1/4 backbone 호출 경로를 연결한다.
- scripts/run.py: 기존 학습 argv 실행을 감싸며 ID/config/seed/commit/PID/status/metrics를 기록.
- scripts/experiment_notes.py: 소급 이관, 결과 표, 명시적 start/sync, festgen/main만 push.
- AGENTS.md와 .codex/commands/: Codex용 정책과 수동 요청 절차. CLAUDE.md/.claude/commands는 호환 진입점.

상세 모델 설명은 기존 model_architecture.md와 대표 config, 실제 source snapshot을 함께 확인한다.
research/ 아래는 festgen의 기존 별도 실험 스냅샷이며 현재 teacher 코드와 혼합하지 않는다.
- scripts/fid_report.py와 configs/fid_protocol_v1.json: 생성 조건을 검사하고 같은 PNG에 대해
  official_fid/valid_fid를 분리 기록한다. reference stats는 git 밖 고정 경로에 캐시하며
  manifest/이미지 fingerprint 불일치 시 재사용을 거절한다. 기존 보호 evaluator는 변경하지 않는다.
