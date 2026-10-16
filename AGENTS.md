# Codex 실험 기록 및 로컬 리뷰 연동

이 파일은 Codex의 저장소 기본 지침입니다. 기존 CODEX.md와 상위 ../CLAUDE.md도 읽습니다.
기존 규칙의 모델 제한(60M 대 100M)이 충돌하므로 새 실제 실험 계획에서 적용 기준과 근거를 명시합니다.
보호된 기본 Model, dataset.py, download_dataset.py, evaluate.py, src/utils.py, split 파일은 변경하지 않습니다.

## 실험 규칙
- 새 실험 = 새 ID(exp-XXX). configs/에 새 yaml을 만들고 실행. 기존 config 덮어쓰기 금지.
- 실행 전 코드와 새 config 변경은 먼저 커밋(git_dirty=false). 이전 실험 결과도 먼저 동기화.
- parent_id와 hypothesis 필수. 첫 실험만 parent_id=null 허용.
- 기존 학습 코드를 래퍼로 감싼다. 실행 중인 프로세스는 ps, nvidia-smi, tmux로 먼저 확인한다.
- 실행 중인 파일/폴더는 이동·이름 변경 금지. 기존 결과 삭제 금지. 이관은 작은 메타데이터만 복사한다.
- 종료 시 sync-notes 절차를 수행한다. 실패해도 meta/metrics를 보존한다.
- CPU smoke는 synthetic 1 optimizer step으로 인프라만 검증하며 FID나 GPU 제한 통과로 해석하지 않는다.

## git 규칙
- push는 오직 festgen 원격의 main으로만: `git push festgen HEAD:refs/heads/main`.
- **origin(원본 FastGen)에는 절대 push하지 않는다.** 기본 원격을 추측해 push하지 않는다.
- push 직전에 항상 `git pull --rebase festgen main`. 충돌 시 자동 덮어쓰기/자동 stash 금지.
- reviews/는 status와 resolution 필드만 갱신. 리뷰 반영 계획을 먼저 제시한다.
- 대용량 파일·체크포인트·원본 로그·원본 데이터 커밋 금지. 새 노트/그림은 각각 1MiB 이하.
- 기존에 추적된 assets PDF/PNG와 split manifest는 사용자 요청으로 유지한다.
- sync에서는 docs/, configs/, runs/*/meta.json, metrics.json, figures/*.png, reviews/만 add.
- 코드/AGENTS.md/명령 문서는 별도 구현 커밋으로 추가하며 sync allowlist를 넓히지 않는다.

## 명시적 명령
- 세션 시작 시 자동 pull이나 리뷰 조회를 하지 않는다.
- 사용자가 `/start` 또는 `start 절차 실행`을 요청하면 .codex/commands/start.md를 읽고 실행한다.
- 사용자가 `/sync-notes` 또는 `sync-notes 절차 실행`을 요청하면 .codex/commands/sync-notes.md를 읽고 실행한다.
- 이는 에이전트용 절차 이름이다. CLI의 내장 slash 명령 자동 등록을 가정하지 않는다.
- 실제 실행 명령은 `python3 scripts/experiment_notes.py start` 또는 `sync`.
- `--local`은 오프라인 검증 전용이며 pull/push 성공으로 보고하면 안 된다.

## 리뷰 처리
- 반영 항목은 반영 계획을 먼저 제시.
- 반영 완료 시 status: done, resolution: exp-XXX 또는 설명.
- /start의 프로세스 사망 판정은 failed 후보로만 보고하며 자동으로 원본 meta를 변경하지 않는다.

새 metrics.json에는 official_fid와 valid_fid를 명시하며 미측정은 null로 둔다.
FID 프로토콜 및 사용법은 docs/EXPERIMENT.md와 configs/fid_protocol_v1.json을 참조한다.
로컬 clone에서는 `git config core.hooksPath .githooks`로 커밋 크기 검사와 origin push 차단을 활성화한다.

실험 meta의 git_commit이 rebase 후에도 pull로 조회되도록 원래 실험 커밋을 main의 merge ancestry로 보존한다.
동기화 스크립트는 `git pull --rebase=merges festgen main`을 사용한다. 수동 pull도 같은 옵션을 사용한다.

업로드 대상은 `JoohoSon24/Experiment_connector`, 브랜치는 main이다.
로컬 원격 이름 `festgen`은 호환을 위해 유지하되 Experiment_connector를 가리킨다. FestGen 저장소에는 push하지 않는다.
