# 실험 규칙

저장소 공통 규칙은 [AGENTS.md](AGENTS.md)에 있다. 반드시 읽고 준수한다.
- 새 실험 = 새 ID(exp-XXX), configs/ 새 yaml. 기존 config 덮어쓰기 금지.
- 실행 전 코드/config를 먼저 커밋하여 git_dirty=false 유지.
- parent_id와 hypothesis를 기록. 실험 종료 시 /sync-notes 실행.

## git 규칙
- push 전 항상 `git pull --rebase festgen main`.
- push 대상은 `festgen/main`뿐. **origin(원본 FastGen)에는 절대 push하지 않는다.**
- reviews/ 내용은 status와 resolution 필드만 갱신.
- 대용량 파일 커밋 금지.

## /start 명령
- /start 입력 시 .claude/commands/start.md 절차 수행.
- 그 외 세션 시작 시 자동 pull/리뷰 확인 금지.

## 리뷰 처리
- 반영 계획 먼저 제시. 반영 완료 시 status: done, resolution: exp-XXX 또는 설명.
