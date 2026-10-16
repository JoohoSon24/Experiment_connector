# start

명시적으로 요청받을 때만 수행한다.
1. AGENTS.md와 기존 실험 허용 규칙을 읽는다.
2. `python3 scripts/experiment_notes.py start` 실행: festgen/main pull --rebase,
   meta 전체 상태 정리, running 프로세스 ps/PID 시작시각 확인(죽었으면 failed 후보),
   RESULTS 미반영 done 및 failed 에러 요약 확인.
3. 미반영 완료 실험에 대해 sync-notes와 같은 절차로 결과 문서 갱신 후 commit/push.
   스크립트는 RESULTS 행만 자동 추가한다. 에이전트는 meta/config/metrics를 직접 읽고
   변경된 세팅/구조/한계를 EXPERIMENT/ARCHITECTURE/LIMITATIONS에도 근거와 함께 갱신한다.
   추가 문서 변경은 허용 경로만 명시적으로 commit한 뒤 pull --rebase 및 festgen push한다.
4. reviews/*.json의 open 항목을 priority 순으로 보고한다. 반영 전 계획을 제시한다.
5. 최종 보고는 실험 현황 / 이번에 갱신한 문서 / 처리 대기 리뷰 / 제안하는 다음 실험.

네트워크/인증 실패는 명확히 보고하고 중단한다. `--local` 검증 결과를 push 성공으로 보고하지 않는다.
