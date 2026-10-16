# sync-notes

1. `git pull --rebase festgen main`. dirty tracked 파일이면 덮어쓰기나 자동 stash 없이 해결한다.
2. RESULTS에 없는 done 실험의 meta.json, metrics.json, config를 읽는다. ID 기준으로 중복 방지.
3. 세팅/구조 변경은 EXPERIMENT.md, ARCHITECTURE.md에 근거와 함께 반영한다.
   새 이상 징후/미확인 가정은 LIMITATIONS.md에 추가한다. 추정은 (추정) 표시.
   자동 스크립트는 RESULTS 행만 갱신하므로 이 검토를 생략하지 않는다.
4. 문서 편집이 있으면 크기·diff를 검토하여 docs 파일만 먼저 commit한다
   (`exp-XXX: 한 줄 요약`). 이는 tracked 편집 때문에 pull --rebase가 막히는 것을 방지한다.
5. `python3 scripts/experiment_notes.py sync`: pull --rebase, RESULTS에
   id/parent/가설/핵심 지표/결론 추가, 허용 경로 검사 후 add/commit,
   push 직전 pull --rebase 후 `git push festgen HEAD:refs/heads/main`.
6. 실제 커밋 파일, hash, 지표 요약, push 성공 여부를 보고한다.

허용: docs/, configs/, runs/*/meta.json, runs/*/metrics.json, runs/*/figures/*.png, reviews/.
금지: 기존 config 수정, 리뷰의 status/resolution 이외 변경, 1MiB 초과 신규/변경 노트,
체크포인트/원본 로그/데이터. 기존 자산은 재추가하거나 제거하지 않는다.
