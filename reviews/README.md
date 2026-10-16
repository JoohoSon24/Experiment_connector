# 로컬 리뷰 연동

로컬 리뷰어가 이 폴더에 JSON 파일을 작성하고 festgen/main으로 push합니다.
서버는 기존 JSON의 `status`, `resolution`만 변경합니다. 다른 필드의 추가/삭제/변경은 금지합니다.
새 리뷰 파일 생성은 로컬 리뷰어의 역할입니다. 반영 계획을 사용자에게 먼저 제시합니다.

예시 (`<review-id>.json`):
```json
{"id":"review-001","priority":"P1","status":"open","title":"검토 항목","summary":"근거와 제안","experiment_id":"exp-001","resolution":null}
```

우선순위는 P0 → P1 → P2 → P3 (또는 critical/high/medium/low)입니다.
반영 완료 시 status=done, resolution=exp-XXX 또는 설명을 기록합니다.
