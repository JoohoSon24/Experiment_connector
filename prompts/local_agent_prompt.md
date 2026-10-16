# 로컬 에이전트용 프롬프트

이 폴더에 "실험 리뷰 워크스페이스"를 구축해줘. 연구실 서버의 에이전트가 실험을 돌리고 git에 기록을 push하면, 여기서 그걸 받아서 아키텍처 시각화, 실험 세팅 정리, 결과 분석, 한계점 검토를 하고 피드백을 서버로 돌려보내는 구조야.

너의 역할은 **리뷰어**야. 실험을 돌리거나 프로젝트 코드를 고치지 않아.

## 입력 정보 (내가 채울 부분)
- 프로젝트 git 레포 SSH 주소: `<REPO_SSH_URL>` (서버 에이전트가 push하는 레포와 같은 주소)
- 브랜치: `main`
- 서버 SSH 별칭: `lab` (~/.ssh/config에 등록되어 있음)
- 서버 프로젝트 경로: `<SERVER_PATH>`

## 1. 디렉터리 구조
```
lab-review/
├── CLAUDE.md
├── .claude/settings.json
├── .claude/commands/start.md
├── .last_start            # 마지막 /start 시점의 sync/ 커밋 해시
├── sync/                  # 프로젝트 레포 clone. 읽기 전용 (reviews/ 제외)
├── arch/
│   ├── overview.mmd       # 데이터 → 전처리 → 모델 → loss → 평가
│   ├── model.mmd          # 블록 단위 내부 구조
│   └── shapes.md          # 단계별 입출력 텐서 shape 표
├── experiments/
│   ├── registry.csv
│   └── exp-XXX/
│       ├── setup.md       # 가설, 부모 실험 대비 diff
│       └── notes.md       # 해석, 이상 징후
├── reports/
│   ├── results_table.md   # 자동 생성. 손으로 수정 금지
│   └── figures/
├── reviews/               # 작성 후 sync/reviews/로 복사해서 push
└── scripts/
    ├── sync.sh
    ├── push_review.sh
    └── build_report.py
```

## 2. 권한 설정
`.claude/settings.json`에 `sync/` 아래 파일 수정을 막는 deny 규칙을 넣어줘(단, `push_review.sh`가 `sync/reviews/`에 복사하는 건 스크립트로 처리). 학습·추론 실행 명령(python train 등)과 서버로의 `ssh lab` 명령 실행도 deny에 넣어. 권한 규칙 문법은 Claude Code 문서로 확인하고 적용해.

## 3. 스크립트
**sync.sh**
- `sync/`가 없으면 clone, 있으면 `git pull --rebase`
- 결과 요약 중 git에 없는 것이 필요할 때만 `rsync -avz lab:<SERVER_PATH>/runs/*/figures/` 형태로 가져오기 (ckpt, logs 제외)
- 마지막으로 새로 생긴 exp-ID 목록 출력

**push_review.sh**
- `reviews/`의 새 파일·변경 파일을 `sync/reviews/`로 복사
- `sync/`에서 `reviews/`만 add, `review: <파일명>`으로 commit, pull --rebase 후 push
- reviews/ 외 경로가 staged되어 있으면 중단

**build_report.py**
- `sync/runs/*/meta.json`, `metrics.json`을 읽어 `experiments/registry.csv` 갱신 (기존 행의 `결론`, `status` 등 수동 입력 컬럼은 보존)
- `reports/results_table.md` 생성: 지표 표, parent 대비 변화량, `git_dirty=true`나 `backfilled=true` 실험은 경고 표시
- 주요 지표 비교 그림을 `reports/figures/`에 저장 (matplotlib)

registry.csv 컬럼:
`id, date, parent_id, hypothesis, change, config_path, git_commit, main_metric, status, conclusion`

## 4. 리뷰 파일 형식
`reviews/YYYY-MM-DD-<주제>.md`
```markdown
---
status: open
priority: high | medium | low
related: exp-012, exp-013
resolution:
---
## 문제
## 근거 (파일 경로, 줄 번호, 수치)
## 제안
```
서버 에이전트는 status와 resolution만 수정해.

## 5. CLAUDE.md에 넣을 규칙
```markdown
## 역할
- 나는 리뷰어. 실험 실행, 프로젝트 코드 수정, 서버 접속 금지
- sync/는 읽기만 한다. 쓰기는 reviews/와 이 워크스페이스 폴더에만

## /start 명령
- 내가 /start를 입력하면 .claude/commands/start.md 절차를 수행
- 그 외에는 세션 시작 시 자동으로 sync하지 않음

## 검토 체크리스트
- config와 docs/EXPERIMENT.md의 일치 여부
- 데이터 누수(train/val/test 중복, 전처리 통계 계산 시점)
- 베이스라인 공정성(같은 데이터, 같은 튜닝 예산, 같은 seed 수)
- seed 1개 결과로 결론을 내린 경우
- 지표 계산 방식과 논문 관례의 차이
- git_dirty, backfilled 실험에 의존한 결론

## 시각화 규칙
- 아키텍처는 Mermaid로, 실제 sync/src 코드에서 근거를 찾아 작성
- 노드에 해당 클래스/함수 이름과 파일 경로 표기
- 코드와 docs/ARCHITECTURE.md가 다르면 코드를 기준으로 그리고 차이를 reviews/에 기록
```

## 6. /start 커스텀 명령
`.claude/commands/start.md`를 만들어. 수행 순서:
1. **새 실험 확인:** `scripts/sync.sh` 실행 → 지난 /start 이후 새로 생긴 exp-ID와 상태가 바뀐 실험 목록 확인 (마지막 확인 시점은 `.last_start`에 커밋 해시로 저장)
2. **업데이트:**
   - `build_report.py` 실행 → registry.csv, results_table.md, figures 갱신
   - 새 실험마다 `experiments/exp-XXX/setup.md` 초안 작성 (부모 대비 config diff 포함)
   - `sync/src`나 `docs/ARCHITECTURE.md`가 바뀌었으면 (`git diff <last>..HEAD`로 확인) arch/ 다이어그램 갱신
3. **정리 및 검토:**
   - 새 완료 실험의 `notes.md` 작성 (결과 해석, parent 대비 변화, 이상 징후)
   - 검토 체크리스트 적용 → 문제가 있으면 reviews/에 초안 작성 (push는 하지 않음)
   - 서버가 처리한 reviews(status: done) 확인
4. 마지막에 "새 실험 / 갱신한 파일 / 주요 발견 / 서버가 처리한 리뷰 / push 대기 리뷰" 5개 항목으로 보고하고, push 대기 리뷰는 내 승인 후 push_review.sh 실행

## 7. Git 연결
- `ssh -T git@github.com`으로 GitHub 인증이 되는지 확인하고, 안 되면 SSH 키 생성과 등록 방법을 안내해 (등록은 내가 직접 함)
- 인증되면 `<REPO_SSH_URL>`을 `sync/`로 clone

## 8. 초기 작업
구축이 끝나면 한 번 전체 흐름을 돌려줘.
1. sync.sh로 레포 받기
2. sync/src를 읽고 arch/ 다이어그램 3종 작성 (shape은 코드에서 추론 가능한 범위만, 추정치는 표시)
3. build_report.py 실행
4. 실험별 setup.md 초안 작성
5. 체크리스트로 검토해서 발견한 문제를 reviews/에 작성
6. push는 하지 말고, 작성한 리뷰 목록을 보여주고 내 승인 후 push_review.sh 실행

## 완료 기준
- 위 6단계가 오류 없이 끝남
- 마지막으로 /start를 한 번 실행해 보고 형식이 정상인지 확인
- 생성한 파일 목록, 발견한 주요 문제 3개 이내 요약, 내가 채워야 할 설정값 보고
