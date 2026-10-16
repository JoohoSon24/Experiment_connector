# 서버 에이전트용 프롬프트

이 레포에 "실험 기록 및 로컬 리뷰 연동 시스템"을 구축해줘. 목표는 실험을 자유롭게 돌리되, 재현과 검토에 필요한 최소 정보만 git에 남겨서 로컬의 리뷰 에이전트가 pull로 받아볼 수 있게 하는 거야.

## 0. 시작 전 주의사항 (반드시 지킬 것)
- 지금 tmux에서 실행 중인 실험이 있을 수 있어. `ps`, `nvidia-smi`로 실행 중인 프로세스를 먼저 확인하고, 그 프로세스가 쓰는 파일·폴더는 이동하거나 이름을 바꾸지 마.
- 바로 수정하지 말고, 먼저 레포 구조(코드, config, 결과 폴더, 기존 실험 기록)를 파악한 뒤 **변경 계획을 나에게 보여주고 승인받은 다음** 진행해.
- 기존 결과물은 삭제하지 마. 옮겨야 하면 복사 후 원본 유지.
- 대용량 파일(체크포인트, 원본 로그, 데이터)은 절대 커밋하지 마.

## 1. 디렉터리 구조
아래 구조를 목표로 해. 기존 구조와 충돌하면 계획 단계에서 대안을 제시해줘.

```
proj/
├── CLAUDE.md
├── .claude/commands/sync-notes.md
├── src/
├── configs/exp-XXX.yaml        # 실험별 config, 덮어쓰기 금지
├── docs/
│   ├── EXPERIMENT.md           # 현재 세팅: 데이터 split, seed, 하이퍼파라미터, 평가 방식
│   ├── ARCHITECTURE.md         # 모델 구조, 코드 흐름, 주요 파일 위치
│   ├── RESULTS.md              # 실험별 핵심 지표 표
│   └── LIMITATIONS.md          # 알려진 문제, 미검증 가정, 의심 지점
├── runs/exp-XXX/
│   ├── meta.json               # 커밋
│   ├── metrics.json            # 커밋
│   ├── figures/*.png           # 커밋 (각 1MB 이하)
│   ├── ckpt/                   # 제외
│   └── logs/                   # 제외
└── reviews/                    # 로컬 리뷰어가 push하는 폴더. 서버는 status 필드만 수정
```

## 2. .gitignore
최소한 아래를 포함하고, 레포에 맞게 추가해줘.
```
runs/*/ckpt/
runs/*/logs/
*.pt
*.pth
*.ckpt
*.safetensors
wandb/
data/
__pycache__/
```
이미 추적 중인 대용량 파일이 있으면 목록을 보여주고 `git rm --cached` 여부를 나에게 물어봐.

## 3. 실행 래퍼 (scripts/run.py 또는 기존 진입점 수정)
실험 시작 시 `runs/exp-XXX/meta.json`을 자동으로 생성해야 해.
- 필드: `id`, `parent_id`, `hypothesis`, `config_path`, `git_commit`, `git_dirty`, `seed`, `gpu`, `host`, `command`, `started_at`, `finished_at`, `status`(running/done/failed)
- 종료 시 `finished_at`, `status`를 갱신하고 최종 지표를 `metrics.json`에 저장
- 실패해도 `status: failed`와 에러 요약이 남도록 try/finally 처리
- ID는 `runs/`의 마지막 번호 + 1로 자동 부여 (3자리 zero-padding)
- 기존 학습 코드는 최소한으로 건드리고, 래퍼에서 감싸는 방식을 우선해

## 4. 기존 실험 이관
이미 돌린 실험이 있으면 가능한 범위에서 exp-ID를 소급 부여하고 `meta.json`을 만들어줘. 알 수 없는 필드는 `null`로 두고 `"backfilled": true`를 표시해.

## 5. CLAUDE.md에 추가할 규칙
```markdown
## 실험 규칙
- 새 실험 = 새 ID(exp-XXX). configs/에 새 yaml을 만들고 실행. 기존 config 덮어쓰기 금지
- 실행 전 코드 변경은 먼저 커밋 (git_dirty=false 유지)
- 실험마다 parent_id와 hypothesis를 반드시 기록
- 실험 종료 시 /sync-notes 실행

## git 규칙
- push 전에 항상 git pull --rebase
- reviews/ 파일은 내용을 수정하지 말고 status와 resolution 필드만 갱신
- 대용량 파일 커밋 금지

## 세션 시작 시
- git pull 후 reviews/에서 status: open 항목을 확인해 요약 보고
- 반영할 항목은 반영 계획을 먼저 제시
- 반영 완료 시 status: done, resolution: exp-XXX 또는 설명 기록
```

## 6. /sync-notes 커스텀 명령
`.claude/commands/sync-notes.md`를 만들어. 수행 순서:
1. `git pull --rebase`
2. 마지막 sync 이후 완료된 실험의 meta.json, metrics.json 읽기
3. `docs/RESULTS.md` 표에 행 추가 (id, parent, 가설, 핵심 지표, 결론 한 줄)
4. 세팅·구조가 바뀌었으면 EXPERIMENT.md, ARCHITECTURE.md 갱신
5. 새로 발견한 한계나 이상 징후는 LIMITATIONS.md에 추가
6. `docs/ configs/ runs/*/meta.json runs/*/metrics.json runs/*/figures/ reviews/`만 add
7. 커밋 메시지 `exp-XXX: 한 줄 요약`으로 commit 후 push
8. 무엇을 커밋했는지 보고

## 7. docs 초기 작성
코드와 기존 결과를 직접 읽고 docs 4종의 초안을 작성해줘. 추측한 내용은 `(추정)`으로 표시하고, 확인이 필요한 부분은 LIMITATIONS.md에 "미확인" 항목으로 남겨.

## 8. 인증
push 권한이 없으면 레포 전용 deploy key 생성 방법(`ssh-keygen -t ed25519 -f ~/.ssh/<repo>_deploy`)과 `~/.ssh/config` 설정을 안내하고, GitHub 등록은 내가 직접 할 테니 공개키만 보여줘.

## 완료 기준
- 테스트용 짧은 실험(1~2 step)을 래퍼로 돌려 meta.json, metrics.json이 정상 생성되는지 확인
- /sync-notes를 한 번 실행해 push까지 성공
- 최종적으로 변경한 파일 목록과 내가 해야 할 일을 정리해서 보고
