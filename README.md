# Experiment Connector

연구실 서버에서 실험을 돌리는 Claude 에이전트와, 로컬에서 실험을 검토하는 Claude 에이전트를 **git으로 연결**하는 워크플로 템플릿입니다.

## 목적

서버(SSH + tmux)에서 에이전트가 실험을 설계·실행하는 동안, 실험 세팅·아키텍처·결과·한계점은 로컬에서 따로 정리하고 검토하고 싶을 때 씁니다.

- **서버 에이전트**는 실험을 돌리고, 재현과 검토에 필요한 최소 정보(config, 메타데이터, 지표, 상태 문서)만 git에 push합니다.
- **로컬 에이전트**는 리뷰어 역할만 합니다. 기록을 pull해서 아키텍처를 시각화하고, 결과 표를 만들고, 문제를 찾아 `reviews/`로 피드백을 돌려보냅니다.
- 두 에이전트는 세션을 공유하지 않고 **문서로만 소통**합니다. 별도 서버나 REST API 없이 git 히스토리가 그대로 실험 기록이 됩니다.

## 구조

```
 서버 (tmux + Claude)                         로컬 (Claude, 리뷰어)
 ─────────────────────                        ─────────────────────
 실험 실행 → runs/exp-XXX/meta.json,          sync.sh → git pull
            metrics.json                       build_report.py → 결과 표·그림
 /sync-notes → docs/ 갱신 → commit/push  ──▶   arch/*.mmd 아키텍처 다이어그램
                                               체크리스트로 검토
 세션 시작 시 reviews/ 확인·반영        ◀──   push_review.sh → reviews/ push
```

| 경로 | 쓰는 쪽 | 내용 |
|---|---|---|
| `src/`, `configs/`, `docs/`, `runs/*/meta.json·metrics.json·figures/` | 서버 | 코드, 실험별 config, 상태 문서, 실험 결과 요약 |
| `reviews/` | 로컬 (서버는 `status`·`resolution`만 수정) | 리뷰 피드백 |

쓰는 경로를 나눠두었기 때문에 같은 브랜치를 써도 충돌이 거의 생기지 않습니다. 체크포인트, 로그, 데이터는 git에 올리지 않습니다.

## 파일

- [`prompts/server_agent_prompt.md`](prompts/server_agent_prompt.md): 서버 프로젝트 레포에 실험 기록 체계를 구축하는 프롬프트 (`.gitignore`, `meta.json`을 남기는 실행 래퍼, `docs/` 4종, `CLAUDE.md` 규칙, `/sync-notes` 명령)
- [`prompts/local_agent_prompt.md`](prompts/local_agent_prompt.md): 로컬 리뷰 워크스페이스 `lab-review/`를 구축하는 프롬프트 (`sync.sh`, `push_review.sh`, `build_report.py`, `registry.csv`, Mermaid 아키텍처 다이어그램, 권한 제한)

## 사용 방법

### 1단계: 서버 세팅

1. 서버의 프로젝트 레포에서 Claude를 실행합니다. (실험이 돌고 있는 tmux 창과는 다른 창을 권장)
2. `prompts/server_agent_prompt.md` 내용을 그대로 붙여넣습니다.
3. 에이전트가 레포 구조와 실행 중인 프로세스를 확인한 뒤 **변경 계획을 보여줍니다.** 검토하고 승인하세요.
4. push 권한이 없으면 에이전트가 deploy key 공개키를 보여줍니다. GitHub 레포의 Settings → Deploy keys에 쓰기 권한으로 등록하세요.
5. 테스트 실험과 `/sync-notes`가 push까지 성공하면 완료입니다.

### 2단계: 로컬 세팅

1. 로컬에 빈 폴더 `lab-review/`를 만들고 그 안에서 Claude를 실행합니다.
2. `prompts/local_agent_prompt.md`의 아래 값을 채웁니다.
   - `<REPO_URL>`: 서버 프로젝트의 git 레포 주소
   - `<SERVER_PATH>`: 서버의 프로젝트 경로
   - `lab`: `~/.ssh/config`에 등록한 서버 별칭 (다르면 수정)
3. 프롬프트를 붙여넣고 실행합니다.
4. 에이전트가 첫 리뷰 목록을 보여주면, 검토 후 승인해서 push합니다.

### 3단계: 일상 루프

| 언제 | 어디서 | 할 일 |
|---|---|---|
| 실험이 끝날 때마다 | 서버 | `/sync-notes` 실행 |
| 결과를 보고 싶을 때 | 로컬 | Claude 실행 → 세션 시작 규칙에 따라 sync, 보고서 갱신 자동 수행 |
| 문제를 발견했을 때 | 로컬 | `reviews/`에 기록 → `scripts/push_review.sh` |
| 서버 세션 시작 시 | 서버 | 에이전트가 `status: open` 리뷰를 확인하고 반영 계획 제시 |

## 실시간 모니터링

이 구조는 기록과 검토용입니다. 학습 중 loss를 실시간으로 보려면 별도 API를 만들지 말고 W&B를 쓰거나, TensorBoard를 SSH 포트포워딩(`ssh -L 6006:localhost:6006 lab`)으로 보는 것을 권장합니다.

## 주의사항

- 서버 프롬프트는 실행 중인 실험의 파일을 옮기지 않도록 되어 있지만, 승인 전에 계획을 직접 확인하세요.
- 공용 서버라면 개인 토큰 대신 레포 전용 deploy key를 쓰세요.
- 로컬 에이전트의 권한 제한(`.claude/settings.json`)이 실제로 적용됐는지 한 번 확인하세요.
