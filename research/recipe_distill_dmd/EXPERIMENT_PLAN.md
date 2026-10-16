# 새 teacher 기반 기존 distillation recipe + DMD 4개 대기열

Teacher: `../workspace_distillation/experiments/teacher_variant_sequence/attention4/attention4_150k/final.pt`. 150K primary EMA 0.9999, 4-head attention, DDIM32 CFG2 Official FID **25.49841494**. Teacher 자체를 추가 학습하지 않는다.

## Student 생성: TEAM_HANDOFF 기존 절차 재사용

1. **Fixed-guidance transfer 50K**: teacher EMA의 특징 추출 가중치를 복사하고, epsilon→v 변경에 맞춰 출력 convolution을 0으로 초기화한다. CFG2의 조건/무조건 teacher 출력을 32-step conditional student에 전달한다.
2. **Progressive 32→16→8→4→2→1**: 단계별 25K. 이전 단계 EMA로 초기화하고, teacher의 두 DDIM 전이를 student 한 전이로 합치는 기존 target inversion/loss를 사용한다.
3. **Budget extension**: n1과 n2 각각 25K full checkpoint의 모델·EMA·AdamW·RNG·batch cursor를 이어서 누적 100K까지 학습한다. 기존과 동일하게 n1 target은 원래 25K n2, n2 target은 원래 25K n4로 고정한다. 25K/50K/75K/100K의 development FID 중 최저 checkpoint를 선택한다.

공통: 실제 training 이미지 16,324개, 고정 holdout 755개, noaug, batch64, constant LR2e-4, warmup0, EMA0.999, clip1, bf16, seed42, v prediction + truncated-SNR loss. 기존 distillation/core.py의 수식은 변경하지 않았다. 원본 workspace 코드는 수정하지 않고 독립 디렉토리에 복사했다. teacher의 **4-head attention**은 student·export wrapper·DMD real/fake score 모두에 유지한다.

Guidance 품질 gate는 기존 기준대로 teacher DDIM32 development FID 41.73750985 + 15 이내이다. 초과하면 실패 상태와 로그를 남기고 후속 학습을 중단한다. 이 경우 무조건 나쁜 student를 DMD로 넘기지 않는다.

## DMD 대기열

모든 실험은 development에서 선택된 **동일한 새 1-step student**에서 시작한다. Endpoint regularization은 동일한 새 teacher에서 만들어진 **25K 4-step student**를 사용한다. 이전 legacy student나 4,096개 teacher endpoint 회귀 cache는 사용하지 않는다.

| 순서 | 실험 | Endpoint 가중치 | Generator LR |
|---|---|---:|---:|
| 1 | A_reg01_lr2e5 | 0.1 | 2e-5 |
| 2 | B_reg00_lr2e5 | 0 | 2e-5 |
| 3 | C_reg03_lr2e5 | 0.3 | 2e-5 |
| 4 | D_reg01_lr1e5 | 0.1 | 1e-5 |

DMD는 제공된 `dmd_handoff_2500`의 VP adaptation을 따른 **DMD-inspired** 구현이다. DMD2/GAN 구현으로 부르지 않는다. Fake score LR2e-4, warmup200 microbatch updates, G2500 updates, batch64=16×4, G EMA0.999, grad clip1, timestep50..950, real score CFG2, seed42. Fake optimizer는 각 microbatch마다 업데이트하므로 generator update당 fake update 4회이다. 각 실험의 모델·optimizer는 새로 구성한다.

평가: 500/1000/2500 step development FID, baseline 포함 최저값 선택. 선택된 PNG 3,020장으로 Official reference 3,020장에 대한 FID를 보고한다. Official FID는 checkpoint 선택에 사용하지 않는다. Student n1/n2/n4의 Official FID도 DMD 전에 기록한다.

## W&B 및 실행

Entity: `fv2z6n929c-kaist-digital-humanities-and-social-sciences-`; project: `pokemon-diffusion`; group: `recipe4head150k_progressive_DMD`.

- Distillation: loss20 updates, qual1000 updates, development FID10K 및 단계 마지막. Budget extension은50K/75K/100K FID.
- DMD: loss20 updates, qual100 updates, development FID500/1000/2500, selected Official FID.
- 하나의 supervisor가 단일 GPU에서 학습·평가·후속 실험을 순차 실행한다. 검증 실패 또는 실행 오류는 `status.json`에 기록하고 중단한다.

```bash
cd /root/Diffusion/recipe_distill_dmd
/root/Diffusion/workspace_distillation/.venv/bin/python -u scripts/session.py
```

tmux session: `recipe_teacher25_distill_DMD4`. 상태: `experiments/recipe/status.json`; 전체 로그: `experiments/recipe/console.log`; 계획: `experiments/recipe/plan.json`; 종료 코드: `experiments/recipe/exit_code.json`.

## 체크포인트 보존 / 재현

Active distillation의 `latest.pt`와 완료된 n1/n2 budget final은 full AdamW/RNG checkpoint이다. 완료된 중간 parent는 EMA+설정+lineage로 축소하며, n1/n2 25K full checkpoint는 해당 budget continuation이 끝날 때까지 유지한다. Budget50K/75K milestone과 DMD best는 inference checkpoint여서 optimizer resume용이 아니다. DMD 중간 오류에서 exact resume은 구현되지 않아 실패 상태로 종료한다.

`experiments/preflight_pass.json`에 실제 GPU batch64 guidance/progressive update, 4-head 1/4-NFE export/sample parity, optimizer/RNG resume 및 DMD update 검증 결과를 기록한다. 사전검증 checkpoint는 테스트 후 정리하고 로그·metrics는 유지한다.

저장 공간 확보를 위해 기존 teacher variant의 완료된 final/latest ZIP **모든 member payload가 byte-identical**인지 비교한 후 중복 latest를 final과 hardlink로 연결했다. 원본 final 가중치와 optimizer/RNG는 보존했다. `storage_dedup_audit.json`에 작업을 기록했다.
