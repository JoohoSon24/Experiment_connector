# Teacher 200K paired LR experiment

사용자 승인에 따라 동일 fresh noaug200K **full checkpoint**에서 두 분기를 시작한다. raw model·AdamW moments·primary/additional EMA·RNG·epoch/cursor를 복원한다. 변경은 LR, 총 step, output bookkeeping뿐이다. 원래 trainer/model/loss는 수정하지 않는다. trainer가 optimizer 상태를 로드한 뒤 config의 constant LR를 매 update 적용한다.

| Arm | LR | 초기 상태 |
|---|---:|---|
| control_lr2e4 | 0.0002 | 동일 200K full state |
| finetune_lr5e5 | 0.00005 | 동일 200K full state |

noaug, batch64, constant LR, warmup0, weight decay0, seed42, EMA0.9999 + extra0.999, bf16, split·noise schedule·UNet을 유지한다. 한 GPU에서 순차 실행하며 먼저205K control,205K lowLR,210K control,210K lowLR를 비교한다. 둘의 시작 checkpoint는 동일하지만 업데이트 후 weight와 optimizer moments는 각 LR에 따라 달라진다.

## 주기적 quality/FID

- 매1000 update: 원 trainer의 고정 class0..3 / seed42 / EMA / CFG1 DDPM50 preview를 W&B `preview/cfg1_ddpm50`에 업로드한다. 이는 요청된 DDIM32 CFG2 holdout 평가와 별도다.
- 추가5K·10K: primary EMA / clipped DDIM32·50 CFG2 / seed42 / batch16 / bf16 / 생성3020장 / holdout reference755장 FID. DDIM32 generated에서 class0,13,50,75,103,133,150 샘플을 `qual/ddim32_cfg2`에 올린다.
- W&B scalar `eval/ddim32_cfg2_fid`, `eval/ddim50_cfg2_fid`의 x축은 누적 `global_step`이다. 기준200K의 이미 측정된 holdout FID를 두 run에 초기 anchor로 기록한다. 재측정으로 표기하지 않는다.
- 두 arm의 결과가 들어올 때마다 `comparison/ddim32_cfg2_fid`와 `comparison/ddim50_cfg2_fid` line chart를 갱신한다. x축은200K 이후 추가 update 수이다.
- 공식 validation으로 후보를 선택하지 않는다.

## 추가25K 조건

210K에서 두 arm 중 가장 좋은 DDIM32 holdout FID가 200K 기준41.94996548997665보다 **0.30 이상** 개선되면 둘 다220K,225K까지 연장하고 두 지점에서 같은 평가를 한다. 아니면210K에서 완료한다. 0.30은 실행 전에 정한 실무 기준이며 통계적 유의성을 뜻하지 않는다. 판단은 `extension_decision.json`에 저장한다.

## 실행·재개·확인

```bash
bash scripts/start_teacher_lr_comparison.sh
cat experiments/teacher_lr_comparison_200k/status.json
tail -n 30 experiments/teacher_lr_comparison_200k/orchestration.log
```

session: `teacher_lr_comparison_200k`. 결과 root: `experiments/teacher_lr_comparison_200k`. arm별 독립 W&B run은 group `teacher_200k_lr_comparison`에 묶는다. arm별 `wandb_identity.json`에 URL을 저장한다. stage마다 `latest.pt`/`final.pt`, FID milestones, source snapshot을 보존한다. background runner를 재실행하면 stage latest에서 복원한다. baseline SHA256 및 recipe/source guard, config 재읽기·재시작 검사를 수행한다.

두 분기를 모두 생성했지만 GPU에서는 현재 arm만 학습한다. 대기 arm에 실제 학습 중이라고 해석할 수 있는 loss를 복사하지 않는다. 완결된 FID는 각 평가 완료 후에만 기록된다.
