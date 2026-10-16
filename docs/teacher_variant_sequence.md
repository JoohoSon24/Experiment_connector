# Teacher variant sequence

승인된 순서: **4-head attention → dropout0.05 → 추가32×32 attention**. 원본 `teacher/` 및 protected dataset/evaluator는 보존했다. 별도 `teacher_variants/`에 모델·config·trainer·evaluator를 복사하고 필요한 변경만 적용했다. `num_attention_heads`를 config와 checkpoint metadata에 기록하며 같은 tensor shape여도 head 수가 다른 checkpoint의 재개를 거부한다.

| 실험 | 단일 변경 | 시작과 budget | 대조군 |
|---|---|---|---|
| attention4 | 기존 attention 위치의1→4 heads | fresh seed42,100K; 유망하면150K→200K | 원 noaug 동일100/150/200K |
| dropout005 | ResBlock dropout0→0.05 | 원본 noaug200K full checkpoint에서 추가10K | 완료된 dropout0/LR2e-4 추가10K arm |
| attention32 | spatial attention에32×32 추가 | fresh seed42, single-head 유지,100K; 유망하면150K→200K | 원 noaug 동일100/150/200K |

**실험을 누적하지 않는다.** dropout은4-head 모델에서 분기하지 않고,32×32 모델도 dropout0·single-head이다. noaug, batch64, constantLR2e-4, warmup0, AdamW moments(재개 실험), noise1000, seed42, primaryEMA0.9999 + extra0.999, bf16와 split을 고정한다.

## Gate와 비용

구조 실험의100K DDIM32 CFG2 holdout FID가 원100K44.11890560635811보다0.30 이상 좋아지면150K와200K까지 진행한다. 아니면100K 평가 후 다음 실험으로 이동한다. operational threshold이며 통계적 유의성 기준은 아니다. 결정은 arm의 `extension_decision.json`에 저장한다. dropout은추가10K로 고정한다.

Parameter 수: attention4/dropout00548,883,107; attention3249,626,147.32×32 variant는 원래 batch64/bf16/AdamW로1 update 실제 GPU smoke를 수행했다. peak allocated memory12,264,184,320bytes(약12.26GB), finite loss·gradient로 통과했다. 검사 checkpoint는 학습 초기화에 사용하지 않는다. 학습 시작은 다시seed42로 초기화한다. CUDA peak20GB 및 parameter60M 제약을 유지한다.

## W&B와 평가

Project `pokemon-diffusion`, group `teacher_variants_single_change`. 각 실험이 시작될 때 독립 run을 만든다.

- 매1000 update: 고정 class0..3의 primaryEMA CFG1 DDPM50 preview.
- milestone 평가: DDIM32/50 CFG2, clipped DDIM, primaryEMA,seed42,batch16,bf16, 생성3020장,development reference755장.
- milestone마다 DDIM32 CFG2의 class0,13,50,75,103,133,150 샘플을 추가한다.
- `eval/ddim32_cfg2_fid`, `eval/ddim50_cfg2_fid` 패널과 matched-budget baseline 비교 bar chart를 기록한다.
- fresh 구조 실험은 새 FID가 나오기 전에 baseline FID를 그 모델의 측정값으로 올리지 않는다.
- official validation을 선택/gate에 사용하지 않는다.

## 검증

원 single-head attention 계산과 exact parity;4-head finite forward/backward; parameter count; config 재읽기;200K dropout fork의 fullstate/mapping/manifest 검증; same-shape head mismatch rejection; 원 teacher/protected source hash 검사;32×32 batch64 실제update memory test를 수행했다. 구조 모델의 장기 학습 개선은 아직 검증되지 않았다.

## 실행과 위치

```bash
bash scripts/start_teacher_variant_sequence.sh
cat experiments/teacher_variant_sequence/status.json
tail -n 30 experiments/teacher_variant_sequence/orchestration.log
```

tmux session `teacher_variant_sequence`. 결과 root `experiments/teacher_variant_sequence`. arm별 `wandb_identity.json`에 URL, stage별 latest/final checkpoint, source snapshot, eval metrics/PNG가 저장된다. `results.json`에 완료된 milestone 결과를 모은다. 재실행 시 stage latest에서 복원한다. 오류는 status에 기록하고 중단해 실패한 실험을 성공으로 간주하지 않는다.
