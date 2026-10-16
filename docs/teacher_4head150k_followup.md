# 4-head150K 후속 실험1·2·3

## EMA 및 sampling seed 비교

4-head150K와 원 single-head200K의 primaryEMA0.9999 / extraEMA0.999를 samplingseed42/1042/2042로 비교한다. 주 지표는 clippedDDIM32 CFG2이며 seed42에서는 DDIM50도 보조 평가한다. 생성3020장, development reference755장, batch16, bf16를 유지한다. seed42 primaryEMA는 기존의 동일 조건 측정값을 재사용하며 새 측정으로 표시하지 않는다.

sampling seed별 FID 평균과 표본 표준편차를 기록한다. 이는 training seed 재현성이나 reference set의 불확실성을 측정하는 실험은 아니다.

## LR 비교

평가가 끝나면 같은4-head150K full checkpoint에서 LR2e-4와1e-4 두 분기를 시작한다. 추가5K(155K)와10K(160K)에서 DDIM32/50 CFG2 holdout FID를 비교한다. raw model,optimizer moments,EMA,RNG,epoch/cursor를 복원한다. noaug,batch64,constantLR,warmup0,noise schedule,seed,구조를 유지한다. 추가10K를 넘어 자동 연장하지 않는다. EMA 평가에서 더 좋은 weight가 나와도 이번 paired training의 시작 weight를 바꾸지 않는다.

## W&B와 checkpoint

매1000update에 기존 CFG1 DDPM50 preview를 기록하고 FID milestone에서 DDIM32 CFG2 qual을 추가한다. 두 LR run과 EMA/seed 평가 run은 별도다. FID 비교 chart와 EMA/seed 결과 table을 업로드한다.

완료 stage의 latest.pt는 검증 후 final.pt를 가리키는 hardlink로 바꾼다. 같은 fullstate를 보존하면서 디스크 중복을 줄인다. 원150Kcheckpoint는 변경하지 않는다.

후보 선택은 development holdout으로만 한다. 별도 대기 reporter가 모든 실험 완료 후 기존DDIM32 생성 이미지를 공식reference3020장으로 score-only 평가해 OfficialFID 비교 panel을 기록한다. 공식 결과는 training/선택/gate에 사용하지 않는다.

## 실행과 결과

```bash
bash scripts/start_teacher_4head150k_followup.sh
cat experiments/teacher_4head150k_followup/status.json
tail -n 30 experiments/teacher_4head150k_followup/orchestration.log
```

결과root는 `experiments/teacher_4head150k_followup`이다. main tmux session은 `teacher_4head150k_followup`, 공식report session은 `teacher_4head150k_official_report`다.

- `evaluation_results.json`: 개별 EMA·seed FID
- `evaluation_summary.json`: samplingseed 평균/표준편차
- `lr_comparison/results.json`:155K·160K 비교
- `official_report/results.json`: 공식FID 보고
- 각 `wandb_identity.json`: run URL

검증: 네 LR config의 재읽기 및 변경 범위 확인,150K fullstate/mapping/manifest/head구조 guard 통과. 실제 평가·학습 완료는 status와 결과 파일을 기준으로 확인한다.
