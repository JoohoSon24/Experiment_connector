# Fresh student + DMD four-experiment queue

선택4-head150K/EMA0.9999 teacher 하나만 사용한다. 기존one_nfe/few_nfe/handed-offDMD2500weight는사용하지않는다. 진행중인synthetic4096pair생성은그대로완료한다. cache를정상로드할수있는시점에기존단일실험runner를종료하고bootstrap-onlyrunner로전환한다. 전환직전작은bootstrapupdate가시작됐으면log를보존하고freshseed42에서bootstrap을다시시작한다. 완성된teacherpair는재사용한다.

## 공통student

student는randominit1-NFE/single-headUNet이다. teacherDDIM32CFG2의noise/class/endpoint4096쌍을batch64/micro16×4로5000updates학습한다. bootstrapLR2e-4constant,EMA0.999,clip1,seed42. `bootstrap_005000.ckpt`의EMA를네DMDarm의공통초기값으로사용한다. 각arm의GAdamW/fakeAdamW를새로만들고RNG를42로설정한다. fake-score는동일4-headteacher의EMA복사본에서시작한다. bootstrapoptimizer를DMD에이어사용하지않아네arm의시작조건을맞춘다.

## 자동순서

| Arm | G LR | endpoint 규제 가중치 | 변경 |
|---|---:|---:|---|
| A |2e-5|0.1|기준|
| B |2e-5|0|endpoint규제제거|
| C |2e-5|0.3|endpoint규제강화|
| D |1e-5|0.1|G update축소|

모든endpointtarget은새teacher의cachedDDIM32CFG2출력이다.legacy4-NFEendpoint를사용하지않는다. 각arm2500Gupdates, fakewarmup200/fakeLR2e-4,constantLR,warmupLR0,weightdecay0,GEMA0.999,gradclip1,t50..950,bf16,batch64/micro16×4를고정한다. DMDtraining은syntheticnoise/image만사용한다. cache는finitepool이며quality는별도의새samplingnoise에서평가한다.

## 평가와보존

매100updatesW&Bqual.250/500/1000/2500에서1-NFE생성3020장/developmentreference755장/batch16/seed42/bf16FID. 각arm은bootstrapbaseline포함최저holdoutFIDcheckpoint를선택하고그generatedPNG로officialreference3020장FID를보고한다. officialFID는선택하지않는다.

디스크사용량때문에각arm의최선inferenceexport만보존한다. 모든milestonePNG/metrics/trainlog는보존한다. optimizer/fake/RNG의exactresume는구현하지않았다. 단일GPU에서각arm의평가·공식보고후즉시다음arm을시작한다. 실패하면status에기록하고잘못된상태로후속실험을강행하지않는다.

원200Kfullcheckpoint와동일SHA의handoff복사본은hardlink로중복저장을줄였다. 파일bytes/SHA와원본checkpoint는변경하지않았다.

## 확인

Queue tmux:`fresh_student_DMD_four_arms`. 결과:`experiments/fresh_student_dmd_queue`.

```bash
cat experiments/fresh_student_dmd_queue/status.json
tail -n 20 experiments/fresh_student_dmd_queue_console.log
```

W&Bgroup:`fresh_student_DMD_four_arms`. 각arm이시작될때독립run을만들고`wandb_identity.json`에URL을저장한다. studentbootstrap은기존W&B `i9yplbdw`에서이어기록한다.teacherpair4096쌍이완료됐고,현재공통studentbootstrap학습이진행중이다.
