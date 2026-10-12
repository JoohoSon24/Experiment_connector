# 선택 teacher 하나에서 새 student 학습

원본teacher:4-head150K,primaryEMA0.9999,OfficialFID25.49841. 기존one_nfe/few_nfe/DMDcheckpoint를로드하지않는다. student는ModelOneNFE의single-headbackbone을seed42로randominitialization한다. real-score는선택teacher, fake-score는그teacher의복사본에서시작해student분포에맞춰학습한다. fake는추가로전달받은teacher가아니라DMD학습용모델이다. 추론에서는student1개만사용한다.

## 순서

1. 선택teacher의clippedDDIM32CFG2로4096개의synthetic(noise,class,image)쌍을생성한다. batch16,seed1000000+batchindex,labels는151classes를순환한다. 동일seed의초기noise와teacherendpoint를쌍으로보존한다. 실제training/validation이미지를loss에사용하지않는다.
2. fresh1-NFEstudent를teacherendpointMSE로5000updates학습한다. batch64/micro16×4,AdamWLR2e-4constant,clip1,EMA0.999.
3. 동일teacherreal-score로DMD2500updates진행한다. fakewarmup200,fakeLR2e-4,G LR2e-5constant,clip1,EMA0.999,t50..950. endpoint규제0.1도기존few_nfe가아니라선택teacher의cachedendpoint를사용한다. 초기단계optimizer의moments는이어사용하고LR만DMD단계값으로바꾼다.

이실험은originalhandoff의이미학습된studentfine-tuning과**초기화·bootstrap단계가다르다**. 따라서완전히같은조건의38.6재현이라고부르지않는다. 목적은기존legacy모델없이새teacher하나에서student를만드는것이다. finite4096noise/classpool은전체noise분포를대표하지않으므로holdoutFID및새samplingnoise에서품질을평가한다.

매100updatesfixedseedqual을W&B에기록한다. bootstrap종료및DMD500/1000/2500에서3020장생성/holdout755장/batch16/bf16/seed42로FID측정. development기준으로checkpoint선택후그PNG를officialreference3020장으로평가한다. officialFID를학습/선택에사용하지않는다.

## 상태와 결과

W&B: https://wandb.ai/fv2z6n929c-kaist-digital-humanities-and-social-sciences-/pokemon-diffusion/runs/i9yplbdw

결과: `experiments/fresh_student_4head150k` / tmux:`fresh_student_only_selected_teacher`.

```bash
cat experiments/fresh_student_4head150k/status.json
tail -n 20 experiments/fresh_student_console.log
```

syntheticpair생성부터시작했으며장기학습품질은아직검증되지않았다. memory20GBguard와finitegradient/lossguard를유지한다. 추론용EMAcheckpoint를보존하고exactoptimizerresume는구현하지않았다.
