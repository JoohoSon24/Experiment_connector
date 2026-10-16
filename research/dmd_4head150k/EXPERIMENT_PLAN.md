# Selected4-head150K teacher DMD experiments

## 전달물과 입력 검증

사용자 지정 `/root/Diffusion/dmd_handoff_2500`의 source overlay SHA256을 검사하고 별도 workspace로 복사했다. 원 handoff는 수정하지 않았다. handoff의 selected `dmd_step_002500.ckpt`는 문서상195,653,282bytes/acc998...여야 하지만 실제약21MB/5d2af...이며 PyTorch zip central directory 로드에 실패한다. 해당파일은사용하지않는다. `handoff_integrity.json`에 expected/actual SHA를 보존했다.

그 대신 handoff의 **재학습 가이드와 같은 원one_nfe 초기화**를 사용한다. 따라서 새실험은 handoff2500student의정확한continuation이아니며, 기존one_nfe에서의새DMDfine-tuning이다. optimizer는새로초기화한다.

## 두 실험

| Arm | Real-score teacher | G 초기값 | Endpoint규제 |
|---|---|---|---:|
| endpoint01 | 선택4-head150K / EMA0.9999 / CFG2 | 기존one_nfe.ckpt |0.1|
| endpoint00 | 동일 | 동일 |0|

새teacher는 `/root/Diffusion/workspace_distillation/experiments/teacher_variant_sequence/attention4/attention4_150k/final.pt`이다. 고정공식seed42 FID25.49841 모델이며 samplingseed1042의최저단일점수를다른weight로취급하지않는다.

Frozen real-score와 trainable fake-score는4-head구조를사용한다. G는기존single-head velocity1-NFE구조를유지한다. 이구조간차이는noise score로분포gradient를제공하는데장애가되지않지만,4-head student구조의직접weight복사는아니다. 각network48,883,107parameters이며 infer시G하나만사용한다.

endpoint01은handoff의legacy4-NFE `few_nfe.ckpt`를weakendpointanchor로사용한다. 이endpoint는새4-headteacher의자식이아니다. 새teacher→G분포전달은real-score를통해서만일어나며endpoint00은legacyanchor없이그효과를비교한다. 원DMD논문의정확한재현이아니라handed-off VP adaptation이다.

## 고정recipe

각2500Gupdates, batch64/micro16 accumulation4, seed42, AdamW G LR2e-5/fakeLR2e-4 constant,warmupLR0,weightdecay0,gradclip1,G EMA0.999. fake200warmupupdates는현재G합성이미지만사용한다. t50..950, epsilonfakeMSE,normalizedx0real/fake scoredifference와0.1/0endpointMSE를사용한다. 두arm은각각동일하게초기화하고baseline평가후RNG를다시42로설정해cache유무의영향을피한다.

reference/official이미지나Inception feature가trainingloss에들어가지않는다. realteacher의trainmanifest·mapping은검증했다. legacyG/endpoint의전체선조holdout제외는추론checkpoint만으로독립검증할수없다는handoff의한계를유지한다.

## 평가·W&B

매100updates 고정classes8개/seed42/EMAqual.500/1000/2500에서1-NFEinferenceexport와developmentFID를저장한다. 생성3020장,reference755장,batch16,seed42+batch_index,bf16을고정한다. hook으로3020장생성의backbone189calls를검증한다. baseline포함minimumdevelopmentFID로각arm의checkpoint를선택하고,선택된동일PNG에officialreference3020장으로FID를보고한다. 공식FID로선택하지않는다.

Projectpokemon-diffusion/groupdmd_selected4head150k_endpoint_ablation. 두arm은단일GPU에서순차실행한다. 각run의 `eval/development_fid`, `eval/selected_official_fid`,qual을기록한다.

## 결과와 실행

```bash
bash scripts/start_dmd_selected_teacher.sh
cat experiments/dmd_4head150k/status.json
tail -n 20 experiments/dmd_4head150k/orchestration.log
```

tmuxsession `dmd_4head150k_two_arms`. 기존재현 `.venv`를사용하며새패키지는설치하지않는다. 각arm폴더에 `train.log`, `results.json`, `dmd_step_000500.ckpt`/001000/002500,qual,evalmetrics,`completed.json`을저장한다. inferenceexports만보존하며optimizer/fakescore/RNG full-resume는구현하지않았다. archivedtrainer역시resumeCLI가없다. 중단된run은자동으로덮어쓰지않고보존한다.

GPUpreflight에서두regloss의fullbatchupdate,finitegradient/output,1-NFEforward,peak4,522,433,024bytes를확인했다.실험G를smoke업데이트에서이어학습하지않고매arm정상초기화한다.
