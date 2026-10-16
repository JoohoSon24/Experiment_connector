# Artifact provenance

This handoff captures the reproduced run in `/root/DMD`, separate from the earlier successful run in `/root/AIC313-Project-FastGen`. The user approved creation at `/root/dmd_handoff_2500`; `teacher_handoff_200k` was not searched for during this build and is not a server dependency.

## Workspace reconstruction

Upstream is `https://github.com/KAIST-Visual-AI-Group/AIC313-Project-FastGen`, commit `e3f5120000cd91037017d86f23993492195101a5`. The pre-existing DMD directory was backed up as `/root/DMD_preexisting_20261007_161552`, a fresh official clone was made, and the team overlay from `/root/team_fastgen_handoff_new/fastgen/fastgen_code_overlay/` was inspected and applied with rsync. The two supplied checkpoints came from that handoff's `checkpoints/` directory. The fresh diffusion and source script came from the older AIC313 workspace. Reference images/manifest were recovered and a local split audit was rebuilt before launch.

The original overlay's 103 file hashes match the inspected source. The earlier successful DMD source snapshot's 17 files matched the corresponding current files during inspection; fresh teacher's 12 Python snapshot files also matched. `model.py` differs from upstream only in the two NFE subclasses; base Model and protected files are unchanged. See `model_vs_upstream.patch`, the original overlay manifest and `verification/source_hashes.json`.

`dmd_reproduce.py` changes only the local manifest lookup and removal of two early stops and a six-hour timeout. The exact diff and launch/preflight excerpt are included. tmux `dmd` was created 2026-10-08 04:31:39 UTC; the complete run report is dated 2026-10-08 08:35:48 UTC. Shell history lacks reliable per-command timestamps and can merge shells; its line order alone is not proof of chronology.

## Model ancestry and trust boundary

Team 1-NFE source metadata names `experiments/low_nfe_budget_100k/n1/final.pt`; team 4-NFE names `experiments/fixed_cfg2_progressive/n4/final.pt`. Team documents describe fixed-CFG progressive lineage. These already-distilled models supply trainable G initialization and frozen endpoint targets respectively. Their complete ancestor checkpoint hash chain and independently verified exclusion of the 755 development images are unavailable.

The fresh diffusion teacher is NOT the parent teacher of the provided team 1-NFE or team 4-NFE checkpoints. It was initialized randomly in the separate two-step teacher_preflight run, then resumed exclusively from that run to total step100000. Historical launcher and metrics support this chain. The DMD real-score uses its primary `ema` (decay0.9999); fake is a trainable deepcopy. Fresh checkpoint `train_paths_sha256` exactly matches the current 16324-path newline-joined development manifest. This verifies path provenance, not hashes of every training pixel throughout all earlier runs.

The selected 2500 inference export contains EMA G only, not optimizer/fake-score continuation state. Its hash equals `selected_one_nfe.ckpt`. Training used three input checkpoints, but final inference opens only the selected generator. Python backbone code lives under `teacher/`; teacher weights are not needed at inference.

## Dataset and reference identity

The fixed course splits are 17079 original train and 3020 validation. Development train/holdout are 16324/755, deterministic seed42, five held out per class. Split subset/disjointness/union and path-duplicate checks pass. The 755 manifest sources equal holdout.txt in order. Every reference PNG hash matches and in-memory reconstruction from current data produces the same 755 hashes. These checks do not prove image-content deduplication or team-ancestor holdout exclusion.

The raw train.txt hash is `99e2aef7b68f4792ce1440c1196dd948310fdd7fc6479068595c96a30558b24e`; the checkpoint hashes newline-joined paths with **no trailing newline**, giving `b4020d6dc65948c7cd2a0de048382284d0fc7a1dd19237583e4f67cc2d353f51`. The reference manifest retains its historical absolute source path as provenance metadata only.

## Results and corrections

Selection is based on development holdout FID, with the team-ancestry limitation above: 65.38627014891784 baseline, 55.72514913187206 at2500, 58.62200371627705 at5000, 60.44414080173647 at10000. Official validation is separate reporting evidence; user-recovered current 2500/5000 numbers lack located numerical artifacts. Older DMD metrics refer to a different run and hash. See `provenance/official_validation_records.md` and `provenance/historical_caveats.md`.

`reports/REPORT.md` is preserved unchanged despite stale conditional stopping text. `config.json` is a script-emitted snapshot, not the trainer's input interface. The handoff-created helpers/checks are labeled new verification. Current `latest.pt` is step10000 and excluded; no exact step2500 optimizer resume state is supplied.

## Portability and included code

The 103-file source_overlay is preserved, supplemented with unmodified official runtime utilities, README, requirements and fixed splits. New helpers resolve the package relative to themselves and take a workspace argument. Historical launch scripts and metadata retain original absolute paths; they are archival, not supported entrypoints. Restore `scripts/dmd_reproduce.py` to `experiments/night_run/` of a **new checkout** so its parent-based ROOT calculation is correct. No symlinks are needed.

## Source workspace state

The source git status before packaging was:

```text
 M model.py
?? .gitignore
?? CODEX.md
?? OVERLAY_MANIFEST.json
?? PROJECT_SPEC.md
?? TEAM_HANDOFF.md
?? configs/
?? consistency/
?? constraints/
?? data/development/
?? distillation/
?? docs/
?? scripts/
?? teacher/
```

`verification/source_before.json.gz` records the source tree before build, including SHA256/size/mode/mtime for every regular file, git status and commit. Final comparison is in `verification/source_unchanged.json`. Atime is excluded because read-only inspection can update it. New files, removals, content changes, symlink changes and metadata differences are checked; no source experiment is used as an output directory. Three empty tool-injected virtual directory entries (`.agents`, `.aws`, `.codex`) differ between sandbox/host views and their metadata is explicitly excluded. No regular file is excluded; raw view differences are preserved in `verification/source_execution_view_differences.json`.

## Artifact inventory

Each row below gives a byte identity. `provenance/artifacts.json` additionally records role, original absolute path, source/parent relationship, frozen/trainable status, parameter count, inclusion and retraining requirement for every copied/registered original artifact. Values not applicable to code/logs are null. New authored documents and verification outputs are covered by `SHA256SUMS`, not presented as historical files.

| Original path | Package path (or excluded) | Bytes | SHA256 |
|---|---|---:|---|
| `/root/DMD/.gitignore` | `source_overlay/.gitignore` | 195 | `304a29f5452da2fb2dd95359f364b4934bd4043f6402080a2d0d80e1cad9442e` |
| `/root/DMD/CODEX.md` | `source_overlay/CODEX.md` | 15255 | `d79ff2468748557885995b5ec6c9b4c3e91383c743275157cb0054abaef9073b` |
| `/root/DMD/PROJECT_SPEC.md` | `source_overlay/PROJECT_SPEC.md` | 12142 | `8ffda390fc55d062c9a2e2d00e60a3a565051e5cd12cc525684a4346fedcb27b` |
| `/root/DMD/TEAM_HANDOFF.md` | `source_overlay/TEAM_HANDOFF.md` | 5599 | `4cccbe29306c3a39d4adc6edd48b544cbd5dc2462366e83c163aef181eb0a6c9` |
| `/root/DMD/configs/consistency/fixed_cfg2.json` | `source_overlay/configs/consistency/fixed_cfg2.json` | 543 | `2c2f343d8589e80fc6f4399a17429599158516c3e42d1a280c0affece41f2e8c` |
| `/root/DMD/configs/consistency/no_extra_guidance.json` | `source_overlay/configs/consistency/no_extra_guidance.json` | 559 | `49647dc9653ce443d56716d3b46e869d1928a3a9ab268b90e1dab5f2d3c7c128` |
| `/root/DMD/configs/distillation/fixed_cfg2_progressive.json` | `source_overlay/configs/distillation/fixed_cfg2_progressive.json` | 789 | `88ebb482e45771bc971d4da10464d3122c0d478e5994b176d3998ea18433d65b` |
| `/root/DMD/configs/distillation/low_nfe_budget_100k.json` | `source_overlay/configs/distillation/low_nfe_budget_100k.json` | 846 | `43438567cabeaebf2e762b3bea1e86bf6ff4a074ce94bd20f124dad08da82c40` |
| `/root/DMD/configs/teacher/debug.json` | `source_overlay/configs/teacher/debug.json` | 886 | `f8fa271cdf583301367a718f9697c36427be805324301ffb82080723581a7c4e` |
| `/root/DMD/configs/teacher/debug_fresh_five.json` | `source_overlay/configs/teacher/debug_fresh_five.json` | 897 | `4891f8d56af12b246e52d955ea636044d94efb5246c0bc7bcaabfa4c0c692f92` |
| `/root/DMD/configs/teacher/debug_resume.json` | `source_overlay/configs/teacher/debug_resume.json` | 893 | `57c33c5e178e759330b8cb94b28c4f7eaeed19177546481a6af6799601d9e7de` |
| `/root/DMD/configs/teacher/initial.json` | `source_overlay/configs/teacher/initial.json` | 925 | `962798bdb25672c7adc6c387c58dc9b24644cbe62415e76f69a96fb588e673e6` |
| `/root/DMD/configs/teacher/initial_probe.json` | `source_overlay/configs/teacher/initial_probe.json` | 918 | `dd5edb299a17616dcdc969ba9efbf459b674adf7fc7093185851ede7cb9b7b78` |
| `/root/DMD/configs/teacher/teacher_b128_100k.json` | `source_overlay/configs/teacher/teacher_b128_100k.json` | 1496 | `57e291d0898886fe6ca0bc42a998773802d9d0de9200dd51271a929b403beb93` |
| `/root/DMD/configs/teacher/teacher_b64_100k.json` | `source_overlay/configs/teacher/teacher_b64_100k.json` | 1484 | `8a2d7e96effd8c235bbdc5290113f1e9fa7e2f98b61449ffcbfbb13ff6db792e` |
| `/root/DMD/configs/teacher/teacher_b64_constant_100k.json` | `source_overlay/configs/teacher/teacher_b64_constant_100k.json` | 1495 | `85c3a78d032ec94c7d6da7b5e33adb019f265b42d8e24b22e06736a4ea71e3c6` |
| `/root/DMD/configs/teacher/teacher_b64_fliponly_constant_100k.json` | `source_overlay/configs/teacher/teacher_b64_fliponly_constant_100k.json` | 1502 | `8ae21975a13ba7427a73ffc42e33e81b954d9d6230a5dd8b209e7af2a1cc60bf` |
| `/root/DMD/configs/teacher/teacher_b64_noaug_constant_100k.json` | `source_overlay/configs/teacher/teacher_b64_noaug_constant_100k.json` | 1499 | `c5f960f142a7df88b7830d69f2968884d9e5ad776cdcb0b8a57af6ef2f9577ce` |
| `/root/DMD/configs/teacher/teacher_ddpm_cfg_v1.json` | `source_overlay/configs/teacher/teacher_ddpm_cfg_v1.json` | 924 | `46715255adfdb53e471f9824f010f1111282608aa43437a1a22b739626ab5cf0` |
| `/root/DMD/configs/teacher/teacher_ddpm_cfg_v1_probe.json` | `source_overlay/configs/teacher/teacher_ddpm_cfg_v1_probe.json` | 919 | `87eb3144b7d9c7bd29896d1dba8b0d44dce2421cf5183d99d66335cd5e88715b` |
| `/root/DMD/configs/teacher/teacher_lr2e4_linear_aug_50k.json` | `source_overlay/configs/teacher/teacher_lr2e4_linear_aug_50k.json` | 1109 | `3de1cff71241d960ef8823047f23c20e97f935ea86a86ffe7b6743fad00044fa` |
| `/root/DMD/configs/teacher/teacher_lr5e5_aug_50k.json` | `source_overlay/configs/teacher/teacher_lr5e5_aug_50k.json` | 1063 | `a9cd6a6a554a2c147186e87c30c0b89989d02280765fac4b92aee564ea17758f` |
| `/root/DMD/configs/teacher/teacher_lr5e5_aug_probe.json` | `source_overlay/configs/teacher/teacher_lr5e5_aug_probe.json` | 1054 | `98ce4d5de200beb72f2addf56ae72c0a9b73a2ae02ffd7ead3bd273e8dacbcb7` |
| `/root/DMD/configs/teacher/teacher_memorization64_constant.json` | `source_overlay/configs/teacher/teacher_memorization64_constant.json` | 1484 | `74f8bbe8f711364101a958ae00339fa8f731c14e64da2e20660b301191483bae` |
| `/root/DMD/consistency/__init__.py` | `source_overlay/consistency/__init__.py` | 80 | `bb8c4edcab0e7c95e553c828c89b50de4273cb93640312511d2256cd5e44bed0` |
| `/root/DMD/consistency/check_submission.py` | `source_overlay/consistency/check_submission.py` | 1600 | `a177c595cd94e2cce430afc8627b1776e212e195cc2f42251bcd4f9e648b6bf5` |
| `/root/DMD/consistency/config.py` | `source_overlay/consistency/config.py` | 2459 | `06661a9d36dfe5ae138cf08221d0f2d4fba6f00d4a913580c058d8016f4549af` |
| `/root/DMD/consistency/core.py` | `source_overlay/consistency/core.py` | 5734 | `34828a734f0410e3c13b08ab266b5155dc710cff6c6bbbf5192506aff0b70732` |
| `/root/DMD/consistency/evaluate.py` | `source_overlay/consistency/evaluate.py` | 6903 | `e896a0e0e59e182651f957cf81c7fe3b32ff61084dfab604ba8bab18dd001ca6` |
| `/root/DMD/consistency/export.py` | `source_overlay/consistency/export.py` | 2799 | `7ffc1724d4cf8db6fd24ff96e6dda10baf3d4d8d5e468d18e117d8c965d90af9` |
| `/root/DMD/consistency/submission.py` | `source_overlay/consistency/submission.py` | 558 | `47e0aa7b76118b691cbe8b6547bb9a83507190e68ee7b2d41487edcccb819fd2` |
| `/root/DMD/consistency/tests.py` | `source_overlay/consistency/tests.py` | 7481 | `59c91a869e223b6e5fb3e98eb7312a77523dd060596baec52bd5613bcaaff271` |
| `/root/DMD/consistency/train.py` | `source_overlay/consistency/train.py` | 16043 | `0e630de9e13f8f2afc657931fac8837d6eae70f5d15443d65851bacbb2e64260` |
| `/root/DMD/constraints/evaluation.txt` | `source_overlay/constraints/evaluation.txt` | 113 | `ac76ddbce734dca5e70abb89090255c880e15ecf6ae24ca422625f15257130e9` |
| `/root/DMD/data/development/seed42_holdout5/holdout.txt` | `source_overlay/data/development/seed42_holdout5/holdout.txt` | 31510 | `a9b672630e28c4dea5fd922a22a1e11dc78155e4c96e88d4bca90bb3ce920a09` |
| `/root/DMD/data/development/seed42_holdout5/metadata.json` | `source_overlay/data/development/seed42_holdout5/metadata.json` | 349 | `78f5c8bba3f23fa4adc81afc278069b546cec9044a7df9a5dbbf1769d5ea4b4b` |
| `/root/DMD/data/development/seed42_holdout5/train.txt` | `source_overlay/data/development/seed42_holdout5/train.txt` | 671861 | `99e2aef7b68f4792ce1440c1196dd948310fdd7fc6479068595c96a30558b24e` |
| `/root/DMD/distillation/__init__.py` | `source_overlay/distillation/__init__.py` | 78 | `05b480c34971f4411ba7e0292a184d2ac4ef89a595cc7b3008f4d7ee0e9936fc` |
| `/root/DMD/distillation/core.py` | `source_overlay/distillation/core.py` | 2646 | `45c4cff601e442f01a42b7af5b19fd52474901c720c9cc76e60957d55441357b` |
| `/root/DMD/distillation/evaluate.py` | `source_overlay/distillation/evaluate.py` | 4591 | `76ccdb7d8f486d969872d9a50ab6935e4737615ddf3fb21e0e57930386a72a70` |
| `/root/DMD/distillation/export.py` | `source_overlay/distillation/export.py` | 2014 | `58d5b04bc1427ce690009049de4dea48ef8d1d242d7c142b9466353fa2cbfb68` |
| `/root/DMD/distillation/pipeline.py` | `source_overlay/distillation/pipeline.py` | 6860 | `bbc2ed349d97263d29439e325e846cddd2e5e11bfb22351c2eb8414bcb26c3d7` |
| `/root/DMD/distillation/smoke.py` | `source_overlay/distillation/smoke.py` | 4291 | `54255dba5e4d84b4123eabf70202bc569ae7836bffa16e7256f893a9a8b09ba0` |
| `/root/DMD/distillation/submission.py` | `source_overlay/distillation/submission.py` | 2068 | `112e835982873923c34fecd0dbb37ef58941c9cc3de6905021225e795c153498` |
| `/root/DMD/distillation/train.py` | `source_overlay/distillation/train.py` | 12048 | `cfcc9f5b926a0ddce5b1760a9c31eeca80bede987d065ed1bc676c8b0169d8d1` |
| `/root/DMD/docs/consistency_distillation.md` | `source_overlay/docs/consistency_distillation.md` | 12456 | `d9d08dca61df40681571b36ee2554965b993b2f353d952d311f9dc70041bec73` |
| `/root/DMD/docs/distillation_baseline.md` | `source_overlay/docs/distillation_baseline.md` | 10286 | `e4b6b98322bf2e4673719ea343b683ebd52bded00b8ba0f2712cd82ebf7b5069` |
| `/root/DMD/docs/distillation_results_review.md` | `source_overlay/docs/distillation_results_review.md` | 8350 | `3086f195a86c9e8528b40a57cf12be8fb39f441f56a24dbecb9a44b7a5699ccb` |
| `/root/DMD/docs/experiment_cleanup.md` | `source_overlay/docs/experiment_cleanup.md` | 4947 | `98a0f926751846c8722fb973ed5aac4eff85bc97ebb71427438027fb389cd1f7` |
| `/root/DMD/docs/low_nfe_budget_experiment.md` | `source_overlay/docs/low_nfe_budget_experiment.md` | 5666 | `30d21d93e3bbdf724dabfe3087f9ccecbc0ce1fd576de748b4a5a4deec95e4ca` |
| `/root/DMD/docs/model_architecture.md` | `source_overlay/docs/model_architecture.md` | 10937 | `6cedc83f0d8c93d75fada0038ba0007793abe371b3b742f8c20e8963b0a55e3f` |
| `/root/DMD/docs/repository_analysis.md` | `source_overlay/docs/repository_analysis.md` | 9020 | `f65b3625497115a781cacb578e453f19f8e199cfdeb88ae7c5ca9d40428a632f` |
| `/root/DMD/docs/sampling_defaults_and_ddim_cfg.md` | `source_overlay/docs/sampling_defaults_and_ddim_cfg.md` | 2981 | `c990c19dfcdec093bae82f5acb9f8ba95d341006b93744afb5b30c851dd5bfaf` |
| `/root/DMD/docs/sampling_steps_seeds_analysis.md` | `source_overlay/docs/sampling_steps_seeds_analysis.md` | 4915 | `a0f8c422f7ad062915756fbcc237966f020efeda504e451ee9f95ec3a9a0ab39` |
| `/root/DMD/docs/teacher_b64_100k.md` | `source_overlay/docs/teacher_b64_100k.md` | 6386 | `bd9aae3f33c46dbd42e8d1552dfe5402ba2ccca5d3112ebff105d28dd6c5e4da` |
| `/root/DMD/docs/teacher_b64_results_and_next.md` | `source_overlay/docs/teacher_b64_results_and_next.md` | 6522 | `9c141447da78665092d27bc6f55125c4bf1645a3a46f3c3fc215d05c8f56f901` |
| `/root/DMD/docs/teacher_baseline.md` | `source_overlay/docs/teacher_baseline.md` | 10621 | `e2625536323532778199de4c1e0f9d8c46b3a4ed29cf98ff7a37921a7c1e0c31` |
| `/root/DMD/docs/teacher_constant_results_and_fliponly.md` | `source_overlay/docs/teacher_constant_results_and_fliponly.md` | 7648 | `3cb1a66c2cc0afb4671ae9ff4f82df20ca00c053b78c6faf5b35825fff40647a` |
| `/root/DMD/docs/teacher_diagnosis.md` | `source_overlay/docs/teacher_diagnosis.md` | 5569 | `b6b136518daa3f80b863ac4cdded94ac896f335e1e773a393878bcace3477052` |
| `/root/DMD/docs/teacher_fliponly_preflight.json` | `source_overlay/docs/teacher_fliponly_preflight.json` | 521 | `3461585d5953850135462ab84dc69e932199d63c53d4133633b869b49d72e091` |
| `/root/DMD/docs/teacher_lr2e4_linear_aug_run.md` | `source_overlay/docs/teacher_lr2e4_linear_aug_run.md` | 2980 | `e12909404958c2976cb70f6715e69fc3919441692aefabc0c515675aaf20a90f` |
| `/root/DMD/docs/teacher_lr5e5_aug_run.md` | `source_overlay/docs/teacher_lr5e5_aug_run.md` | 3374 | `692e346ff883bc715375f85a966fde5436727b5f3fe1aa565b2090657a8cf8f9` |
| `/root/DMD/docs/teacher_remaining_experiments.md` | `source_overlay/docs/teacher_remaining_experiments.md` | 5020 | `77d4aeec3db03e84fae5bb34aae03340432a77f3b65ce007b97fbe9332f529df` |
| `/root/DMD/docs/teacher_review_50k.md` | `source_overlay/docs/teacher_review_50k.md` | 12246 | `d0f7bda1cd274152c8f97361933610c3516b7c9b40f260ccfa4586d3999ad9de` |
| `/root/DMD/docs/teacher_training_run.md` | `source_overlay/docs/teacher_training_run.md` | 5013 | `c38350b28cf8a1400aec6d9a9e7dc090ebfc35f091bb168e092a3abe14f52401` |
| `/root/DMD/model.py` | `source_overlay/model.py` | 3742 | `7f78c718bdbee8a00c783ed5d8be98f144edff405544b0ed0120050d256899e2` |
| `/root/DMD/scripts/audit_teacher.py` | `source_overlay/scripts/audit_teacher.py` | 8545 | `81794f20b575c6e61c2e8b7f0bf202984c747e9c63dbc29c2519f029d394a5c1` |
| `/root/DMD/scripts/benchmark_teacher.py` | `source_overlay/scripts/benchmark_teacher.py` | 3714 | `3ebfd7c2e5010cc0e99d9fec80bf4c9c70404e8d2f6403aa32ba9482045fbdf2` |
| `/root/DMD/scripts/compare_teacher_steps_seeds.py` | `source_overlay/scripts/compare_teacher_steps_seeds.py` | 2385 | `cb514daae05f96ee71813b1cbcb9c1f8876c1cdcb553e9e130e8acdb6b1f160d` |
| `/root/DMD/scripts/create_handoff.py` | `source_overlay/scripts/create_handoff.py` | 3686 | `c27cbdfa37dc059039983702153e6917319793cfc868ffea8ef3dd29c1db728e` |
| `/root/DMD/scripts/diagnose_teacher.py` | `source_overlay/scripts/diagnose_teacher.py` | 3283 | `4f75b45fc44c14136d06060733386784bf3cba4e6fe2bc00834d25ccd8f90673` |
| `/root/DMD/scripts/make_student_class_grids.py` | `source_overlay/scripts/make_student_class_grids.py` | 3570 | `afafa9a0fc3f252c92f45527df010142da46e21c11a325561091e79490af188c` |
| `/root/DMD/scripts/report_teacher_official.py` | `source_overlay/scripts/report_teacher_official.py` | 2842 | `be8a6c40b0a01a6c541fcb9f12a061a9abb18bdf24d67b5f800bc5d52f0b4c81` |
| `/root/DMD/scripts/review_ddim_cfg.sh` | `source_overlay/scripts/review_ddim_cfg.sh` | 395 | `c41bacc5233b17348f9526e4885b157df45722ed23f0433f215a805966819271` |
| `/root/DMD/scripts/review_teacher_run.py` | `source_overlay/scripts/review_teacher_run.py` | 2868 | `df8986e5034ef85ade20d6293c750649e31deefacedbffa14eb5c4d9a4a84b59` |
| `/root/DMD/scripts/run_consistency.sh` | `source_overlay/scripts/run_consistency.sh` | 1109 | `d0239d1f2b391c670a0a9bfd11c563fc62123c2542789eba9cf9e9af58c5c7f7` |
| `/root/DMD/scripts/run_fixed_guidance_distillation.sh` | `source_overlay/scripts/run_fixed_guidance_distillation.sh` | 670 | `642dfb2d457cd4c682390ee04bc768cb5437c847602013ca39dfc6bce9c7ba4c` |
| `/root/DMD/scripts/run_low_nfe_budget.py` | `source_overlay/scripts/run_low_nfe_budget.py` | 7016 | `8ee102272e32a951b595be159412c87076da5494b6f7bfcc62385468d9eb84b2` |
| `/root/DMD/scripts/run_low_nfe_budget.sh` | `source_overlay/scripts/run_low_nfe_budget.sh` | 380 | `fae62768ff5881bed10e38850a62bb4e3228465c0049b440d3a34817c01d8545` |
| `/root/DMD/scripts/run_teacher_next_experiments.sh` | `source_overlay/scripts/run_teacher_next_experiments.sh` | 464 | `d4d3b3c683c74d979783cbbeb893a2e9ff5fe36b67e7516c73668fb95fb72b40` |
| `/root/DMD/scripts/run_teacher_to_students.sh` | `source_overlay/scripts/run_teacher_to_students.sh` | 693 | `74d449cf6a2b0c003faa6fe988a4ccd608d27db84b1f0f78df7a935b66926765` |
| `/root/DMD/scripts/supplement_ddim_diagnostics.py` | `source_overlay/scripts/supplement_ddim_diagnostics.py` | 2531 | `a84cd44e7744daa65f986fe254638610dc0be3bc0c140345b0f919ebecc38dd5` |
| `/root/DMD/scripts/train_teacher_b128.sh` | `source_overlay/scripts/train_teacher_b128.sh` | 1104 | `cb6fb155c159a0a8f41a00bb718f8d9723d75fec1516fa43c54b272a07a7dbdd` |
| `/root/DMD/scripts/train_teacher_b64.sh` | `source_overlay/scripts/train_teacher_b64.sh` | 1102 | `c34d60e150c333506eaf68a4530b4ac3f0432f023280e1eed927438dd36abef3` |
| `/root/DMD/scripts/train_teacher_b64_constant.sh` | `source_overlay/scripts/train_teacher_b64_constant.sh` | 1120 | `c1a0e1f7f2c02fb1b9f02354ca42f2cbe299ab622d504841e7a2e7d3dadee3ce` |
| `/root/DMD/scripts/train_teacher_b64_fliponly_constant.sh` | `source_overlay/scripts/train_teacher_b64_fliponly_constant.sh` | 1138 | `74ec4f69606433b6ef31821abb275d5c35227eb86d20efa36dcc8b7a0936e341` |
| `/root/DMD/scripts/train_teacher_b64_noaug_constant.sh` | `source_overlay/scripts/train_teacher_b64_noaug_constant.sh` | 1132 | `e0040de77808e194d35961eefa4be4f3604d6c9e7f2bd20399d9ce37e0e6c582` |
| `/root/DMD/scripts/train_teacher_lr2e4_linear_aug.sh` | `source_overlay/scripts/train_teacher_lr2e4_linear_aug.sh` | 797 | `eaa8c5f1d88fa33f18d6f9dd9daf9ad0ad07027b1cfdd9a49323fd61658ea96a` |
| `/root/DMD/scripts/train_teacher_lr5e5_aug.sh` | `source_overlay/scripts/train_teacher_lr5e5_aug.sh` | 783 | `4d171b1ccfbeea0f4fa4095aee551bff5196a818aa86cc7558d656561457dcd4` |
| `/root/DMD/scripts/train_teacher_memorization64.sh` | `source_overlay/scripts/train_teacher_memorization64.sh` | 1152 | `b468b9743b49966a5a369e2ce737ab3a215987e0f98f715d7b54f52d93062fbc` |
| `/root/DMD/scripts/train_teacher_v1.sh` | `source_overlay/scripts/train_teacher_v1.sh` | 784 | `79f886a3f58d40a39114c0bc03c3809cfbffed4591a0a9883117da4b53bbdff3` |
| `/root/DMD/teacher/__init__.py` | `source_overlay/teacher/__init__.py` | 52 | `b5a93aff89ca44a04577ba300e7b1e3ec405f18678045b892d2d8e512d687227` |
| `/root/DMD/teacher/cfg.py` | `source_overlay/teacher/cfg.py` | 509 | `74ef535faba677a09f6fce384a4f02d7c3ad3ae4cb85362d4edfc7354c019f30` |
| `/root/DMD/teacher/config.py` | `source_overlay/teacher/config.py` | 3907 | `7cb5df078b38aeb8a6fdb78c232167fa5c2dee982a1718deb1926c93f576cf31` |
| `/root/DMD/teacher/data.py` | `source_overlay/teacher/data.py` | 3009 | `39140525868a7aa6e4cc48ea4e253dd4f68c545d41a317650ca36bc582e07ea1` |
| `/root/DMD/teacher/diffusion.py` | `source_overlay/teacher/diffusion.py` | 3941 | `022413888e08ca0c5a1c954f8eee38455cccf237c78c3250c2f5367ec32fead4` |
| `/root/DMD/teacher/evaluate.py` | `source_overlay/teacher/evaluate.py` | 8679 | `5ae76f9d3910500336450fe41ab10d2759370714a74863a4bcb4b4c0d45b80c7` |
| `/root/DMD/teacher/model.py` | `source_overlay/teacher/model.py` | 7323 | `6180f84fceccabf9164dda8a71cf5db29794ac700cf0e6b3e064131364adb9de` |
| `/root/DMD/teacher/sample.py` | `source_overlay/teacher/sample.py` | 2594 | `43f2e25af503a2fb5bffa52269db68422357328805f5ad2a3d782e0e51f8735f` |
| `/root/DMD/teacher/smoke.py` | `source_overlay/teacher/smoke.py` | 5668 | `c2ac1858d3ad7460e8fb8c805139c27e2ba7873ea779369c856d0636c9cdf281` |
| `/root/DMD/teacher/splits.py` | `source_overlay/teacher/splits.py` | 2393 | `f8c2b7a77545e784b71dbffa4d29ed45fd4f948dcff92ab46946784e73e3f5a1` |
| `/root/DMD/teacher/train.py` | `source_overlay/teacher/train.py` | 16107 | `9c76cd0946689d03b93c4abb9570e964c6b86fc16a96293ea5d01b11a98b23d8` |
| `/root/DMD/teacher/utils.py` | `source_overlay/teacher/utils.py` | 2526 | `4854bf81955d0b76c6022f7cba21b0425bd12da171878d6eab6fa2c6fb3948fc` |
| `/root/DMD/OVERLAY_MANIFEST.json` | `source_overlay/OVERLAY_MANIFEST.json` | 15646 | `04a741993d4e5c0d34ede8b7a2c5b96f87ba9d5a3b2b7cfac4abffbc1ee7707f` |
| `/root/DMD/README.md` | `source_overlay/README.md` | 19357 | `03e47b87bb042bb133300395a9d40857958daa9b7127114c64ebdc10f86b1af0` |
| `/root/DMD/requirements.txt` | `source_overlay/requirements.txt` | 50 | `e30940e3e890e6e5463f617475b97d7415e73f349fbf4070936aa284f0573eb2` |
| `/root/DMD/evaluate.py` | `source_overlay/evaluate.py` | 5578 | `f201320f3505ad478096313eedf657388a0c6aca4af456095509e13807228adb` |
| `/root/DMD/dataset.py` | `source_overlay/dataset.py` | 5923 | `47aea2b4ae6833198cbfa0801e0cc63fd5425dbe72bb77aec7f19a97ad90d063` |
| `/root/DMD/download_dataset.py` | `source_overlay/download_dataset.py` | 3553 | `e8e882dec2a83dcf5bdbfc1eb592df50948966a66d2a147ea87332652a8190a0` |
| `/root/DMD/src/__init__.py` | `source_overlay/src/__init__.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `/root/DMD/src/utils.py` | `source_overlay/src/utils.py` | 1027 | `f5a02c07d8ad190969468b5ef947f0037a6943deb98e7903cd193361f742988f` |
| `/root/DMD/data/pokemon-generation-one-22k/train_split.txt` | `source_overlay/data/pokemon-generation-one-22k/train_split.txt` | 703371 | `c4068bdec8c5994fa14887d3335e762dd6a4da1bc6394a6df7fae96d293138c0` |
| `/root/DMD/data/pokemon-generation-one-22k/val_split.txt` | `source_overlay/data/pokemon-generation-one-22k/val_split.txt` | 153787 | `6dfba46a82c1bb329287d6da0f87be1b3b71d26c217150bf681f37cce4264df0` |
| `/root/DMD/data/pokemon-generation-one-22k/category_to_id.json` | `source_overlay/data/pokemon-generation-one-22k/category_to_id.json` | 2656 | `7c27ec2a15fa4f70b5729319d596d3a80bc528cdeeb02f36b88877a6545b77e8` |
| `/root/DMD/checkpoints/one_nfe.ckpt` | `EXCLUDED` | 195649098 | `8173c059ca3dda5d63c2c3a9d463b100218867c5fc08b995f5c93162844f7266` |
| `/root/DMD/checkpoints/few_nfe.ckpt` | `EXCLUDED` | 195646254 | `d3c6d0cccc1b81d1709ed090abd68961197ef7417a7c6ae67724835bf7bfa596` |
| `/root/DMD/experiments/night_run/fresh_teacher/final.pt` | `EXCLUDED` | 978259214 | `b5c09c58954352f71291a2c4421c5f542ac6abb392acda4074c1f118ab9c3a27` |
| `/root/DMD/experiments/night_run/dmd_preserve_team/dmd_step_002500.ckpt` | `checkpoints/dmd_step_002500.ckpt` | 195653282 | `acc99889d73d905c1c31da57e1f2434b396a6640dd854a96cfdf1b9cc07d708c` |
| `/root/DMD/experiments/night_run/dmd_preserve_team/dmd_step_005000.ckpt` | `EXCLUDED` | 195653282 | `0043973b738684c2e703d7c0e303fafcba5815048179aa42ff3d141edd48ed44` |
| `/root/DMD/experiments/night_run/dmd_preserve_team/dmd_step_010000.ckpt` | `EXCLUDED` | 195653282 | `b0eb8d869dd63ab526f7f957aed10ec96743d199807c742ea7dedd84e377d328` |
| `/root/DMD/experiments/night_run/dmd_preserve_team/selected_one_nfe.ckpt` | `EXCLUDED` | 195653282 | `acc99889d73d905c1c31da57e1f2434b396a6640dd854a96cfdf1b9cc07d708c` |
| `/root/DMD/experiments/night_run/dmd_preserve_team/latest.pt` | `EXCLUDED` | 1369590460 | `a568884c06e0c0aa1bda6952c7b683fb6e06fd88d974b208416c474185cf51e4` |
| `/root/DMD/experiments/night_run/dmd_reproduce.py` | `scripts/dmd_reproduce.py` | 15509 | `27f44ddccb02cbd9ebca69bfbe94863908efebc8b2ac6348968edb4246248769` |
| `/root/DMD/experiments/night_run/dmd_preserve_team.py` | `scripts/dmd_preserve_team.py` | 15695 | `32cd2510ef82e2512ce57fd1370d52f6e27d4df6974e5c0249fc6f80f8a355d5` |
| `/root/DMD/experiments/night_run/dmd_preserve_team/config.json` | `configs/dmd_config.json` | 1410 | `c5e418711f104f28b8b936e495606b4ef5ce4991fb2345527c856287c44d3ce9` |
| `/root/DMD/experiments/night_run/dmd_preserve_team/REPORT.md` | `reports/REPORT.md` | 4435 | `403bf6bbe92298670693b34cd1c1285b08b7a9202235dbc1ad9ce6000461658c` |
| `/root/DMD/experiments/night_run/dmd_preserve_team/results.json` | `reports/results.json` | 1732 | `cf9824e6f614f80d98f79a23ca14b445d14269c81acc406dc8f323164bf60554` |
| `/root/DMD/experiments/night_run/dmd_preserve_team/selected_model.json` | `reports/selected_model.json` | 701 | `f994052bd8ebafc665f50e656a3c4cac71206555c45f8f43045c214ac35275e2` |
| `/root/DMD/experiments/night_run/dmd_preserve_team/fake_score_warmup.json` | `reports/fake_score_warmup.json` | 197 | `1ba353b5f9c5fb6f376be6eb86e69d77d54fef49d3c10f90fe64638241dbf048` |
| `/root/DMD/experiments/night_run/dmd_preserve_team/input_checkpoint_hashes.json` | `reports/input_checkpoint_hashes.json` | 165 | `6085e2ef7aec757dd9ba5c5a9dfe12e749e0a45026633129474ce42bd41b6634` |
| `/root/DMD/experiments/night_run/dmd_preserve_team/train.log` | `reports/train.log` | 2564836 | `c1ec92d0aa28e913b040669c86842881ab8c827db8c29301b50b02b624a37711` |
| `/root/DMD/experiments/night_run/dmd_reproduce_console.log` | `reports/dmd_reproduce_console.log` | 181976 | `e6278628e5ea9a8de44e863cb4203f7be18572a5835c58539be6803d7de3d112` |
| `/root/DMD/experiments/night_run/dmd_preserve_team/supplied_baseline/metrics.json` | `reports/metrics/supplied_baseline.json` | 383 | `14d28f0bab26196affdb0f257b546eaf1bca08a56612c89f673fd441871c6701` |
| `/root/DMD/experiments/night_run/dmd_preserve_team/dmd_002500/metrics.json` | `reports/metrics/dmd_002500.json` | 404 | `fe4de23b37eea58655dafd5d7f03ad2b10e2370bcce9db43d6f8730d5a49176d` |
| `/root/DMD/experiments/night_run/dmd_preserve_team/dmd_005000/metrics.json` | `reports/metrics/dmd_005000.json` | 404 | `776fe34a608fb9f08447841f1c722fe0a044e4446a60b4bda33102b5ee17e0ca` |
| `/root/DMD/experiments/night_run/dmd_preserve_team/dmd_010000/metrics.json` | `reports/metrics/dmd_010000.json` | 406 | `2647c02a0c4b3b1a2441c2c3fc38881d385711834874b45ef373b296f3b2537a` |
| `/root/DMD/experiments/night_run/audit_results.json` | `provenance/audit_results.json` | 1230 | `1e7442a02be3ac081f4c0f461d55362518744a766c7d73ce9ea8548a3750bc0d` |
| `/root/DMD/experiments/night_run/reference_manifest.json` | `provenance/reference_manifest.json` | 141955 | `a833ea989bab29e1751805f23d8b855c03add77ceb5d99f8eaf4250cb7b16e65` |
| `/root/AIC313-Project-FastGen/experiments/night_run/fresh_teacher/config.json` | `provenance/historical_lineage_evidence/fresh_teacher_config.json` | 1640 | `2a35da044d68976d00eea01a9c17ef8631cf837863f4a829fbe35a013021e440` |
| `/root/AIC313-Project-FastGen/experiments/night_run/fresh_teacher/metrics.json` | `provenance/historical_lineage_evidence/fresh_teacher_metrics.json` | 3464 | `a9a543f9a5250488cc4b19305d8ae8dc531926479fe18a6abdeb157f053f90d9` |
| `/root/AIC313-Project-FastGen/experiments/night_run/teacher_preflight/metrics.json` | `provenance/historical_lineage_evidence/teacher_preflight_metrics.json` | 3333 | `9e02b189d68a10125b20ce827ec6f19a4f17e34a252e57d046c6d6f9b43c6af8` |
| `/root/AIC313-Project-FastGen/experiments/night_run/teacher_preflight/config.json` | `provenance/historical_lineage_evidence/teacher_preflight_config.json` | 1638 | `90b70b513dff1af7d899431ac9568dffd8dac73d509fc3734fdfda0648ec32e6` |
| `/root/AIC313-Project-FastGen/experiments/night_run/USER_CHECKPOINT_ATTESTATION.json` | `provenance/historical_lineage_evidence/USER_CHECKPOINT_ATTESTATION.json` | 915 | `7ea35c0144f0960da6581595af800fa27e87fbcfc45ec53d273be76fc1ba17c8` |
| `/root/AIC313-Project-FastGen/experiments/night_run/COURSE_RULES_OVERRIDE.md` | `provenance/historical_lineage_evidence/COURSE_RULES_OVERRIDE.md` | 1920 | `584e915604c900cd1c6f26b309edbfbf1e174205fa0dcd67400860aaa04c4665` |
| `/root/AIC313-Project-FastGen/experiments/night_run/LINEAGE_REAUDIT.md` | `provenance/historical_lineage_evidence/LINEAGE_REAUDIT.md` | 4679 | `2cf85840830a446d6dec8a983835eaa2214860568abe3e6b08cf1591007c892d` |
| `/root/AIC313-Project-FastGen/experiments/night_run/lineage_reaudit.json` | `provenance/historical_lineage_evidence/lineage_reaudit.json` | 1946 | `7678a75ed8e3ea760b63adb8552df5c141555807fa5aa0f569f19edd73e4270d` |
| `/root/AIC313-Project-FastGen/experiments/night_run/overnight.py` | `provenance/historical_lineage_evidence/overnight.py` | 13151 | `3653154364bbff2afbf05f32f41e61b8934857b6fd4c6332ab65b6575a91a3cf` |
| `/root/AIC313-Project-FastGen/experiments/night_run/dmd_preserve_team/reproduction/MANIFEST.json` | `provenance/historical_lineage_evidence/previous_run_reproduction_manifest.json` | 3121 | `f98cf81467f4ec3ad2484f7f83d77667c3968b6641a53896b6af8c665cb7b6be` |
| `/root/AIC313-Project-FastGen/experiments/night_run/three_frozen_models_official_comparison/team_one_nfe/metrics.json` | `provenance/historical_lineage_evidence/official_baseline_metrics.json` | 772 | `6343eb1c250ab424ea01dbe19cd4c496e297d7a1aaf6b582e4e4afd3c26597de` |
| `/root/AIC313-Project-FastGen/experiments/night_run/three_frozen_models_official_comparison/dmd_one_nfe/metrics.json` | `provenance/historical_lineage_evidence/official_older_different_dmd_metrics.json` | 813 | `f9890e6be557bd7a2d7106e86882856e40269346e6261fed9d0c5b7aea7dd3af` |
| `/root/AIC313-Project-FastGen/experiments/night_run/fresh_teacher/config.json` | `configs/fresh_teacher_config.json` | 1640 | `2a35da044d68976d00eea01a9c17ef8631cf837863f4a829fbe35a013021e440` |
