# FastGen DMD-inspired post-distillation fine-tuning

## 1. Overview

This run fine-tunes the supplied team 1-NFE generator with a normalized diffusion-score-derived gradient and weak online 4-NFE endpoint MSE. The selected output is the generator's 2500-update EMA. Its architecture and one-forward sampler are preserved. Development FID improves from 65.38627014891784 to 55.72514913187206. This is an observed single-run result, not a causal attribution to any individual component.

## 2. Original DMD in brief

Original DMD initializes G and a fake denoiser from a pretrained base denoiser, freezes the base for real scores, and trains fake scores on generated images. Its generator initialization is `G(z)=mu_base(z,T-1)`; G has no varying time-conditioning. Noised generated images receive a real/fake score-difference update. [Yin et al., CVPR 2024, sections 3.1–3.2](https://arxiv.org/html/2311.18828v4#S3)

## 3. Canonical flow

```text
pretrained base diffusion
    +-- G initialization
    +-- frozen real score
    +-- fake-score initialization
    +-- deterministic multi-step offline regression pairs
                     |
                     v
                 one-step G
```

The main regression uses LPIPS and default weight 0.25; Appendix I also evaluates L2. Real-score CFG is supported while fake-score formulation remains unchanged. [Original DMD, sections 3.3–3.4 and Appendix I](https://arxiv.org/html/2311.18828v4#S3.SS3)

## 4. Why this is not an exact reproduction

| Component | Original DMD | This experiment |
|---|---|---|
| G initialization | Base diffusion | Already-distilled team 1-NFE |
| Real score | Same base | Independent fresh diffusion |
| Fake init | Base diffusion | Fresh primary EMA deepcopy |
| Regression | Base multi-step offline pairs | Team 4-NFE online endpoints |
| Main distance/weight | LPIPS / 0.25 | Pixel MSE / 0.1 |
| Purpose | One-step distillation | Post-distillation refinement |
| Inference | One-step G | 1-NFE G |

Combining paper equations (4), (5), (7), (8) literally in VP notation leaves `a_t * (x0_fake-x0_real)/normalizer`. Our code omits that extra `a_t`; its weighting is an adaptation. This is an algebraic comparison, not a claim about inaccessible original training code. [Paper equations](https://arxiv.org/html/2311.18828v4#S3.SS2)

## 5. Available model lineage

```text
original team diffusion lineage [documented; full parent hash chain unavailable]
             |
      guided/progressive distillation
             +----------------------+
             |                      |
       existing team 1-NFE     existing team 4-NFE
             |                      |
       trainable G init        frozen endpoint target
             |                      |
             +----------+-----------+
                        v
                 refined 1-NFE G
                        ^
                        | normalized real/fake guidance
independent fresh diffusion
    +-- primary EMA --> frozen real-score model
    +-- primary EMA deepcopy --> trainable fake-score model
```

**The fresh diffusion teacher is NOT the parent teacher of the provided team 1-NFE or team 4-NFE checkpoints.** Team source metadata identifies `low_nfe_budget_100k/n1/final.pt` and `fixed_cfg2_progressive/n4/final.pt`. Team documents connect these to fixed-CFG progressive distillation. Complete original ancestor checkpoint hashes are not available in this package.

Fresh diffusion has a separate root: seed-42 random initialization, two `teacher_preflight` updates without resume, then `fresh_teacher` resumes that state through total step 100000. The train manifest contains 16324 paths. Its newline-joined digest matches local development training exactly. Historical preflight metrics, launcher and source evidence are included under `provenance/historical_lineage_evidence/`.

## 6. Our variant

The executable source of truth is `scripts/dmd_reproduce.py`, particularly lines 60–110. G and the endpoint model are class-conditional velocity predictors; real and fake score models predict epsilon. Every backbone contains 48,883,107 parameters. No adversarial loss or learned perceptual loss is used. DMD updates use synthetic images and labels; development references enter evaluation, not the loss.

## 7. Generator initialization

G loads the supplied `one_nfe.ckpt` state exactly. The optimizer is newly initialized. This is not a continuation of its earlier AdamW state. The U-Net uses image size 64, RGB, 151 classes, base channels 96, channel multipliers `(1,2,3,4)`, two residual blocks per level, attention at resolution 16, dropout 0. The null-class embedding remains counted.

With `a_t=sqrt(alpha_bar[t])`, `s_t=sqrt(1-alpha_bar[t])`:

```text
G(z,y) = clamp(a_999*z - s_999*v_theta(z,999,y), -1,1)
```

The linear VP schedule has 1000 steps, beta 0.0001→0.02. Time-conditioning architecture is retained and fixed to 999 for G. No architecture change, new decoder or inference CFG is introduced.

## 8. Real-score model

`fresh_teacher/final.pt['ema']` is loaded into a model constructed from that checkpoint's config; `.eval().requires_grad_(False)` freezes it. This is the primary EMA, decay **0.9999**, not the separately stored `ema_tracks['0.999']`. During DMD it uses CFG scale 2, irrespective of the fresh checkpoint's diagnostic `guidance_scale=1` setting.

```text
eps_real = eps_uncond + 2*(eps_cond-eps_uncond)
```

The unconditional class id is 151. CFG arithmetic is defined in `teacher/cfg.py`, with low-precision predictions cast to float before the subtraction.

## 9. Fake-score model

`fake=deepcopy(real).train().requires_grad_(True)`. This network estimates the current synthetic distribution, not an additional strong teacher. Each fit draws independent integer timesteps uniformly in `[50,950]` and Gaussian noise:

```text
x_noisy = a_t*stopgrad(image) + s_t*noise
L_fake = mean((fake(x_noisy,t,y).float()-noise)^2)
```

There is no class dropout, fake CFG, fake EMA, or explicit additional timestep weighting in this loss. The whole fake model is optimized immediately for each batch of 16.

## 10. Fake-score warmup

Before any G update, 200 fake updates fit images from the initial G, batch 16. This differs from learning-rate warmup: both learning rates are constant with zero LR warmup. Warmup peak allocated memory was 3,691,061,760 bytes. During main training, fake is updated once per G microbatch, hence four times per G optimizer step. Final saved optimizer counters independently show 10000 G and 40200 fake steps.

## 11. Distribution-matching gradient

For each microbatch, code first generates detached images, fits fake, and then draws a new timestep/noise for score guidance. Both score models see the **same** noised tensor:

```text
x_det = stopgrad(G(z,y))
t ~ randint(50,951)
x_t = a_t*x_det + s_t*epsilon
x0_real = (x_t-s_t*eps_real)/a_t
x0_fake = (x_t-s_t*eps_fake)/a_t
n = clamp_min(mean_CHW(abs(x_det-x0_real)), 0.001)
g = (x0_fake-x0_real)/n

x = G(z,y)                         # new forward with autograd
target = stopgrad(stopgrad(x)-g)
L_DMD = 0.5*mean((x-target)^2)
```

The score calculation, normalizer and g run under `no_grad`. Neither real nor fake receives G-loss gradients. Reconstructed score x0 values are not clamped. G's image clamp is in the differentiation path, so saturation affects its gradients. Tensor means average batch, channels and pixels. No extra `a_t` or `s_t` factor is multiplied into g. This surrogate delivers a prescribed gradient; its numerical loss is not a measured KL divergence.

## 12. Endpoint regularization

For G update `step`, zero-based microbatch `micro`, code sets `seed=1000000+step*4+micro`. The G initial noise and `teacher.sample(...,seed=seed)` use the same CUDA generator seed, shape and labels. The frozen endpoint teacher performs deterministic v/DDIM transitions at `[999,741,483,226]` and a clean endpoint at -1. Intermediate x0 estimates are not clipped; final output is clamped.

```text
desired = stopgrad(G_4NFE(z,y))
L_endpoint = mean((G(z,y)-desired)^2)
L_micro = (L_DMD + 0.1*L_endpoint)/4
```

Endpoints are created online on the same noise stream as DMD. There is no offline pair dataset. The 4-NFE network supplies no DMD score.

## 13. Why a team 4-NFE endpoint was used

The contemporary source/report documents preserving existing team weights and transferring a matched-noise/class 4-NFE endpoint, with weak regression intended to stabilize the change. See archived `REPORT.md` and `dmd_preserve_team.py`. No inspected contemporary evidence explicitly says latent correspondence motivated the original choice.

## 14. Verified facts and retrospective interpretation

**Verified implementation facts:** distinct score/endpoint models, existing 1-NFE initialization, matched seeds/classes, four teacher forwards, and 0.1 pixel MSE. Team documents report a shared progressive lineage and better 4-NFE FID; these documents are weaker than a complete ancestor hash chain.

**Retrospective interpretation:** independent fresh diffusion need not map a given z to the same image semantics as team G. Per-z regression to its many-step outputs could therefore compete with the existing mapping. A same-lineage endpoint may be a useful preservation heuristic. Shared lineage does not prove identical latent semantics. Four target forwards also cost fewer network calls than a many-step full diffusion target. These explanations are plausible, not established historical intent or ablation findings.

## 15. Optimization and hyperparameters

| Setting | G | Fake |
|---|---|---|
| AdamW LR | 2e-5 | 2e-4 |
| Betas / eps | (0.9,0.999) / 1e-8 | same |
| Weight decay | 0 | 0 |
| Batch / accumulation | 16 / 4 (effective 64) | 16 / 1 |
| Gradient norm clip | 1 | 1 |
| LR schedule / LR warmup | constant / none | constant / none |
| EMA | 0.999 | none |
| Precision | bf16 forward, float loss arithmetic | same |

Global torch and CUDA seeds are 42. Labels, score times and noise use the global RNG; G latent noise uses the explicit per-step seed above. EMA updates once per G optimizer step by `target.lerp_(student,0.001)`. Training uses current student, evaluation/export uses EMA. No strict bitwise reproducibility across devices/software versions is promised.

## 16. Training schedule

The archived reproduction replaces the historical absolute manifest lookup, removes the two conditional early stops and removes the six-hour timeout. It runs 200 fake warmup updates, then 10000 G updates unconditionally, evaluating baseline and 2500/5000/10000. Those three EMA exports are retained; selection minimizes development FID including baseline.

`latest.pt` is overwritten atomically within the original run at steps 1, 2, each 500, and milestones. It currently holds 10000, including current student, EMA, fake model, both optimizers and RNG states. It is excluded from the handoff. A 2500 optimizer resume snapshot is not available here. The original training script has no resume CLI.

## 17. Evaluation

Protected `evaluate.generate_samples` and `compute_fid` use clean-fid defaults. Reference: 755 PNG derived from the training-only development split. Generated: 3020 PNG, 20/class, batches of 16, seed `42+batch index`, bf16. There are 189 sample calls/backbone forwards. New build checks verify the reference hashes and in-memory byte regeneration without performing FID.

Split disjointness is path-based. It does not establish image-content deduplication or complete holdout exclusion in all team ancestors. Fresh checkpoint train-path digest matches the 16324-entry manifest. Team ancestry exclusion remains documented/user-attested rather than independently verified.

Official rules forbid validation for training, distillation and hyperparameter selection and designate it as FID reference. They do not state that merely viewing a validation score is prohibited. Checkpoint selection here uses development results; official metrics are separate reporting evidence. See `provenance/validation_policy.md`.

## 18. Results

| Step | Development FID | Recorded elapsed training seconds |
|---|---:|---:|
| 0 | 65.38627014891784 | 0 |
| **2500** | **55.72514913187206** | 3643.402763916005 |
| 5000 | 58.62200371627705 | 7264.994736391003 |
| 10000 | 60.44414080173647 | 14320.559109403985 |

All models have 48,883,107 parameters and NFE 1. Recorded main peak allocated VRAM is 4,522,762,752 bytes, below the script's 20,000,000,000-byte bound. Elapsed timing starts before warmup; later milestones include intervening evaluation/checkpoint/report work. It is not pure optimizer compute time. `train.log` is continuous through 10000. Its `loss` sums all four scaled microbatch losses; `dmd_loss`, `dmd_regression`, `fake_score_loss` describe the last microbatch only. Logged gradient norm is measured before clipping.

Official baseline 58.15948330602913 is supported by identical-hash historical metrics. User-recovered current 2500/5000 values 38.605112/41.246506 lack located numerical artifacts; 10000 is not recorded. An older, different DMD checkpoint has official 38.43229103196154 and must not be substituted.

## 19. Interpretation

An already-distilled G can receive this implemented refinement gradient; its initial state need not be a raw base denoiser for the code to run. Refining an existing generator near the target distribution is a plausible explanation of the observed improvement. It is retrospective and not a paper guarantee for this hybrid setting. Later scores worsening in one run do not establish a general overfitting mechanism.

## 20. Limitations

There is one training seed, no component ablation, imperfect team ancestry provenance, no fully independent unseen-holdout claim, and no verified current DMD official numerical artifacts. G initialization, independent score lineage, VP weighting, timestep interval `[50,950]`, online endpoint MSE and update schedule differ from the canonical setup. Historical report text contains stale stopping-policy and broad validation statements, preserved and annotated rather than edited. The recorded config is emitted by hardcoded Python; editing only JSON does not alter this trainer.

## 21. Possible future ablations

These are proposals, not executed results: (A) current hybrid; (B) team G with recovered original-team diffusion real score and its multi-step targets; (C) team G with fresh real score and fresh multi-step targets. Hold budgets, seeds and evaluation policy fixed to test endpoint and lineage effects; also consider repeated training seeds and a regression-weight ablation. Do not reinterpret these as current experimental evidence.
