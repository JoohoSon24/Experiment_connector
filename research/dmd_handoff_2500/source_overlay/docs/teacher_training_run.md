# Teacher training run: `teacher_ddpm_cfg_v1`

Completed: 100,000 optimizer steps, 22,842.83 seconds, final checkpoint written on 2026-09-27 around 21:58 UTC. The tmux training session ended normally. Quality remains insufficient; see [teacher_diagnosis.md](/root/workspace_distillation/docs/teacher_diagnosis.md) for controlled sample comparisons and FID reporting.

## Source settings and local choices

The [DDPM paper, Appendix B](https://arxiv.org/pdf/2006.11239) reports 1,000 diffusion steps, a linear beta schedule from `1e-4` to `0.02`, Adam at learning rate `2e-4`, EMA decay `0.9999`, two residual blocks per resolution and attention at 16×16. We use no learning-rate warmup so the reported rate applies from the first step; the paper does not specify a warmup. The [classifier-free guidance paper](https://arxiv.org/pdf/2207.12598) trains one epsilon-prediction network with null conditioning and studies unconditional probabilities `0.1`, `0.2`, and `0.5`; `0.1` and `0.2` performed similarly in its ImageNet 64×64 experiment. OpenAI's [guided-diffusion configuration](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/script_util.py) uses `(1,2,3,4)` channel multipliers for 64×64.

| Setting | This run | Basis |
| --- | --- | --- |
| Input / architecture | 64×64 RGB, 96 base channels, `(1,2,3,4)`, two blocks/level, 16×16 attention | Architecture pattern from DDPM/OpenAI; 96 channels is a local size reduction to 48,883,107 parameters |
| Objective / noise | Epsilon MSE, 1,000 linear steps, beta `1e-4→0.02` | DDPM and CFG papers |
| Optimizer | AdamW, `weight_decay=0`, learning rate `2e-4`, no warmup | Equivalent to Adam at zero weight decay; DDPM learning rate |
| EMA | `0.9999` | DDPM |
| Class dropout | `0.1`, null ID 151 | CFG paper |
| Batch | 8 microbatch × 2 accumulation = 16 images/update | Local 20GB GPU and iteration-speed choice; not a paper batch size |
| Total updates | 100,000 (~94 passes over 17,079 training images) | Local first-run budget; assess training curve and train-only samples before extension |
| Precision / clipping | bf16, gradient clip 1.0 | Stability and GPU-efficiency choices; not attributed to DDPM |
| Diagnostic samples | 50-step deterministic DDIM, conditional scale `1.0` | Development diagnostic, not a paper quality setting or selected optimal CFG scale |
| Seed | 42 | Reproducibility choice |

The CFG paper writes guided epsilon as `(1+w)·cond − w·uncond`. This repository's `guidance_scale` is `s=1+w`: `s=1` means class-conditioned without extra guidance, `s=0` means unconditional. The ImageNet guidance results should not be transferred as a best scale for Pokémon. Choose later CFG settings using only a held-out subset of `train_split.txt`; the fixed validation split cannot be used for hyperparameter selection under the FastGen README.

## Launch and records

The run uses [teacher_ddpm_cfg_v1.json](/root/workspace_distillation/configs/teacher/teacher_ddpm_cfg_v1.json) and [train_teacher_v1.sh](/root/workspace_distillation/scripts/train_teacher_v1.sh). It was launched inside tmux as session `fastgen_teacher_v1` at 2026-09-27 15:37:18 UTC. The script appends all stdout/stderr to `experiments/teacher_ddpm_cfg_v1/console.log` and writes an exit code when the process ends. A later launch automatically resumes from `latest.pt` if present.

- `experiments/teacher_ddpm_cfg_v1/config.json`: exact launch config.
- `experiments/teacher_ddpm_cfg_v1/source_snapshot/teacher/`: teacher source at launch.
- `experiments/teacher_ddpm_cfg_v1/train.log`: JSONL for **every optimizer step**, including loss, learning rate, gradient norm, images seen, elapsed time, and peak allocated VRAM.
- `experiments/teacher_ddpm_cfg_v1/metrics.json`: latest machine-readable summary.
- `experiments/teacher_ddpm_cfg_v1/latest.pt`: full checkpoint every 500 steps; `final.pt` at completion.
- `experiments/teacher_ddpm_cfg_v1/samples_step_*.png`: diagnostic grid every 1,000 steps.

The train split alone feeds the optimizer. A finite initial loss and falling trend over time indicate that the run is operating, but do not establish generation quality. A checkpoint, diagnostic samples and train-only evaluation are required before assessing the teacher.

The two-step full-size probe finished with losses `0.999755` and `0.990052`, learning rate `2e-4`, finite gradient norms, and peak allocated VRAM `2,225,084,928` bytes. The tmux run passed 1,000 steps and created loadable 500/1,000-step checkpoints. The 1,000-step diagnostic grid was generated at 50 NFE with class-conditioned scale `1.0`, but visually remains colored noise. A diagnostic sample from the raw, non-EMA 1,000-step weights was also noisy. A zero-image oracle epsilon predictor returned exactly zero through the same 50-step sampler, which checks the DDIM update arithmetic. The measured peak allocated VRAM after the automatic sample was `2,227,817,472` bytes. Early training loss and a completed sample call establish execution, not image quality.
