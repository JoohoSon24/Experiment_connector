## Current implementation update

The inspection below describes the original repository. After the user authorized distillation, ModelOneNFE and ModelFewNFE bodies in model.py were implemented for fixed-CFG progressive students. The base Model class, official dataset/evaluator/manifests and parameter-count utilities remain unchanged. See [distillation baseline](distillation_baseline.md) for execution order, checkpoints, exact 1/4-NFE smoke checks and official evaluation.

# FastGen repository analysis (Phase 0)

Upstream: [KAIST-Visual-AI-Group/AIC313-Project-FastGen](https://github.com/KAIST-Visual-AI-Group/AIC313-Project-FastGen), commit `e3f5120`. This repository root also contains `CODEX.md` and `PROJECT_SPEC.md`, which impose the stricter project rules described below.

## 1. Current repository structure

| Path | Purpose | Editing policy |
| --- | --- | --- |
| `dataset.py` | `PokemonDataset`, `PokemonDataModule`, fixed split and normalization | Upstream says do not modify |
| `download_dataset.py` | Kaggle download, white background compositing for alpha images | Upstream says do not modify |
| `data/pokemon-generation-one-22k/{train_split.txt,val_split.txt,category_to_id.json}` | Immutable manifests and class IDs | Do not modify |
| `model.py` | Base `Model`, unfinished `ModelOneNFE` and `ModelFewNFE` | Preserve base class; subclasses are for a later phase |
| `evaluate.py` | Official student sampling and clean-FID evaluation | Do not modify |
| `src/utils.py` | Parameter accounting | Do not modify |
| `teacher/`, `configs/teacher/` | Phase 1 teacher implementation | New local code |

No pretrained model, U-Net, teacher trainer, sampling trainer, or checkpoints were supplied. The original repository had no training script. `README.md` is the challenge contract.

## 2. Existing training pipeline

`PokemonDataModule(..., return_category=True).train_dataloader()` constructs a shuffled, `drop_last=True` loader from `train_split.txt`. `PokemonDataset` reads image paths relative to `data_root`, downloads the Kaggle archive automatically if `PokemonData/` is absent, converts to RGB, resizes to **64×64**, then `ToTensor` and `Normalize((0.5,)*3,(0.5,)*3)` produce **[-1, 1]** floats. `download_dataset.py` first composites transparency onto white, including palette transparency. The fixed mapping has **151 classes**, integer IDs **0–150**. The manifest has **17,079 training** and **3,020 validation** paths, with no overlap. Labels are returned as `torch.long` scalar tensors when `return_category=True`.

There is no upstream optimizer or training loop. The new `teacher.data.TrainImages` adapts `PokemonDataset` using only `train_split.txt` and the immutable mapping. It never constructs a validation loader. Only the official evaluation path reads validation images.

## 3. Existing sampling pipeline

`evaluate.py` calls `Model.load_checkpoint(path, evaluate_mode, device)`, then `model.sample((B,3,64,64), device=device, category=integer_ids)`. It writes 20 PNG samples per category (3,020 total), remaps returned `[-1,1]` tensors to RGB `[0,1]`, and compares with validation PNGs using `cleanfid.fid.compute_fid`. The reference images are exported from `val_dataloader()`. There is no separate teacher sampling pipeline upstream; Phase 1 adds `teacher.sample` using deterministic DDIM.

## 4. FastGen submission interface

`ModelOneNFE` and `ModelFewNFE` are stubs inside `model.py`. Their constructors accept `device` and `**kwargs`; `sample(shape, *, device="cuda", category=None, **kwargs)` must return the requested image tensor. `Model.load_checkpoint` selects a class based on `one_nfe` or `few_nfe`, loads either a raw state dict or a dict under `state_dict`, calls `.to(device).eval()`, and the evaluator counts parameters before sampling. Expected final paths are `checkpoints/one_nfe.ckpt` and `checkpoints/few_nfe.ckpt`.

The upstream evaluator caps each submitted model at **100M parameters**, counting frozen parameters but excluding buffers (`src.utils.count_parameters(module, trainable_only=False)`). The project instructions are stricter: **<60M** inference parameters, **exactly 1** backbone evaluation for `ModelOneNFE`, **≤4** for `ModelFewNFE`, and **<20GB** allocated training VRAM. A conditional plus unconditional CFG pair consumes **2 NFE** per denoising step. Any trained auxiliary network used during inference also counts. Phase 1 teacher DDIM sampling uses many NFEs and is **not** a submission implementation.

## 5. Reusable code

- `PokemonDataset` image loading, RGB/resize/normalization, fixed train manifest, and `category_to_id.json`.
- `download_dataset.py` Kaggle archive materialization and transparency compositing.
- `src.utils.count_parameters` for authoritative submitted-model parameter accounting.
- `evaluate.py` for a later, unchanged official student evaluation.
- `Model` base checkpoint loader and submission method signature, for later student integration.

## 6. Code needed beyond upstream

Phase 1 adds a class-conditioned pixel U-Net, learned null class, epsilon diffusion loss, linear noise schedule, shared CFG formula, deterministic DDIM teacher sampler, EMA, mixed precision training, resumable checkpoints, config files, seed management, and smoke checks. Later phases require a student trainer, one/few-NFE wrappers and checkpoints, and teacher FID sweep outside `evaluate.py`; those are deliberately absent now.

## 7. Implementation risks

- `PokemonDataset` automatically downloads images when missing. Kaggle access, local storage, and a successful white-compositing pass are prerequisites for a real training batch.
- The class mapping must be kept identical at inference. The teacher checkpoint records its class names and resume rejects mismatches.
- The upstream evaluator cannot evaluate a teacher checkpoint, because it instantiates only unfinished student classes. Teacher FID requires a separate evaluation utility in a later stage.
- The project spec says <60M parameters even though the upstream evaluator permits 100M. Apply the stricter project limit to eventual submission models.
- CFG costs two teacher forwards per step except scales 0 and 1, which can use a single branch. Teacher sampling NFE must be reported separately from future student NFE.
- The smoke script has a synthetic fallback if images are absent; that fallback alone cannot prove data conventions or VRAM. In this run, the real training batch and one full-size training step were measured. Class fidelity and FID still require a properly trained teacher.

## 8. README rules and submission checklist

- **Environment setup:** The README recommends Python 3.10 in conda and PyTorch/torchvision/torchaudio 2.6.0 from the CUDA 12.4 index, followed by the supplied `requirements.txt`. Our local probe used Python 3.12 and PyTorch/torchvision 2.6.0+cu124; Python 3.10 in the TA environment has not been tested. The teacher config uses standard-library JSON and adds no package to the provided dependencies.
- **What to implement:** Final work must implement both subclasses and their `sample()` methods in `model.py`, with exactly 1 and at most 4 NFE. It must provide independent compatible checkpoints at `checkpoints/one_nfe.ckpt` and `checkpoints/few_nfe.ckpt`. This Phase 1 teacher is an intermediate model and intentionally does not satisfy that submission checklist yet.
- **Important rules:** Do not change the base `Model`, provided dataset/downloader/evaluator/utilities, or fixed manifests; do not use pretrained generation models; do not train, distill, or select hyperparameters on validation images; keep peak training allocation under 20 GB; do not require extra libraries in the TA environment. Open-source code and papers require citation in the final write-up. This implementation uses the provided dependencies and does not modify protected files.
- **Evaluation:** `evaluate.py` loads each submitted class through `Model.load_checkpoint`, checks the 100M upstream parameter cap, generates 20 samples for each of 151 classes (3,020 total), and computes clean-FID against all 3,020 validation references. The project-local <60M inference parameter rule remains stricter. README TA reference FIDs are 39.15 for one NFE and 30.90 for few NFE; these are reference scores, not results for this teacher.
- **Self-evaluation and final delivery:** Test end-to-end training, both checkpoint loads, exact NFE budgets, sample generation, and official evaluation. Include both checkpoints, all required source, and a final write-up of at most two A4 pages excluding references. The write-up needs method and training details, loss curves/time, around eight early-training samples, and citations. The README lists missing required items or citation violations as grading failures.

For future teacher CFG selection, use a held-out subset **drawn only from `train_split.txt`**. The fixed validation split is reserved for final FID reporting and must not select guidance scales, training steps, or other hyperparameters.
