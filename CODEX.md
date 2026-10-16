# FastGen Project — Codex Implementation Instructions

## 0. Role

이 repository에서 너의 역할은 다음 연구 아이디어를 **재현 가능하고 실험 가능한 코드**로 구현하는 것이다.

> Class-conditional diffusion teacher를 먼저 충분히 학습한 뒤, teacher의 classifier-free guidance(CFG) behavior까지 포함하여 one/few-step student에 distill한다.

최종 목표는 FastGen benchmark에서:

- `ModelOneNFE`: 정확히 1 NFE
- `ModelFewNFE`: 최대 4 NFE

로 높은 generation quality를 얻는 것이다.

현재 가장 중요한 기준 논문은:

1. **On Distillation of Guided Diffusion Models — Meng et al., CVPR 2023**
2. **Consistency Models — Song et al., ICML 2023**
3. 이후 필요하면 **Improved Techniques for Training Consistency Models**

이다.

처음부터 새로운 방법을 임의로 설계하지 말고, **baseline reproduction → 검증 → modification** 순서를 반드시 지킨다.

---

# 1. Hard Constraints

아래 조건은 어떠한 구현에서도 위반하면 안 된다.

## Dataset

- FastGen에서 제공된 training split만 사용한다.
- validation split은 학습에 사용하지 않는다.
- 외부 이미지 dataset을 사용하지 않는다.
- pretrained image/model feature를 training에 사용하지 않는다.
- class 수와 dataset 구조는 제공 code에서 직접 확인한다.

## Model

최종 sampling model 각각:

- parameter count < 60M
- inference 시 사용하는 모든 learned network를 parameter count에 포함
- auxiliary teacher는 inference에서 사용하지 않는 경우 제한에서 제외

## Training

- single GPU peak allocated VRAM < 20GB
- 반드시 실제 값 측정:
  `torch.cuda.max_memory_allocated()`
- target environment: A100 20GB 이하
- AMP/bfloat16 사용 가능
- gradient accumulation 사용 가능

## Sampling

### ModelOneNFE
- backbone forward 정확히 1회
- CFG를 위해 conditional/unconditional forward를 각각 호출하면 안 됨

### ModelFewNFE
- 총 backbone evaluation ≤ 4
- 기본 목표는 4 NFE
- 가능하면 2/3/4 NFE를 모두 평가

FastGen에서 backbone의 부분적인 evaluation도 1 NFE로 계산된다고 가정한다.

---

# 2. Development Principles

## 반드시 단계적으로 구현한다

한 번에 teacher + CFG distillation + consistency distillation을 모두 구현하지 않는다.

각 stage를 독립적으로 검증한다.

순서:

1. repository 및 evaluator 분석
2. diffusion teacher baseline
3. CFG teacher
4. guided-distillation reproduction
5. consistency-distillation reproduction
6. 두 방법 통합
7. FastGen Model wrapper 통합
8. ablation 및 optimization

각 stage가 동작하지 않으면 다음 stage로 진행하지 않는다.

---

# 3. Repository Inspection

코드 수정 전에 다음을 확인하여 문서화한다.

- dataset interface
- image normalization range
- class-label format
- image resolution
- provided `Model` interface
- `ModelOneNFE`
- `ModelFewNFE`
- `.sample()` signature
- FID evaluation interface
- checkpoint loading mechanism
- parameter-count utility
- 수정 금지 파일

기존 architecture를 가능한 한 재사용한다.

수정 금지 파일은 절대 변경하지 않는다.

repository 구조를 이해하기 전에 대규모 refactor를 하지 않는다.

---

# 4. Phase A — Diffusion Teacher

첫 번째 목표는 **신뢰할 수 있는 class-conditional diffusion teacher**이다.

## Teacher requirements

입력:

- noisy image `x_t`
- noise/time level `t`
- class label `c`

출력 parameterization은 repository와 구현 난이도를 고려해 하나를 선택하되 프로젝트 전체에서 일관되게 유지한다.

가능한 후보:

- epsilon prediction
- x0 prediction
- v prediction

선택한 parameterization을 configuration과 문서에 명확히 기록한다.

---

# 5. Classifier-Free Guidance Teacher

training 시 class label dropout을 지원한다.

예:

```text
p_uncond ≈ 0.1 ~ 0.2
```

unconditional token 또는 null class embedding을 둔다.

teacher inference prediction:

```text
pred_uncond = teacher(x_t, t, null_class)
pred_cond   = teacher(x_t, t, class)

pred_cfg =
    pred_uncond
    + guidance_scale * (pred_cond - pred_uncond)
```

프로젝트 코드에서는 CFG 정의를 **한 곳에만 구현**한다.

예:

```text
apply_cfg(uncond, cond, scale)
```

여러 training script에서 식을 중복 구현하지 않는다.

---

# 6. Teacher Validation

distillation을 시작하기 전에 teacher를 검증한다.

필수 평가:

- unguided generation
- CFG generation
- multiple CFG scale

최소 guidance sweep:

```text
0
0.5
1
2
3
4
6
```

기록:

- FID
- sampling NFE
- random seed
- checkpoint
- guidance scale

teacher quality가 충분하지 않으면 student distillation을 진행하지 않는다.

Student는 teacher 이상의 정보를 배울 수 없다는 것을 기본 원칙으로 한다.

---

# 7. Phase B — Guided Diffusion Distillation Reproduction

먼저 Meng et al.의 핵심 아이디어를 독립적으로 재현한다.

목적:

```text
teacher unconditional prediction
+
teacher conditional prediction
+
guidance scale
            ↓
       guided target
            ↓
 single student forward
```

Student는 inference에서 conditional/unconditional 두 번 호출하지 않아야 한다.

---

# 8. Guidance-Conditioned Student

가능하면 student interface를 다음과 같이 설계한다.

```text
student(x_t, t, class_label, guidance_scale)
```

guidance scale `s`는 embedding하여 time/class embedding과 결합한다.

초기 구현은 간단하게 유지한다.

예:

```text
s
↓
Fourier / sinusoidal / small MLP embedding
↓
conditioning embedding
```

architecture 변경보다 reproduction 정확성을 우선한다.

---

# 9. Fixed-Guidance Baseline

variable guidance를 구현하기 전에 반드시 fixed-guidance baseline을 만든다.

Teacher:

```text
target = CFG(teacher, s_fixed)
```

Student:

```text
student(x_t, t, class)
```

여기서 `s_fixed`는 teacher sweep에서 성능이 좋았던 값으로 선택한다.

이 실험은 variable-guidance conditioning의 필요성을 판단하는 baseline이다.

---

# 10. Variable-Guidance Distillation

그 다음:

```text
s ~ p(s)
```

를 sampling하고 student가 `s`에 condition되도록 한다.

초기 distribution:

```text
Uniform(0, s_max)
```

또는 discrete:

```text
{0, 1, 2, 3, 4}
```

둘 중 단순한 것부터 사용한다.

초기 단계에서 복잡한 adaptive distribution을 만들지 않는다.

---

# 11. Phase C — Consistency Distillation

Guided distillation이 검증된 이후 consistency distillation을 구현한다.

핵심 목표:

teacher의 probability-flow / deterministic denoising trajectory 상에서 인접한 noise level이 동일한 clean sample에 대응하도록 student를 학습한다.

개념적으로:

```text
x_t
 |
 | teacher trajectory
 v
x_s

student(x_t, t, ...)
≈
target_student(x_s, s, ...)
```

원 논문의:

- timestep/noise schedule
- boundary condition
- target-network/EMA mechanism
- consistency loss

를 가능한 한 충실히 재현한다.

처음부터 임의 변경하지 않는다.

---

# 12. Combined Method

최종 핵심 모델은:

```text
f_theta(x_t, t, class, guidance_scale)
```

이다.

Teacher guided prediction:

```text
T_cfg(x_t,t,c,s)
 =
T_uncond(x_t,t)
+
s * (T_cond(x_t,t,c) - T_uncond(x_t,t))
```

이 guided teacher가 정의하는 trajectory를 student가 consistency distillation으로 학습한다.

목표:

```text
multi-step CFG teacher
        ↓
guided trajectory
        ↓
consistency distillation
        ↓
CFG-aware consistency student
        ↓
1 / 2 / 4 NFE
```

Inference 시 teacher는 사용하지 않는다.

---

# 13. One-NFE Model

`ModelOneNFE`는 별도의 checkpoint 사용을 허용한다.

우선순위:

1. 1-step quality
2. FID
3. parameter budget
4. speed

반드시 한 번의 student backbone evaluation으로 image를 생성한다.

Sampling 과정 내부에서:

- teacher 호출 금지
- unconditional secondary forward 금지
- learned decoder/refiner 추가 금지

---

# 14. Few-NFE Model

`ModelFewNFE`는 최대 4번의 backbone evaluation을 사용한다.

기본 목표:

```text
4 NFE
```

추가 평가:

```text
2 NFE
3 NFE
4 NFE
```

가능하면 같은 checkpoint가 여러 NFE를 지원하도록 구현하되, 성능이 크게 떨어지면 별도 few-NFE checkpoint를 허용한다.

---

# 15. LoRA

LoRA는 메인 구현이 아니다.

기본 student training:

```text
full fine-tuning
```

LoRA는 다음 research ablation에서만 사용한다.

```text
Full FT vs LoRA
```

LoRA를 사용한다고 해서 inference backbone parameter가 줄어드는 것으로 간주하지 않는다.

Teacher weight initialization을 student에 복사한 뒤:

- full FT
- optional LoRA rank 8
- optional LoRA rank 16

정도로 비교 가능하다.

Main result 확보 전에는 LoRA 실험을 하지 않는다.

---

# 16. Architecture Policy

처음에는 architecture novelty를 최소화한다.

권장:

```text
small class-conditional U-Net
~20M–50M parameters
```

목표는 60M에 최대한 가까운 모델을 만드는 것이 아니라:

```text
stable training
+
teacher quality
+
distillation quality
+
multiple experiments
```

이다.

VRAM과 iteration speed를 중요하게 본다.

---

# 17. Configuration

모든 중요한 hyperparameter를 config로 이동한다.

최소:

```text
model_channels
channel_mult
num_res_blocks
attention_resolutions

prediction_type

batch_size
gradient_accumulation

learning_rate
weight_decay

num_train_steps

ema_decay

class_dropout

noise_schedule

guidance_distribution
guidance_min
guidance_max

distillation_type

num_consistency_steps

seed
```

magic number를 training code에 흩뿌리지 않는다.

---

# 18. Experiment Tracking

모든 실험은 재현 가능해야 한다.

각 run마다 기록:

```text
experiment name
git commit
config
seed
parameter count
peak VRAM
training steps
checkpoint
FID
NFE
CFG scale
notes
```

최소 directory:

```text
experiments/
    <experiment_name>/
        config.yaml
        metrics.json
        train.log
```

가능하면 자동으로 생성한다.

---

# 19. Checkpoint Policy

항상 다음을 구분한다.

```text
latest
best_fid
final
```

Teacher와 student checkpoint를 별도 directory에 저장한다.

예:

```text
checkpoints/
    teacher/
    guided_student/
    consistency_student/
    final/
```

최종 FastGen 파일:

```text
checkpoints/one_nfe.ckpt
checkpoints/few_nfe.ckpt
```

---

# 20. Evaluation Policy

모델 변경 후 subjective sample만 보고 판단하지 않는다.

최소 평가:

```text
FID
parameter count
NFE
peak VRAM
```

그리고 동일 seed/sample count로 baseline과 비교한다.

빠른 개발용 FID와 최종 official FID를 분리한다.

예:

```text
quick_eval
official_eval
```

---

# 21. Required Ablations

메인 방법이 동작한 뒤 다음 순서로 실험한다.

### A0
Diffusion teacher, no CFG

### A1
Diffusion teacher + inference CFG

### A2
Standard consistency distillation

### A3
Fixed-CFG consistency distillation

### A4
Guidance-conditioned consistency distillation

이 결과가 확보된 이후에만:

### A5
CFG sampling distribution 변경

### A6
guidance curriculum

### A7
Full FT vs LoRA

를 실행한다.

---

# 22. Guidance Curriculum — Optional Research Extension

기본 variable-guidance distillation이 동작한 경우에만 구현한다.

예:

```text
early:
s ~ U(0, 1)

middle:
s ~ U(0, 3)

late:
s ~ U(0, 5)
```

그러나 이것은 논문 reproduction이 아니라 프로젝트 extension임을 명확히 구분한다.

---

# 23. Coding Requirements

- type hint 적극 사용
- 핵심 tensor shape 주석
- training / sampling logic 분리
- teacher/student abstraction 분리
- CFG function 공통화
- noise schedule 공통화
- sampler 공통화

가능한 structure:

```text
src/
    models/
        unet.py
        diffusion.py
        consistency.py

    diffusion/
        schedule.py
        cfg.py
        sampler.py

    distillation/
        guided.py
        consistency.py

    training/
        teacher_trainer.py
        student_trainer.py

    evaluation/
        metrics.py

configs/
    teacher/
    distillation/
```

기존 repository 구조와 충돌한다면 기존 구조를 우선하고 필요한 최소 변경만 한다.

---

# 24. Tests

최소 unit/smoke test:

### Shape
```text
input → correct output shape
```

### Conditional
class condition 변경 시 forward 정상 작동

### Unconditional
null condition 정상 작동

### CFG
scale=0일 때 unconditional과 동일한지 확인

### NFE
`ModelOneNFE.sample()` backbone call == 1

`ModelFewNFE.sample()` backbone call <= 4

### Parameter count
< 60M

### VRAM
training smoke test < 20GB

### Determinism
동일 seed inference consistency 확인

---

# 25. Do Not Do

다음은 명시적인 이유 없이 하지 않는다.

- pretrained models 사용
- external dataset 사용
- validation split training
- evaluator 수정
- forbidden FastGen files 수정
- baseline 검증 전 architecture 대규모 변경
- baseline 없이 여러 아이디어 동시에 추가
- undocumented hyperparameter 변경
- teacher 검증 전에 student 장시간 학습
- LoRA를 기본 방법으로 채택
- CFG를 inference에서 두 번의 student forward로 구현

---

# 26. Decision Rule

새로운 아이디어를 구현하기 전에 항상 다음 질문을 한다.

1. 현재 baseline이 정상 동작하는가?
2. 변경하려는 요소가 하나인가?
3. 무엇과 비교할 것인가?
4. 어떤 metric으로 효과를 판단할 것인가?
5. FastGen NFE/parameter/VRAM 규칙을 위반하지 않는가?

하나라도 답할 수 없으면 먼저 baseline 또는 evaluation을 정리한다.

---

# 27. Current Priority

## User override: future learning-rate policy (2026-09-28)

New training experiments use a constant learning rate, with no LR decay and no
warmup (`learning_rate_schedule="constant"`, `warmup_steps=0`), unless the user
explicitly requests another policy. Preserve historical configs/checkpoints and
their scheduler behavior for reproducibility. This does not change the diffusion
noise schedule.

현재 구현 우선순위는 다음과 같다.

```text
P0 repository inspection
↓
P1 class-conditioned diffusion teacher
↓
P2 CFG teacher + guidance sweep
↓
P3 guided-distillation reproduction
↓
P4 consistency-distillation reproduction
↓
P5 CFG-aware consistency distillation
↓
P6 ModelOneNFE / ModelFewNFE integration
↓
P7 ablations
↓
P8 optional improvements
```

항상 이 순서를 기본 계획으로 사용한다.

## User scope expansion: sequential experiments and distillation

The user now authorizes: (2) DDIM CFG2 sampling-step/seed comparison, then (3) no-augmentation teacher ablation, then fixed-guidance distillation and 1-/few-NFE FID evaluation. Earlier Phase 0–1-only and no-distillation restrictions are superseded for these tasks. Preserve the base Model class and protected data/evaluation files. Implement the fixed-guidance baseline before variable guidance; use training-only holdout selection and constant LR.
