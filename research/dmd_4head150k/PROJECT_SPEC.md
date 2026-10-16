# FastGen — Guided Consistency Distillation Project Specification

## 1. Project Goal

FastGen challenge에서 제한된 sampling computation으로 높은 품질의 class-conditional Pokémon image를 생성한다.

주 연구 방향:

> **Strong CFG diffusion teacher의 생성 능력과 guidance behavior를 one/few-step consistency student에 distill한다.**

최종적으로 teacher 없이:

```text
1 NFE
or
≤ 4 NFE
```

generation을 수행한다.

---

# 2. Core Research Question

기본 질문:

> Multi-step class-conditional diffusion teacher의 CFG behavior까지 student가 직접 학습하면, inference-time CFG 없이 높은 품질의 1–4 step generation이 가능한가?

세부 질문:

1. unguided consistency distillation보다 guided teacher distillation이 좋은가?
2. 특정 CFG scale 하나를 distill하는 것이 좋은가?
3. 여러 CFG scale을 하나의 student가 학습하는 것이 좋은가?
4. 1-step과 4-step에서 최적 training strategy가 다른가?
5. full fine-tuning과 parameter-efficient adaptation의 차이가 있는가?

---

# 3. Main Method

전체 pipeline:

```text
Training Data
     │
     ▼
Class-Conditional Diffusion Teacher
     │
     ├── conditional prediction
     │
     └── unconditional prediction
                  │
                  ▼
         Classifier-Free Guidance
                  │
                  ▼
          Guided Trajectory
                  │
                  ▼
       Consistency Distillation
                  │
                  ▼
 Guidance-Conditioned Student
                  │
          ┌───────┴───────┐
          ▼               ▼
       1 NFE            ≤4 NFE
```

---

# 4. Teacher

Teacher는 class-conditional diffusion model이다.

Training 시 class label dropout을 사용하여 한 network에서:

```text
conditional prediction
unconditional prediction
```

을 모두 수행할 수 있도록 한다.

Inference CFG:

```text
T_cfg =
T_uncond
+
s * (T_cond - T_uncond)
```

Teacher는 training/distillation에서만 사용한다.

최종 submission inference에는 포함하지 않는다.

---

# 5. Student

최종 연구 모델은 기본적으로:

```text
f_theta(x_t, t, class, guidance_scale)
```

형태이다.

Student는 teacher의 conditional/unconditional network를 각각 호출하는 대신 **guided result 자체를 한 번의 forward로 학습**한다.

따라서 inference에서는:

```text
student(...)
```

한 번이 하나의 NFE이다.

---

# 6. Why Guidance Distillation

일반 CFG는 한 timestep마다:

```text
conditional forward
+
unconditional forward
```

가 필요하다.

1-NFE FastGen setting에서는 이것을 직접 사용할 수 없다.

따라서:

```text
CFG computation
```

자체를 inference에서 수행하는 대신,

```text
CFG behavior
```

를 student parameter에 흡수시키는 방향을 사용한다.

---

# 7. Why Consistency Distillation

Diffusion teacher는 좋은 generation quality를 얻기 위해 많은 denoising steps가 필요하다.

Consistency model은 teacher trajectory에서 서로 다른 noise level의 상태를 동일한 clean sample에 대응시키도록 학습하여:

```text
noise → image
```

를 매우 적은 number of function evaluations로 수행할 수 있다.

FastGen의:

```text
1 NFE
≤4 NFE
```

평가와 직접적으로 잘 맞는다.

---

# 8. Main References

## Core

### On Distillation of Guided Diffusion Models
Meng et al., CVPR 2023

프로젝트에서 가져오는 핵심:

- CFG teacher distillation
- conditional + unconditional combination을 single model에 흡수
- guidance strength conditioning
- few-step guided generation

### Consistency Models
Song et al., ICML 2023

프로젝트에서 가져오는 핵심:

- diffusion → consistency distillation
- one-step generation
- multistep sampling
- ImageNet 64×64 class-conditional experiments

## Secondary

### Improved Techniques for Training Consistency Models
Song & Dhariwal, ICLR 2024

필요할 경우 consistency training 안정화 및 성능 개선 참고.

### Progressive Distillation for Fast Sampling of Diffusion Models

few-step diffusion distillation의 기본 reference.

### Consistency Trajectory Models

추후 arbitrary trajectory jump 및 1–4 NFE optimization 참고.

### Shortcut Models

distillation 없는 multi-step-size model의 comparison/reference.

---

# 9. FastGen Constraints

프로젝트 전 기간 동안 유지해야 하는 hard constraints:

```text
No pretrained models

Training:
single GPU
peak allocated VRAM < 20GB

Sampling model:
< 60M parameters

ModelOneNFE:
exactly 1 NFE

ModelFewNFE:
≤ 4 NFE

Training data:
training split only
```

Teacher와 auxiliary model은 최종 sampling에서 호출되지 않는 한 sampling parameter budget에 포함시키지 않는다.

---

# 10. Compute Constraint

가용 자원:

```text
GPU:
A100 20GB

Experiment period:
approximately one month
```

따라서 연구 전략은:

```text
large architecture search
```

보다

```text
small/medium model
+
reliable baseline
+
controlled ablation
```

을 우선한다.

---

# 11. Architecture Direction

현재 기본 방향:

```text
class-conditional U-Net
~20M–50M parameters
64×64 image generation
```

정확한 크기는 repository baseline을 확인한 뒤 결정한다.

60M parameter limit을 최대한 채우는 것이 목표가 아니다.

Iteration speed와 실험 횟수를 중요하게 본다.

---

# 12. LoRA Decision

기본 방향:

```text
Full fine-tuning = main method
LoRA = optional ablation
```

이유:

- pretrained foundation model adaptation 문제가 아님
- student가 teacher와 다른 few-step mapping을 배워야 함
- inference parameter budget을 실질적으로 줄여주지 않음

따라서 main experiment가 확보되기 전에는 LoRA를 우선하지 않는다.

---

# 13. Experimental Baselines

반드시 아래 hierarchy를 유지한다.

## B0 — Diffusion

```text
class-conditional diffusion
no CFG
```

## B1 — CFG Diffusion

```text
class-conditional diffusion
+
CFG
```

Strong teacher baseline.

## B2 — Standard Consistency Distillation

```text
diffusion
→
consistency student
```

Guidance distillation 없음.

## B3 — Fixed Guided Consistency Distillation

```text
CFG teacher(s_fixed)
→
consistency student
```

Student는 guidance scale을 입력받지 않는다.

## B4 — Guidance-Conditioned Consistency Distillation

```text
CFG teacher(s)
→
student(x_t,t,class,s)
```

Main proposed method.

---

# 14. Main Ablation Axis

가장 중요한 ablation:

```text
No Guidance Distillation
vs
Fixed Guidance Distillation
vs
Variable Guidance Distillation
```

다음 우선순위:

```text
guidance distribution
```

예:

```text
fixed s
Uniform
discrete scales
curriculum
```

그 다음:

```text
Full FT
vs
LoRA
```

---

# 15. Guidance Scale

CFG는 quality/fidelity와 diversity 사이의 trade-off를 만든다.

따라서 가장 큰 guidance scale을 무조건 사용하는 것을 피한다.

먼저 teacher에서 sweep:

```text
s ∈ {
0,
0.5,
1,
2,
3,
4,
6
}
```

을 실시한다.

Teacher의 best-performing scale을 fixed-guidance baseline으로 사용한다.

---

# 16. Dataset Characteristic

Dataset은 상대적으로 sample 수가 작으면서 class가 많은 class-conditional generation 문제이다.

따라서 다음 위험을 항상 고려한다.

```text
high guidance
→ stronger class fidelity
→ possible diversity reduction

low guidance
→ more diversity
→ weaker class identity
```

따라서 CFG scale 또는 CFG distribution은 중요한 연구 변수이다.

---

# 17. One-NFE Strategy

1-NFE model의 최우선 목표:

```text
noise
↓
single student evaluation
↓
image
```

Inference-time CFG는 사용하지 않는다.

필요하면 one-NFE 전용 student를 별도로 학습한다.

---

# 18. Few-NFE Strategy

Few-NFE model은 우선:

```text
4 NFE
```

성능을 최적화한다.

가능하면:

```text
2
3
4
```

NFE를 하나의 student가 지원하도록 한다.

그러나 4-NFE 성능 손실이 크다면 별도 checkpoint를 허용한다.

---

# 19. Experiment Priority

## Stage 1
Teacher 확보

```text
Diffusion
→
CFG
→
CFG scale sweep
```

## Stage 2
Distillation baseline

```text
guided distillation
consistency distillation
```

각각 독립적으로 검증.

## Stage 3
Main method

```text
CFG-aware consistency distillation
```

## Stage 4
FastGen optimization

```text
1 NFE
4 NFE
```

## Stage 5
Ablations

```text
fixed vs variable CFG
CFG distributions
guidance curriculum
LoRA
architecture tuning
```

---

# 20. Success Criteria

각 단계는 다음 조건을 만족해야 다음으로 넘어간다.

### Teacher
CFG teacher가 unguided teacher보다 적절한 FID 개선 또는 명확한 qualitative improvement를 보여야 함.

### Guided Distillation
Single-forward student가 teacher의 guided behavior를 근사해야 함.

### Consistency
1/few-step generation이 정상적으로 동작해야 함.

### Main method
Standard consistency baseline보다 FID 개선 여부를 확인.

Main method가 baseline보다 나쁘다면 원인을 조사하고 baseline 결과를 보존한다.

---

# 21. Metrics

기본 metric:

```text
FID
```

항상 함께 기록:

```text
NFE
parameter count
peak VRAM
training steps
wall-clock training cost
```

필요하다면 연구 분석용으로:

```text
per-class sample visualization
class-conditioned diversity
```

를 추가한다.

하지만 FastGen optimization의 최우선 판단 기준은 official evaluation metric이다.

---

# 22. Reproducibility

모든 결과는:

```text
config
seed
checkpoint
git commit
metrics
```

와 연결되어야 한다.

실험 결과를 notebook이나 기억에만 남기지 않는다.

---

# 23. Research Discipline

한 번에 하나의 변수를 변경한다.

나쁜 예:

```text
architecture 변경
+
optimizer 변경
+
CFG 변경
+
loss 변경
```

후 baseline과 비교.

좋은 예:

```text
baseline
→ guidance conditioning만 추가
→ 비교
```

모든 proposed improvement에는 명시적인 baseline이 있어야 한다.

---

# 24. Current Hypothesis

현재 핵심 가설:

> FastGen처럼 sampling NFE가 매우 제한된 환경에서는 inference-time CFG를 사용하는 것보다, strong multi-step teacher의 CFG-induced trajectory를 student 내부에 distill하는 것이 효율적이다.

추가 가설:

> 하나의 fixed guidance scale만 distill하는 것보다 guidance scale을 condition으로 학습하면 작은 class-conditional dataset에서 fidelity/diversity trade-off를 더 유연하게 제어할 수 있다.

이들은 **검증할 가설이지 이미 사실로 확정된 결론이 아니다.**

---

# 25. Optional Novel Contributions

Baseline reproduction이 완료된 뒤 고려할 수 있다.

### Guidance curriculum

Training progress에 따라 guidance range 증가.

### Guidance distribution optimization

Uniform 대신 성능에 유리한 scale에 sampling probability 집중.

### Class-dependent guidance

class별 데이터 수 또는 난이도에 따라 guidance behavior 조절.

이는 baseline 결과가 확보된 후에만 고려한다.

---

# 26. Things We Are Not Prioritizing

현재 우선순위가 낮은 것:

```text
large transformer backbone
latent diffusion
external autoencoder
pretrained features
complex GAN hybrid objective
DMD2-scale reproduction
full CTM reproduction
large architecture search
```

제한된 GPU와 기간을 고려하여 reproduction risk가 낮은 방법부터 진행한다.

---

# 27. Current Final Direction

현재 프로젝트의 기본 설계는 다음 한 줄로 유지한다.

> **Train a strong class-conditional CFG diffusion teacher from scratch, then distill its guided denoising behavior into a compact guidance-aware consistency model optimized for 1- and 4-NFE generation.**

이 방향은 명확한 실험적 근거 없이 변경하지 않는다.

방향을 변경할 경우 반드시 이 문서의:

- hypothesis
- method
- baseline
- experiment priority

를 함께 수정한다.