# Consistency Distillation 구현과 원격 학습 실행

2026년 10월 5일 추가한 구현은 기존 epsilon teacher를 사용하는 고정 CFG Consistency
Distillation입니다. 현재 상태는 코드와 CPU 검증이며, 실제 데이터 GPU 학습 및 FID 개선은
아직 검증하지 않았습니다. 기존 Progressive 모델과 체크포인트는 유지합니다.

## 구현 범위

- `consistency/core.py`: VP 좌표 변환, teacher Euler 전이, 경계 조건, L2 consistency loss,
  1~4 NFE 생성. CFG 수식은 기존 `teacher.cfg.apply_cfg`를 재사용합니다.
- `consistency/train.py`: 고정 학습률, microbatch 누적, target EMA와 sampling EMA,
  원자적 체크포인트 저장, 종료 신호 처리, 같은 명령으로 재개, 중복 학습 잠금.
- `consistency/evaluate.py`: 기존 공식 함수로 이미지 생성과 FID 계산. 모델 선택은
  development 1-NFE 평가에만 허용합니다. `best_fid.pt`와 선택 근거를 저장합니다.
- `consistency/export.py`: 선택한 모델을 기존 ModelOneNFE와 ModelFewNFE 형식으로 내보냅니다.
  출력 파일이 이미 있으면 중단하므로 기존 모델이 덮어써지지 않습니다.
- `model.py`: 두 subclass의 state 로딩만 확장했습니다. 공통 Model 클래스는 그대로입니다.
  `consistency_sigma_data` buffer가 있는 체크포인트만 consistency sampler를 사용합니다.
  기존 Progressive 체크포인트는 기존 sampler를 계속 사용합니다.

Variable guidance, 교사 교체 실험, Pseudo-Huber, LoRA는 이번 구현 범위에 포함하지 않았습니다.

## 방법과 논문과의 차이

[Consistency Models, Song et al., ICML 2023](https://proceedings.mlr.press/v202/song23a.html)의
Algorithm 1과 2를 현재 teacher의 이산 VP schedule에 맞춘 구현입니다. 원 논문의 EDM
실험을 그대로 재현했다는 의미는 아닙니다.

`a=sqrt(alpha_bar)`, `s=sqrt(1-alpha_bar)`일 때 `y=x_VP/a`, `sigma=s/a`로 변환합니다.
Teacher 입력은 기존과 같은 `a*y`와 원래 timestep이며, epsilon 출력으로
`y_next=y+(sigma_next-sigma)*epsilon`을 계산합니다. 이는 동일한 두 timestep 사이의
deterministic DDIM 전이와 같습니다. 기본 설정은 teacher의 x0 clipping을 유지합니다.

학생 출력은 `c_skip*y+c_out*F(a*y,t,class)`입니다.
`c_skip=sd²/((sigma-sigma_min)²+sd²)`,
`c_out=sd*(sigma-sigma_min)/sqrt(sigma²+sd²)`로 두어 최소 노이즈에서 항등 함수가 됩니다.
학생의 backbone은 consistency residual을 출력하며 epsilon이나 v로 해석하지 않습니다.
Teacher의 특징 가중치를 복사한 뒤 출력 convolution만 0으로 초기화합니다.

기본 학습 설정은 원래 1,000개 timestep에서 128개를 등간격으로 선택하고 인접한 쌍을
균등하게 뽑는 것입니다. L2 loss, 고정 LR 2e-4, batch 64/microbatch 16, 100K 업데이트,
두 EMA decay 0.999를 사용합니다. 이는 아직 최적화하지 않은 프로젝트 기준 설정입니다.
Teacher, 데이터, 출력 정의를 고정한 뒤 한 요소씩 비교하십시오.

학습 입력은 `실제 이미지 + sigma*noise`입니다. 추론은 `sigma_max*noise`에서 시작하며,
유한한 최대 노이즈에서 실제 신호 성분을 무시하는 근사는 남아 있습니다. 다단계 생성은
denoise와 re-noise를 번갈아 수행하며 backbone을 정확히 요청한 횟수만 호출합니다.
1/2/4 NFE는 기존 32-node grid의 timestep을 재사용하고, 3 NFE는 같은 grid에서 선택합니다.
이 timestep 배치는 FID에 맞춰 최적화한 값이 아닙니다.

외부 pretrained feature나 LPIPS는 사용하지 않습니다. iCT의 target EMA 제거는 별도의
Consistency Training 결과이므로 이번 distillation에 적용하지 않습니다.
[Improved Techniques for Training Consistency Models](https://arxiv.org/html/2310.14189v1)

## 실행 전 준비

이 폴더는 원본 FastGen 저장소에 적용하는 overlay입니다. 원본 `dataset.py`, `evaluate.py`,
`src/utils.py`, requirements와 실제 포켓몬 이미지가 필요합니다. 원래 TEAM_HANDOFF.md의
지정 commit과 환경 설치 절차를 따르십시오. 아래 명령은 overlay가 적용된 원본 저장소
루트에서 실행합니다. 파일 배포 시 새 `consistency/`, `configs/consistency/`도 포함하십시오.

기존 생성용 `one_nfe.ckpt`와 `few_nfe.ckpt`로 새 학습을 시작할 수는 없습니다. 원래 학습한
선생님 `experiments/teacher_b64_noaug_constant_100k/final.pt`를 팀원에게 받아야 합니다.
이미지 데이터, training/holdout manifest, class mapping도 기존 실험과 같아야 합니다.
학습 상태에 기록된 SHA256과 클래스 순서가 다르면 실행을 거부합니다.

CPU 수치 테스트:

```bash
python -m consistency.tests
```

GPU에서 먼저 5회 업데이트를 실행해 finite loss와 실제 VRAM을 확인합니다. 이 명령은
아래 전체 학습과 동일한 설정을 쓰며, 전체 학습이 이 저장 지점부터 이어집니다.

```bash
python -m consistency.train \
  --config configs/consistency/fixed_cfg2.json \
  --teacher-checkpoint experiments/teacher_b64_noaug_constant_100k/final.pt \
  --output-dir experiments/consistency_cfg2 \
  --max-updates-this-run 5
```

`metrics.json`의 peak_vram_bytes를 확인하십시오. 학습기는 20,000,000,000 bytes 이상이면
실패 처리합니다. GPU 측정 전에는 메모리 제한 충족을 주장하지 않습니다. 기본 실험은
선택된 no-augmentation teacher에 제한되어 있으며, 데이터 증강을 재실험하지 않습니다.

## 노트북을 닫아도 계속 학습하기

학습은 노트북이 아닌 원격 GPU 서버에서 실행해야 합니다. 서버에 SSH로 접속한 다음:

```bash
tmux new -s fastgen-cd
cd /path/to/assembled/FastGen
export FASTGEN_PYTHON=/path/to/your/environment/bin/python
bash scripts/run_consistency.sh
```

실행 후 `Ctrl+b`를 누르고 손을 뗀 다음 `d`를 누르면 세션에서 빠져나옵니다. 이때 노트북을
닫거나 노트북의 와이파이가 끊겨도 원격 서버에서 학습은 계속됩니다. 다시 SSH로 접속해서:

```bash
tmux attach -t fastgen-cd
```

`tmux` 세션은 SSH 단절 후에도 유지됩니다.
[tmux 공식 매뉴얼](https://man.openbsd.org/tmux.1)

서버 자체의 종료, 재부팅, GPU 작업 시간 제한, 디스크 손실까지 tmux가 막아주지는 않습니다.
체크포인트는 지속 저장소에 두십시오. 서버를 다시 켠 뒤 **같은 실행 명령**을 사용하면
`latest.pt`부터 재개합니다. 서버 재부팅 후 자동으로 프로세스를 실행하는 서비스는 이번에
설정하지 않았습니다. 원격 클라우드 서버를 켜두는 동안 서비스 요금은 계속 발생할 수 있습니다.

노트북에서 직접 학습할 경우 절전 상태에서는 계산이 멈출 수 있습니다. Colab도 별도 VM
수명과 사용 제한이 있으므로 무기한 실행을 보장하지 않습니다.
[Colab 공식 FAQ](https://research.google.com/colaboratory/faq.html)

## 저장과 재개

`latest.pt`는 기본 500회 업데이트마다 저장합니다. SIGINT(Ctrl+C) 또는 SIGTERM을 받으면
현재 optimizer update를 마친 뒤 저장하고 나옵니다. 갑작스러운 전원 차단이나 SIGKILL은
마지막으로 완료된 저장 이후의 작업을 잃을 수 있습니다. 저장 도중에는 임시 파일에 쓴 뒤
교체하므로 기존 정상 체크포인트를 보존합니다.

학생, target EMA, sampling EMA, optimizer, RNG, epoch와 batch 위치를 저장합니다.
교사 파일 내용, 설정, 소스 코드, 데이터 경로 목록, class mapping, device type이 바뀌면
같은 실험의 재개를 거부합니다. CPU 합성 데이터 테스트에서 저장 후 재개와 연속 실행의
가중치·optimizer·RNG 일치를 검사합니다. GPU 결과의 비트 단위 일치는 별도 검증 대상입니다.

`train.log`와 실행별 `console_*.log`는 진단용으로 추가 기록됩니다. 마지막 체크포인트보다
나중에 기록된 로그는 장애 후 다시 나타날 수 있으므로 완료 update는 체크포인트를 기준으로
판단합니다. `final.pt`는 100K 완료 때만 생성합니다. 디스크 용량은 full checkpoint 여러 개를
저장할 수 있도록 확보하십시오. 같은 run 디렉터리의 중복 trainer는 파일 잠금으로 차단합니다.

## 비교 평가와 내보내기

먼저 기존 training-only holdout reference PNG를 준비합니다. 이미 서버에 있으면 재생성하지
않습니다. 없으면 기존 `teacher.evaluate --reference-split development` 실행 절차를 사용합니다.
755장 내부 reference와 3,020장 공식 reference의 FID는 서로 직접 비교하지 않습니다.

```bash
python -m consistency.evaluate \
  --checkpoint experiments/consistency_cfg2/step_025000.pt \
  --output-dir experiments/consistency_cfg2/eval_025000_n1 \
  --reference-dir data/development/seed42_holdout5/reference \
  --nfe 1 --select-best
```

다른 milestone도 별도 출력 디렉터리에서 평가합니다. `--nfe 4`로 4회 생성도 비교할 수 있습니다.
`--select-best`는 동일 설정·시드·reference의 development 1-NFE 점수에만 사용할 수 있습니다.
중단돼 이미지가 일부만 저장된 평가는 새 output-dir에서 재실행합니다. 평가 결과가 이미
완료된 경우 입력 digest가 같을 때만 재사용합니다. 모델 선택 평가는 같은 run에서 순차 실행합니다.
Reference 디렉터리는 명시한 split에서 만든 것을 사용해야 하며 파일 개수와 내용 digest만으로
잘못된 split을 완전히 식별할 수는 없습니다.

CFG 영향만 비교하려면 `configs/consistency/no_extra_guidance.json`으로 별도 디렉터리에서
학습합니다. CFG=1은 종류 조건은 유지하고 추가 guidance만 사용하지 않는 설정입니다.
CFG=2와 나머지 학습 조건은 같습니다. 코드/설정 변경 후 기존 run의 resume로 섞지 마십시오.

선택한 모델을 기존 baseline과 다른 경로로 내보냅니다:

```bash
python -m consistency.export \
  --checkpoint experiments/consistency_cfg2/best_fid.pt \
  --mode one_nfe --output checkpoints/consistency/one_nfe.ckpt
python -m consistency.export \
  --checkpoint experiments/consistency_cfg2/best_fid.pt \
  --mode few_nfe --output checkpoints/consistency/few_nfe.ckpt
python evaluate.py --model_checkpoint checkpoints/consistency/one_nfe.ckpt \
  --evaluate_mode one_nfe --reference_dir results/reference \
  --output_dir results/consistency_one --batch_size 16
python evaluate.py --model_checkpoint checkpoints/consistency/few_nfe.ckpt \
  --evaluate_mode few_nfe --reference_dir results/reference \
  --output_dir results/consistency_four --batch_size 16
```

공식 평가 전에 실제 GPU에서 두 exported 모델의 NFE와 출력 shape를 확인하십시오.
기존 `checkpoints/one_nfe.ckpt`, `checkpoints/few_nfe.ckpt`는 이 절차에서 바꾸지 않습니다.

```bash
python -m consistency.check_submission --checkpoint checkpoints/consistency/one_nfe.ckpt \
  --mode one_nfe --device cuda
python -m consistency.check_submission --checkpoint checkpoints/consistency/few_nfe.ckpt \
  --mode few_nfe --device cuda
```

## 로컬 검증 결과

macOS CPU, Python 3.12, PyTorch 2.6.0에서 수치·gradient·1~4 NFE·동일 seed 생성,
중간 batch 저장 후 epoch를 넘어 재개한 가중치/optimizer/RNG 일치 테스트를 통과했습니다.
설정이나 teacher digest가 바뀐 재개를 거부하는 검사도 통과했습니다.

전달된 실제 Progressive 체크포인트 두 개는 각각 48,883,107 parameters로 로드됐고,
같은 seed에서 기존 sample_velocity 경로와 bitwise 같은 결과를 생성했습니다.
새 CD 체크포인트 형식은 합성 형식 fixture로 export/load와 정확한 1/4 NFE를 검사했습니다.
이 fixture는 실제로 학습한 CD 모델이 아니며 품질 검증에 사용하지 않았습니다.

Overlay에 없는 `src.utils`의 파라미터 집계 함수만 로컬 호환성 검사에서 대체했습니다.
실제 backbone, checkpoint loader, 두 sampler는 그대로 검사했습니다. 공식 전체 환경,
실제 데이터 학습, CUDA 최대 메모리, FID, 원격 tmux 실행은 이 노트북에서 검증하지 않았습니다.
