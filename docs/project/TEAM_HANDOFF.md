# FastGen 프로젝트 팀 공유 안내

## 전달 파일

1. `fastgen_code_overlay.zip`: 이 프로젝트에서 추가·수정한 코드, 설정, 문서, training-only holdout manifest. 원본 FastGen 제공 파일 위에 적용한다. 데이터 이미지와 checkpoint는 없다.
2. `fastgen_inference_checkpoints.zip`: 지금까지의 **최선 1-NFE** 모델(100K budget, 공식 FID 58.1595)과 4-NFE 모델(공식 FID 33.0517). ZIP 내부 경로는 `checkpoints/one_nfe.ckpt`, `checkpoints/few_nfe.ckpt`다. 원래 작업 폴더의 `checkpoints/one_nfe.ckpt`는 이전 25K 모델이므로 이 ZIP에는 개선된 100K 모델을 넣었다.
3. `SHA256SUMS.txt`: 전달 후 ZIP 무결성 확인용 해시.

원본 기준: `https://github.com/KAIST-Visual-AI-Group/AIC313-Project-FastGen`, commit `e3f5120000cd91037017d86f23993492195101a5`. 이 overlay는 **정확히 이 commit의 깨끗한 checkout**에 적용한다. 기존 작업이 있는 폴더에는 덮어쓰지 않는다.

## 팀원 또는 AI agent가 적용하는 순서

```bash
git clone https://github.com/KAIST-Visual-AI-Group/AIC313-Project-FastGen.git workspace_distillation
cd workspace_distillation
git checkout e3f5120000cd91037017d86f23993492195101a5
git status --short

# ZIP 세 개가 있는 전달 폴더를 지정한다.
handoff_dir=/path/to/received_files
(cd "$handoff_dir" && sha256sum -c SHA256SUMS.txt)
python3 -m zipfile -e "$handoff_dir/fastgen_code_overlay.zip" .
python3 -m zipfile -e "$handoff_dir/fastgen_inference_checkpoints.zip" .
```

`git status --short`는 overlay 적용 전 깨끗해야 한다. 적용 후에는 `model.py` 수정과 새 파일만 나타나야 한다. `dataset.py`, `download_dataset.py`, `evaluate.py`, `src/utils.py`, 원본 train/val split은 ZIP에 없으며 변경하면 안 된다. `model.py`에서는 `ModelOneNFE`와 `ModelFewNFE` subclass만 수정했다. ZIP 내부의 `TEAM_HANDOFF.md`도 이 안내의 사본이다.

원본 `README.md`의 환경 설치 절차를 따르고, Kaggle 데이터셋은 원본 저장소의 `download_dataset.py`로 각자 준비한다. 이미지 데이터와 `.venv`는 공유 ZIP에 포함하지 않았다. 제공된 `requirements.txt` 이외의 새 의존성은 추가하지 않았다.

## 최소 확인

```bash
python - <<'PY'
from model import Model
for mode, path in [('one_nfe','checkpoints/one_nfe.ckpt'), ('few_nfe','checkpoints/few_nfe.ckpt')]:
    model = Model.load_checkpoint(path, mode, device='cpu')
    print(mode, model.count_parameters())
PY
```

두 모델 모두 48,883,107 parameters여야 한다. 데이터셋 준비 후 원본 evaluator로 FID를 재계산할 수 있다.

```bash
python evaluate.py --model_checkpoint checkpoints/one_nfe.ckpt --evaluate_mode one_nfe --reference_dir results/reference --output_dir results/evaluation_one
python evaluate.py --model_checkpoint checkpoints/few_nfe.ckpt --evaluate_mode few_nfe --reference_dir results/reference --output_dir results/evaluation_few
```

평가 결과는 환경과 실제 이미지 준비 상태를 확인하고 비교한다. 공식 보고치: 1 NFE 58.1595, 4 NFE 33.0517. 두 checkpoint는 CFG scale 2의 guided teacher를 distill한 결과이며 추론 시 teacher가 필요하지 않다.

## 추가 학습을 함께 이어갈 경우

Inference checkpoint에는 optimizer·EMA·RNG 학습 상태가 없다. 다음 연구를 이어가려면 큰 파일을 **별도로** 전송한다. 최소한 선택된 teacher의 `experiments/teacher_b64_noaug_constant_100k/final.pt`가 필요하다. 기존 progressive 결과를 연장하려면 `experiments/fixed_cfg2_progressive/{n4,n2,n1}/final.pt`, 100K budget 실험에서 이어가려면 `experiments/low_nfe_budget_100k/n1/final.pt`와 `n2/step_075000.pt` 또는 `n2/final.pt`도 필요하다. 경로를 그대로 보존한다. 이 학습 checkpoint들은 총 수 GB라 기본 ZIP에 넣지 않았다.

`data/development/seed42_holdout5/{train.txt,holdout.txt,metadata.json}`은 코드 ZIP에 포함되어 있다. `reference/`의 755 PNG는 포함되지 않았다. 이 경로는 training split 내부 holdout 평가에 필요하며, teacher checkpoint와 원본 데이터에서 `teacher.evaluate --reference-split development`로 다시 만들 수 있다. `docs/low_nfe_budget_experiment.md`와 `docs/distillation_results_review.md`에 현 결과와 다음 실험의 비교 기준이 있다. Consistency distillation은 아직 구현되지 않았다.

## Git을 쓰고 싶다면

fork 없이도 이 checkout에서 로컬 브랜치와 commit을 만들 수 있다. GitHub에 push할 권한이 없을 때도, 변경 사항을 commit한 뒤 `git bundle` 파일로 오프라인 전달할 수 있다. 이번 ZIP은 아직 commit하지 않은 파일까지 안전하게 전달하도록 만든 것이다. Git bundle은 checkpoint와 데이터처럼 Git에 넣지 않은 파일을 자동으로 포함하지 않으므로 가중치는 별도 전달한다.
