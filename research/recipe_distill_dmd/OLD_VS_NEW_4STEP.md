# 기존 4-step과 새 4-step 비교

기존 checkpoint를 현재 복원된 환경에서 다시 생성·평가한 Official FID는 **33.05174275**이다. 기존 보고33.0517을 반올림 정밀도 안에서 재현했다. 새25K 4-step의 Official FID는 **36.03177398**로, 기존보다 **2.98003123 높다**.

## 조건과 결과

| 항목 | 기존 4-step | 새 4-step |
|---|---|---|
| Teacher budget | 100K | 150K |
| Teacher/Student attention heads | 1 | 4 |
| Teacher Official FID (기측정) | 26.65365 | 25.49841 |
| CFG transfer budget | 50K | 50K |
| Progressive | 32→16→8→4, 단계별25K | 동일 |
| LR / batch / EMA | 2e-4 constant / 64 / 0.999 | 동일 |
| Prediction / loss | v / truncated-SNR | 동일 |
| Data augmentation | noaug | noaug |
| Student parameters | 48,883,107 | 동일 |
| 실제 inference NFE | 4 | 4 |
| Official FID (이번 확인) | 33.05174 | 36.03177 |
| Development FID | 47.62077 (handoff 기록) | 48.66473 |

평가:151클래스×20=3,020 generated PNG, 동일한3,020 official reference PNG, seed42+batch index, batch16, EMA, bf16 생성. 기존은 원본 workspace의1-head ModelFewNFE를 사용했고 새 모델은 독립 디렉토리의4-head 모델을 사용했다. FID는 동일한 보호된 evaluate.compute_fid이다. 기존 final teacher와 student 학습 원본 전체는 복구되지 않았으므로 기존 학습 조건은 handoff 기록 및 inference 구현 기준이다.

Teacher budget과 attention heads가 함께 다르므로 결과의 개선/악화 원인을 teacher FID 또는 attention head 하나에만 귀속할 수 없다. 이번 결과에서는 더 낮은 teacher FID가 더 낮은4-NFE FID로 이어지지 않았다.

## 어느 단계에서 차이가 생겼는가

아래 표는755장 training holdout에 대한 **development FID**이며 Official FID와 구분한다. 기존 값은 handoff 결과 기록이다.

| 단계 | 기존 | 새 | 새−기존 |
|---|---:|---:|---:|
| Guidance32 | 43.25620 | 42.61387 | -0.64233 |
| Progressive16 | 45.26116 | 44.25382 | -1.00734 |
| Progressive8 | 46.38606 | 45.81403 | -0.57203 |
| Progressive4 | 47.62077 | 48.66473 | +1.04396 |

8-step까지는 새 student가 holdout에서 더 낮지만4-step에서는 순위가 뒤집힌다. 8→4 holdout 악화량은 기존 **+1.23471**, 새 **+2.85070**이다. 다음 진단은8→4 압축 단계의 target/output 오차와 학습 budget에 집중할 근거가 있다. 이것만으로 원인이 확정된 것은 아니다. 기존 예약 학습/DMD4개는 변경하지 않았다.

## 같은 입력의 이미지 비교

같은 클래스·sample index0의 이미지다. 모든 칸에서 **위가기존 / 아래가새 모델**이다. seed42+batch index와batch16을 맞췄다. 원본64×64를 nearest-neighbor2배로 표시했으며 FID용 PNG는 수정하지 않았다.

- [16클래스 비교](../../reports/recipe_distill_dmd/compare_old_new_n4/comparison_16_classes.png)
- [151클래스 전체 비교](../../reports/recipe_distill_dmd/compare_old_new_n4/comparison_all151.png)
- [원시 수치·체크포인트SHA256](../../reports/recipe_distill_dmd/compare_old_new_n4/comparison.json)
