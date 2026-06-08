# CNUH_DISEASE Version History

이 문서는 `CNUH_DISEASE` 계열 실험에서 version을 올릴 때마다 코드와 설정을 왜 수정했는지 누적 기록하는 changelog다.

각 version section은 다음 관점으로 업데이트한다.

- 이전 version에서 확인한 병목
- 새 version에서 바꾼 구조와 loss/config
- 변경 의도와 trade-off
- 코드 영향 범위
- diagnostics에서 확인해야 할 기준

## VER10 -> VER11

### 목적

`CNUH_DISEASE_VER11`은 `CNUH_DISEASE_VER10`에서 남은 Airway 병목을 줄이기 위한 실험이다. 핵심은 모델 capacity를 키우고, Airway top-branch teacher를 더 이른 시점부터 강화하며, `top_support_score_margin`이 어려운 Airway sample에 더 강하게 작동하도록 만드는 것이다.

`VER10` 후반 분석에서 확인한 문제는 다음과 같았다.

- Airway에서 `top_branch_margin`이 충분한 sample도 있었지만, 일부 sample은 branch teacher 품질 자체가 약했다.
- `top_support_score`가 Airway를 충분히 올리지 못하면 `branch_direct_score -> branch_support_score -> class_evidence_logits` 경로가 다시 Normal 쪽으로 기울었다.
- `top_support_score_margin`에 Airway label multiplier와 support-conditioned multiplier가 있었지만, hard sample에서 추가로 강해지는 장치는 부족했다.
- `class_evidence_logits`가 wrong class를 과신하는 경우를 제한하는 label-agnostic cap이 없었다.
- 사용자는 현재 모델이 AST 계열과 비교해 parameter 수가 작다고 보고 있었고, capacity 부족 가능성을 실험적으로 확인하고 싶어 했다.

따라서 `VER11`은 branch/gate 구조를 크게 바꾸기보다, capacity와 loss weighting을 조정해 Airway branch teacher 및 evidence anchor를 강화하는 방향으로 설계했다.

### 핵심 변경 요약

#### 1. Model capacity 확장

`model.encoder.architecture`를 다음처럼 키웠다.

```json
"architecture": {
  "hidden_size": 384,
  "adapter_depth": 6,
  "shared_stem_depth": 1
}
```

의도는 AST와의 capacity gap을 줄이고, 특히 evidence scorer와 branch-support 변환부가 더 넓은 hidden representation을 사용할 수 있게 하는 것이다.

`classifier`와 residual combiner capacity는 유지했다. 현재 병목은 final combiner보다 `branch/top support -> class_evidence_logits` 앞단에 있다고 판단했기 때문이다.

#### 2. Airway top-branch margin ramp를 앞당김

`top_branch_margin.phase_weight_schedule`을 Airway에 대해 epoch 21부터 31까지 `1.0 -> 2.0`으로 ramp하도록 바꿨다.

```json
"phase_weight_schedule": {
  "enabled": true,
  "start_epoch": 21,
  "end_epoch": 31,
  "start_multiplier_by_label": {
    "Normal": 1.0,
    "Lung_Parenchymal": 1.0,
    "Airway": 1.0
  },
  "label_multiplier_by_label": {
    "Normal": 1.0,
    "Lung_Parenchymal": 1.0,
    "Airway": 2.0
  }
}
```

`VER10` 후보 논의에서는 Phase 3에서 Airway top branch를 강화하는 방향이 맞지만, epoch 31부터만 강화하면 branch teacher가 너무 늦게 개선될 수 있다고 봤다. 그래서 start epoch을 21로 앞당기고, abrupt step 대신 ramp로 적용했다.

의도는 다음과 같다.

- epoch 21부터 Airway branch teacher 품질을 조금씩 강화한다.
- Phase 2 후반과 Phase 3 진입부에서 Airway margin pressure를 자연스럽게 키운다.
- Normal/Lung multiplier는 1.0으로 유지해 normal false positive 증가를 제한한다.

#### 3. `top_support_score_margin`에 hardness weighting 추가

`top_support_score_margin`은 이미 Airway label multiplier와 support-conditioned multiplier를 가지고 있었다.

`VER11`에서는 여기에 hard sample weighting을 추가했다.

```json
"hardness_weighting": {
  "enabled": true,
  "source": "top_support_gap",
  "mode": "negative_gap",
  "gain": 1.0,
  "cap": 3.0
}
```

수식은 다음 의미다.

```text
hardness = relu(-top_support_gap)
hardness_weight = 1 + gain * clamp(hardness, 0, cap)
```

즉, `top_support_score`가 true class를 hardest negative보다 낮게 두는 sample일수록 `top_support_score_margin` penalty를 더 키운다.

기존 multiplier와 함께 적용되는 전체 multiplier chain은 다음이다.

```text
total_multiplier =
  label_multiplier
  * support_conditioned_multiplier
  * hardness_weight
```

Airway의 경우 `label_weight_by_label.Airway=2.0`을 유지했으므로, Airway hard sample은 label multiplier와 hardness weighting을 동시에 받는다.

#### 4. `class_evidence_gap_cap_regularization` 추가

wrong class overconfidence를 label-agnostic하게 제한하기 위해 새 loss를 추가했다.

```json
"class_evidence_gap_cap_regularization": {
  "enabled": true,
  "weight": 0.02,
  "target": "class_evidence_logits",
  "mode": "negative_gap_hinge",
  "negative_gap_cap": 3.0,
  "class_weighted": true,
  "reduction": "mean",
  "warmup_epochs": 10
}
```

수식은 다음과 같다.

```text
evidence_gap =
  class_evidence_logits[true]
  - max(class_evidence_logits[negative])

penalty = relu(-evidence_gap - negative_gap_cap)
```

이 loss는 모든 오답을 강하게 누르는 margin loss가 아니다. `evidence_gap`이 이미 크게 음수인 경우, 즉 wrong class overconfidence가 심한 경우만 제한한다.

의도는 다음과 같다.

- label-specific target 없이 wrong overconfidence를 제한한다.
- Airway가 Normal에 크게 밀리는 extreme case를 완화한다.
- 기존 `class_evidence_margin`과 `b2e`가 담당하는 true-label anchor와 중복되지 않도록 약한 weight `0.02`로 시작한다.

### 유지한 설정

`VER11`은 구조 전체를 다시 바꾸는 실험이 아니다. 다음 설정은 유지했다.

- `class_axis_attention` evidence scorer
- branch direct score의 top/gated/reliability mixture 구조
- `top_support_score_margin.label_weight_by_label.Airway=2.0`
- `top_support_score_margin.support_conditioned_multiplier`
- `branch_direct_score_margin`
- `branch_support_score_margin`
- `branch_to_evidence_ranking_consistency`
- `class_evidence_margin`
- `gate_branch_regret`
- `gate_bad_branch_suppression`
- bounded zero-mean residual combiner
- confidence-aware residual gate
- `global_residual_anti_veto`

이렇게 유지한 이유는 `VER10`의 병목이 final combiner 전체가 아니라 Airway teacher/support/evidence 경로의 강도와 capacity에 있다고 판단했기 때문이다.

### 코드 영향 범위

#### Config schema

`src/utils/config.py`에서 다음 surface를 확장했다.

- `TopBranchMarginPhaseWeightScheduleConfig.end_epoch`
- `TopBranchMarginPhaseWeightScheduleConfig.start_multiplier_by_label`
- `TopSupportScoreMarginConfig.hardness_weighting`
- `ClassEvidenceGapCapRegularizationConfig`

validation은 다음을 검사한다.

- `phase_weight_schedule.end_epoch >= start_epoch`
- `start_multiplier_by_label`과 `label_multiplier_by_label` label key가 config label과 일치
- hardness weighting source/mode/gain/cap이 유효
- gap cap weight/cap/warmup/reduction/class_weighted 조건이 유효

#### CLI wiring

`src/cli/training.py`와 `src/cli/cv.py`에서 label-name mapping을 class-index tuple로 resolve하도록 했다.

특히 `phase_weight_schedule.start_multiplier_by_label`과 `label_multiplier_by_label`은 public config에서는 label name으로 쓰고, trainer 내부에서는 class index 순서의 tuple로 사용한다.

#### Trainer loss path

`src/training/trainer.py`에서 다음을 추가했다.

- `_top_branch_margin_phase_multipliers()`가 start/end epoch 사이에서 label별 multiplier를 선형 보간한다.
- `_compute_top_support_score_margin_loss()`가 label multiplier, support multiplier, hardness multiplier를 모두 곱한다.
- `_compute_class_evidence_gap_cap_regularization_loss()`를 추가한다.
- total loss에는 `class_evidence_margin` 이후, branch-support 계열 loss 전에 gap cap loss를 더한다.

loss ordering 의도는 다음이다.

```text
class_evidence_margin
class_evidence_gap_cap_regularization
top_support_score_margin
branch_direct_score_margin
branch_support_score_margin
branch_to_evidence_ranking_consistency
...
```

#### Diagnostics and logging

sample-level diagnostics에 다음 field를 추가했다.

- `top_support_score_margin_hardness`
- `top_support_score_margin_hardness_weight`
- `top_support_score_margin_total_multiplier`
- `class_evidence_gap_cap_gap`
- `class_evidence_gap_cap_penalty`
- `class_evidence_gap_cap_cap`
- `class_evidence_gap_cap_eligible`

epoch log에는 다음 component가 추가된다.

- `evidence_gap_cap raw=... loss=...`
- `top_support_mult=label/support/hardness`

이 field들은 `VER11` 후반 분석에서 Airway hard sample이 실제로 더 강한 multiplier를 받고 있는지, 그리고 wrong overconfidence cap이 작동하는지를 확인하기 위한 것이다.

### 실험 해석 기준

`VER11`을 해석할 때는 다음 순서로 본다.

1. Airway `top_branch_margin_value`가 epoch 21 이후 증가하는지 확인한다.
2. Airway `top_branch_margin_violation`이 Phase 3에서 줄어드는지 확인한다.
3. Airway `top_support_score_margin_hardness_weight`가 hard sample에서 실제로 커지는지 확인한다.
4. Airway `top_support_score_margin_total_multiplier`가 label/support/hardness multiplier를 모두 반영하는지 확인한다.
5. `class_evidence_gap_cap_penalty`가 extreme wrong-overconfidence sample에서만 켜지는지 확인한다.
6. Airway `top_support_score_gap -> branch_direct_score_gap -> branch_support_score_gap -> class_evidence_gap -> final_gap` chain이 개선되는지 확인한다.
7. Normal false positive가 지나치게 증가하는지 confusion matrix와 per-class recall을 같이 확인한다.

### 기대 효과

`VER11`이 성공한다면 다음 현상이 나타나야 한다.

- Airway top branch teacher가 Phase 3 진입 전부터 더 안정적으로 커진다.
- Airway hard sample에서 `top_support_score_margin`의 effective pressure가 커진다.
- `class_evidence_logits`가 Normal을 과도하게 확신하는 extreme case가 줄어든다.
- Airway recall이 오르면서 Normal recall이 과도하게 붕괴하지 않는다.

### 리스크

- Airway multiplier `2.0`은 normal false positive를 늘릴 수 있다.
- model capacity 증가로 train 성능은 오르지만 validation 균형이 더 흔들릴 수 있다.
- `class_evidence_gap_cap_regularization`은 weight가 작아 wrong overconfidence를 감지하더라도 제어력이 부족할 수 있다.
- `top_support_score_margin`이 강해져도 branch-support 변환 구조 자체가 틀어져 있으면 `class_evidence_logits`로 전달되지 않을 수 있다.

### 현재 후속 관찰 포인트

`VER11` 중간 분석에서는 `epoch 45`가 현재 best macro F1 / macro recall이고, `epoch 46`에서는 Normal 쏠림이 다시 강해졌다. 특히 Airway 샘플 단위 분석에서 다음 병목을 계속 지켜봐야 한다.

- `top_branch_margin`은 충분한데 `top_support_score_gap`이 작거나 음수인지
- `top_support_score_gap`이 양수인데도 `gated_support_score_gap` 때문에 `branch_direct_score_gap`이 음수로 뒤집히는지
- Airway 정답이 branch path가 아니라 `embedding_score_gap`에 의존해서 맞는지
- final combiner가 아니라 evidence 전 단계가 주 병목인지

이 관찰은 후속 `VER12` 후보에서 gated path를 더 제한하거나, top-support path를 branch evidence의 primary path로 재설계해야 하는지 판단하는 기준이 된다.
