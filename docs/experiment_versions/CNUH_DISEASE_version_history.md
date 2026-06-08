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

### 극단 sample에 대한 의도

`VER11`의 목적은 극단적인 sample까지 모두 억지로 맞추는 것이 아니다. 더 정확한 의도는 Airway hard sample이 평균 loss 안에서 묻히지 않도록 하고, bounded multiplier 안에서 evidence path를 더 강하게 교정하는 것이다.

현재 설정이 공격적으로 보이는 이유는 다음 항목들이 Airway hard sample에 동시에 작동하기 때문이다.

- `top_branch_margin.phase_weight_schedule`은 Airway branch teacher를 epoch 21부터 강화한다.
- `top_support_score_margin.label_weight_by_label.Airway=2.0`은 Airway top-support score 오류를 더 크게 본다.
- `top_support_score_margin.support_conditioned_multiplier`는 top branch support가 있는 sample을 더 중요하게 본다.
- `top_support_score_margin.hardness_weighting`은 `top_support_gap < 0`인 hard sample의 penalty를 더 키운다.

하지만 이 설정은 무한히 극단 sample을 쫓는 구조는 아니다.

- `support_conditioned_multiplier.cap=2.0`으로 support 기반 증폭을 제한한다.
- `hardness_weighting.cap=3.0`으로 hard sample 증폭을 제한한다.
- `class_evidence_gap_cap_regularization`은 wrong class overconfidence가 너무 커지는 경우를 label-agnostic하게 제한한다.
- residual combiner는 bounded, zero-mean, confidence-aware gate로 제한되어 있다.
- `top_branch_margin.auto_margin_by_train_stats`도 Airway max margin을 `0.8`로 제한한다.

따라서 `VER11`의 의도는 다음 문장으로 정리할 수 있다.

```text
Airway hard sample을 더 강하게 보되,
무한히 맞추려 하지 않고,
bounded multiplier와 evidence gap cap 안에서 evidence path를 교정한다.
```

다만 실제 학습에서는 label noise, acoustic ambiguity, dataset artifact가 있는 Airway sample까지 과하게 따라갈 수 있다. 후반부 분석에서 다음 현상이 보이면 `VER11`은 hard sample을 잘 교정한 것이 아니라 극단 sample을 과하게 쫓은 것으로 해석해야 한다.

- Airway recall은 안정적으로 오르지 않는데 Normal false positive가 증가한다.
- `top_support_score_margin_penalty`가 소수 Airway sample에 과도하게 집중된다.
- `top_support_score_margin_total_multiplier`가 큰 sample에서 `class_evidence_gap` 개선으로 이어지지 않는다.
- `class_evidence_gap_cap_penalty`가 자주 켜지면서도 validation loss가 흔들린다.

### 현재 후속 관찰 포인트

`VER11` 중간 분석에서는 `epoch 45`가 현재 best macro F1 / macro recall이고, `epoch 46`에서는 Normal 쏠림이 다시 강해졌다. 특히 Airway 샘플 단위 분석에서 다음 병목을 계속 지켜봐야 한다.

- `top_branch_margin`은 충분한데 `top_support_score_gap`이 작거나 음수인지
- `top_support_score_gap`이 양수인데도 `gated_support_score_gap` 때문에 `branch_direct_score_gap`이 음수로 뒤집히는지
- Airway 정답이 branch path가 아니라 `embedding_score_gap`에 의존해서 맞는지
- final combiner가 아니라 evidence 전 단계가 주 병목인지

이 관찰은 후속 `VER12` 후보에서 gated path를 더 제한하거나, top-support path를 branch evidence의 primary path로 재설계해야 하는지 판단하는 기준이 된다.

### 후반부 추가 관찰: score scale 폭주

`VER11` 후반부에서는 metric 기준으로 개선이 있었다. `epoch 73`은 현재 best macro F1을 갱신했고, `epoch 71`은 macro recall 기준으로 가장 좋았다. 하지만 validation loss는 크게 증가했다. 이는 단순히 성능이 좋아졌다는 의미가 아니라, `class_evidence_logits` scale이 커지면서 correct sample은 매우 큰 positive gap을 만들고, wrong sample은 매우 큰 negative gap을 만드는 방향으로 학습이 진행됐다는 뜻이다.

대표적으로 `epoch 73` diagnostics에서는 다음 현상이 관찰됐다.

- `main`, `class_evidence_margin`, `branch_to_evidence`, `class_evidence_gap_cap_regularization`이 크게 증가했다.
- `top_support_score_margin`, `branch_direct_score_margin`, `branch_support_score_margin`은 상대적으로 작은 범위에 머물렀다.
- Airway correct sample에서도 `branch_direct_score_gap`과 `branch_support_score_gap`은 여전히 음수인 경우가 많았다.
- Airway 정답은 branch path가 직접 만든 것이 아니라 `embedding_score_gap`과 `interaction_score_gap`이 매우 커지면서 만들어졌다.
- wrong sample에서는 `class_evidence_gap_cap_penalty`가 크게 켜졌고, 일부 sample은 confidence가 거의 1.0인 상태로 잘못 예측됐다.

따라서 `VER11`의 후반부 개선은 다음처럼 해석해야 한다.

```text
Metric은 개선됐지만,
branch/top support -> evidence 전달 문제가 해결된 것은 아니다.
후반부 정답 개선의 상당 부분은 embedding/interaction score scale 증가에 의존했다.
```

이 관찰은 후속 version에서 반드시 두 가지 제약을 고려해야 함을 의미한다.

#### 1. Embedding/interaction score scale control

`embedding_score`와 `interaction_score`가 지나치게 큰 gap을 만들면, branch path가 실제로 개선됐는지 확인하기 어렵고, wrong sample의 loss가 폭주한다. 따라서 후속 version에서는 다음 방식 중 하나를 검토해야 한다.

- `embedding_score`와 `interaction_score`에 logit norm penalty 또는 gap cap을 적용한다.
- `embedding_score_gap`과 `interaction_score_gap`이 일정 cap을 넘으면 soft penalty를 준다.
- `class_evidence_logits` 전체가 아니라 decomposition component별 scale을 제한한다.
- `interaction_scale` 또는 embedding head output에 bounded scale을 둔다.
- evidence scorer output에 temperature 또는 normalized logit head를 도입한다.

목표는 metric을 낮추는 것이 아니라, evidence gap이 과도하게 커져 wrong-overconfidence를 만드는 현상을 제어하는 것이다.

#### 2. Branch path dominance constraint

`VER11`에서는 Airway 정답 sample에서도 `branch_direct_score_gap`과 `branch_support_score_gap`이 음수인 경우가 남았다. 이 상태에서는 Airway recall이 좋아져도 branch signal이 evidence로 전달됐다고 보기 어렵다.

후속 version에서는 branch path가 evidence decision에 최소한의 방향성을 갖도록 제약해야 한다.

검토 후보는 다음과 같다.

- `branch_support_score_gap`이 음수인데 `class_evidence_gap`만 큰 양수인 sample에 penalty를 준다.
- `class_evidence_gap`이 positive일 때 `branch_direct_score_gap` 또는 `branch_support_score_gap`도 일정 수준 이상이 되도록 consistency loss를 둔다.
- `class_evidence_logits = branch_primary_score + bounded_embedding_correction + bounded_interaction_correction` 형태로 evidence scorer를 재구성한다.
- top-support path를 primary path로 두고 gated/embedding/interaction path는 correction으로 제한한다.
- Airway에만 국한하지 않고 label-agnostic하게 `branch_support_gap -> evidence_gap`의 monotonic consistency를 강화한다.

핵심 목표는 다음이다.

```text
Airway를 맞추는 이유가 embedding/interaction shortcut이 아니라,
branch/top support path가 true class evidence를 실제로 끌어올렸기 때문이어야 한다.
```

따라서 후속 version을 설계할 때는 `macro_f1`만 보지 말고 다음 diagnostics chain을 반드시 같이 확인한다.

```text
top_branch_margin
-> top_support_score_gap
-> branch_direct_score_gap
-> branch_support_score_gap
-> embedding_score_gap / interaction_score_gap
-> class_evidence_gap
-> final_gap
```

특히 `class_evidence_gap`이 크게 양수인데 `branch_support_score_gap`이 음수라면, metric이 좋아도 branch path 병목은 해결되지 않은 것으로 판단한다.

## VER11 -> VER12

### 목적

`CNUH_DISEASE_VER12`는 `VER11`에서 확인된 후반부 병목을 label-specific 보정이 아니라 label-agnostic 구조 보정으로 다루는 실험이다.

`VER11` 최종 분석에서 가장 중요한 결론은 final combiner가 주 병목이 아니라는 점이었다. `final_gap`은 대체로 `class_evidence_gap`을 따라갔고, 실제 문제는 `top/branch support -> class_evidence_logits` 경로에서 발생했다. 특히 `embedding_score`와 `interaction_score`가 branch path보다 큰 scale로 evidence gap을 뒤집거나, wrong sample에서 매우 큰 negative gap을 만들어 validation loss와 overconfidence를 키웠다.

따라서 `VER12`는 Airway만 따로 밀어주는 방식 대신 다음 원칙을 적용한다.

- 모든 label에 동일한 구조 제약을 적용한다.
- top/branch support가 evidence gap으로 전달되는 최소 방향성을 보장한다.
- embedding/interaction score는 branch path를 완전히 덮지 못하도록 bounded correction으로 제한한다.
- final combiner는 유지하고 evidence scorer 쪽 병목만 직접 수정한다.

### 핵심 변경 요약

#### 1. Embedding / interaction score bounding

`class_gate.evidence_scorer.score_decomposition.score_bounding`을 추가했다.

```json
"score_bounding": {
  "enabled": true,
  "embedding": {
    "enabled": true,
    "mode": "tanh_bound",
    "bound": 8.0,
    "temperature": 1.0
  },
  "interaction": {
    "enabled": true,
    "mode": "tanh_bound",
    "bound": 6.0,
    "temperature": 1.0
  }
}
```

`class_evidence_logits`는 raw component가 아니라 bounded component를 사용한다.

```text
bounded_embedding = embedding_bound * tanh(raw_embedding / embedding_temperature)
bounded_interaction = interaction_bound * tanh(raw_interaction / interaction_temperature)

class_evidence_logits =
  bounded_embedding
  + branch_scale * branch_support_score
  + interaction_effective_scale * bounded_interaction
```

의도는 `embedding_score_gap`과 `interaction_score_gap`이 branch path를 완전히 덮거나 wrong overconfidence를 폭주시킬 위험을 줄이는 것이다.

#### 2. Top-support direct path 직접화

`branch_direct_score.top_support_direct_path`를 추가해 top support를 monotonic raw/relative path로 계산한다.

```text
top_support_score =
  raw_scale * softplus(top_raw)
  + relative_positive_scale * softplus(top_relative)
  - relative_negative_scale * softplus(-top_relative)
  + small_residual_correction
```

이 변경은 top branch teacher가 살아 있을 때 top path가 gated path mismatch에 덜 무너지도록 만드는 장치다. `raw`와 `relative`는 label-specific 값이 아니라 모든 class에 같은 shared scale을 쓰는 feature family다.

#### 3. Branch path dominance constraint

새 loss `branch_path_dominance_constraint`를 추가했다.

```text
branch_gap = branch_support_score[y] - max(branch_support_score[j != y])
evidence_gap = class_evidence_logits[y] - max(class_evidence_logits[j != y])
loss = support_weight * relu(branch_gap - evidence_gap - allowed_drop)
```

이 loss는 branch support가 이미 true class를 지지하는데 evidence gap이 그 방향을 과도하게 잃는 경우를 막는다. `allowed_drop=0.5`를 둬 branch teacher 오류가 evidence를 과하게 고정하지 않도록 완충한다.

#### 4. Top-branch hardness weighting

`top_branch_margin.hardness_weighting`을 추가했다.

```text
hardness = relu(target_margin[label] - top_branch_margin)
hardness_weight = 1 + gain * clamp(hardness, 0, cap)
top_branch_margin_penalty *= hardness_weight
```

이 설정은 특정 label multiplier가 아니라 margin deficit이 큰 sample을 label-agnostic하게 더 강하게 학습시키는 목적이다.

#### 5. VER11 label-specific 보정 중립화

`VER11`에서 추가했던 Airway-only 보정은 label-agnostic 원칙에 맞게 중립화했다.

- `top_branch_margin.phase_weight_schedule.enabled=false`
- `top_branch_margin.phase_weight_schedule.label_multiplier_by_label`은 모든 label `1.0`
- `top_support_score_margin.label_weight_by_label`은 모든 label `1.0`

### Code Impact

이번 변경은 다음 위치에 영향을 준다.

- `src/models/multiscale_rdt_ast.py`
  - score decomposition의 raw/bounded embedding and interaction score 분리
  - monotonic top-support direct path 추가
  - diagnostics용 score component output field 추가
- `src/utils/config.py`
  - `score_bounding`, `top_support_direct_path`, `branch_path_dominance_constraint`, `top_branch_margin.hardness_weighting` schema 및 validation 추가
- `src/training/trainer.py`
  - branch dominance loss 추가
  - top-branch hardness weighting 적용
  - epoch/history/checkpoint extra state component 확장
- `src/evaluation/diagnostics.py`
  - raw/bounded score gap, top direct component, branch dominance diagnostic field 추가
- `configs/training_CNUH_disease_3classes.json`
  - `experiment.name=CNUH_DISEASE_VER12`
- `configs/training_CNUH_new_test_CNUH_3classes.json`
  - `experiment.name=new_test_CNUH_3classes_ver28`

### Diagnostics Checkpoints

`VER12` 분석에서는 다음 chain을 대표 epoch별로 확인한다.

```text
top_branch_margin
-> top_support_score_gap
-> branch_direct_score_gap
-> branch_support_score_gap
-> raw_embedding_score_gap / bounded_embedding_score_gap
-> raw_interaction_score_gap / bounded_interaction_score_gap
-> class_evidence_gap
-> final_gap
```

성공 기준은 다음이다.

- bounded embedding/interaction gap이 raw gap보다 과도하게 폭주하지 않는다.
- `branch_support_score_gap`이 양수인 sample에서 `class_evidence_gap`이 불필요하게 크게 하락하지 않는다.
- `branch_path_dominance_penalty`가 학습 후반으로 갈수록 줄어든다.
- top-branch hardness가 특정 label multiplier 없이 hard sample에만 선택적으로 켜진다.
- final combiner는 여전히 evidence gap을 크게 뒤집지 않는다.

주의할 점은 score bounding이 너무 강하면 underfit이 생길 수 있다는 것이다. 따라서 `embedding bound=8.0`, `interaction bound=6.0`은 첫 실험값이며, 후속 분석에서 `bounded_embedding_score_gap`과 `bounded_interaction_score_gap`이 너무 작게 묶이는지 확인해야 한다.
