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

## VER12 -> VER13

### 목적

`CNUH_DISEASE_VER13`은 `VER12`에서 추가한 label-agnostic evidence-path 보정이 모든 label의 난이도 차이를 충분히 반영하지 못한 문제를 보완하는 실험이다.

`VER12` diagnostics에서 확인한 병목은 두 가지였다.

- Airway hard sample에서는 `top_branch_margin -> top_support_score -> branch_support_score -> class_evidence_logits` chain이 계속 약해졌다.
- Lung/Normal sample에서는 `class_evidence_logits`, 특히 embedding/interaction path가 과도하게 커져 validation loss를 키웠다.

따라서 `VER13`은 구조 자체는 `VER12`를 유지하되, 다음 원칙을 적용한다.

- Airway branch/top-support hard sample에는 `VER11`에서 효과가 있었던 label-aware 압력을 복원한다.
- `branch_path_dominance_constraint`와 `class_evidence_gap_cap_regularization`은 label별 cap/drop/weight를 지원한다.
- Lung/Normal overconfidence는 positive evidence gap cap과 interaction gap cap으로 제한한다.
- final combiner, bounded zero-mean residual, confidence-aware gate는 유지한다.

### 핵심 변경 요약

#### 1. Airway top-support / top-branch 강화 복원

`top_support_score_margin.label_weight_by_label`은 Airway만 `2.0`으로 복원했다.

```json
"label_weight_by_label": {
  "Normal": 1.0,
  "Lung_Parenchymal": 1.0,
  "Airway": 2.0
}
```

`top_branch_margin.phase_weight_schedule`도 다시 활성화했다. epoch 21부터 31까지 Airway multiplier가 `1.0 -> 2.0`으로 ramp되고, Normal/Lung은 `1.0`으로 유지된다.

이 설정은 branch teacher 자체가 약한 Airway hard sample을 더 일찍 보정하기 위한 것이다.

#### 2. Branch dominance label-aware 확장

`branch_path_dominance_constraint`는 다음 label-aware 값을 사용한다.

```json
"label_weight_by_label": {
  "Normal": 0.75,
  "Lung_Parenchymal": 0.75,
  "Airway": 1.5
},
"allowed_drop_by_label": {
  "Normal": 0.5,
  "Lung_Parenchymal": 0.75,
  "Airway": 0.3
}
```

해석은 다음과 같다.

- Airway는 branch support gap이 evidence gap으로 전달되어야 하는 압력을 더 크게 둔다.
- Airway는 `allowed_drop=0.3`으로 더 엄격하게 둔다.
- Lung은 이미 overconfidence가 강하게 발생하므로 `allowed_drop=0.75`, label multiplier `0.75`로 완화한다.

diagnostics에서는 다음 값을 같이 확인한다.

```text
branch_path_dominance_branch_gap
branch_path_dominance_evidence_gap
branch_path_dominance_allowed_drop_effective
branch_path_dominance_label_multiplier
```

#### 3. Negative evidence gap cap label-aware 확장

`class_evidence_gap_cap_regularization`은 wrong overconfidence, 즉 true class evidence gap이 너무 음수로 내려가는 경우를 제한한다.

`VER13`에서는 Airway를 더 엄격하게 둔다.

```json
"negative_gap_cap_by_label": {
  "Normal": 3.0,
  "Lung_Parenchymal": 3.0,
  "Airway": 2.5
},
"label_weight_by_label": {
  "Normal": 1.0,
  "Lung_Parenchymal": 1.0,
  "Airway": 1.5
}
```

이 설정은 Airway가 wrong class에 강하게 눌리는 sample의 penalty를 더 빨리 키우기 위한 것이다.

#### 4. Positive evidence gap cap 추가

새 loss `class_evidence_positive_gap_cap_regularization`을 추가했다.

```text
penalty = relu(evidence_gap - positive_gap_cap)
```

첫 실험값은 다음이다.

```json
"weight": 0.01,
"positive_gap_cap": 8.0,
"class_weighted": false
```

이 loss는 맞은 sample에서도 evidence gap이 지나치게 커져 validation loss와 calibration을 악화시키는 현상을 줄이는 보조 regularizer다.

#### 5. Interaction gap cap 추가

새 loss `interaction_gap_cap_regularization`을 추가했다.

```text
penalty = relu(abs(interaction_gap) - gap_cap)
```

첫 실험값은 다음이다.

```json
"weight": 0.01,
"gap_cap": 6.0,
"class_weighted": false
```

이 loss는 `interaction_score`가 branch support를 안정적으로 보정하기보다 이미 생긴 방향을 과도하게 증폭하는 경우를 제한한다.

### Code Impact

이번 변경은 다음 위치에 영향을 준다.

- `src/utils/config.py`
  - label-aware cap/drop/weight mapping validation
  - `class_evidence_positive_gap_cap_regularization` schema
  - `interaction_gap_cap_regularization` schema
- `src/cli/training.py`, `src/cli/cv.py`
  - label name mapping을 class-index tuple로 resolve
- `src/training/trainer.py`
  - label-aware negative gap cap
  - label-aware branch dominance
  - positive gap cap loss
  - interaction gap cap loss
  - diagnostics/history/checkpoint extra state 확장
- `src/training/epoch_logging.py`
  - `positive_gap_cap`, `interaction_gap_cap` epoch summary 출력
- `configs/training_CNUH_disease_3classes.json`
  - `experiment.name=CNUH_DISEASE_VER13`
- `configs/training_CNUH_new_test_CNUH_3classes.json`
  - `experiment.name=new_test_CNUH_3classes_ver29`

### Diagnostics Checkpoints

`VER13` 분석에서는 다음 chain을 label별로 확인한다.

```text
top_branch_margin
-> top_support_score_gap
-> branch_direct_score_gap
-> branch_support_score_gap
-> branch_path_dominance_allowed_drop_effective
-> class_evidence_gap
-> class_evidence_gap_cap_effective_cap
-> class_evidence_positive_gap_cap_penalty
-> interaction_gap_cap_penalty
-> final_gap
```

성공 기준은 다음이다.

- Airway hard sample에서 `top_support_score_gap`과 `branch_support_score_gap`이 덜 음수로 내려간다.
- Airway error sample에서 `branch_path_dominance_penalty`와 `class_evidence_gap_cap_penalty`가 실제로 켜진다.
- Lung/Normal correct sample에서 `class_evidence_positive_gap_cap_penalty`가 extreme evidence gap만 제한한다.
- `interaction_gap_cap_penalty`가 interaction path의 과도한 증폭 sample에 선택적으로 켜진다.
- final combiner는 여전히 evidence gap을 크게 뒤집지 않는다.

주의할 점은 label-aware 압력을 되살렸기 때문에 Airway recall은 좋아질 수 있지만 Normal false positive가 증가할 수 있다는 것이다. 따라서 `macro_f1`뿐 아니라 Normal specificity와 Airway recall을 함께 확인해야 한다.

## VER13 -> VER14

### 목적

`CNUH_DISEASE_VER14`는 `VER13`에서 확인된 남은 병목을 더 직접적으로 보정하는 실험이다.

`VER13`의 핵심 관찰은 loss pressure가 부족한 것이 아니라, `top_branch_margin`이 true class를 지지하는 sample에서도 `top_support_score_gap`이 음수로 뒤집히는 경우가 남는다는 점이었다. 특히 Airway hard sample에서는 다음 chain이 계속 병목으로 남았다.

```text
top_branch_margin
-> top_support_score_gap
-> branch_direct_score_gap
-> branch_support_score_gap
-> class_evidence_gap
-> final_gap
```

즉, `top_support_score_margin`, `branch_path_dominance_constraint`, `class_evidence_gap_cap_regularization`이 실제로 켜져도, `top_support_score` 자체가 branch teacher의 sign을 충분히 보존하지 못하면 evidence path는 여전히 Normal 또는 다른 negative class로 기울 수 있다.

따라서 `VER14`는 다음 세 가지를 결정 사항으로 둔다.

- `top_support_score`를 learned correction 중심이 아니라 teacher-derived monotonic direct score 중심으로 바꾼다.
- `top_support_score_margin`보다 더 직접적으로 `top_support_gap`의 sign 또는 최소 gap을 강제한다.
- embedding/interaction path는 단순 cap이 아니라 branch-support disagreement가 있을 때만 강하게 제한한다.

### 핵심 변경 방향

#### 1. Teacher-derived monotonic top support 강화

`top_support_score`는 `top_raw`와 `top_relative`가 만드는 branch teacher 신호를 primary path로 사용한다.

적용 수식은 다음과 같다.

```text
direct_top_score[c] =
  raw_scale * top_raw[c]
  + relative_positive_scale * softplus(top_relative[c])
  - relative_negative_scale * softplus(-top_relative[c])
  + class_bias[c]

top_support_score[c] =
  direct_top_score[c]
  + top_support_residual_scale * bounded_top_support_residual[c]
```

해석은 다음과 같다.

- `top_raw[c]`가 높으면 해당 class의 top support가 구조적으로 커져야 한다.
- `top_relative[c] > 0`이면 해당 class가 다른 class보다 top-branch support에서 우세하므로 support가 더 커져야 한다.
- `top_relative[c] < 0`이면 해당 class가 경쟁에서 밀리므로 support가 낮아져야 한다.
- learned residual은 작은 보정만 담당하며, direct teacher score의 sign을 쉽게 뒤집지 못해야 한다.

이를 위해 `a`, `b`, `c`는 non-negative parameter로 두고, learned residual은 bounded correction으로 제한한다. `VER13`에서 `top_margin > 0`인데도 `top_support_gap < 0`이 남았기 때문에, `VER14`에서는 top support의 기본 sign을 learned residual이 아니라 branch teacher가 결정하도록 만든다.

구현에서는 direct top score가 tanh 변환 후 feature가 아니라 raw `class_top_branch_margin_features`와 raw relative top support를 사용한다. 반면 bounded residual correction, gated support, class-axis attention은 기존 transformed branch feature path를 유지한다.

#### 2. `top_support_gap` sign 직접 강제

기존 `top_support_score_margin`은 softplus true-vs-hardest-negative ranking이다. 안정적이지만, `top_support_score_margin_penalty`가 큰데도 gap sign이 바뀌지 않는 경우에는 너무 부드럽다.

따라서 `VER14`에서는 support-conditioned 또는 label-aware hinge를 추가한다.

적용 수식은 다음과 같다.

```text
top_support_gap =
  top_support_score[y]
  - max(top_support_score[j != y])

target_min_gap =
  base_min_gap_by_label[y]
  + support_gain * clamp(relu(top_branch_margin[y]), 0, support_cap)

loss =
  relu(target_min_gap - top_support_gap)
```

이 loss의 목적은 `top_branch_margin`이 true class를 충분히 지지하는 sample에서 `top_support_gap`이 최소한 0 또는 label별 최소 gap 이상이 되도록 강제하는 것이다.

중요한 점은 이 변경이 Airway-only hardcoding이 아니라는 것이다. 구조는 모든 label에 공통으로 적용하고, label별 난이도 차이는 `base_min_gap_by_label` 또는 multiplier config로 조절한다.

#### 3. Branch-support disagreement conditioned embedding/interaction cap

`VER13`에서 `class_evidence_positive_gap_cap_regularization`과 `interaction_gap_cap_regularization`을 추가했지만, 첫 실험 weight는 `0.01`이었다. 단순히 weight를 올리면 correct sample의 큰 positive gap까지 눌러 underfit 위험이 있다.

따라서 `VER14`에서는 embedding/interaction path를 항상 제한하지 않고, branch path와 disagreement가 있을 때만 강하게 제한한다.

적용 수식은 다음과 같다.

```text
branch_support_disagreement =
  1 + condition_gain
      * clamp(relu(disagreement_threshold - branch_support_gap), 0, condition_cap)

embedding_cap_loss =
  branch_support_disagreement
  * relu(abs(embedding_gap) - embedding_gap_cap)

interaction_cap_loss =
  branch_support_disagreement
  * relu(abs(interaction_gap) - interaction_gap_cap)
```

또는 더 직접적으로 다음 조건을 사용할 수 있다.

```text
if top_support_gap < threshold or branch_support_gap < threshold:
  penalize large embedding_gap
  penalize large interaction_gap
```

이 방식은 branch support가 이미 좋은 correct sample은 덜 건드리고, branch path가 약하거나 반대 방향인데 embedding/interaction이 큰 확신을 만드는 sample만 제한한다.

### Code Impact

이번 변경은 다음 위치에 영향을 준다.

- `src/utils/config.py`
  - monotonic top-support direct path config 확장
  - `top_support_gap_min_constraint` 또는 equivalent hinge loss schema 추가
  - branch-support-disagreement conditioned cap schema 추가
- `src/models/multiscale_rdt_ast.py`
  - `top_support_score` 계산을 teacher-derived direct score 중심으로 재구성
  - learned residual correction이 direct score를 과도하게 뒤집지 못하도록 bounded residual scale 적용
  - diagnostics용 `direct_top_score`, `top_support_residual_score`, scale/weight field 출력
- `src/training/trainer.py`
  - `top_support_gap` hinge loss 추가
  - embedding/interaction conditioned cap loss 추가
  - 기존 `top_support_score_margin`, `branch_path_dominance_constraint`, `class_evidence_gap_cap_regularization`과의 loss ordering 정리
- `src/evaluation/diagnostics.py`
  - `direct_top_score_gap`
  - `top_support_residual_gap`
  - `top_support_gap_min_target`
  - `top_support_gap_min_penalty`
  - `branch_support_disagreement`
  - `embedding_disagreement_cap_penalty`
  - `interaction_disagreement_cap_penalty`
- `src/training/epoch_logging.py`
  - 새 hinge/cap loss의 epoch raw/loss summary 출력
- `configs/training_CNUH_disease_3classes.json`
  - `experiment.name=CNUH_DISEASE_VER14`
- `configs/training_CNUH_new_test_CNUH_3classes.json`
  - `experiment.name=new_test_CNUH_3classes_ver30`

### Diagnostics Checkpoints

`VER14` 분석에서는 다음 chain을 대표 epoch과 label별로 확인한다.

```text
top_branch_margin
-> direct_top_score_gap
-> top_support_residual_gap
-> top_support_score_gap
-> branch_direct_score_gap
-> branch_support_score_gap
-> branch_support_disagreement
-> bounded_embedding_score_gap
-> bounded_interaction_score_gap
-> class_evidence_gap
-> final_gap
```

성공 기준은 다음이다.

- `top_branch_margin > 0`인 sample에서 `top_support_score_gap < 0`인 비율이 줄어든다.
- `direct_top_score_gap`이 양수인데 learned residual 때문에 `top_support_score_gap`이 음수로 뒤집히는 case가 줄어든다.
- `top_support_gap_min_penalty`가 학습 후반으로 갈수록 줄어든다.
- `branch_support_disagreement`가 큰 sample에서만 embedding/interaction conditioned cap이 선택적으로 켜진다.
- embedding/interaction cap이 correct high-confidence sample 전체를 누르지 않는다.
- final combiner는 여전히 `class_evidence_gap`을 크게 뒤집지 않는다.

### 주의점

`VER14`는 loss weight를 단순히 키우는 실험이 아니다. `VER13`에서 이미 penalty가 커도 `top_support_gap` sign이 해결되지 않는 현상이 확인되었기 때문에, primary 수정은 top-support score 구조 자체에 있다.

또한 label-aware 설정은 유지할 수 있지만, 핵심 구조는 label-agnostic이어야 한다. 즉 Airway만 따로 처리하는 것이 아니라 모든 label에 동일한 teacher-derived top-support path와 disagreement-conditioned cap을 적용하고, label별 난이도는 config multiplier와 min-gap 값으로만 조절한다.

## CNUH_DISEASE_VER14 -> CNUH_DISEASE_VER15

### 목적

`VER14` diagnostics에서 hard FN의 병목이 top-support/evidence 뒤쪽이 아니라 더 앞단의 teacher-relative branch path에서 시작되는 것이 확인됐다.

핵심 관찰은 다음이다.

- `top_branch_margin`이 일부 hard sample에서 약하거나 양수여도, `class_top_branch_margin_features[true] - max(negative)`가 먼저 음수로 무너진다.
- 이 상태에서는 `top_support_score_gap`, `branch_direct_score_gap`, `branch_support_score_gap`, `class_evidence_gap`, `final_gap`이 연쇄적으로 음수로 내려간다.
- `final_gap`은 대부분 `class_evidence_gap`을 따라가므로 final combiner를 더 수정하는 우선순위는 낮다.

따라서 `VER15`는 teacher 자체의 class-relative ranking을 직접 보정하는 실험으로 분리한다.

### 설정 변경

#### 1. Branch/adaptor teacher capacity 증가

```json
"architecture": {
  "hidden_size": 512,
  "adapter_depth": 8,
  "shared_stem_depth": 1
}
```

`hidden_size`와 `adapter_depth`를 키워 branch/adaptor teacher가 hard sample에서 class-relative support를 더 잘 만들 수 있는지 확인한다. `shared_stem_depth`는 유지해 class/branch-specific adaptor capacity 증가를 우선한다.

#### 2. `class_top_branch_relative_margin` 추가

새 loss는 `class_top_branch_margin_features` 자체의 true-vs-hardest-negative gap을 직접 본다.

```text
teacher_gap =
  class_top_branch_margin_features[true]
  - max(class_top_branch_margin_features[negative])

penalty = relu(margin - teacher_gap)
```

실험값은 다음으로 둔다.

```json
"class_top_branch_relative_margin": {
  "enabled": true,
  "weight": 0.05,
  "target": "class_top_branch_margin_features",
  "mode": "true_vs_hardest_negative_hinge",
  "margin": 0.3,
  "class_weighted": true,
  "reduction": "mean",
  "warmup_epochs": 10,
  "support_weighting": {
    "enabled": true,
    "source": "top_branch_margin",
    "mode": "linear",
    "gain": 0.5,
    "cap": 3.0
  },
  "hardness_weighting": {
    "enabled": true,
    "source": "teacher_gap_deficit",
    "mode": "linear",
    "gain": 1.0,
    "cap": 3.0
  }
}
```

`support_weighting`은 true class top branch support가 있는 sample을 더 중요하게 본다.
`hardness_weighting`은 `teacher_gap`이 목표 margin보다 부족한 sample을 hard-FN proxy로 보고 penalty를 키운다.

#### 3. `top_teacher_gap_min_constraint` 추가

`top_support_gap_min_constraint`가 top-support scorer 뒤쪽을 보정했다면, 새 constraint는 teacher 앞단에 직접 최소 gap을 요구한다.

```text
target_min_gap =
  base_min_gap
  + support_gain * clamp(max(class_top_branch_margin_features[true], 0), 0, support_cap)

penalty = relu(target_min_gap - teacher_gap)
```

실험값은 다음이다.

```json
"top_teacher_gap_min_constraint": {
  "enabled": true,
  "weight": 0.05,
  "target": "class_top_branch_margin_features",
  "mode": "support_conditioned_min_gap",
  "base_min_gap": 0.0,
  "support_source": "top_branch_margin",
  "support_gain": 0.5,
  "support_cap": 2.0,
  "class_weighted": true,
  "reduction": "mean",
  "warmup_epochs": 10
}
```

#### 4. Positive/interaction cap 보조 강화

`class_evidence_positive_gap_cap_regularization.weight`와 `interaction_gap_cap_regularization.weight`는 `0.01 -> 0.015`로 올린다.
cap 값은 그대로 둔다.

- `positive_gap_cap=8.0`
- `interaction_gap_cap=6.0`

이 변경은 primary fix가 아니라, teacher-relative 보정 이후 남는 evidence overconfidence를 약하게 정리하기 위한 보조 장치다.

### Code Impact

- `src/utils/config.py`
  - `ClassTopBranchRelativeMarginConfig`
  - `TopTeacherGapMinConstraintConfig`
  - `teacher_gap_deficit` hardness weighting
- `src/training/trainer.py`
  - `_compute_class_top_branch_relative_margin_loss()`
  - `_compute_top_teacher_gap_min_constraint_loss()`
  - epoch aggregation/history/diagnostics component 추가
- `src/training/epoch_logging.py`
  - `top_teacher_rel raw/loss`
  - `top_teacher_min raw/loss`
- `configs/training_CNUH_disease_3classes.json`
  - `experiment.name=CNUH_DISEASE_VER15`
  - `hidden_size=512`, `adapter_depth=8`
- `configs/training_CNUH_new_test_CNUH_3classes.json`
  - `experiment.name=new_test_CNUH_3classes_ver31`

### Diagnostics Checkpoints

`VER15` 분석에서는 다음 chain을 집중적으로 확인한다.

```text
top_branch_margin
-> class_top_branch_relative_gap
-> top_teacher_gap_min_target
-> direct_top_score_gap
-> top_support_score_gap
-> branch_direct_score_gap
-> branch_support_score_gap
-> class_evidence_gap
-> final_gap
```

성공 기준은 다음이다.

- hard FN에서 `class_top_branch_relative_gap < 0` 비율이 줄어든다.
- `class_top_branch_relative_margin_penalty`와 `top_teacher_gap_min_penalty`가 후반으로 갈수록 감소한다.
- `class_top_branch_relative_gap`이 양수인 sample에서 downstream `top_support_score_gap`이 음수로 뒤집히는 빈도가 줄어든다.
- `final_gap`은 계속 `class_evidence_gap`을 크게 뒤집지 않는다.

### 주의점

`VER15`는 downstream support scorer를 더 세게 누르는 실험이 아니라 teacher 앞단을 보정하는 실험이다. 따라서 새 loss가 줄지 않거나 `class_top_branch_relative_gap`이 개선되지 않으면, 다음 병목은 branch/adaptor teacher 구조 또는 데이터 자체의 class-relative signal 한계로 봐야 한다.

## CNUH_DISEASE_VER15 -> CNUH_DISEASE_VER16

### 수정 이유

`VER15` 최종 분석에서는 모델 capacity 자체보다 teacher-relative objective의 힘 배분이 더 직접적인 병목으로 확인됐다. 대표 hard sample에서는 `top_branch_margin_value > 0`으로 true class branch 내부 support가 약하게 살아 있는데도 `class_top_branch_relative_gap < 0`으로 class-wise teacher ranking이 먼저 무너졌고, 이후 `direct_top_score_gap -> top_support_score_gap -> branch_support_score_gap -> class_evidence_gap -> final_gap`이 같은 방향으로 따라갔다.

따라서 `VER16`에서는 capacity 증가는 후순위로 두고, 기존 teacher-relative loss 2개의 역할을 분리한다.

- `top_teacher_gap_min_constraint`: primary correction
- `class_top_branch_relative_margin.hardness_weighting`: auxiliary hard-sample weighting

### Config 변경

#### 1. `top_teacher_gap_min_constraint` 강화

`top_teacher_gap_min_constraint`는 `top_branch_margin_value`가 양수인 sample에서 `class_top_branch_relative_gap`의 최소 목표를 더 강하게 요구한다.

```text
teacher_gap =
  class_top_branch_margin_features[true]
  - max(class_top_branch_margin_features[negative])

support =
  clamp(relu(top_branch_margin_value), 0, support_cap)

target_min_gap =
  base_min_gap + support_gain * support

penalty =
  relu(target_min_gap - teacher_gap)
```

실험값은 다음이다.

```json
"top_teacher_gap_min_constraint": {
  "enabled": true,
  "weight": 0.075,
  "target": "class_top_branch_margin_features",
  "mode": "support_conditioned_min_gap",
  "base_min_gap": 0.0,
  "support_source": "top_branch_margin",
  "support_gain": 0.75,
  "support_cap": 2.0,
  "class_weighted": true,
  "reduction": "mean",
  "warmup_epochs": 10
}
```

별도 weak-positive band multiplier는 추가하지 않는다. `top_branch_margin_value`가 약하게 양수인 sample도 `support_gain=0.75`를 통해 `target_min_gap`을 직접 키우도록 처리한다.

#### 2. `class_top_branch_relative_margin` hardness 완화

`class_top_branch_relative_margin`은 teacher gap이 크게 무너진 sample을 더 보게 하는 보조 loss로 유지한다. outlier 과집중을 줄이기 위해 hardness multiplier는 완화한다.

```json
"hardness_weighting": {
  "enabled": true,
  "source": "teacher_gap_deficit",
  "mode": "linear",
  "gain": 0.5,
  "cap": 2.0
}
```

`weight=0.05`, `margin=0.3`, `support_weighting.gain=0.5`, `support_weighting.cap=3.0`은 유지한다.

### Code Impact

- `configs/training_CNUH_disease_3classes.json`
  - `experiment.name=CNUH_DISEASE_VER16`
  - teacher-relative loss 계수 조정
- `configs/training_CNUH_new_test_CNUH_3classes.json`
  - `experiment.name=new_test_CNUH_3classes_ver32`
  - 동일 설정 반영
- `src/training/trainer.py`
  - `top_teacher_gap_min_constraint.support_source="top_branch_margin"` 의미에 맞게 support 계산을 `_top_branch_support_gap()` 기반으로 통일
  - `top_teacher_gap_min_support_value` diagnostics도 같은 source로 기록

### Diagnostics Checkpoints

`VER16` 분석에서는 다음을 우선 확인한다.

- `top_branch_margin_value > 0`인데 `class_top_branch_relative_gap < 0`인 sample 비율이 줄어드는가?
- `top_teacher_gap_min_target`이 weak-positive support sample에서 충분히 커지는가?
- `top_teacher_gap_min_penalty`와 `class_top_branch_relative_margin_penalty`가 후반으로 갈수록 같이 줄어드는가?
- downstream `top_support_score_gap`, `branch_support_score_gap`, `class_evidence_gap`, `final_gap`이 teacher gap 개선을 따라오는가?

### 주의점

`VER16`은 새 loss를 추가하지 않는다. 같은 teacher-relative objective 안에서 primary loss와 auxiliary hard-sample weighting의 역할을 재조정하는 실험이다.

## CNUH_DISEASE_VER16 -> CNUH_DISEASE_VER17

### 수정 이유

`VER16` 최종 분석에서는 `top_branch_margin_value`가 약하게 양수인 sample에서 `class_top_branch_relative_gap`이 여전히 음수로 무너지는 현상이 반복됐다. 즉 branch 내부 true-class support는 완전히 죽지 않았지만, class-wise top-branch teacher ranking으로 넘어가는 순간 hardest negative가 true class를 이겼고, 이후 `top_support_score_gap -> branch_support_score_gap -> class_evidence_gap -> final_gap`도 같은 방향으로 따라갔다.

`VER17`에서는 model capacity와 final combiner를 유지하고, teacher-front loss 두 개에 weak-positive support band weighting을 추가한다. 이 변경은 label-specific correction이 아니라 `0.2 <= top_branch_margin_value <= 1.0`인 weak-positive teacher sample을 label-agnostic하게 더 강하게 보는 구조 보정이다.

### Config 변경

#### 1. `top_teacher_gap_min_constraint` primary correction 강화

```json
"top_teacher_gap_min_constraint": {
  "enabled": true,
  "weight": 0.1,
  "target": "class_top_branch_margin_features",
  "mode": "support_conditioned_min_gap",
  "base_min_gap": 0.0,
  "support_source": "top_branch_margin",
  "support_gain": 0.75,
  "support_cap": 2.0,
  "weak_positive_support_weighting": {
    "enabled": true,
    "min_support": 0.2,
    "max_support": 1.0,
    "multiplier": 1.5
  },
  "class_weighted": true,
  "reduction": "mean",
  "warmup_epochs": 10
}
```

#### 2. `class_top_branch_relative_margin` auxiliary hard-sample pressure 재조정

```json
"class_top_branch_relative_margin": {
  "enabled": true,
  "weight": 0.05,
  "target": "class_top_branch_margin_features",
  "mode": "true_vs_hardest_negative_hinge",
  "margin": 0.3,
  "class_weighted": true,
  "reduction": "mean",
  "warmup_epochs": 10,
  "support_weighting": {
    "enabled": true,
    "source": "top_branch_margin",
    "mode": "linear",
    "gain": 0.75,
    "cap": 2.0
  },
  "hardness_weighting": {
    "enabled": true,
    "source": "teacher_gap_deficit",
    "mode": "linear",
    "gain": 0.75,
    "cap": 2.0
  },
  "weak_positive_support_weighting": {
    "enabled": true,
    "min_support": 0.2,
    "max_support": 1.0,
    "multiplier": 1.25
  }
}
```

### Code Impact

- `src/utils/config.py`
  - `WeakPositiveSupportWeightingConfig` 추가
  - `class_top_branch_relative_margin` 및 `top_teacher_gap_min_constraint` nested parser/validation 추가
- `src/training/trainer.py`
  - `_weak_positive_support_weights()` helper 추가
  - 두 teacher-front loss에 weak-positive multiplier 적용
  - diagnostics에 `class_top_branch_relative_margin_weak_positive_weight`, `top_teacher_gap_min_weak_positive_weight` 추가
- `configs/training_CNUH_disease_3classes.json`
  - `experiment.name=CNUH_DISEASE_VER17`
- `configs/training_CNUH_new_test_CNUH_3classes.json`
  - `experiment.name=new_test_CNUH_3classes_ver33`

### Diagnostics Checkpoints

`VER17` 분석에서는 다음을 우선 확인한다.

- weak-positive band sample에서 `class_top_branch_relative_margin_weak_positive_eligible=true` 또는 `top_teacher_gap_min_weak_positive_eligible=true`가 정상 기록되는가?
- `top_branch_margin_value`가 `0.2~1.0`인 wrong sample에서 `class_top_branch_relative_gap < 0`의 magnitude가 줄어드는가?
- `top_teacher_gap_min_penalty`가 단순히 커지기만 하지 않고 후반부에 감소하는가?
- downstream `top_support_score_gap`, `branch_support_score_gap`, `class_evidence_gap`, `final_gap`이 teacher gap 개선을 따라오는가?

## CNUH_DISEASE_VER17 -> CNUH_DISEASE_VER18

### 수정 이유

`VER17`은 evidence/final 성능을 개선했지만, diagnostics에서는 hard sample에서 두 가지 문제가 남았다.

1. `top_branch_margin_value`가 약하게 양수인데도 `class_top_branch_relative_gap`이 음수로 무너지는 sample이 계속 남았다.
2. best branch가 존재하지만 true-class gate가 다른 branch를 선택하는 gate mismatch가 다시 커지는 신호가 있었다.

따라서 `VER18`에서는 model capacity, evidence scorer 구조, final combiner는 유지하고, weak-positive teacher gap과 condition-based gate mismatch만 직접 보정한다.

### Config 변경

#### 1. `top_teacher_gap_min_constraint.weak_positive_target_boost`

```json
"weak_positive_target_boost": {
  "enabled": true,
  "min_support": 0.2,
  "max_support": 1.0,
  "boost": 0.3
}
```

`0.2 <= top_branch_margin_value <= 1.0`인 weak-positive sample에서는 target minimum teacher gap을 직접 올린다.

```text
target_min_gap =
  base_min_gap
  + support_gain * clamp(relu(top_branch_margin_value), 0, support_cap)
  + weak_positive_target_boost
```

#### 2. `class_top_branch_relative_margin.weak_positive_margin_boost`

```json
"weak_positive_margin_boost": {
  "enabled": true,
  "min_support": 0.2,
  "max_support": 1.0,
  "boost": 0.2
}
```

`class_top_branch_relative_margin`은 기존 weak-positive multiplier를 유지하되, 같은 band에서 hinge target 자체를 올린다.

```text
effective_margin =
  margin
  + weak_positive_margin_boost

penalty =
  relu(effective_margin - class_top_branch_relative_gap)
  * support_weight
  * hardness_weight
  * weak_positive_weight
```

#### 3. `gate_best_branch_alignment`

```json
"gate_best_branch_alignment": {
  "enabled": true,
  "weight": 0.03,
  "target": "true_class_gate",
  "source": "branch_logits",
  "mode": "weak_positive_best_branch_alignment",
  "margin_mode": "true_vs_hardest_negative",
  "min_best_margin": 0.2,
  "max_best_margin": 1.0,
  "mismatch_margin_drop": 0.5,
  "loss": "negative_log_best_gate",
  "detach_branch_margin": true,
  "class_weighted": true,
  "reduction": "mean",
  "warmup_epochs": 10
}
```

이 loss는 `regret/bad_suppress`를 대체하지 않는다. 역할은 다음과 같이 더 좁다.

```text
eligible =
  min_best_margin <= best_margin <= max_best_margin
  and selected_branch != best_branch
  and selected_margin + mismatch_margin_drop < best_margin

penalty = -log(true_class_gate[best_branch])
```

Branch margin은 detach하고 gate만 업데이트한다.

### Diagnostics Checkpoints

`VER18` 분석에서는 다음 chain을 우선 확인한다.

- weak-positive sample에서 `top_teacher_gap_min_weak_positive_target_boost`와 `class_top_branch_relative_margin_weak_positive_margin_boost`가 기대대로 기록되는가?
- `class_top_branch_relative_margin_effective_target`이 weak-positive sample에서 0.5 수준으로 올라가는가?
- `gate_best_branch_alignment_best_branch`, `gate_best_branch_alignment_selected_branch`, `gate_best_branch_alignment_eligible`가 gate mismatch를 제대로 잡는가?
- `gate_best_branch_alignment_loss`가 감소하면서 `gate_branch_regret`와 `gate_bad_branch_suppression`이 같이 안정되는가?
- `final_gap`은 계속 `class_evidence_gap`을 따라가는가? 그렇다면 final combiner는 유지한다.

### 주의점

`VER18`은 새 구조 변경이 아니라 loss target과 gate mismatch 조건을 더 직접화하는 실험이다. Checkpoint tensor shape는 크게 바뀌지 않지만 loss semantics가 달라지므로 fresh run으로 비교한다.

## CNUH_DISEASE_VER18 -> CNUH_DISEASE_VER19

### 수정 이유

`VER18` 최종 분석에서는 label-agnostic weak-positive 보정만으로는 Airway hard sample의 teacher-relative gap 붕괴가 충분히 해결되지 않았다. 특히 Airway는 `top_branch_margin_value > 0`인데도 `class_top_branch_relative_gap < 0`인 sample이 남았고, 이 경우 downstream `top_support_score_gap -> branch_support_score_gap -> class_evidence_gap -> final_gap`이 같은 방향으로 무너졌다.

동시에 gate path에서는 `gate_best_branch_alignment`의 동작 여부를 synthetic test와 diagnostics로 더 명확히 검증할 필요가 있었다. 다만 Airway의 primary 병목은 gate alignment가 아니라 teacher-relative min gap이므로, gate alignment는 Normal/Lung 중심의 condition-based 보정으로 두고 Airway weight는 낮춘다.

### 결정 사항

#### 1. `top_teacher_gap_min_constraint` label-aware 확장

Airway는 더 높은 base min gap, 더 큰 support gain, 더 큰 weak-positive target boost를 사용한다.

```json
"top_teacher_gap_min_constraint": {
  "weight": 0.1,
  "base_min_gap_by_label": {
    "Normal": 0.0,
    "Lung_Parenchymal": 0.0,
    "Airway": 0.2
  },
  "support_gain_by_label": {
    "Normal": 0.5,
    "Lung_Parenchymal": 0.5,
    "Airway": 1.0
  },
  "weak_positive_target_boost": {
    "boost_by_label": {
      "Normal": 0.2,
      "Lung_Parenchymal": 0.2,
      "Airway": 0.6
    },
    "support_band_by_label": {
      "Airway": {
        "min_support": 0.2,
        "max_support": 1.2
      }
    }
  }
}
```

Effective target:

```text
target_min_gap =
  base_min_gap[label]
  + support_gain[label] * clamp(relu(top_branch_margin_value), 0, support_cap)
  + weak_positive_target_boost[label] * I(label-specific weak-positive band)
```

#### 2. `class_top_branch_relative_margin` label-aware 확장

Airway는 larger margin, larger weak-positive margin boost, stronger hardness weighting을 사용한다.

```json
"class_top_branch_relative_margin": {
  "margin_by_label": {
    "Normal": 0.3,
    "Lung_Parenchymal": 0.3,
    "Airway": 0.5
  },
  "weak_positive_margin_boost": {
    "boost_by_label": {
      "Normal": 0.1,
      "Lung_Parenchymal": 0.2,
      "Airway": 0.4
    },
    "support_band_by_label": {
      "Airway": {
        "min_support": 0.2,
        "max_support": 1.2
      }
    }
  },
  "hardness_weighting_by_label": {
    "Normal": {
      "gain": 0.5,
      "cap": 2.0
    },
    "Lung_Parenchymal": {
      "gain": 0.75,
      "cap": 2.5
    },
    "Airway": {
      "gain": 1.0,
      "cap": 3.0
    }
  }
}
```

Effective margin:

```text
effective_margin =
  margin[label]
  + weak_positive_margin_boost[label] * I(label-specific weak-positive band)
```

#### 3. Airway top-support anchor 복원

`top_support_score_margin.label_weight_by_label.Airway=2.0`과 `top_support_gap_min_constraint.base_min_gap_by_label.Airway=0.2`를 유지한다. Teacher-front 보정이 성공하더라도 top-support scorer가 Airway support를 다시 음수 gap으로 뒤집는지 계속 확인하기 위한 anchor다.

#### 4. `gate_best_branch_alignment` label-aware 보정

`gate_best_branch_alignment`는 class weighting을 끄고 explicit label weight를 사용한다.

```json
"gate_best_branch_alignment": {
  "class_weighted": false,
  "label_weight_by_label": {
    "Normal": 1.0,
    "Lung_Parenchymal": 0.75,
    "Airway": 0.25
  },
  "max_best_margin_by_label": {
    "Airway": 1.2
  }
}
```

이 설정의 목적은 Airway teacher correction과 gate alignment가 서로 과도하게 충돌하지 않도록 하면서, Normal/Lung gate mismatch는 condition-based로 계속 추적하는 것이다.

### Code Impact

- `src/utils/config.py`
  - weak-positive boost에 `boost_by_label`, `support_band_by_label` 추가
  - `class_top_branch_relative_margin.margin_by_label`, `hardness_weighting_by_label` 추가
  - `top_teacher_gap_min_constraint.base_min_gap_by_label`, `support_gain_by_label`, `support_cap_by_label` 추가
  - `gate_best_branch_alignment.label_weight_by_label`, `min_best_margin_by_label`, `max_best_margin_by_label`, `mismatch_margin_drop_by_label` 추가
- `src/cli/training.py`, `src/cli/cv.py`
  - 새 label mapping을 class-index tuple로 resolve
- `src/training/trainer.py`
  - teacher-relative target/margin/hardness/band를 label별로 계산
  - gate alignment threshold/multiplier를 label별로 계산
  - diagnostics에 effective label-specific values 기록
- `configs/training_CNUH_disease_3classes.json`
  - `experiment.name=CNUH_DISEASE_VER19`
- `configs/training_CNUH_new_test_CNUH_3classes.json`
  - `experiment.name=new_test_CNUH_3classes_ver35`

### Diagnostics Checkpoints

`VER19` 분석에서는 다음을 우선 확인한다.

- Airway sample에서 `top_teacher_gap_min_base_min_gap=0.2`, `top_teacher_gap_min_support_gain=1.0`, `top_teacher_gap_min_weak_positive_target_boost=0.6`이 기대대로 기록되는가?
- Airway weak-positive support band가 `0.2~1.2`로 적용되어 `top_branch_margin_value=1.1` sample도 eligible이 되는가?
- `class_top_branch_relative_margin_label_margin`, `class_top_branch_relative_margin_effective_target`, `class_top_branch_relative_margin_hardness_gain`, `class_top_branch_relative_margin_hardness_cap`이 label별 설정을 반영하는가?
- `gate_best_branch_alignment_label_multiplier`, `gate_best_branch_alignment_max_best_margin`, `gate_best_branch_alignment_eligible`가 loss와 같은 조건으로 기록되는가?
- final combiner는 계속 `final_gap ~= class_evidence_gap`인지 확인한다. 이 조건이 유지되면 병목은 final combiner가 아니라 teacher/support/evidence path다.

### 주의점

`VER19`부터는 label-specific correction을 허용한다. 따라서 비교 시 단순 macro metric뿐 아니라 label별 teacher/gate diagnostics를 같이 확인해야 한다. 특히 Airway recall 개선이 Normal false positive 증가로만 나타나는지, 또는 `class_top_branch_relative_gap -> top_support_score_gap -> class_evidence_gap` chain이 실제로 개선되는지를 분리해서 봐야 한다.

## CNUH_DISEASE_VER19 -> CNUH_DISEASE_VER20

### Date

2026-06-15

### Motivation

`VER19` diagnostics에서 Airway hard sample은 gate가 best branch를 거의 선택해도, weak-positive top teacher 구간에서 `class_top_branch_relative_gap`과 `top_support_score_gap`이 크게 음수로 뒤집히는 문제가 남았다. 따라서 `VER20`은 final combiner나 capacity가 아니라 teacher-relative target과 top-support direct path를 더 직접적으로 조정한다.

### Changes

#### 1. Airway teacher-relative target 강화

`class_top_branch_relative_margin`의 Airway target을 강화한다.

```json
"margin_by_label": {
  "Normal": 0.3,
  "Lung_Parenchymal": 0.3,
  "Airway": 0.8
},
"weak_positive_margin_boost": {
  "boost_by_label": {
    "Normal": 0.1,
    "Lung_Parenchymal": 0.2,
    "Airway": 0.8
  }
},
"hardness_weighting_by_label": {
  "Airway": {
    "gain": 1.5,
    "cap": 4.0
  }
}
```

#### 2. `top_teacher_gap_min_constraint` 강화

Airway teacher-front primary correction을 더 강하게 둔다.

```json
"top_teacher_gap_min_constraint": {
  "weight": 0.12,
  "base_min_gap_by_label": {
    "Normal": 0.0,
    "Lung_Parenchymal": 0.0,
    "Airway": 0.4
  },
  "support_gain_by_label": {
    "Normal": 0.5,
    "Lung_Parenchymal": 0.5,
    "Airway": 1.25
  },
  "weak_positive_target_boost": {
    "boost_by_label": {
      "Normal": 0.2,
      "Lung_Parenchymal": 0.2,
      "Airway": 1.0
    }
  }
}
```

#### 3. Top-support negative relative cap

`top_support_direct_path.negative_relative_cap`을 추가한다. 이 항목은 loss가 아니라 구조적 guardrail이다.

```text
uncapped_negative = relative_negative_scale * softplus(-top_relative)
cap_value = max_negative_fraction[label] * relu(raw_positive_component) + negative_cap[label]
capped_negative = min(uncapped_negative, cap_value)
direct_top_score = raw_positive_component + relative_positive_component - capped_negative + class_bias
```

초기 설정은 전체 label에 `max_negative_fraction=0.75`, `negative_cap=1.5`를 적용하고 Airway만 `max_negative_fraction=0.5`, `negative_cap=1.0`으로 더 엄격하게 제한한다.

#### 4. Gate alignment은 보조로 유지

`gate_best_branch_alignment`는 Normal/Lung 보조 loss로 유지하고 Airway 비중은 낮춘다.

```json
"gate_best_branch_alignment": {
  "weight": 0.04,
  "label_weight_by_label": {
    "Normal": 1.5,
    "Lung_Parenchymal": 1.0,
    "Airway": 0.1
  }
}
```

### Config Names

- `configs/training_CNUH_disease_3classes.json`
  - `experiment.name=CNUH_DISEASE_VER20`
- `configs/training_CNUH_new_test_CNUH_3classes.json`
  - `experiment.name=new_test_CNUH_3classes_ver36`

### Diagnostics Checkpoints

`VER20` 분석에서는 다음을 확인한다.

- Airway weak-positive sample에서 `class_top_branch_relative_margin_effective_target`과 `top_teacher_gap_min_effective_target`이 강화된 값을 반영하는가?
- `class_evidence_top_support_relative_negative_component_uncapped`가 큰 sample에서 `class_evidence_top_support_relative_negative_component_capped`가 `class_evidence_top_support_relative_negative_cap_value`로 제한되는가?
- cap 활성 샘플에서 `direct_top_score_gap -> top_support_score_gap -> class_evidence_gap` chain이 실제로 덜 무너지는가?
- `gate_best_branch_alignment`는 Normal/Lung mismatch를 보조하고 Airway teacher correction과 충돌하지 않는가?
- final combiner는 계속 `final_gap ~= class_evidence_gap`인지 확인한다.

## CNUH_DISEASE_VER20 -> CNUH_DISEASE_VER21

### Date

2026-06-16

### Motivation

`VER20` diagnostics에서는 Airway hard sample에서 `top_branch_margin_value > 0`이어도 `class_top_branch_relative_gap < 0`으로 무너지는 병목이 계속 남았다. 특히 true Airway top teacher가 약하게 살아 있어도 hardest negative, 주로 Lung/Normal 쪽 top teacher가 더 커지면 이후 `top_support_score_gap -> branch_support_score_gap -> class_evidence_gap -> final_gap` chain이 그대로 따라 무너졌다.

`top_teacher_gap_min_constraint`는 true-vs-negative gap의 target을 높이는 역할을 유지하지만, 그것만으로는 true top을 올리는 힘과 hardest negative top을 낮추는 힘이 분리되지 않았다. 따라서 `VER21`은 teacher-front stage에서 hardest negative top teacher를 직접 누르는 `hard_negative_top_teacher_suppression`을 추가한다.

### Changes

#### 1. `hard_negative_top_teacher_suppression` 추가

새 loss는 `class_top_branch_margin_features` 단계에서 true class top teacher와 hardest negative top teacher를 비교한다.

```text
true_top = class_top_branch_margin_features[y]
neg_top = max(class_top_branch_margin_features[j != y])
required_gap = base_required_gap[label] + weak_band_boost[label] + moderate_band_boost[label]
loss = relu(neg_top - detach(true_top) + required_gap)
```

기본 설정은 다음과 같다.

```json
"hard_negative_top_teacher_suppression": {
  "enabled": true,
  "weight": 0.05,
  "target": "class_top_branch_margin_features",
  "mode": "detach_true_top_hardest_negative_hinge",
  "support_source": "top_branch_margin",
  "base_required_gap_by_label": {
    "Normal": 0.1,
    "Lung_Parenchymal": 0.1,
    "Airway": 0.3
  },
  "detach_true_top": true,
  "class_weighted": true,
  "reduction": "mean",
  "warmup_epochs": 10
}
```

`detach_true_top=true`이므로 이 loss는 true top을 직접 올리는 대신 hardest negative top을 낮추는 데 집중한다. true top을 올리는 역할은 기존 `top_branch_margin`, `class_top_branch_relative_margin`, `top_teacher_gap_min_constraint`가 계속 맡는다.

#### 2. Weak-positive / moderate-positive support band boost

Airway hard FN은 weak-positive support band뿐 아니라 그보다 약간 큰 support 구간에도 분포하므로 두 band를 둔다.

```json
"weak_positive_band": {
  "enabled": true,
  "min_support": 0.2,
  "max_support": 1.0,
  "boost_by_label": {
    "Normal": 0.1,
    "Lung_Parenchymal": 0.1,
    "Airway": 0.5
  },
  "support_band_by_label": {
    "Airway": {
      "min_support": 0.2,
      "max_support": 1.2
    }
  }
},
"moderate_positive_band": {
  "enabled": true,
  "min_support": 1.2,
  "max_support": 2.0,
  "boost_by_label": {
    "Normal": 0.0,
    "Lung_Parenchymal": 0.0,
    "Airway": 0.2
  }
}
```

### Config Names

- `configs/training_CNUH_disease_3classes.json`
  - `experiment.name=CNUH_DISEASE_VER21`
- `configs/training_CNUH_new_test_CNUH_3classes.json`
  - `experiment.name=new_test_CNUH_3classes_ver37`

### Diagnostics Checkpoints

`VER21` 분석에서는 다음을 확인한다.

- `hard_negative_top_teacher_negative_class`가 Airway FN에서 어떤 label로 집중되는가?
- weak-positive Airway sample에서 `hard_negative_top_teacher_required_gap`이 `base_required_gap + weak_band_boost`를 반영하는가?
- moderate-positive Airway sample에서 `hard_negative_top_teacher_moderate_band_boost`가 활성화되는가?
- `hard_negative_top_teacher_penalty`가 후반으로 갈수록 감소하고, 동시에 `class_top_branch_relative_gap`이 음수로 무너지는 비율이 줄어드는가?
- downstream chain인 `top_support_score_gap -> branch_support_score_gap -> class_evidence_gap -> final_gap`이 teacher-front 개선을 실제로 따라가는가?
- final combiner는 계속 `final_gap ~= class_evidence_gap`인지 확인한다. 이 조건이 유지되면 병목은 final path가 아니라 teacher/support/evidence path다.
