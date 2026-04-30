# Codex Instructions: 4-Class Supervised Pretraining with Weighted CE and Branch Binary Auxiliary

## Target Repository and Branch

Repository:

```text
https://github.com/jjhong0608/RespiratoryClassification.git
```

Base branch:

```text
MIL-RDT-AST
```

This task is based on the latest `MIL-RDT-AST` branch.

The current branch is an event-MIL-first respiratory sound classifier with a single supported encoder type:

```text
multiscale_rdt_ast
```

Do **not** reintroduce Hugging Face `ASTModel`.  
Do **not** reintroduce latent-query pooling.  
Do **not** rewrite the model architecture from scratch.

This task modifies the **training objective, model outputs, loss construction, validation loss semantics, checkpointing, metrics, and canonical 4-class pretraining config**.

---

## Objective

Implement a 4-class supervised pretraining objective:

```text
Main task:
  normal / crackle / wheeze / rhonchi
  4-class single-label classification

Auxiliary task:
  branch-level normal-vs-abnormal binary classification
```

The desired total loss is:

```text
L_total = L_4cls_weighted_CE
        + lambda_branch_bin * mean_s BCE(branch_binary_logit_s, binary_target)
```

where:

```text
lambda_branch_bin = 0.3 by default
```

Do **not** add a global binary auxiliary head.

Do **not** use the old multiclass branch auxiliary loss for this pretraining config.

---

## Key Design Decisions

### 1. Main 4-class head

The main classifier predicts:

```text
normal
crackle
wheeze
rhonchi
```

Output:

```text
logits: [B, 4]
```

Loss:

```text
weighted CrossEntropyLoss
```

Class weights are computed from the **training split only** using:

```text
w_c = (1 / sqrt(n_c)) / mean_j(1 / sqrt(n_j))
```

Equivalent:

```python
raw = 1.0 / torch.sqrt(class_counts.float())
weights = raw / raw.mean()
```

The normalization must make the mean class weight equal to 1.

---

### 2. Branch binary auxiliary

Each branch should produce a normal-vs-abnormal logit.

Output:

```text
branch_binary_logits: [B, num_branches]
```

Target is built using a config-provided binary label map:

```json
"label_to_index": {
  "normal": 0,
  "crackle": 1,
  "wheeze": 1,
  "rhonchi": 1
}
```

Do not hard-code `"normal" != class` unless the config says so.  
Use the binary auxiliary label map from config.

Branch binary loss:

```text
L_branch_bin = mean over branches of BCEWithLogitsLoss(branch_binary_logit_s, binary_target)
```

`pos_weight` is computed from the **training split only** using the binary auxiliary label map:

```text
pos_weight = sqrt(n_0 / n_1)
```

where:

```text
n_0 = number of training examples mapped to binary label 0
n_1 = number of training examples mapped to binary label 1
```

---

### 3. No global binary auxiliary head

Do **not** implement:

```text
pooled_embedding -> global binary aux head
```

The auxiliary task must be branch-level:

```text
branch embedding/logit per branch -> binary auxiliary logits
```

---

### 4. Do not extend the old `branch_auxiliary` config

Do **not** overload or extend the existing config block:

```json
"branch_auxiliary": { ... }
```

Add a new block:

```json
"branch_binary_auxiliary": { ... }
```

For the new 4-class pretraining config, set the old branch auxiliary off:

```json
"branch_auxiliary": {
  "enabled": false
}
```

The old multiclass branch auxiliary must not contribute to the new pretraining loss.

---

### 5. Validation loss semantics

Validation loss must use the **same loss structure** as training:

```text
val_loss = weighted 4-class CE
         + lambda_branch_bin * branch binary BCE
```

But all weights must be derived from the **training split**:

```text
class weights: train split only
binary pos_weight: train split only
```

Do not compute validation class weights from validation labels.

Use the requested config shape:

```json
"val": {
  "loss": {
    "use_train_loss_config": true,
    "class_weight_source": "train",
    "binary_pos_weight_source": "train"
  }
}
```

---

### 6. No early stopping

For pretraining, disable early stopping:

```json
"early_stopping": {
  "enabled": false
}
```

Use long fixed training and save multiple checkpoint candidates.

---

### 7. Multi-monitor top-k checkpointing

Save:

```text
top 3 best_macro_f1 checkpoints
top 3 best_macro_recall checkpoints
top 3 best_loss checkpoints
last 3 checkpoints
```

Definitions:

```text
best_macro_f1:
  4-class validation macro F1, mode=max

best_macro_recall:
  4-class validation macro recall, mode=max

best_loss:
  total validation loss, mode=min
  total validation loss = weighted CE + lambda * branch binary BCE

last:
  latest 3 epochs
```

---

## Current Architecture to Preserve

Keep the current event-MIL architecture:

```text
input_values: [B, 1024, 128]
  -> unsqueeze channel
[B, 1, 1024, 128]
  -> four patch tokenizers
  -> learned patch position + scale embeddings
  -> shared branch-wise transformer stem
  -> scale-specific transformer adapters
  -> reshape each branch to [B, T_s, F_s, D]
  -> learned frequency-attention pooling
  -> branch event tokens
  -> branch MIL attention heads
  -> top-k evidence tokens per branch
U0: [B, 8, D], H_ctx: [B, 1916, D]
  -> optional recurrent RDT refinement over U using H_ctx
  -> evidence pooling
  -> fuse evidence embedding, mean branch embedding, and branch logits
  -> fusion projector
[B, D]
  -> classifier
```

Do not change:

```text
patch geometry
RDT block
RDT settings
evidence pooling implementation
final fusion inputs
classifier output shape
data preprocessing
augmentation behavior
```

Only add/modify what is necessary for the new 4-class pretraining objective.

---

# Part A — Add / Update Config Schema

Modify the config parser/schema, likely in:

```text
src/utils/config.py
```

The exact class names may differ. Adapt to the current codebase.

---

## A1. Add class weighting config

Add under:

```text
train.loss.class_weighting
```

Recommended dataclass:

```python
@dataclass(frozen=True)
class ClassWeightingConfig:
    enabled: bool = False
    type: Literal["none", "sqrt_inverse_frequency"] = "none"
    normalize: Literal["none", "mean_one"] = "mean_one"
    source: Literal["train"] = "train"
```

Validation:

```text
if enabled:
  type must be "sqrt_inverse_frequency"
  normalize must be "mean_one" or supported value
  source must be "train"
```

For this task, only `sqrt_inverse_frequency` is required.

---

## A2. Add branch binary auxiliary config

Add a new config block:

```text
train.loss.branch_binary_auxiliary
```

Do **not** extend old `branch_auxiliary`.

Recommended dataclasses:

```python
@dataclass(frozen=True)
class BinaryAuxiliaryPosWeightConfig:
    enabled: bool = True
    type: Literal["sqrt_normal_over_abnormal"] = "sqrt_normal_over_abnormal"
    source: Literal["train"] = "train"


@dataclass(frozen=True)
class BranchBinaryAuxiliaryConfig:
    enabled: bool = False
    weight: float = 0.3
    label_to_index: Mapping[str, int] = field(default_factory=dict)
    pos_weight: BinaryAuxiliaryPosWeightConfig = field(
        default_factory=BinaryAuxiliaryPosWeightConfig
    )
    aggregation: Literal["mean"] = "mean"
```

Validation:

```text
if branch_binary_auxiliary.enabled:
  weight > 0
  aggregation == "mean"
  label_to_index must not be empty
  label_to_index values must be only 0 or 1
  both binary labels 0 and 1 must appear
  label_to_index keys must match the main 4-class label names
  pos_weight.type == "sqrt_normal_over_abnormal"
  pos_weight.source == "train"
```

### Binary label map validation

Valid:

```json
{
  "normal": 0,
  "crackle": 1,
  "wheeze": 1,
  "rhonchi": 1
}
```

Invalid examples:

```json
{
  "normal": 0,
  "crackle": 2,
  "wheeze": 1,
  "rhonchi": 1
}
```

```json
{
  "normal": 0,
  "crackle": 0,
  "wheeze": 0,
  "rhonchi": 0
}
```

```json
{
  "normal": 1,
  "crackle": 1,
  "wheeze": 1,
  "rhonchi": 1
}
```

If main labels include exactly:

```text
normal, crackle, wheeze, rhonchi
```

then the binary auxiliary label map must include exactly those keys.

If the codebase supports arbitrary class names, implement this generically:

```text
set(branch_binary_auxiliary.label_to_index.keys()) == set(main_label_to_index.keys())
```

---

## A3. Add validation loss config

Add:

```text
val.loss
```

Recommended dataclass:

```python
@dataclass(frozen=True)
class ValidationLossConfig:
    use_train_loss_config: bool = True
    class_weight_source: Literal["train"] = "train"
    binary_pos_weight_source: Literal["train"] = "train"
```

Validation:

```text
use_train_loss_config must be true for this pretraining setup
class_weight_source must be "train"
binary_pos_weight_source must be "train"
```

If the current config system has no top-level `"val"` section, add one without breaking existing configs. Missing `val.loss` should default to:

```text
use_train_loss_config = true
class_weight_source = train
binary_pos_weight_source = train
```

---

## A4. Checkpoint monitor config

Add or extend checkpointing config.

Recommended:

```json
"checkpointing": {
  "monitors": [
    {
      "name": "val_macro_f1",
      "mode": "max",
      "keep_top_k": 3,
      "filename_prefix": "best_macro_f1"
    },
    {
      "name": "val_macro_recall",
      "mode": "max",
      "keep_top_k": 3,
      "filename_prefix": "best_macro_recall"
    },
    {
      "name": "val_loss",
      "mode": "min",
      "keep_top_k": 3,
      "filename_prefix": "best_loss"
    },
    {
      "name": "last",
      "mode": "last",
      "keep_top_k": 3,
      "filename_prefix": "last"
    }
  ]
}
```

If the existing checkpoint manager uses a different schema, adapt while preserving the required semantics.

Existing configs should continue to work.

---

# Part B — Create Train-Derived Loss Weights

Implement a setup step that computes and stores loss weights from the training split.

Likely files:

```text
src/training/ast_setup.py
src/training/trainer.py
src/data/loaders.py
```

Exact location depends on current code.

---

## B1. Compute 4-class weights from train labels

Use training dataset class counts.

Pseudo-code:

```python
def compute_sqrt_inverse_class_weights(
    class_counts: Tensor,
    *,
    normalize: str = "mean_one",
) -> Tensor:
    counts = class_counts.float().clamp_min(1.0)
    raw = 1.0 / torch.sqrt(counts)
    if normalize == "mean_one":
        raw = raw / raw.mean()
    return raw
```

Requirements:

```text
Use train split labels only.
Order must match main class index order.
Device must be moved to training device before loss computation.
Log class counts and weights.
Store class weights in checkpoint metadata if practical.
```

---

## B2. Compute binary auxiliary mapping

Build a tensor mapping main class index to binary target.

Pseudo-code:

```python
def build_main_index_to_binary_target(
    *,
    main_label_to_index: Mapping[str, int],
    binary_label_to_index: Mapping[str, int],
) -> Tensor:
    num_classes = len(main_label_to_index)
    mapping = torch.empty(num_classes, dtype=torch.long)

    for label, main_idx in main_label_to_index.items():
        mapping[main_idx] = int(binary_label_to_index[label])

    return mapping
```

Example:

```text
main_label_to_index:
  normal: 0
  crackle: 1
  wheeze: 2
  rhonchi: 3

binary_label_to_index:
  normal: 0
  crackle: 1
  wheeze: 1
  rhonchi: 1

main_index_to_binary_target:
  [0, 1, 1, 1]
```

Trainer usage:

```python
binary_target = main_index_to_binary_target[targets].to(device).float()
```

---

## B3. Compute binary pos_weight from train labels

Use the binary mapping and train class counts.

Pseudo-code:

```python
binary_counts = torch.zeros(2, dtype=torch.long)
for class_idx, count in enumerate(class_counts):
    binary_idx = main_index_to_binary_target[class_idx]
    binary_counts[binary_idx] += count

n0 = binary_counts[0].float()
n1 = binary_counts[1].float()
pos_weight = torch.sqrt(n0 / n1.clamp_min(1.0))
```

Requirements:

```text
Use train split only.
Do not compute pos_weight from validation labels.
Log binary counts and pos_weight.
Store in checkpoint metadata if practical.
```

---

# Part C — Modify Model Output and Branch Binary Head

Likely file:

```text
src/models/multiscale_rdt_ast.py
```

---

## C1. Add branch binary logits output

Extend the model output dataclass.

Current output likely includes fields like:

```text
logits
pooled_embedding
branch_logits
branch_attention_weights
selected_evidence_tokens
selected_evidence_indices
selected_evidence_scores
selected_evidence_branch_ids
...
```

Add:

```python
branch_binary_logits: Tensor | None = None
```

Expected shape:

```text
[B, num_branches]
```

---

## C2. Add branch binary head

Add branch-level binary auxiliary heads.

Recommended approach:

Use branch embeddings already produced by branch MIL heads.

If branch embeddings are available as:

```text
branch_embeddings: [B, S, D]
```

then add:

```python
self.branch_binary_head = nn.Linear(hidden_size, 1)
```

applied to each branch embedding:

```python
branch_binary_logits = self.branch_binary_head(branch_embeddings).squeeze(-1)
# [B, S]
```

This can be a shared binary head across branches.

If current code has separate branch modules and branch embeddings are only in a list:

```python
branch_binary_logits = []
for branch_embedding in branch_embeddings:
    logit = self.branch_binary_head(branch_embedding).squeeze(-1)
    branch_binary_logits.append(logit)
branch_binary_logits = torch.stack(branch_binary_logits, dim=1)
```

Do not reuse final 4-class branch logits as binary logits.

Do not use global pooled embedding.

Do not add global binary auxiliary head.

---

## C3. Keep existing branch_logits

Do not remove existing `branch_logits` if they are still used by final fusion or diagnostics.

For this new pretraining config, old branch auxiliary loss will be disabled, but branch logits may still be used by the fusion projector.

Keep:

```text
branch_logits
```

Add:

```text
branch_binary_logits
```

They have different meanings.

---

# Part D — Modify Trainer Loss Computation

Likely file:

```text
src/training/trainer.py
```

---

## D1. Loss components

Trainer must compute:

```text
loss_4cls = CrossEntropyLoss(weight=class_weights)(output.logits, targets)

loss_branch_binary = mean over branches of BCEWithLogitsLoss(pos_weight=...)(output.branch_binary_logits[:, s], binary_target)

total_loss = loss_4cls + lambda_branch_bin * loss_branch_binary
```

Only compute branch binary loss when:

```text
cfg.train.loss.branch_binary_auxiliary.enabled == true
```

If enabled but `output.branch_binary_logits is None`, raise a clear error.

Do not compute old branch auxiliary loss when `branch_auxiliary.enabled == false`.

---

## D2. Pseudo-code

```python
def compute_loss(output: AstModelOutput, targets: Tensor) -> LossOutput:
    loss_4cls = ce_criterion(output.logits, targets)
    total = loss_4cls

    loss_branch_binary = None
    if cfg.train.loss.branch_binary_auxiliary.enabled:
        if output.branch_binary_logits is None:
            raise ValueError("branch_binary_auxiliary enabled but model output lacks branch_binary_logits")

        binary_target = main_index_to_binary_target[targets].to(
            output.branch_binary_logits.device
        ).float()

        # branch_binary_logits: [B, S]
        # binary_target: [B]
        expanded_target = binary_target.unsqueeze(1).expand_as(output.branch_binary_logits)

        loss_branch_binary_raw = bce_criterion(
            output.branch_binary_logits,
            expanded_target,
        )
        # If BCE criterion reduction='none', reduce over batch and branches.
        loss_branch_binary = loss_branch_binary_raw.mean()

        total = total + cfg.train.loss.branch_binary_auxiliary.weight * loss_branch_binary

    return LossOutput(
        total=total,
        main_4cls=loss_4cls,
        branch_binary=loss_branch_binary,
    )
```

Whether you use `BCEWithLogitsLoss(reduction="mean")` directly or `reduction="none"` with manual reduction is up to the existing trainer style. Ensure it equals mean over batch and branches.

---

## D3. Training and validation must share criterion weights

The same `class_weights` and `binary_pos_weight` computed from the training split must be used for both:

```text
train epoch loss
validation epoch loss
```

Validation must not recompute weights.

---

## D4. Logging loss components

Log separately:

```text
train_loss
train_loss_4cls
train_loss_branch_binary

val_loss
val_loss_4cls
val_loss_branch_binary
```

Where:

```text
train_loss / val_loss = total loss
```

---

# Part E — Metrics

Update metric computation and/or logging.

Likely files:

```text
src/evaluation/metrics.py
src/training/trainer.py
src/evaluation/diagnostics.py
```

Exact file names may differ.

---

## E1. 4-class metrics

Ensure validation logs contain:

```text
accuracy
macro_precision
macro_recall
macro_f1
weighted_f1
per_class_precision
per_class_recall
per_class_f1
confusion_matrix
one-vs-rest ROC-AUC
one-vs-rest PR-AUC
```

If the current multiclass metric code already computes these, preserve it.

For checkpointing, ensure the following keys exist in epoch metrics:

```text
val_macro_f1
val_macro_recall
val_loss
```

or equivalent names. If names differ, either adapt checkpoint config names or standardize names.

---

## E2. Branch binary auxiliary metrics

When `branch_binary_logits` is present, compute binary metrics.

Per branch:

```text
branch_binary_auc_{s}
branch_binary_ap_{s}
branch_binary_f1_{s}
branch_binary_brier_{s}
```

Aggregate:

```text
branch_binary_auc_mean
branch_binary_ap_mean
branch_binary_f1_mean
branch_binary_brier_mean
```

Binary target is the same config-derived target:

```python
binary_target = main_index_to_binary_target[targets]
```

For F1, use threshold 0.5 on sigmoid probabilities.

If a metric cannot be computed due to only one class being present in a validation split, return `nan` and log a warning rather than crashing.

---

## E3. Diagnostics

If diagnostics JSONL is produced, add optional fields:

```text
branch_binary_logits
branch_binary_probabilities
binary_auxiliary_target
```

Do not remove existing diagnostics.

---

# Part F — Checkpointing

Modify checkpoint manager to support top-3 multi-monitor checkpointing.

Likely file:

```text
src/utils/checkpoint.py
```

or current checkpoint manager location.

---

## F1. Required monitors

Support:

```text
best_macro_f1:
  metric key: val_macro_f1
  mode: max
  keep_top_k: 3

best_macro_recall:
  metric key: val_macro_recall
  mode: max
  keep_top_k: 3

best_loss:
  metric key: val_loss
  mode: min
  keep_top_k: 3

last:
  mode: last
  keep_top_k: 3
```

---

## F2. File naming

Recommended file names:

```text
best_macro_f1_rank1_0.538691_epoch063.pt
best_macro_f1_rank2_0.534113_epoch059.pt
best_macro_f1_rank3_0.530417_epoch055.pt

best_macro_recall_rank1_0.519300_epoch063.pt
...

best_loss_rank1_0.890722_epoch036.pt
...

last_epoch118.pt
last_epoch119.pt
last_epoch120.pt
```

Exact formatting can follow the existing code style, but monitor name, rank, score, and epoch should be visible.

---

## F3. Last checkpoints

Keep only the latest 3 last checkpoints.

When epoch 121 is saved, remove the oldest last checkpoint beyond top 3.

---

## F4. Backward compatibility

Existing binary configs that use current `best_loss`, `best_f1`, and `last` should not break.

If the new monitor config is missing, use existing behavior.

---

# Part G — Add Canonical Config

Create a new config.

Recommended filename:

```text
configs/training_4class_pretrain_weighted_ce_branch_bin_aux.json
```

Use an existing working multiclass config as the base.

Do not invent dataset paths. Copy paths from the existing multiclass config.

---

## G1. Required key settings

Main task remains 4-class:

```json
"task": "normal_vs_crackle_vs_wheeze_vs_rhonchi",
"label_to_index": {
  "normal": 0,
  "crackle": 1,
  "wheeze": 2,
  "rhonchi": 3
}
```

Loss:

```json
"train": {
  "epochs": 120,
  "loss": {
    "type": "cross_entropy",
    "class_weighting": {
      "enabled": true,
      "type": "sqrt_inverse_frequency",
      "normalize": "mean_one",
      "source": "train"
    },
    "branch_auxiliary": {
      "enabled": false
    },
    "branch_binary_auxiliary": {
      "enabled": true,
      "weight": 0.3,
      "label_to_index": {
        "normal": 0,
        "crackle": 1,
        "wheeze": 1,
        "rhonchi": 1
      },
      "pos_weight": {
        "enabled": true,
        "type": "sqrt_normal_over_abnormal",
        "source": "train"
      },
      "aggregation": "mean"
    }
  },
  "early_stopping": {
    "enabled": false
  }
}
```

Validation loss:

```json
"val": {
  "loss": {
    "use_train_loss_config": true,
    "class_weight_source": "train",
    "binary_pos_weight_source": "train"
  }
}
```

Checkpointing:

```json
"checkpointing": {
  "monitors": [
    {
      "name": "val_macro_f1",
      "mode": "max",
      "keep_top_k": 3,
      "filename_prefix": "best_macro_f1"
    },
    {
      "name": "val_macro_recall",
      "mode": "max",
      "keep_top_k": 3,
      "filename_prefix": "best_macro_recall"
    },
    {
      "name": "val_loss",
      "mode": "min",
      "keep_top_k": 3,
      "filename_prefix": "best_loss"
    },
    {
      "name": "last",
      "mode": "last",
      "keep_top_k": 3,
      "filename_prefix": "last"
    }
  ]
}
```

Model:

Use the current best H0/H0-gated pretraining model settings unless existing multiclass config already defines them.

Key requirement:

```text
Do not use old multiclass branch_auxiliary.
Use branch_binary_auxiliary instead.
```

---

# Part H — README Updates

Update README with a section:

```text
4-Class Supervised Pretraining
```

Document:

```text
Main objective:
  weighted 4-class CE

Class weights:
  sqrt inverse frequency, mean-one normalized, computed from train split

Branch auxiliary objective:
  branch-level normal-vs-abnormal BCE

Binary auxiliary label map:
  normal -> 0
  crackle/wheeze/rhonchi -> 1

Validation loss:
  same as train loss, using train-derived weights

Early stopping:
  disabled for pretraining

Checkpoints:
  top-3 best_macro_f1
  top-3 best_macro_recall
  top-3 best_loss
  last-3
```

Clarify:

```text
This is not multi-label classification.
This is 4-class single-label supervised pretraining with a branch-level binary auxiliary loss.
```

Also clarify:

```text
The old multiclass branch_auxiliary is disabled in this config.
branch_binary_auxiliary is a separate config block and should not be confused with branch_auxiliary.
```

---

# Part I — Tests

Add or update tests.

## I1. Config validation tests

Valid config:

```text
training_4class_pretrain_weighted_ce_branch_bin_aux.json
```

Invalid binary auxiliary label maps:

```text
value not in {0,1}
all labels mapped to 0
all labels mapped to 1
missing label
extra unknown label
```

Invalid class weighting:

```text
unsupported type
normalize invalid
source != train
```

Invalid pos weight:

```text
unsupported type
source != train
```

---

## I2. Weight computation tests

Use synthetic class counts.

Test:

```python
counts = torch.tensor([3108, 554, 514, 235])
weights = compute_sqrt_inverse_class_weights(counts)
assert torch.allclose(weights.mean(), torch.tensor(1.0))
```

Test known approximate behavior:

```text
normal weight < crackle weight ~ wheeze weight < rhonchi weight
```

Test binary pos_weight:

```python
normal = 3108
abnormal = 554 + 514 + 235
expected = sqrt(normal / abnormal)
```

---

## I3. Main-to-binary target mapping tests

Given:

```python
main_label_to_index = {"normal": 0, "crackle": 1, "wheeze": 2, "rhonchi": 3}
binary_label_to_index = {"normal": 0, "crackle": 1, "wheeze": 1, "rhonchi": 1}
```

Expect:

```python
main_index_to_binary_target = [0, 1, 1, 1]
```

Given targets:

```python
targets = [0, 1, 2, 3]
```

Expect:

```python
binary_target = [0, 1, 1, 1]
```

---

## I4. Model output tests

For multiclass model:

```text
output.logits.shape == [B, 4]
output.branch_binary_logits.shape == [B, num_branches]
```

Do not require global binary auxiliary output.

---

## I5. Loss computation tests

Test:

```text
total_loss = weighted_ce + lambda * branch_binary_bce
```

Test that old branch auxiliary is not included when disabled.

Test that enabling branch_binary_auxiliary without model output `branch_binary_logits` raises a clear error.

---

## I6. Validation weight reuse tests

Ensure validation loss uses the training-derived class weights and binary pos_weight.

Do not recompute from validation labels.

---

## I7. Checkpoint tests

Test top-3 saving behavior:

```text
best_macro_f1 keeps top 3
best_macro_recall keeps top 3
best_loss keeps top 3
last keeps last 3
```

Test old checkpoint behavior remains valid if monitor config is absent.

---

## I8. Metrics tests

Test multiclass metric keys include:

```text
macro_f1
macro_recall
```

Test branch binary metric computation with valid binary targets.

Handle one-class metric edge cases gracefully.

---

# Part J — Acceptance Criteria

The task is complete when:

1. `class_weighting` config is supported.
2. `branch_binary_auxiliary` config is supported.
3. Old `branch_auxiliary` is not extended for binary auxiliary.
4. Binary auxiliary label map is validated.
5. `main_index_to_binary_target` mapping is built.
6. Class weights use sqrt inverse frequency from train split.
7. Class weights are mean-one normalized.
8. Binary pos_weight uses sqrt(n0/n1) from train split.
9. Validation loss uses train-derived weights.
10. Model returns `branch_binary_logits`.
11. No global binary auxiliary head is added.
12. Total loss equals weighted 4-class CE + lambda * branch binary BCE.
13. Existing old branch auxiliary is disabled in the new pretraining config.
14. Early stopping is disabled in the new pretraining config.
15. Top-3 checkpointing works for macro F1, macro recall, loss, and last.
16. Multiclass and branch-binary metrics are logged.
17. README documents the new pretraining objective.
18. Tests pass.

---

# Part K — Avoid These Mistakes

- Do not treat this as multi-label classification.
- Do not change the main 4-class label map.
- Do not hard-code normal-vs-abnormal mapping; use config-provided binary map.
- Do not allow binary auxiliary label values outside `{0, 1}`.
- Do not compute validation weights from validation labels.
- Do not add a global binary auxiliary head.
- Do not extend old `branch_auxiliary` for this purpose.
- Do not compute old multiclass branch auxiliary in the new config.
- Do not remove existing `branch_logits` if final fusion still uses them.
- Do not confuse `branch_logits` with `branch_binary_logits`.
- Do not enable early stopping in the new pretraining config.
- Do not save only one best checkpoint per monitor.
- Do not break existing binary or multiclass configs.

---

# Part L — Suggested Implementation Pseudo-Code

## L1. Build weights

```python
class_counts = get_train_class_counts(train_dataset, num_classes=4)

if cfg.train.loss.class_weighting.enabled:
    class_weights = compute_sqrt_inverse_class_weights(class_counts)
else:
    class_weights = None

main_to_binary = build_main_index_to_binary_target(
    main_label_to_index=cfg.data.label_to_index,
    binary_label_to_index=cfg.train.loss.branch_binary_auxiliary.label_to_index,
)

if cfg.train.loss.branch_binary_auxiliary.pos_weight.enabled:
    binary_counts = torch.zeros(2)
    for class_idx, count in enumerate(class_counts):
        binary_idx = main_to_binary[class_idx]
        binary_counts[binary_idx] += count

    pos_weight = torch.sqrt(binary_counts[0] / binary_counts[1].clamp_min(1.0))
else:
    pos_weight = None
```

---

## L2. Model output

```python
branch_binary_logits = self.branch_binary_head(branch_embeddings).squeeze(-1)
# [B, S]

return AstModelOutput(
    logits=logits,  # [B, 4]
    pooled_embedding=pooled_embedding,
    branch_logits=branch_logits,
    branch_binary_logits=branch_binary_logits,
    ...
)
```

---

## L3. Loss

```python
loss_4cls = ce_loss(output.logits, targets)
total_loss = loss_4cls

loss_branch_binary = None
if cfg.train.loss.branch_binary_auxiliary.enabled:
    if output.branch_binary_logits is None:
        raise ValueError("branch_binary_auxiliary enabled but model output lacks branch_binary_logits")

    binary_target = main_to_binary[targets].to(output.branch_binary_logits.device).float()
    expanded_target = binary_target.unsqueeze(1).expand_as(output.branch_binary_logits)

    loss_branch_binary = bce_loss(output.branch_binary_logits, expanded_target)
    total_loss = total_loss + cfg.train.loss.branch_binary_auxiliary.weight * loss_branch_binary
```

---

## L4. Metrics

```python
# 4-class
metrics.update(multiclass_metrics(output.logits, targets))

# branch binary
if output.branch_binary_logits is not None:
    binary_target = main_to_binary[targets].float()
    branch_probs = output.branch_binary_logits.sigmoid()
    metrics.update(branch_binary_metrics(branch_probs, binary_target))
```

---

# Final Note

The intended pretraining setup is:

```text
4-class supervised pretraining
weighted CE for main classifier
branch-level binary normal-vs-abnormal auxiliary BCE
no global binary auxiliary
no old multiclass branch auxiliary
no early stopping
top-3 checkpoint retention for macro F1, macro recall, total loss, and last checkpoints
```

The key conceptual goal is:

```text
Each branch should learn scale-specific abnormal evidence detection.
The final head should learn the 4-class subtype decision.
```
