# Codex Instructions: Branch Binary Auxiliary Cosine-Floor Schedule with Fixed Monitor Loss

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

Do **not** rewrite the model architecture.  
Do **not** reintroduce Hugging Face `ASTModel`.  
Do **not** reintroduce latent-query pooling.  
Do **not** add a global binary auxiliary head.  
Do **not** re-enable the old multiclass `branch_auxiliary` for the new 4-class pretraining config.

This task modifies the **branch binary auxiliary loss weighting schedule**, **validation loss logging**, and **checkpoint monitor loss** for the existing 4-class supervised pretraining setup.

---

## Objective

Implement an epoch-dependent schedule for the branch binary auxiliary weight:

\[
\lambda_{\mathrm{train}}(t)
=
\lambda_{\min}
+
(\lambda_{\max}-\lambda_{\min})
\cdot
\frac{1}{2}
\left(1+\cos(\pi t/T)\right),
\]

with:

```text
lambda_max = 0.4
lambda_min = 0.1
T = 120 epochs
```

But do **not** use this scheduled total validation loss as the checkpoint loss monitor.

Instead, compute a separate fixed-weight monitor validation loss:

\[
\mathcal{L}_{val,monitor}
=
\mathcal{L}_{4cls,val}^{weightedCE}
+
\lambda_{monitor}\mathcal{L}_{branch\_bin,val},
\]

with:

```text
lambda_monitor = 0.3
```

This avoids the problem where the scheduled total validation loss decreases partly because \(\lambda_{\mathrm{train}}(t)\) decreases, rather than because the model improves.

---

## Conceptual Summary

### Training objective

During both training and validation logging, compute the scheduled objective:

\[
\mathcal{L}_{total,scheduled}(t)
=
\mathcal{L}_{4cls}
+
\lambda_{\mathrm{train}}(t)
\mathcal{L}_{branch\_bin}.
\]

### Checkpoint monitor loss

For loss-based checkpoint selection, compute:

\[
\mathcal{L}_{total,monitor}
=
\mathcal{L}_{4cls}
+
0.3
\mathcal{L}_{branch\_bin}.
\]

Use this fixed monitor loss for the `best_loss` checkpoint.

### Important rule

```text
Do not use val_loss_total_scheduled for best_loss checkpointing.
Use val_loss_total_monitor instead.
```

---

## Current Pretraining Objective to Preserve

This task assumes the current 4-class pretraining setup already uses:

```text
main 4-class weighted CE
+ branch-level binary normal-vs-abnormal auxiliary BCE
```

The main task is:

```text
normal / crackle / wheeze / rhonchi
```

The branch binary auxiliary target is config-derived, for example:

```json
"label_to_index": {
  "normal": 0,
  "crackle": 1,
  "wheeze": 1,
  "rhonchi": 1
}
```

The old multiclass `branch_auxiliary` should remain disabled in the relevant pretraining config:

```json
"branch_auxiliary": {
  "enabled": false
}
```

The new branch binary auxiliary is separate:

```json
"branch_binary_auxiliary": {
  "enabled": true,
  ...
}
```

---

# Part A — Add Branch Binary Auxiliary Schedule Config

Modify the config schema, likely in:

```text
src/utils/config.py
```

The exact class names may differ. Adapt to the current codebase.

---

## A1. Add schedule config

Add a schedule config under:

```text
train.loss.branch_binary_auxiliary.schedule
```

Recommended dataclass:

```python
@dataclass(frozen=True)
class BranchBinaryAuxiliaryScheduleConfig:
    enabled: bool = False
    type: Literal["none", "cosine_floor"] = "none"
    max_weight: float = 0.4
    min_weight: float = 0.1
    total_epochs: int | None = None
```

### Validation

```text
if schedule.enabled:
  schedule.type must be "cosine_floor"
  max_weight > 0
  min_weight >= 0
  max_weight >= min_weight
  total_epochs is None or total_epochs > 0
```

If `total_epochs` is `None`, use `train.epochs`.

For the canonical config in this task, set:

```text
max_weight = 0.4
min_weight = 0.1
total_epochs = 120
```

---

## A2. Add monitor config

Add a monitor config under:

```text
train.loss.branch_binary_auxiliary.monitor
```

Recommended dataclass:

```python
@dataclass(frozen=True)
class BranchBinaryAuxiliaryMonitorConfig:
    loss_weight: float = 0.3
```

### Validation

```text
loss_weight >= 0
```

For this task:

```text
loss_weight = 0.3
```

---

## A3. Updated branch binary auxiliary config

Extend the existing `BranchBinaryAuxiliaryConfig` or equivalent with:

```python
@dataclass(frozen=True)
class BranchBinaryAuxiliaryConfig:
    enabled: bool = False
    weight: float = 0.3
    label_to_index: Mapping[str, int] = field(default_factory=dict)
    pos_weight: BinaryAuxiliaryPosWeightConfig = field(
        default_factory=BinaryAuxiliaryPosWeightConfig
    )
    aggregation: Literal["mean"] = "mean"
    schedule: BranchBinaryAuxiliaryScheduleConfig = field(
        default_factory=BranchBinaryAuxiliaryScheduleConfig
    )
    monitor: BranchBinaryAuxiliaryMonitorConfig = field(
        default_factory=BranchBinaryAuxiliaryMonitorConfig
    )
```

The existing `weight` field remains the fallback fixed scalar weight when:

```text
schedule.enabled = false
```

When:

```text
schedule.enabled = true
```

use the scheduled weight instead of the fixed `weight`.

---

# Part B — Cosine-Floor Schedule Calculation

Add a helper function.

Possible location:

```text
src/training/trainer.py
src/training/losses.py
src/training/ast_setup.py
```

Choose the location that best fits the current code.

---

## B1. Required formula

Use:

\[
\lambda(t)
=
\lambda_{\min}
+
(\lambda_{\max}-\lambda_{\min})
\cdot
\frac{1}{2}
(1+\cos(\pi p)),
\]

where:

```text
p = (epoch - 1) / (total_epochs - 1)
```

This gives:

```text
epoch 1              -> lambda = max_weight
epoch total_epochs   -> lambda = min_weight
```

---

## B2. Pseudo-code

```python
import math

def resolve_branch_binary_aux_weight(
    cfg: BranchBinaryAuxiliaryConfig,
    *,
    epoch: int,
    total_epochs: int,
) -> float:
    if not cfg.enabled:
        return 0.0

    if not cfg.schedule.enabled:
        return float(cfg.weight)

    schedule = cfg.schedule

    if schedule.type != "cosine_floor":
        raise ValueError(f"Unsupported branch binary auxiliary schedule: {schedule.type}")

    horizon = int(schedule.total_epochs or total_epochs)

    if horizon <= 1:
        return float(schedule.min_weight)

    progress = (epoch - 1) / (horizon - 1)
    progress = min(max(progress, 0.0), 1.0)

    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(schedule.min_weight + (schedule.max_weight - schedule.min_weight) * cosine)
```

---

## B3. Expected values for canonical config

For:

```text
max_weight = 0.4
min_weight = 0.1
total_epochs = 120
```

approximately:

| Epoch | Weight |
|---:|---:|
| 1 | 0.400 |
| 30 | 0.357 |
| 40 | 0.329 |
| 60 | 0.255 |
| 80 | 0.181 |
| 100 | 0.124 |
| 120 | 0.100 |

The implementation should be monotone non-increasing.

---

# Part C — Apply Scheduled Weight in Training and Validation

Modify the trainer loss computation, likely in:

```text
src/training/trainer.py
```

---

## C1. Training loss

Current intended loss:

```text
loss_total = loss_4cls + lambda_branch_bin * loss_branch_binary
```

Change to:

```python
branch_binary_aux_weight = resolve_branch_binary_aux_weight(
    cfg.train.loss.branch_binary_auxiliary,
    epoch=current_epoch,
    total_epochs=cfg.train.epochs,
)

loss_total_scheduled = (
    loss_4cls
    + branch_binary_aux_weight * loss_branch_binary
)
```

### Required behavior

```text
If branch_binary_auxiliary.enabled = false:
  branch_binary_aux_weight = 0
  no branch binary loss contributes

If branch_binary_auxiliary.enabled = true and schedule.enabled = false:
  use scalar branch_binary_auxiliary.weight

If branch_binary_auxiliary.enabled = true and schedule.enabled = true:
  use cosine-floor scheduled weight
```

---

## C2. Validation scheduled loss

During validation for epoch `e`, compute the scheduled validation total using the same scheduled weight for that epoch:

```python
val_loss_total_scheduled = (
    val_loss_4cls
    + branch_binary_aux_weight * val_loss_branch_binary
)
```

This is for logging and for tracking the actual epoch-dependent validation objective.

Do **not** use this as the `best_loss` checkpoint monitor.

---

## C3. Validation monitor loss

Compute a separate fixed-weight monitor loss:

```python
monitor_weight = cfg.train.loss.branch_binary_auxiliary.monitor.loss_weight

val_loss_total_monitor = (
    val_loss_4cls
    + monitor_weight * val_loss_branch_binary
)
```

For the canonical schedule config:

```text
monitor_weight = 0.3
```

This is the loss used for `best_loss` checkpointing.

---

## C4. Important train/val weight source rule

The branch binary auxiliary weight schedule is separate from class weights and binary pos_weight.

Continue to enforce:

```text
4-class class weights are computed from train split only
binary pos_weight is computed from train split only
validation reuses train-derived weights
```

The fixed monitor loss changes only the scalar multiplying `val_loss_branch_binary`.

---

# Part D — Metric Naming and Logging

Update logging so the two validation total losses are distinct.

---

## D1. Training logging

Log:

```text
train_loss
train_loss_total_scheduled
train_loss_4cls
train_loss_branch_binary
train_branch_binary_aux_weight
```

Recommended compatibility:

```text
train_loss = train_loss_total_scheduled
```

---

## D2. Validation logging

Log:

```text
val_loss
val_loss_total_scheduled
val_loss_total_monitor
val_loss_4cls
val_loss_branch_binary
val_branch_binary_aux_weight
val_branch_binary_aux_monitor_weight
```

Recommended compatibility:

```text
val_loss = val_loss_total_monitor
```

This allows existing checkpointing or logging that expects `val_loss` to use the fixed monitor loss, not the scheduled total loss.

However, if using explicit monitor keys is easy, prefer:

```text
best_loss monitor -> val_loss_total_monitor
```

---

## D3. Why this matters

Do not let `val_loss_total_scheduled` be the loss-based checkpoint metric.  
Because \(\lambda_{\mathrm{train}}(t)\) decreases over epochs, `val_loss_total_scheduled` can decrease even if model quality does not improve.

---

# Part E — Checkpointing Changes

Modify checkpoint configuration/manager if needed.

Likely file:

```text
src/utils/checkpoint.py
```

or current checkpoint manager.

---

## E1. Required monitors

Use:

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
  metric key: val_loss_total_monitor
  mode: min
  keep_top_k: 3

last:
  mode: last
  keep_top_k: 3
```

If current checkpointing must use `val_loss`, set:

```text
val_loss = val_loss_total_monitor
```

and document this clearly.

---

## E2. Recommended checkpoint config

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
      "name": "val_loss_total_monitor",
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

---

## E3. Checkpoint metadata

If practical, include the schedule and the current epoch weight in checkpoint metadata:

```json
"branch_binary_auxiliary_schedule": {
  "type": "cosine_floor",
  "max_weight": 0.4,
  "min_weight": 0.1,
  "total_epochs": 120
}
```

For each checkpoint:

```json
"branch_binary_aux_weight": 0.255,
"branch_binary_aux_monitor_weight": 0.3
```

This is recommended for later analysis.

---

# Part F — Add Canonical Config

Create a new config based on the existing 4-class pretraining config.

Recommended filename:

```text
configs/training_4class_pretrain_branch_bin_cosine_040_010_monitor030.json
```

Do not invent dataset paths. Copy dataset paths and local settings from the current working 4-class pretraining config.

---

## F1. Required loss config

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
      "aggregation": "mean",
      "schedule": {
        "enabled": true,
        "type": "cosine_floor",
        "max_weight": 0.4,
        "min_weight": 0.1,
        "total_epochs": 120
      },
      "monitor": {
        "loss_weight": 0.3
      }
    }
  },
  "early_stopping": {
    "enabled": false
  }
}
```

---

## F2. Required validation config

```json
"val": {
  "loss": {
    "use_train_loss_config": true,
    "class_weight_source": "train",
    "binary_pos_weight_source": "train"
  }
}
```

This means:

```text
validation uses train-derived class weights
validation uses train-derived binary pos_weight
validation logs both scheduled and fixed-monitor total loss
```

---

## F3. Required checkpointing config

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
      "name": "val_loss_total_monitor",
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

---

## F4. Model settings

Keep the same model settings as the current working 4-class pretraining config, including:

```text
multiscale_rdt_ast
4-scale
RDT settings
evidence pooling settings
augmentation settings if present
```

Do not change model architecture in this task.

---

# Part G — README Updates

Add a section under 4-class pretraining:

```text
Branch Binary Auxiliary Schedule
```

Document the training schedule:

\[
\lambda_{\mathrm{train}}(t)
=
0.1
+
(0.4-0.1)
\cdot
\frac{1}{2}
(1+\cos(\pi t/T))
\]

with:

```text
T = 120
epoch 1 -> 0.4
epoch 120 -> 0.1
```

Document fixed monitor loss:

\[
\mathcal{L}_{monitor}
=
\mathcal{L}_{4cls}
+
0.3\mathcal{L}_{branch\_bin}
\]

Clarify:

```text
The scheduled total validation loss is logged but is not used for best_loss checkpointing.
best_loss uses val_loss_total_monitor with fixed branch binary monitor weight 0.3.
```

Add the relevant config filename:

```text
configs/training_4class_pretrain_branch_bin_cosine_040_010_monitor030.json
```

---

# Part H — Tests

Add or update tests.

---

## H1. Schedule value tests

For canonical schedule:

```text
max_weight = 0.4
min_weight = 0.1
total_epochs = 120
```

Assert:

```python
weight(epoch=1) == approx(0.4)
weight(epoch=120) == approx(0.1)
weight(epoch=60) == approx(0.25, tolerance reasonable)
```

Assert:

```text
weights are monotonically non-increasing
```

---

## H2. Schedule disabled test

If:

```text
schedule.enabled = false
```

then:

```text
branch_binary_auxiliary.weight is used
```

---

## H3. Monitor loss test

Given:

```text
val_loss_4cls = 1.2
val_loss_branch_binary = 0.8
scheduled_weight at epoch 120 = 0.1
monitor_weight = 0.3
```

Expect:

```text
val_loss_total_scheduled = 1.2 + 0.1 * 0.8 = 1.28
val_loss_total_monitor   = 1.2 + 0.3 * 0.8 = 1.44
```

Ensure checkpoint monitor uses `val_loss_total_monitor`.

---

## H4. Train/val same scheduled epoch weight test

For a given epoch:

```text
train scheduled loss and validation scheduled loss use the same lambda_train(epoch)
```

Do not use one epoch's weight for training and another for validation.

---

## H5. Logging keys test

Ensure metrics include:

```text
branch_binary_aux_weight
branch_binary_aux_monitor_weight
train_loss_total_scheduled
val_loss_total_scheduled
val_loss_total_monitor
val_loss_4cls
val_loss_branch_binary
```

Names can follow existing naming conventions, but all values must be available.

---

## H6. Config parsing test

Ensure the new config parses:

```text
configs/training_4class_pretrain_branch_bin_cosine_040_010_monitor030.json
```

Invalid schedule configs should be rejected:

```text
max_weight < min_weight
min_weight < 0
max_weight <= 0
total_epochs <= 0
unsupported schedule type
monitor.loss_weight < 0
```

---

## H7. Backward compatibility test

Existing configs without `branch_binary_auxiliary.schedule` must still work.

Existing configs without `branch_binary_auxiliary.monitor` must still work, using reasonable defaults.

---

# Part I — Acceptance Criteria

The task is complete when:

1. `branch_binary_auxiliary.schedule` config exists.
2. `cosine_floor` schedule is supported.
3. Canonical schedule uses max 0.4, min 0.1, total epochs 120.
4. Epoch 1 scheduled weight is 0.4.
5. Final epoch scheduled weight is 0.1.
6. Scheduled weight is used for training total loss.
7. Scheduled weight is used for validation scheduled total loss.
8. Fixed monitor weight 0.3 is used for validation monitor loss.
9. `best_loss` checkpointing uses fixed monitor loss, not scheduled total loss.
10. `val_loss_total_scheduled` is logged.
11. `val_loss_total_monitor` is logged.
12. `val_loss_4cls` and `val_loss_branch_binary` are logged.
13. `branch_binary_aux_weight` and monitor weight are logged.
14. New canonical config exists.
15. README documents the schedule and monitor loss distinction.
16. Tests pass.

---

# Part J — Avoid These Mistakes

- Do not use scheduled validation total loss for `best_loss` checkpointing.
- Do not compute validation weights from validation labels.
- Do not change class weighting formula.
- Do not change binary pos_weight formula.
- Do not change binary label mapping.
- Do not add a global binary auxiliary head.
- Do not re-enable old multiclass `branch_auxiliary`.
- Do not apply the schedule to the main 4-class CE.
- Do not use different scheduled weights for train and validation within the same epoch.
- Do not make epoch 1 weight lower than `max_weight`.
- Do not make final epoch weight higher than `min_weight`.
- Do not break configs that use fixed branch binary auxiliary weight.
- Do not remove top-3 checkpointing for macro F1, macro recall, loss, and last.

---

# Part K — Suggested Implementation Pseudo-Code

## K1. Schedule

```python
def resolve_branch_binary_aux_weight(cfg, *, epoch: int, total_epochs: int) -> float:
    if not cfg.enabled:
        return 0.0

    if not cfg.schedule.enabled:
        return float(cfg.weight)

    schedule = cfg.schedule
    if schedule.type != "cosine_floor":
        raise ValueError(f"Unsupported schedule type: {schedule.type}")

    horizon = schedule.total_epochs or total_epochs

    if horizon <= 1:
        return float(schedule.min_weight)

    progress = (epoch - 1) / (horizon - 1)
    progress = min(max(progress, 0.0), 1.0)

    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(
        schedule.min_weight
        + (schedule.max_weight - schedule.min_weight) * cosine
    )
```

---

## K2. Training loss

```python
lambda_train = resolve_branch_binary_aux_weight(
    cfg.train.loss.branch_binary_auxiliary,
    epoch=current_epoch,
    total_epochs=cfg.train.epochs,
)

loss_4cls = ce_loss(output.logits, targets)
loss_branch_binary = branch_binary_bce(output.branch_binary_logits, binary_target)

loss_total_scheduled = loss_4cls + lambda_train * loss_branch_binary
```

---

## K3. Validation losses

```python
lambda_train = resolve_branch_binary_aux_weight(
    cfg.train.loss.branch_binary_auxiliary,
    epoch=current_epoch,
    total_epochs=cfg.train.epochs,
)

lambda_monitor = cfg.train.loss.branch_binary_auxiliary.monitor.loss_weight

val_loss_4cls = ce_loss(output.logits, targets)
val_loss_branch_binary = branch_binary_bce(output.branch_binary_logits, binary_target)

val_loss_total_scheduled = (
    val_loss_4cls + lambda_train * val_loss_branch_binary
)

val_loss_total_monitor = (
    val_loss_4cls + lambda_monitor * val_loss_branch_binary
)
```

Metrics:

```python
metrics["val_loss_total_scheduled"] = val_loss_total_scheduled
metrics["val_loss_total_monitor"] = val_loss_total_monitor
metrics["val_loss_4cls"] = val_loss_4cls
metrics["val_loss_branch_binary"] = val_loss_branch_binary
metrics["branch_binary_aux_weight"] = lambda_train
metrics["branch_binary_aux_monitor_weight"] = lambda_monitor
```

For checkpointing:

```text
best_loss monitor -> val_loss_total_monitor
```

---

# Final Note

The key purpose of this task is to support the hypothesis:

```text
Use strong branch-level abnormality learning early,
then reduce branch binary pressure later so the final 4-class subtype classifier can dominate.
```

But because the scheduled loss scale changes over epochs, checkpoint loss monitoring must use a fixed-weight validation monitor loss.

The intended final behavior is:

```text
Training:
  L = L_4cls + lambda_train(epoch) * L_branch_binary

Validation logging:
  val_loss_total_scheduled = L_4cls + lambda_train(epoch) * L_branch_binary
  val_loss_total_monitor   = L_4cls + 0.3 * L_branch_binary

Checkpoint:
  best_loss uses val_loss_total_monitor
```
