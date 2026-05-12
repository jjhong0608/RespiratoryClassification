# Codex Instructions: Add Weighted-Prior Classifier Bias Initialization for FSD50K Multi-label Fine-tuning

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

The current task concerns **Stage 2 FSD50K supervised multi-label fine-tuning**.

Do **not** change the FSD50K SSL pretraining implementation.  
Do **not** change the CNUH transfer/fine-tuning behavior.  
Do **not** change the model architecture.  
Do **not** reintroduce Hugging Face `ASTModel`.  
Do **not** add branch auxiliary losses for FSD50K supervised training.

---

## Objective

Add configurable classifier bias initialization for the FSD50K supervised multi-label classifier head.

The immediate goal is to stabilize early multi-label training by initializing the final classifier bias using the **weighted prior** implied by the FSD50K training label distribution and the configured BCE `pos_weight`.

The new initialization should help avoid the current behavior where the classifier starts with logits near zero, causing sigmoid probabilities near 0.5 for many classes.

---

## Background

The FSD50K supervised fine-tuning task is multi-label:

```text
num_classes = 200
target shape = [B, 200]
loss = BCEWithLogitsLoss
```

The current class imbalance correction is:

\[
pos\_weight_c
=
\min\left(
\sqrt{\frac{n_c^-}{n_c^+}},
10
\right),
\]

where:

```text
n_c^+ = number of positive training examples for class c
n_c^- = number of negative training examples for class c
```

The classifier head is randomly initialized. If its bias is approximately zero, initial logits are near zero and:

\[
\sigma(0)=0.5.
\]

For a sparse 200-class multi-label task, this causes too many initial positive predictions.

Implement a **weighted-prior bias initialization**:

\[
b_c
=
\log
\frac{
pos\_weight_c \cdot n_c^+
}{
n_c^-
}
\]

with numerical stabilizers and optional clipping.

This is the constant-logit optimum for class \(c\) under weighted BCE with `pos_weight`.

---

# Part A — Add Config Schema

Modify the config parser/schema, likely in:

```text
src/utils/config.py
```

Add a classifier bias initialization config for the FSD50K supervised multi-label stage.

Recommended config path:

```text
model.classifier.bias_init
```

or, if the existing config style places classifier settings elsewhere, adapt to the current style while preserving the semantics.

Recommended config:

```json
"classifier": {
  "num_classes": 200,
  "bias_init": {
    "enabled": true,
    "type": "weighted_prior",
    "source": "train",
    "eps": 1e-6,
    "clamp_min": -10.0,
    "clamp_max": 5.0
  }
}
```

Supported values:

```text
enabled: bool
type: "none" | "prior" | "weighted_prior"
source: "train"
eps: positive float
clamp_min: float
clamp_max: float
```

Validation rules:

```text
if bias_init.enabled:
  type must be one of {"prior", "weighted_prior"}
  source must be "train"
  eps > 0
  clamp_min < clamp_max
```

If `bias_init` is missing:

```text
disabled by default for backward compatibility
```

For the FSD50K supervised config, enable:

```text
type = "weighted_prior"
```

---

# Part B — Compute Train Label Counts

The bias initialization must use **training split only**.

Use the same multi-hot training targets used to compute BCE `pos_weight`.

For FSD50K supervised:

```text
train_targets: Tensor [N, C]
C = 200
```

Compute:

```python
positive_counts = train_targets.sum(dim=0)          # [C]
negative_counts = train_targets.shape[0] - positive_counts
```

Requirements:

```text
- Use train split only.
- Do not use validation labels.
- Do not use eval labels.
- The order must match the classifier output class order.
- Reuse the same label_to_index/vocabulary ordering used by the dataloader.
```

If any class has zero positives in train:

```text
raise a clear error or handle via eps, but log a warning.
```

Given FSD50K train data, all 200 classes are expected to have positives, but keep the code robust.

---

# Part C — Reuse or Compute pos_weight

The weighted-prior bias must use the same `pos_weight` as the BCE loss.

Current formula:

\[
pos\_weight_c
=
\min\left(
\sqrt{\frac{n_c^-}{n_c^+}},
cap
\right)
\]

Default cap:

```text
cap = 10
```

Pseudo-code:

```python
positive_counts = train_targets.sum(dim=0).float()
negative_counts = train_targets.shape[0] - positive_counts

pos_weight = torch.sqrt(
    negative_counts / positive_counts.clamp_min(eps)
)
pos_weight = torch.clamp(pos_weight, max=cap)
```

If the code already computes `pos_weight`, reuse that tensor to avoid mismatch.

Important:

```text
The pos_weight used in the bias calculation must be identical to the pos_weight passed to BCEWithLogitsLoss.
```

---

# Part D — Bias Initialization Types

Implement two useful types.

## D1. `prior`

Simple empirical prior:

\[
b_c
=
\log
\frac{
n_c^+
}{
n_c^-
}
\]

Pseudo-code:

```python
bias = torch.log(
    (positive_counts + eps)
    / (negative_counts + eps)
)
```

## D2. `weighted_prior`

Weighted BCE prior:

\[
b_c
=
\log
\frac{
pos\_weight_c \cdot n_c^+
}{
n_c^-
}
\]

Pseudo-code:

```python
bias = torch.log(
    (pos_weight * positive_counts + eps)
    / (negative_counts + eps)
)
```

Then clamp:

```python
bias = torch.clamp(bias, min=clamp_min, max=clamp_max)
```

Default for FSD50K supervised:

```text
type = "weighted_prior"
eps = 1e-6
clamp_min = -10.0
clamp_max = 5.0
```

---

# Part E — Apply Bias to Final Multi-label Classifier Head

Apply the computed bias only to the final FSD50K multi-label classifier head.

Expected classifier output:

```text
logits: [B, 200]
```

If classifier is an `nn.Linear(hidden_dim, num_classes)`:

```python
with torch.no_grad():
    model.classifier.bias.copy_(bias.to(model.classifier.bias.device))
```

If the classifier is nested, find the final `nn.Linear` that outputs `num_classes=200`.

Requirements:

```text
- Only initialize the final classifier bias.
- Do not initialize SSL decoder biases.
- Do not initialize CNUH classifier bias unless explicitly configured.
- Do not overwrite pretrained encoder/body weights.
- Bias initialization should run after the FSD50K classifier head is created.
- Bias initialization should run after checkpoint loading if the classifier head is newly initialized.
```

Important checkpoint-loading order:

```text
1. Build FSD50K supervised model.
2. Load Stage 1 SSL checkpoint into compatible modules, strict=false.
3. Create/keep FSD50K classifier head [B, 200].
4. Apply classifier bias initialization.
```

If an FSD50K supervised checkpoint is being resumed, do **not** reinitialize the classifier bias unless an explicit `force_reinit` flag is added and enabled. For this task, default behavior should avoid reinitializing on resume.

---

# Part F — Add Logging

Log bias initialization details at setup time.

Required logs:

```text
classifier_bias_init.enabled
classifier_bias_init.type
classifier_bias_init.source
classifier_bias_init.eps
classifier_bias_init.clamp_min
classifier_bias_init.clamp_max
positive_counts min/mean/median/max
negative_counts min/mean/median/max
pos_weight min/mean/median/max
bias min/mean/median/max
number of classes clamped at clamp_min
number of classes clamped at clamp_max
```

Example log:

```text
FSD50K classifier bias init | type=weighted_prior | eps=1e-6 | clamp=[-10,5]
positive_counts min=... mean=... max=...
pos_weight min=... mean=... max=...
bias min=... mean=... median=... max=...
```

Also log the first few class examples:

```text
class_name, positive_count, negative_count, pos_weight, initialized_bias
```

This helps diagnose class-prior issues.

---

# Part G — Save Metadata in Checkpoints

Store bias initialization metadata in checkpoint metadata.

Recommended metadata:

```json
"classifier_bias_init": {
  "enabled": true,
  "type": "weighted_prior",
  "source": "train",
  "eps": 1e-6,
  "clamp_min": -10.0,
  "clamp_max": 5.0,
  "bias_min": "...",
  "bias_mean": "...",
  "bias_median": "...",
  "bias_max": "...",
  "num_clamped_min": "...",
  "num_clamped_max": "...",
  "values": [...]
}
```

Also ensure existing `pos_weight` metadata is saved:

```json
"fsd50k_pos_weight": {
  "type": "sqrt_negative_over_positive",
  "cap": 10,
  "values": [...]
}
```

If storing full bias vector is too verbose, at least save summary stats and config. Prefer saving full values if the existing checkpoint metadata supports it.

---

# Part H — Update FSD50K Supervised Config

Update or create:

```text
configs/fsd50k_supervised_multilabel_finetune.json
```

Add:

```json
"model": {
  "encoder": {
    "type": "multiscale_rdt_ast",
    "init_from": "<ssl_checkpoint_path>"
  },
  "classifier": {
    "num_classes": 200,
    "bias_init": {
      "enabled": true,
      "type": "weighted_prior",
      "source": "train",
      "eps": 1e-6,
      "clamp_min": -10.0,
      "clamp_max": 5.0
    }
  }
}
```

Keep existing settings:

```json
"train": {
  "loss": {
    "type": "bce_with_logits",
    "pos_weight": {
      "type": "sqrt_negative_over_positive",
      "cap": 10,
      "source": "train",
      "save_to_checkpoint_metadata": true
    },
    "branch_auxiliary": {
      "enabled": false
    },
    "branch_binary_auxiliary": {
      "enabled": false
    }
  }
}
```

Do not change:

```text
encoder_lr = 1e-5
body_lr = 3e-5
head_lr = 3e-4
scheduler = linear warmup + cosine
macro AP / micro AP monitors
F1 threshold = 0.5
```

---

# Part I — Add Tests

Add or update tests.

## I1. Bias formula tests

Synthetic counts:

```python
positive_counts = torch.tensor([100.0, 1000.0])
negative_counts = torch.tensor([9900.0, 9000.0])
pos_weight = torch.tensor([10.0, 3.0])
```

Expected weighted prior:

```python
expected = torch.log((pos_weight * positive_counts) / negative_counts)
```

Assert computed bias matches expected.

---

## I2. Prior bias test

For `type="prior"`:

```python
expected = torch.log(positive_counts / negative_counts)
```

---

## I3. Clamp test

Use extreme counts to ensure:

```text
bias >= clamp_min
bias <= clamp_max
```

and clamped class counts are logged or returned.

---

## I4. Classifier application test

Create a dummy classifier:

```python
head = nn.Linear(16, 200)
```

Apply bias init.

Assert:

```python
torch.allclose(head.bias, expected_bias)
```

---

## I5. No reinitialization on resume test

If the training code supports resuming from FSD50K supervised checkpoint:

```text
bias init should not overwrite checkpoint classifier bias on resume
```

unless explicit force behavior is implemented.

For this task, default should be:

```text
do not reinitialize on resume
```

---

## I6. Config parsing test

Ensure config parses:

```text
bias_init.enabled = true
bias_init.type = weighted_prior
```

Invalid configs should fail:

```text
unknown type
source != train
eps <= 0
clamp_min >= clamp_max
```

---

## I7. Setup integration test

Use a tiny synthetic multi-label dataset with class counts.

Verify:

```text
pos_weight computed
classifier bias initialized
BCEWithLogitsLoss receives same pos_weight
metadata contains classifier_bias_init
```

---

# Part J — Expected Runtime Diagnostics After This Change

After applying weighted-prior bias initialization, the first validation epoch should no longer show:

```text
prob_mean ≈ 0.5
mean_predicted_positives@0.5 ≈ 90+
```

Expected behavior:

```text
prob_mean should start much lower than 0.5
mean_predicted_positives@0.5 should be much smaller
probability scale should be less chaotic in early epochs
```

Do **not** expect F1@0.5 to be high immediately.  
The main expected improvement is stable early logit scaling and reduced class-prior collapse.

Continue logging:

```text
val_macro_AP
val_micro_AP
val_macro_F1@0.5
prob_mean
prob_std
prob_max_mean
mean_predicted_positives@0.5
mean_predicted_positives@0.3
mean_predicted_positives@0.1
top-k predicted label distribution
```

---

# Part K — Avoid These Mistakes

- Do not apply bias initialization to SSL decoders.
- Do not apply this to the CNUH classifier unless separately configured.
- Do not compute bias from validation or eval labels.
- Do not use simple prior when config says weighted_prior.
- Do not use a different pos_weight for bias than the BCE loss uses.
- Do not overwrite classifier bias when resuming supervised training.
- Do not change branch auxiliary settings.
- Do not change model architecture.
- Do not remove existing AP/F1/probability diagnostics.
- Do not silently ignore classes with zero positives.
- Do not forget to save bias initialization metadata.

---

# Acceptance Criteria

The task is complete when:

1. FSD50K supervised classifier supports configurable bias initialization.
2. `type="weighted_prior"` is implemented.
3. Bias formula uses `log((pos_weight * positive_count + eps) / (negative_count + eps))`.
4. The same `pos_weight` is used for BCE loss and bias initialization.
5. Bias is clamped to configured bounds.
6. Bias is applied only to the final FSD50K multi-label classifier head.
7. Bias initialization is not applied on resume by default.
8. Bias initialization stats are logged.
9. Bias initialization metadata is saved in checkpoints.
10. Config file enables weighted-prior bias init for FSD50K supervised fine-tuning.
11. Tests pass.
