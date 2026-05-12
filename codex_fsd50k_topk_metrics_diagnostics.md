# Codex Instructions: Add Top-k Multi-label Diagnostics for FSD50K Supervised Fine-tuning

## Target Repository and Branch

Repository:

```text
https://github.com/jjhong0608/RespiratoryClassification.git
```

Base branch:

```text
MIL-RDT-AST
```

This task targets the current **FSD50K supervised multi-label fine-tuning** pipeline.

Do **not** change the model architecture.  
Do **not** change the FSD50K SSL pretraining code.  
Do **not** change the CNUH transfer/fine-tuning code.  
Do **not** change the current primary checkpoint monitors.  
Do **not** add class-wise threshold optimization.

---

## Objective

Add top-k multi-label ranking diagnostics to the FSD50K supervised trainer.

The current main monitors are:

```text
val_macro_AP
val_micro_AP
```

These should remain the primary and secondary monitors.

This task adds diagnostic metrics:

```text
val_hit_at_5
val_hit_at_10
val_recall_at_5
val_recall_at_10
```

and diagnostic summaries:

```text
top5_label_frequency
top10_label_frequency
true_label_frequency_top20
```

These metrics are intended to diagnose whether the model is learning sample-specific semantic ranking or collapsing to a small set of frequent/prior labels such as `Alarm`, `Vehicle`, `Music`, or `Musical_instrument`.

---

## Background

FSD50K supervised fine-tuning is a multi-label task:

```text
logits: [N, 200]
targets: [N, 200] multi-hot
```

Threshold-based F1 at `0.5` can be uninformative early in training because predicted probabilities may be poorly calibrated. For example:

```text
mean_true_positives_per_sample ≈ 3
mean_predicted_positives@0.5 may be near 0 or very large
```

Top-k metrics are threshold-free sample-wise ranking diagnostics.

They answer:

```text
Hit@K:
  Does at least one true label appear in the top-K predictions?

Recall@K:
  What fraction of the sample's true labels appear in the top-K predictions?
```

---

# Part A — Add Top-k Metric Computation

Add a utility function, preferably in the existing metrics module, e.g.:

```text
src/evaluation/metrics.py
```

or wherever FSD50K multi-label metrics are currently implemented.

---

## A1. Definitions

For each sample \(i\):

```text
Y_i = set of true positive labels
TopK_i = set of K highest-scoring predicted labels
```

Hit@K:

\[
Hit@K_i = 1[TopK_i \cap Y_i \neq \emptyset]
\]

\[
Hit@K = mean_i Hit@K_i
\]

Recall@K:

\[
Recall@K_i =
\frac{|TopK_i \cap Y_i|}{|Y_i|}
\]

\[
Recall@K = mean_i Recall@K_i
\]

---

## A2. Required metric keys

Use logger-friendly names:

```text
hit_at_5
hit_at_10
recall_at_5
recall_at_10
```

During validation, expose them as:

```text
val_hit_at_5
val_hit_at_10
val_recall_at_5
val_recall_at_10
```

---

## A3. Pseudo-code

```python
import torch

def multilabel_topk_metrics(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    ks: tuple[int, ...] = (5, 10),
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    Args:
        y_true:
            Multi-hot labels, shape [N, C], values 0/1.
        y_score:
            Prediction scores or probabilities, shape [N, C].
            Use sigmoid probabilities for consistency with other metrics.
        ks:
            K values for top-k metrics.

    Returns:
        {
            "hit_at_5": float,
            "recall_at_5": float,
            ...
        }
    """
    if y_true.ndim != 2 or y_score.ndim != 2:
        raise ValueError("Expected y_true and y_score to have shape [N, C].")

    if y_true.shape != y_score.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_score={y_score.shape}")

    y_true_bool = y_true.bool()
    n, c = y_true_bool.shape

    true_counts = y_true_bool.sum(dim=1)
    valid_samples = true_counts > 0

    if not valid_samples.all():
        # FSD50K should normally have at least one positive label per clip.
        # Exclude empty-label samples defensively.
        y_true_bool = y_true_bool[valid_samples]
        y_score = y_score[valid_samples]
        true_counts = true_counts[valid_samples]

    true_counts = true_counts.clamp_min(1)

    metrics: dict[str, float] = {}

    for k in ks:
        k_eff = min(k, c)
        topk_idx = torch.topk(y_score, k=k_eff, dim=1).indices  # [N, K]

        pred_topk = torch.zeros_like(y_true_bool, dtype=torch.bool)
        pred_topk.scatter_(dim=1, index=topk_idx, value=True)

        hits_per_sample = (pred_topk & y_true_bool).sum(dim=1)

        hit_at_k = (hits_per_sample > 0).float().mean().item()
        recall_at_k = (hits_per_sample.float() / true_counts.float()).mean().item()

        metrics[f"hit_at_{k}"] = hit_at_k
        metrics[f"recall_at_{k}"] = recall_at_k

    return metrics
```

Use sigmoid probabilities:

```python
probs = torch.sigmoid(logits)
metrics.update(multilabel_topk_metrics(targets, probs, ks=(5, 10)))
```

Ranking is identical for logits and sigmoid probabilities, but use probabilities for consistency.

---

# Part B — Add Top-k Label Frequency Diagnostics

Add diagnostic summaries that count how often each class appears in top-k predictions.

This is important because a model may have non-trivial Hit@K only because it always predicts frequent labels.

---

## B1. Top-k predicted label frequency

Implement a helper:

```python
from collections import Counter

def topk_label_frequency(
    y_score: torch.Tensor,
    index_to_label: list[str] | None,
    k: int = 5,
    top_n: int = 20,
) -> list[dict]:
    """
    Args:
        y_score:
            Prediction probabilities or logits, shape [N, C].
        index_to_label:
            Class name lookup. If unavailable, use stringified indices.
        k:
            Top-k predictions per sample.
        top_n:
            Number of most frequent labels to return.

    Returns:
        [
            {"class_index": int, "class_name": str, "count": int},
            ...
        ]
    """
    n, c = y_score.shape
    k_eff = min(k, c)

    topk_idx = torch.topk(y_score, k=k_eff, dim=1).indices.cpu()

    counter = Counter()
    for row in topk_idx:
        for idx in row.tolist():
            counter[int(idx)] += 1

    results = []
    for idx, count in counter.most_common(top_n):
        class_name = index_to_label[idx] if index_to_label is not None else str(idx)
        results.append({
            "class_index": int(idx),
            "class_name": class_name,
            "count": int(count),
        })

    return results
```

Required diagnostics:

```text
top5_label_frequency
top10_label_frequency
```

---

## B2. True label frequency

Add a helper to summarize validation target distribution:

```python
def true_label_frequency(
    y_true: torch.Tensor,
    index_to_label: list[str] | None,
    top_n: int = 20,
) -> list[dict]:
    """
    Args:
        y_true: [N, C] multi-hot labels.

    Returns:
        top-N true label frequencies.
    """
    counts = y_true.sum(dim=0).cpu()

    sorted_idx = torch.argsort(counts, descending=True)

    results = []
    for idx in sorted_idx[:top_n].tolist():
        class_name = index_to_label[idx] if index_to_label is not None else str(idx)
        results.append({
            "class_index": int(idx),
            "class_name": class_name,
            "count": int(counts[idx].item()),
        })

    return results
```

Required diagnostic:

```text
true_label_frequency_top20
```

---

# Part C — Integrate into FSD50K Validation Loop

Modify the FSD50K supervised trainer, likely:

```text
src/training/multilabel_trainer.py
```

or wherever the FSD50K validation logic currently aggregates logits and labels.

At the end of validation, after collecting all logits and targets:

```python
all_logits: Tensor [N, 200]
all_targets: Tensor [N, 200]
```

compute:

```python
probs = all_logits.sigmoid()
```

Then compute:

```python
topk_metrics = multilabel_topk_metrics(all_targets, probs, ks=(5, 10))
```

Add to validation metrics:

```python
metrics["val_hit_at_5"] = topk_metrics["hit_at_5"]
metrics["val_hit_at_10"] = topk_metrics["hit_at_10"]
metrics["val_recall_at_5"] = topk_metrics["recall_at_5"]
metrics["val_recall_at_10"] = topk_metrics["recall_at_10"]
```

Keep existing metrics:

```text
val_macro_AP
val_micro_AP
val_macro_F1@0.5
val_loss
```

Do not replace them.

---

# Part D — Logging

Add a concise epoch-level log line.

Example:

```text
FSD50K top-k metrics | epoch=10 |
hit@5=0.399 | hit@10=0.499 |
recall@5=0.178 | recall@10=0.235
```

Also log top-k frequency summaries.

Recommended log format:

```text
FSD50K top5 label frequency | epoch=10 |
[{"class_index": ..., "class_name": "...", "count": ...}, ...]
```

```text
FSD50K top10 label frequency | epoch=10 |
[{"class_index": ..., "class_name": "...", "count": ...}, ...]
```

```text
FSD50K true label frequency top20 | epoch=10 |
[{"class_index": ..., "class_name": "...", "count": ...}, ...]
```

Keep the existing probability stats log:

```text
prob_mean
prob_std
prob_max_mean
mean_predicted_positives@0.5
mean_predicted_positives@0.3
mean_predicted_positives@0.1
mean_true_positives
```

---

# Part E — Diagnostics JSONL

The current trainer saves validation diagnostics such as:

```text
checkpoints/fsd50k_supervised_multilabel_finetune/diagnostics/val_epoch_010.jsonl
```

Extend diagnostics to include an epoch-level summary record.

If the existing JSONL contains one row per sample, add either:

1. A separate summary file:

```text
diagnostics/val_epoch_010_summary.json
```

or

2. A final JSONL row with:

```json
{
  "record_type": "summary",
  "epoch": 10,
  "topk_metrics": {
    "hit_at_5": 0.399,
    "hit_at_10": 0.499,
    "recall_at_5": 0.178,
    "recall_at_10": 0.235
  },
  "top5_label_frequency": [
    {"class_index": 0, "class_name": "Alarm", "count": 4170}
  ],
  "top10_label_frequency": [],
  "true_label_frequency_top20": []
}
```

Preferred:

```text
Write a separate val_epoch_XXX_summary.json file.
```

Reason:

```text
Keeps sample-level diagnostics and epoch-level summaries cleanly separated.
```

---

# Part F — Do Not Use Top-k Metrics as Primary Checkpoint Monitors

Do not change the current checkpoint monitors.

Keep:

```text
primary: val_macro_AP
secondary: val_micro_AP
```

Top-k metrics are diagnostic metrics, not primary checkpoint criteria.

Rationale:

```text
Hit@K can be inflated by frequent-label priors.
Recall@K depends on K.
macro AP remains the better primary multi-label ranking monitor.
```

However, it is acceptable to log top-k metrics to TensorBoard/W&B/CSV if the existing logging framework supports it.

---

# Part G — Optional Config

Add optional metric config if the project uses config-driven metrics.

Recommended:

```json
"metrics": {
  "primary": "macro_AP",
  "secondary": "micro_AP",
  "f1_threshold": 0.5,
  "topk": {
    "enabled": true,
    "ks": [5, 10],
    "save_label_frequency": true,
    "top_n_frequency": 20
  },
  "save_per_class_AP": true,
  "zero_positive_class_policy": "exclude_with_warning"
}
```

If no config-driven metric selection exists, just enable top-k diagnostics by default for FSD50K supervised training.

---

# Part H — Tests

Add or update tests.

## H1. Top-k metric correctness

Synthetic example:

```python
y_true = torch.tensor([
    [1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1],
], dtype=torch.float32)

y_score = torch.tensor([
    [0.9, 0.8, 0.7, 0.1, 0.0],
    [0.6, 0.5, 0.4, 0.3, 0.2],
])
```

For `k=1`:

```text
sample 1 top1 = class 0, hit = 1, recall = 1/2
sample 2 top1 = class 0, hit = 0, recall = 0
hit@1 = 0.5
recall@1 = 0.25
```

For `k=3`, compute expected manually and assert.

---

## H2. Shape validation

Test error handling for:

```text
y_true shape != y_score shape
not 2D inputs
K > num_classes
```

If `K > num_classes`, use:

```text
k_eff = min(k, num_classes)
```

---

## H3. Empty-label defensive handling

Although FSD50K should have at least one label per sample, test behavior when one sample has no positives.

Expected:

```text
empty-label sample is ignored or handled with clamp, according to implementation.
```

Prefer:

```text
exclude empty-label samples and log warning
```

---

## H4. Top-k frequency test

Use small synthetic predictions.

Verify:

```text
top5_label_frequency counts class occurrences correctly
true_label_frequency counts target positives correctly
class names are attached correctly
```

---

## H5. Trainer integration test

With a small dummy multi-label validation set:

```text
val_hit_at_5
val_hit_at_10
val_recall_at_5
val_recall_at_10
```

must appear in the validation metrics dictionary.

If diagnostics are enabled, check that summary file includes:

```text
topk_metrics
top5_label_frequency
top10_label_frequency
true_label_frequency_top20
```

---

# Part I — Acceptance Criteria

The task is complete when:

1. `val_hit_at_5` is computed and logged.
2. `val_hit_at_10` is computed and logged.
3. `val_recall_at_5` is computed and logged.
4. `val_recall_at_10` is computed and logged.
5. Top-5 predicted label frequency is saved or logged.
6. Top-10 predicted label frequency is saved or logged.
7. True label frequency top-20 is saved or logged.
8. Existing `macro_AP`, `micro_AP`, `F1@0.5`, and probability stats remain unchanged.
9. Checkpoint monitors remain based on `val_macro_AP`, `val_micro_AP`, `val_loss`, and `last`.
10. Tests pass.

---

# Part J — Avoid These Mistakes

- Do not replace macro AP or micro AP with top-k metrics.
- Do not use top-k metrics as primary checkpoint monitors.
- Do not apply a sigmoid threshold when computing top-k; use ranking.
- Do not compute Hit@K as exact match of the full label set.
- Do not compute Recall@K as `hits / K`; that is Precision@K, not Recall@K.
- Do not ignore true label cardinality in Recall@K.
- Do not omit top-k label frequency diagnostics.
- Do not hide class-prior collapse by reporting only Hit@K.
- Do not implement class-wise threshold optimization in this task.
- Do not break existing validation diagnostics.

---

# Final Intended Behavior

At each FSD50K supervised validation epoch, logs should include:

```text
Val macro AP
Val micro AP
Val macro F1@0.5
Val hit@5
Val hit@10
Val recall@5
Val recall@10
Probability statistics
Top-5 predicted label frequency
Top-10 predicted label frequency
True label frequency top-20
```

These diagnostics should make it clear whether the model is learning sample-specific semantic ranking or simply predicting the same frequent/prior labels for most clips.
