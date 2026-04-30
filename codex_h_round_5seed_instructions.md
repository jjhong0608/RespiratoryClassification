# Codex Instructions: H Round 5-Seed Experiments for Final Candidate Selection

## Target Repository and Branch

Repository:

```text
https://github.com/jjhong0608/RespiratoryClassification.git
```

Base branch:

```text
MIL-RDT-AST
```

This instruction is based on the latest `MIL-RDT-AST` branch.

The current branch is already an **event-MIL-first** respiratory sound classifier. Do **not** rewrite the architecture from scratch. Do **not** reintroduce Hugging Face `ASTModel`. Do **not** reintroduce latent-query pooling.

The goal of this task is to prepare **H Round**, a 5-seed ablation round used to narrow the final Event-MIL-RDT candidate before adding more data and augmentation.

---

## Current Branch State to Preserve

The latest README describes the active model as:

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
  -> evidence mean pooling
  -> fuse evidence embedding, mean branch embedding, and branch logits
  -> fusion projector
[B, D]
  -> classifier
```

The active encoder type must remain:

```text
multiscale_rdt_ast
```

The current config conventions must remain:

```text
RDT off == model.encoder.architecture.rdt.enabled = false
branch auxiliary settings live under train.loss.branch_auxiliary
classifier.pooling remains latent_mean
legacy latent_query_count is rejected
legacy summary_tokens_per_scale is rejected
legacy flat rdt_steps is rejected
best_loss checkpoint is the primary comparison checkpoint
```

Do not change these conventions.

---

## Why H Round Is Needed

Round G showed that the D3/G1 mainline is a strong custom Event-MIL baseline, but three structural choices remain unresolved:

1. **Is RDT actually needed?**
   - G2 no-RDT was strong in a single seed.
   - Need 5 seeds.

2. **Is branch auxiliary loss actually needed?**
   - G5 no-aux was strong in a single seed.
   - Need 5 seeds.

3. **Is the 2×128 branch actually needed?**
   - G3 3-scale was strong in a single seed.
   - Need 5 seeds.

H Round should answer those three questions.

---

## H Round Overview

Use the already-completed G1/D3 5-seed mainline as the reference if available.

Reference H0 / G1 mainline:

```text
4-scale
RDT on
RDT steps = 3
top_tokens_per_branch = 2
BCE loss
branch auxiliary enabled, weight = 0.1
encoder_lr = 1e-5
head_lr = 3e-4
no warm-start
5 seeds
```

H Round adds:

| Round | Purpose | Difference from H0/G1 |
|---|---|---|
| **H1** | Test RDT necessity | RDT off |
| **H2** | Test branch auxiliary necessity | branch auxiliary off |
| **H3** | Test 2×128 branch necessity | 3-scale only |

Use 5 seeds for each:

```text
0, 1, 2, 42, 43
```

---

## Fixed Common Settings for H Round

All H-round configs should use direct training only.

Do **not** warm-start.

Use:

```json
"train": {
  "epochs": 40,
  "optimizer": {
    "encoder_lr": 0.00001,
    "head_lr": 0.0003,
    "weight_decay": 0.01
  },
  "loss": {
    "type": "bce",
    "auto_pos_weight": false,
    "pos_weight": null,
    "gamma": 2.0
  },
  "early_stopping": {
    "enabled": true,
    "monitor": "val_loss",
    "patience": 8,
    "min_delta": 0.00001
  },
  "initialization": {
    "checkpoint_path": null,
    "load_model_state": false,
    "strict": false,
    "load_optimizer_state": false
  }
}
```

Keep the existing data paths, label mapping, preprocessing, split paths, sampler settings, analysis outputs, checkpoint settings, and output directory conventions from the current working configs.

Do **not** invent local paths.

---

## Default 4-Scale Geometry

For H1 and H2, keep the default 4 branches:

```json
"patch_branches": [
  { "patch_size": [16, 16], "stride": [8, 16] },
  { "patch_size": [8, 32], "stride": [4, 32] },
  { "patch_size": [4, 64], "stride": [2, 64] },
  { "patch_size": [2, 128], "stride": [1, 128] }
]
```

Expected default shapes:

```text
Temporal event lengths:
  branch 0: 127
  branch 1: 255
  branch 2: 511
  branch 3: 1023

H_ctx length = 127 + 255 + 511 + 1023 = 1916

top_tokens_per_branch = 2
selected evidence length = 4 branches × 2 = 8
U0 = [B, 8, D]
```

---

## H1 — No-RDT 5 Seeds

### Purpose

Determine whether RDT is necessary.

Compare:

```text
H0/G1: RDT on
H1:    RDT off
```

### Configs to create

```text
configs/training_h1_no_rdt_seed0.json
configs/training_h1_no_rdt_seed1.json
configs/training_h1_no_rdt_seed2.json
configs/training_h1_no_rdt_seed42.json
configs/training_h1_no_rdt_seed43.json
```

### Shared H1 settings

Use 4-scale geometry.

Set:

```json
"model": {
  "encoder": {
    "type": "multiscale_rdt_ast",
    "adaptation": {
      "mode": "full",
      "num_layers": 0
    },
    "architecture": {
      "hidden_size": 192,
      "num_attention_heads": 4,
      "mlp_ratio": 2.0,
      "hidden_dropout_prob": 0.1,
      "attention_probs_dropout_prob": 0.1,
      "layer_norm_eps": 1e-6,
      "shared_stem_depth": 2,
      "adapter_depth": 1,
      "rdt": {
        "enabled": false,
        "steps": 3,
        "top_tokens_per_branch": 2,
        "gated_residual": true,
        "layerscale_init": 0.01
      }
    }
  }
}
```

Keep branch auxiliary enabled:

```json
"branch_auxiliary": {
  "enabled": true,
  "weight": 0.1,
  "aggregation": "mean"
}
```

Each config should differ only by experiment name and seed.

Example:

```json
"experiment": {
  "name": "respiratory_h1_no_rdt_seed0",
  "seed": 0
}
```

Use the repository’s current seed field. If the seed field is not under `experiment`, use the actual schema from the repo. Do not invent an unsupported field.

### Interpretation

| Result | Interpretation |
|---|---|
| H1 ≈ H0/G1 | RDT is not essential |
| H1 > H0/G1 | RDT may be hurting |
| H1 < H0/G1 | RDT should remain in final candidate |
| H1 has lower std | RDT-off may be more stable |

---

## H2 — Branch Auxiliary Off 5 Seeds

### Purpose

Determine whether branch auxiliary supervision is necessary.

Compare:

```text
H0/G1: branch auxiliary on, weight 0.1
H2:    branch auxiliary off
```

### Configs to create

```text
configs/training_h2_no_aux_seed0.json
configs/training_h2_no_aux_seed1.json
configs/training_h2_no_aux_seed2.json
configs/training_h2_no_aux_seed42.json
configs/training_h2_no_aux_seed43.json
```

### Shared H2 settings

Use 4-scale geometry.

Keep RDT on:

```json
"rdt": {
  "enabled": true,
  "steps": 3,
  "top_tokens_per_branch": 2,
  "gated_residual": true,
  "layerscale_init": 0.01
}
```

Disable branch auxiliary loss:

```json
"branch_auxiliary": {
  "enabled": false,
  "weight": 0.1,
  "aggregation": "mean"
}
```

If the current validator requires `weight > 0` even when disabled, keep `weight = 0.1` and ensure the trainer ignores auxiliary loss when `enabled = false`.

Do not set `enabled = true` with `weight = 0.0`.

Each config should differ only by experiment name and seed.

Example:

```json
"experiment": {
  "name": "respiratory_h2_no_aux_seed0",
  "seed": 0
}
```

### Interpretation

| Result | Interpretation |
|---|---|
| H2 ≈ H0/G1 | branch auxiliary is not essential for final performance |
| H2 > H0/G1 | auxiliary supervision may constrain final fusion/RDT |
| H2 < H0/G1 | auxiliary supervision should remain |
| H2 has high ROC/PR but low F1@0.5 | threshold/calibration issue; inspect Brier and opt threshold |

---

## H3 — 3-Scale 5 Seeds

### Purpose

Determine whether the `2×128` branch is necessary.

Compare:

```text
H0/G1: 4-scale
H3:    3-scale without 2×128
```

### Configs to create

```text
configs/training_h3_3scale_seed0.json
configs/training_h3_3scale_seed1.json
configs/training_h3_3scale_seed2.json
configs/training_h3_3scale_seed42.json
configs/training_h3_3scale_seed43.json
```

### Shared H3 settings

Use only the first 3 branches:

```json
"patch_branches": [
  { "patch_size": [16, 16], "stride": [8, 16] },
  { "patch_size": [8, 32], "stride": [4, 32] },
  { "patch_size": [4, 64], "stride": [2, 64] }
]
```

Keep RDT on:

```json
"rdt": {
  "enabled": true,
  "steps": 3,
  "top_tokens_per_branch": 2,
  "gated_residual": true,
  "layerscale_init": 0.01
}
```

Keep branch auxiliary enabled:

```json
"branch_auxiliary": {
  "enabled": true,
  "weight": 0.1,
  "aggregation": "mean"
}
```

Expected dynamic shapes:

```text
Temporal event lengths:
  branch 0: 127
  branch 1: 255
  branch 2: 511

H_ctx length = 127 + 255 + 511 = 893

top_tokens_per_branch = 2
selected evidence length = 3 branches × 2 = 6
U0 = [B, 6, D]
```

The model, trainer, diagnostics, and tests must not assume exactly 4 branches or exactly 8 evidence tokens.

Each config should differ only by experiment name and seed.

Example:

```json
"experiment": {
  "name": "respiratory_h3_3scale_seed0",
  "seed": 0
}
```

### Interpretation

| Result | Interpretation |
|---|---|
| H3 ≈ H0/G1 | 2×128 branch is not essential |
| H3 > H0/G1 | 2×128 branch may be noisy or unnecessary |
| H3 < H0/G1 | keep 4-scale |
| H3 has lower std | 3-scale may be a more stable compact candidate |

---

## H0 / G1 Reference

Do not recreate H0 if G1 configs and results already exist.

H0/G1 reference is:

```text
4-scale
RDT on
RDT steps = 3
top_tokens_per_branch = 2
BCE
branch auxiliary enabled, weight = 0.1
encoder_lr = 1e-5
head_lr = 3e-4
no warm-start
seeds = 0, 1, 2, 42, 43
```

If the codebase changed after G1, or seed handling/reproducibility was modified, then also regenerate H0 configs:

```text
configs/training_h0_mainline_seed0.json
configs/training_h0_mainline_seed1.json
configs/training_h0_mainline_seed2.json
configs/training_h0_mainline_seed42.json
configs/training_h0_mainline_seed43.json
```

Otherwise, reference existing G1 results.

---

# Code Checks and Required Fixes

H Round should be mostly config-only. Still, check and fix the following if necessary.

## 1. Seed Handling

H Round requires meaningful 5-seed comparisons.

Inspect the current schema and training entrypoint.

If seed support already exists, use it.

If seed support does not exist or is incomplete, add it.

Preferred config location:

```json
"experiment": {
  "seed": 42
}
```

Required behavior:

```python
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

If safe in this codebase, also support optional deterministic mode:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Do not force `torch.use_deterministic_algorithms(True)` unconditionally if it breaks CUDA ops. If added, put it behind a config flag.

DataLoader reproducibility should be checked:

- seeded generator if the current project supports it,
- stable worker initialization,
- deterministic shuffling given seed.

Do not break existing pipeline behavior.

---

## 2. Dynamic Branch Count

H3 requires 3 branches.

Verify:

- model forward works with 3 branches,
- branch logits can be `[B, 3]` for binary classification,
- branch auxiliary loss works with 3 branches,
- final fusion works with 3 branch logits/embeddings,
- diagnostics can serialize 3 branches,
- selected evidence length can be 6,
- tests do not hard-code 4 branches.

Fix any hard-coded branch count.

---

## 3. Dynamic Selected Evidence Length

H1/H2 use selected evidence length 8.

H3 uses selected evidence length 6.

Verify:

- RDT accepts `[B, 6, D]` for H3,
- evidence mean pooling works for arbitrary selected length,
- diagnostics support arbitrary selected length,
- output dataclass fields support dynamic selected length.

Do not assume selected evidence length is always 8.

---

## 4. RDT-Off Forward Path

H1 requires `rdt.enabled = false`.

Verify:

- forward succeeds,
- no RDT block is called,
- selected evidence tokens are still produced,
- evidence embedding is computed from selected evidence tokens,
- final fusion/classifier works,
- diagnostics still include selected evidence metadata if present.

---

## 5. Branch-Aux-Off Trainer Path

H2 requires:

```json
"branch_auxiliary": {
  "enabled": false
}
```

Verify:

```text
total_loss == final_loss
```

when disabled.

Do not require branch logits for auxiliary loss when disabled.

The model may still return branch logits for fusion/diagnostics.

---

# README Updates

Update README with a new section:

```text
H Round: 5-Seed Final Candidate Ablation
```

Add a table:

| Round | Configs | Purpose |
|---|---|---|
| H1 | `configs/training_h1_no_rdt_seed*.json` | Test RDT necessity |
| H2 | `configs/training_h2_no_aux_seed*.json` | Test branch auxiliary necessity |
| H3 | `configs/training_h3_3scale_seed*.json` | Test 2×128 branch necessity |

Document:

```text
H0/G1 is the reference mainline.
Use best_loss_*.pt as the primary checkpoint.
Compare mean, std, min, max, and worst seed.
Do not choose final candidate by a single best seed.
```

Add interpretation rules:

```text
H1 vs H0/G1 -> RDT decision
H2 vs H0/G1 -> auxiliary decision
H3 vs H0/G1 -> scale/2×128 branch decision
```

Also document dynamic shapes:

```text
4-scale top-2 -> U0 [B, 8, D], H_ctx length 1916
3-scale top-2 -> U0 [B, 6, D], H_ctx length 893
```

---

# Optional Utility Script

If convenient, add or update a summary utility script to aggregate H Round results.

Suggested path:

```text
scripts/summarize_h_round.py
```

Purpose:

```text
Collect best_loss eval metrics for H1/H2/H3 and G1/H0 reference.
Report mean/std/min/max across seeds.
```

Suggested output columns:

```text
group
seed
checkpoint
f1_at_0.5
f1_opt
roc_auc
pr_auc
brier
balanced_accuracy
optimal_threshold
```

Aggregate:

```text
mean
std
min
max
worst_seed
```

If such a script already exists, extend it instead of creating a new one.

This utility is optional but recommended.

---

# Tests to Add or Update

## 1. Config Parsing Tests

Ensure all new H configs parse:

```text
training_h1_no_rdt_seed0.json
training_h1_no_rdt_seed1.json
training_h1_no_rdt_seed2.json
training_h1_no_rdt_seed42.json
training_h1_no_rdt_seed43.json

training_h2_no_aux_seed0.json
training_h2_no_aux_seed1.json
training_h2_no_aux_seed2.json
training_h2_no_aux_seed42.json
training_h2_no_aux_seed43.json

training_h3_3scale_seed0.json
training_h3_3scale_seed1.json
training_h3_3scale_seed2.json
training_h3_3scale_seed42.json
training_h3_3scale_seed43.json
```

If H0 configs are regenerated, test those too.

---

## 2. H1 RDT-Off Forward Test

Use a small or default config with:

```text
rdt.enabled = false
```

Assert:

- model forward succeeds,
- logits shape is correct,
- pooled embedding shape is correct,
- selected evidence metadata is present if the model normally returns it,
- no RDT module is required in the forward path.

---

## 3. H2 Branch-Aux-Off Loss Test

Use trainer loss composition.

Assert:

```text
branch_auxiliary.enabled = false
total_loss == final_loss
```

Also assert no error is raised if branch logits are absent when auxiliary loss is disabled.

---

## 4. H3 3-Scale Dynamic Shape Test

Use 3 patch branches.

Assert:

```text
selected_evidence_tokens.shape[1] == 6
selected_evidence_indices.shape[1] == 6
selected_evidence_branch_ids.shape[1] == 6
```

If context length is exposed, assert:

```text
H_ctx length == 893
```

Also verify branch logits shape for binary classification:

```text
branch_logits.shape == [B, 3]
```

or equivalent current shape.

---

## 5. Seed Parsing Test

If seed support is added or changed, test that:

- each config exposes the intended seed,
- seed values are `0, 1, 2, 42, 43`,
- training setup consumes the seed without error.

---

# Acceptance Criteria

The task is complete when:

1. H1/H2/H3 5-seed config files exist.
2. Each H1 config uses RDT off.
3. Each H2 config uses branch auxiliary off.
4. Each H3 config uses 3-scale patch branches.
5. All H configs use direct training with no warm-start.
6. All H configs use BCE, low LR, early stopping patience 8.
7. Dataset paths and preprocessing are copied from existing working configs.
8. The model supports H1 RDT-off forward.
9. The trainer supports H2 branch-aux-off loss.
10. The model supports H3 3-branch dynamic shape.
11. README documents H Round.
12. Tests pass.

---

# Commands to Run

Run:

```bash
pytest
```

Also run if configured:

```bash
ruff check src tests
```

If available:

```bash
mypy src tests
```

If any command fails due to environment limitations, document:

```text
- exact command
- exact failure
- what remains unverified
```

---

# Avoid These Mistakes

- Do not use staged warm-start in H Round.
- Do not accidentally copy C3 Stage 2 initialization into H configs.
- Do not use focal loss in H Round.
- Do not change `top_tokens_per_branch = 2`.
- Do not remove the 2×128 branch except in H3.
- Do not assume selected evidence length is always 8.
- Do not assume branch count is always 4.
- Do not reintroduce latent-query pooling.
- Do not set `rdt.steps = 0` to disable RDT; use `rdt.enabled = false`.
- Do not move branch auxiliary settings out of `train.loss.branch_auxiliary`.
- Do not choose final candidate by a single best seed.
- Do not use `best_f1` as the primary checkpoint-selection criterion.
- Do not break binary logits shape `[B]`.

---

# Recommended Execution Order

Run H Round in this order:

1. **H1** — RDT off 5 seeds.
2. **H2** — branch auxiliary off 5 seeds.
3. **H3** — 3-scale 5 seeds.

Reason:

```text
RDT necessity is the most important unresolved question.
Auxiliary loss is second.
2×128 branch/scale simplification is third.
```

After H Round:

```text
1. Compare H1/H2/H3 against H0/G1 using best_loss checkpoints.
2. Select 1–2 final structure candidates.
3. Apply more data and augmentation to those candidates.
4. After augmentation, re-check RDT/aux/scale ablations on the narrowed candidate set.
5. Only then move to 5-fold CV for the final model.
```

---

# Final Note

H Round is a **final candidate narrowing round**, not a new model-design round.

It should answer:

```text
Does Event-MIL need RDT?
Does Event-MIL need branch auxiliary loss?
Does Event-MIL need the 2×128 branch?
```

Use G1/D3 as the reference mainline, and compare H1/H2/H3 by:

```text
mean
std
min
max
worst seed
Brier
optimal-threshold stability
```

The final model should not be the one with only the best single seed. It should be the one with the best balance of mean performance and stability.
