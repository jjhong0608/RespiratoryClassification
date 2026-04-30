# Codex Instructions: Round G Experiments After D/E/F Analysis

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

The current branch is already an **event-MIL-first** model. Do **not** rewrite the architecture from scratch. Do **not** reintroduce Hugging Face `ASTModel`. Do **not** reintroduce latent-query pooling.

The goal of this task is to prepare **Round G** experiments after the D/E/F result analysis.

Round G should treat the following configuration as the current mainline:

```text
D3 direct B3 low-LR
```

Meaning:

```text
RDT on
RDT steps = 3
top_tokens_per_branch = 2
4-scale branches
BCE loss
branch auxiliary enabled with weight = 0.1
encoder_lr = 1e-5
head_lr = 3e-4
no warm-start
best_loss checkpoint as primary model-selection criterion
```

---

## Current Branch State to Preserve

The current active encoder type is:

```text
multiscale_rdt_ast
```

The active architecture is event-MIL-first:

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

Default branch geometry:

| Branch | Patch size | Stride | Patch tokens | Temporal event tokens |
|---:|---:|---:|---:|---:|
| 0 | `(16, 16)` | `(8, 16)` | `1016` | `127` |
| 1 | `(8, 32)` | `(4, 32)` | `1020` | `255` |
| 2 | `(4, 64)` | `(2, 64)` | `1022` | `511` |
| 3 | `(2, 128)` | `(1, 128)` | `1023` | `1023` |

Default context and evidence state:

```text
H_ctx length = 127 + 255 + 511 + 1023 = 1916
top_tokens_per_branch = 2
U0 = [B, 8, D]
```

Keep these current conventions:

```text
RDT off == rdt.enabled = false
branch auxiliary config lives under train.loss.branch_auxiliary
classifier.pooling remains latent_mean
legacy latent_query_count is rejected
best_loss checkpoint is the primary comparison checkpoint
```

---

# Background: What D/E/F Showed

The D/E/F results indicate:

1. **D3 direct B3 low-LR is the current best candidate.**
   - It outperformed the staged C3 variants and the E/F variants.
   - It produced the best overall combination of:
     - F1@0.5,
     - F1@opt,
     - ROC-AUC,
     - PR-AUC,
     - Brier score,
     - balanced accuracy.

2. **Staged warm-start is no longer the default mainline.**
   - D2 showed warm-start + low LR with RDT off is not sufficient.
   - D3 showed direct low-LR RDT training can be much stronger.

3. **top_tokens_per_branch = 2 should remain the default.**
   - top-k = 1, 3, and 4 did not improve over top-k = 2.

4. **4-scale should remain the default.**
   - 3-scale staged experiments underperformed.
   - Branch-level diagnostics showed the `2×128` branch can be highly informative.

5. **Evidence-score source changes should be deprioritized for now.**
   - attention_logit, instance_logit, attention temperature, and entropy regularization did not beat D3 in the previous round.
   - Keep `evidence_score_source = "attention_weight"` for Round G unless a config explicitly tests otherwise.

6. **The next priority is robustness and direct ablation of the D3 mainline.**
   - D3 is still single-seed evidence.
   - It must be tested across seeds.
   - It also needs direct low-LR no-RDT and 3-scale controls.

---

# Round G Goals

Add config files and any required code support for these Round G experiments:

| ID | Purpose |
|---|---|
| **G1** | Repeat D3 direct low-LR across multiple seeds |
| **G2** | Direct low-LR, RDT off, no warm-start |
| **G3** | Direct low-LR, 3-scale only |
| **G4** | Direct low-LR, aux weight = 0.05 |
| **G5** | Direct low-LR, branch auxiliary off |
| **G6** | Direct low-LR, focal loss gamma = 1.0 |
| **G7** | Direct low-LR, focal loss gamma = 2.0 |

These experiments should be runnable by config only.

---

# Part A — Add Round G Configs

Create these configs:

```text
configs/training_g1_d3_seed0.json
configs/training_g1_d3_seed1.json
configs/training_g1_d3_seed2.json
configs/training_g1_d3_seed42.json
configs/training_g1_d3_seed43.json

configs/training_g2_direct_low_lr_no_rdt.json
configs/training_g3_direct_low_lr_3scale.json
configs/training_g4_direct_low_lr_aux005.json
configs/training_g5_direct_low_lr_no_aux.json
configs/training_g6_direct_low_lr_focal_gamma1.json
configs/training_g7_direct_low_lr_focal_gamma2.json
```

If the project naming style prefers shorter names, keep the same semantic content.

Use existing working configs as templates. Preserve:

- dataset paths,
- label mapping,
- preprocessing,
- sampler,
- output directory conventions,
- evaluation/diagnostic settings,
- checkpoint settings,
- threshold-optimization behavior.

Do **not** invent local paths.

---

## Shared D3-Mainline Base Settings

All G-series configs should start from this base unless explicitly overridden.

### Model

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
        "enabled": true,
        "steps": 3,
        "top_tokens_per_branch": 2,
        "gated_residual": true,
        "layerscale_init": 0.01,
        "evidence_score_source": "attention_weight",
        "exclude_branches_from_evidence": []
      },
      "patch_branches": [
        { "patch_size": [16, 16], "stride": [8, 16] },
        { "patch_size": [8, 32], "stride": [4, 32] },
        { "patch_size": [4, 64], "stride": [2, 64] },
        { "patch_size": [2, 128], "stride": [1, 128] }
      ]
    }
  },
  "classifier": {
    "type": "linear",
    "hidden_dim": 256,
    "dropout": 0.1,
    "pooling": "latent_mean"
  }
}
```

If the current branch does not yet support `evidence_score_source` or `exclude_branches_from_evidence`, either:

1. add support according to the previous D/E/F instruction, or
2. omit those fields from Round G configs and keep the existing default behavior.

Preferred: keep the fields if already supported.

### Training

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
    "gamma": 2.0,
    "branch_auxiliary": {
      "enabled": true,
      "weight": 0.1,
      "aggregation": "mean"
    }
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

Important:

```text
Round G direct experiments should not warm-start unless explicitly stated.
For G1-G7, checkpoint_path should stay null.
load_model_state should be false.
```

---

# G1 — D3 Repeat Seeds

Purpose:

```text
Check whether the D3 direct B3 low-LR result is robust across seeds.
```

Create:

```text
configs/training_g1_d3_seed0.json
configs/training_g1_d3_seed1.json
configs/training_g1_d3_seed2.json
configs/training_g1_d3_seed42.json
configs/training_g1_d3_seed43.json
```

Each config should be identical to the D3-mainline base except for:

```json
"experiment": {
  "name": "respiratory_g1_d3_seed0"
}
```

and the seed.

Use the existing seed mechanism from the repository.

Do **not** invent an unsupported seed field.

Inspect existing configs and config schema first. If the current schema uses:

```json
"experiment": {
  "seed": 42
}
```

then set:

```json
"experiment": {
  "name": "respiratory_g1_d3_seed0",
  "seed": 0
}
```

If the seed field is elsewhere, use the existing location.

Expected seed values:

```text
0, 1, 2, 42, 43
```

Primary comparison:

```text
mean ± std over best_loss checkpoints
```

Metrics:

```text
F1@0.5
F1@opt
ROC-AUC
PR-AUC
Brier
balanced accuracy
optimal threshold
```

---

# G2 — Direct Low-LR No-RDT Control

Purpose:

```text
Test whether D3 improves because of RDT or simply because of the low-LR schedule.
```

File:

```text
configs/training_g2_direct_low_lr_no_rdt.json
```

Settings:

```json
"experiment": {
  "name": "respiratory_g2_direct_low_lr_no_rdt"
}
```

Change only:

```json
"rdt": {
  "enabled": false,
  "steps": 3,
  "top_tokens_per_branch": 2,
  "gated_residual": true,
  "layerscale_init": 0.01,
  "evidence_score_source": "attention_weight",
  "exclude_branches_from_evidence": []
}
```

Keep:

```text
no warm-start
BCE
branch_auxiliary.enabled = true
branch_auxiliary.weight = 0.1
encoder_lr = 1e-5
head_lr = 3e-4
4-scale
top_tokens_per_branch = 2
```

Interpretation:

```text
If G2 ~= G1/D3, RDT is not the main cause.
If G2 < G1/D3, RDT contributes under low-LR direct training.
```

---

# G3 — Direct Low-LR 3-Scale Control

Purpose:

```text
Test whether the 2×128 branch is truly helpful under the D3 direct low-LR setting.
```

File:

```text
configs/training_g3_direct_low_lr_3scale.json
```

Settings:

```json
"experiment": {
  "name": "respiratory_g3_direct_low_lr_3scale"
}
```

Use RDT on, same as D3:

```json
"rdt": {
  "enabled": true,
  "steps": 3,
  "top_tokens_per_branch": 2,
  "gated_residual": true,
  "layerscale_init": 0.01,
  "evidence_score_source": "attention_weight",
  "exclude_branches_from_evidence": []
}
```

Remove the final `2×128` branch:

```json
"patch_branches": [
  { "patch_size": [16, 16], "stride": [8, 16] },
  { "patch_size": [8, 32], "stride": [4, 32] },
  { "patch_size": [4, 64], "stride": [2, 64] }
]
```

Expected dynamic shapes:

```text
H_ctx length = 127 + 255 + 511 = 893
selected evidence length = 3 branches × 2 = 6
U0: [B, 6, D]
```

The model must not assume exactly 4 branches or exactly 8 selected evidence tokens.

Interpretation:

```text
If G3 < D3, keep 2×128 branch.
If G3 ~= or > D3, revisit 2×128 branch.
```

---

# G4 — Direct Low-LR Aux Weight 0.05

Purpose:

```text
Test whether branch auxiliary weight 0.1 is too strong under D3.
```

File:

```text
configs/training_g4_direct_low_lr_aux005.json
```

Settings:

```json
"experiment": {
  "name": "respiratory_g4_direct_low_lr_aux005"
}
```

Change:

```json
"branch_auxiliary": {
  "enabled": true,
  "weight": 0.05,
  "aggregation": "mean"
}
```

Keep everything else the same as D3-mainline.

Interpretation:

```text
If G4 > D3, lower aux weight is better.
If G4 < D3, keep aux weight 0.1.
```

---

# G5 — Direct Low-LR No Branch Auxiliary Loss

Purpose:

```text
Test whether branch auxiliary loss is still necessary under D3.
```

File:

```text
configs/training_g5_direct_low_lr_no_aux.json
```

Settings:

```json
"experiment": {
  "name": "respiratory_g5_direct_low_lr_no_aux"
}
```

Set:

```json
"branch_auxiliary": {
  "enabled": false,
  "weight": 0.1,
  "aggregation": "mean"
}
```

or, if the current validator requires positive weight only when enabled:

```json
"branch_auxiliary": {
  "enabled": false,
  "weight": 0.0,
  "aggregation": "mean"
}
```

Use the currently valid schema. Prefer not to break validation.

Keep:

```text
RDT on
steps = 3
top_tokens_per_branch = 2
BCE
encoder_lr = 1e-5
head_lr = 3e-4
4-scale
no warm-start
```

Interpretation:

```text
If G5 ~= D3, branch aux may be unnecessary.
If G5 < D3, branch aux helps.
```

---

# G6 — Direct Low-LR Focal Loss Gamma 1.0

Purpose:

```text
Test whether focal loss can recover ranking strength under the low-LR D3 setting without severe calibration collapse.
```

File:

```text
configs/training_g6_direct_low_lr_focal_gamma1.json
```

Settings:

```json
"experiment": {
  "name": "respiratory_g6_direct_low_lr_focal_gamma1"
}
```

Change:

```json
"loss": {
  "type": "focal",
  "auto_pos_weight": false,
  "pos_weight": null,
  "gamma": 1.0,
  "branch_auxiliary": {
    "enabled": true,
    "weight": 0.1,
    "aggregation": "mean"
  }
}
```

Keep all other D3-mainline settings.

Interpretation:

```text
If G6 improves ROC/PR without hurting Brier/threshold too much, focal gamma=1 may be useful.
```

---

# G7 — Direct Low-LR Focal Loss Gamma 2.0

Purpose:

```text
Retest focal gamma=2.0 under low-LR D3 conditions.
```

File:

```text
configs/training_g7_direct_low_lr_focal_gamma2.json
```

Settings:

```json
"experiment": {
  "name": "respiratory_g7_direct_low_lr_focal_gamma2"
}
```

Change:

```json
"loss": {
  "type": "focal",
  "auto_pos_weight": false,
  "pos_weight": null,
  "gamma": 2.0,
  "branch_auxiliary": {
    "enabled": true,
    "weight": 0.1,
    "aggregation": "mean"
  }
}
```

Keep all other D3-mainline settings.

Interpretation:

```text
If G7 improves ranking but hurts Brier/threshold, keep BCE or gamma=1.
```

---

# Part B — Required Code Checks and Possible Fixes

Round G should be mostly config-only. Still, check the following code behavior and fix if needed.

---

## 1. Seed Handling

G1 requires deterministic seed variation.

Inspect the current config schema and training entrypoint.

If seed support already exists:

```text
Use the existing seed field.
```

If seed support does not exist, add it.

Preferred location:

```json
"experiment": {
  "seed": 42
}
```

Required behavior:

- seed Python `random`,
- seed NumPy,
- seed PyTorch CPU,
- seed PyTorch CUDA if available,
- make DataLoader generator deterministic if the current project supports it,
- keep the current worker init behavior if already implemented.

Do not add seed logic in a way that breaks existing reproducibility behavior.

---

## 2. Dynamic Branch Count

G3 requires 3-scale direct low-LR.

Verify:

- model forward supports 3 branches,
- trainer supports branch logits of shape `[B, 3]` for binary,
- branch auxiliary loss supports 3 branches,
- diagnostics support 3 branches,
- selected evidence length can be 6,
- README/tests do not hard-code 4 branches.

Fix any hard-coding.

Expected G3 shapes:

```text
H_ctx length = 893
selected evidence length = 6
```

---

## 3. Dynamic Selected Evidence Length

The model must support selected evidence lengths:

```text
4-scale top-2 -> 8
3-scale top-2 -> 6
```

Even if Round G does not use top-1/top-3/top-4, keep the previous dynamic support intact.

Do not assume selected evidence length is always 8.

---

## 4. RDT Off Direct Path

G2 requires:

```json
"rdt": {
  "enabled": false
}
```

with no warm-start and low LR.

Verify forward still works when RDT is off:

- selected evidence tokens are still produced,
- evidence embedding is computed from selected evidence tokens,
- final fusion/classifier still works,
- diagnostics still include evidence indices/scores/branch IDs if available.

---

## 5. Branch Auxiliary Off Path

G5 requires:

```json
"branch_auxiliary": {
  "enabled": false
}
```

Verify trainer loss behavior:

```text
total_loss == final_loss
```

and no code tries to require branch logits for auxiliary loss when disabled.

The model can still return branch logits for diagnostics/fusion.

---

## 6. Focal Loss Under Low LR

G6/G7 require focal loss under the D3 low-LR setup.

Verify current loss builder supports:

```json
"loss": {
  "type": "focal",
  "gamma": 1.0
}
```

and:

```json
"loss": {
  "type": "focal",
  "gamma": 2.0
}
```

with branch auxiliary loss using the same criterion.

---

## 7. Initialization Must Stay Off

G1-G7 direct configs must not warm-start.

Make sure each config uses:

```json
"initialization": {
  "checkpoint_path": null,
  "load_model_state": false,
  "strict": false,
  "load_optimizer_state": false
}
```

or equivalent valid schema.

This prevents accidentally running staged training.

---

# Part C — README Updates

Update README with a new section:

```text
Round G Experiments
```

Add a table:

| Study | Config | Purpose |
|---|---|---|
| G1 seed0/1/2/42/43 | `configs/training_g1_d3_seed*.json` | Repeat D3 direct low-LR |
| G2 | `configs/training_g2_direct_low_lr_no_rdt.json` | Direct no-RDT control |
| G3 | `configs/training_g3_direct_low_lr_3scale.json` | Direct 3-scale control |
| G4 | `configs/training_g4_direct_low_lr_aux005.json` | Aux weight 0.05 |
| G5 | `configs/training_g5_direct_low_lr_no_aux.json` | No branch aux |
| G6 | `configs/training_g6_direct_low_lr_focal_gamma1.json` | Focal gamma 1.0 |
| G7 | `configs/training_g7_direct_low_lr_focal_gamma2.json` | Focal gamma 2.0 |

Document interpretation:

```text
G1 checks robustness.
G2 tests RDT contribution under low LR.
G3 tests the 2×128 branch.
G4/G5 test branch auxiliary importance.
G6/G7 retest focal loss under low LR.
```

Document that all comparisons should prioritize:

```text
best_loss_*.pt
```

and report:

```text
F1@0.5
F1@opt
ROC-AUC
PR-AUC
Brier
balanced accuracy
optimal threshold
```

---

# Part D — Tests to Add or Update

Add tests or update existing tests.

## 1. Config Parsing Tests

Ensure all Round G configs parse:

```text
training_g1_d3_seed0.json
training_g1_d3_seed1.json
training_g1_d3_seed2.json
training_g1_d3_seed42.json
training_g1_d3_seed43.json
training_g2_direct_low_lr_no_rdt.json
training_g3_direct_low_lr_3scale.json
training_g4_direct_low_lr_aux005.json
training_g5_direct_low_lr_no_aux.json
training_g6_direct_low_lr_focal_gamma1.json
training_g7_direct_low_lr_focal_gamma2.json
```

## 2. Seed Config Test

If seed support is added or modified, test that config parsing exposes the seed and training setup applies it.

## 3. G2 Forward Test

Use a tiny or normal config with:

```text
rdt.enabled = false
```

Assert:

- forward succeeds,
- output logits shape is correct,
- selected evidence metadata is present if diagnostics are enabled.

## 4. G3 3-Scale Shape Test

Use 3 patch branches.

Assert:

```text
temporal context length = 893
selected evidence length = 6
```

If the code does not expose context length directly, test output selected evidence shape:

```text
selected_evidence_tokens.shape[1] == 6
selected_evidence_indices.shape[1] == 6
selected_evidence_branch_ids.shape[1] == 6
```

## 5. G5 Aux Off Trainer Loss Test

Assert:

```text
branch_auxiliary.enabled = false
total_loss == final_loss
```

## 6. Focal Loss Config Test

Assert `gamma = 1.0` and `gamma = 2.0` focal configs build loss successfully.

---

# Part E — Acceptance Criteria

The task is complete when:

1. All G1-G7 config files exist.
2. G1 configs differ only by seed and experiment name.
3. G2 is direct low-LR RDT-off control.
4. G3 is direct low-LR 3-scale RDT-on control.
5. G4 uses branch aux weight 0.05.
6. G5 disables branch auxiliary loss.
7. G6 uses focal loss gamma 1.0.
8. G7 uses focal loss gamma 2.0.
9. All G-series configs avoid warm-start.
10. Model and trainer still support RDT on/off.
11. Model and trainer support branch aux on/off.
12. Model supports 3-scale branch count.
13. Model supports dynamic selected evidence length.
14. README documents Round G.
15. Tests pass.

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

- Do not use staged warm-start for G1-G7.
- Do not accidentally copy C3 stage2 initialization into G configs.
- Do not remove the `2×128` branch except in G3.
- Do not change `top_tokens_per_branch = 2` in Round G.
- Do not reintroduce latent-query pooling.
- Do not assume selected evidence length is always 8.
- Do not assume number of branches is always 4.
- Do not use `best_f1` as the primary comparison criterion.
- Do not set branch aux weight to `0.0` with `enabled=true`; use `enabled=false` for G5 unless the validator requires a placeholder positive weight.
- Do not break binary logits shape `[B]`.

---

# Suggested Execution Order After Implementation

Run in this order:

1. **G2** — direct low-LR no-RDT control.
2. **G3** — direct low-LR 3-scale control.
3. **G4/G5** — branch aux sensitivity.
4. **G6/G7** — focal loss sweep.
5. **G1 seeds** — repeat the final mainline D3/G1 setting.

If compute allows, run G1 seeds earlier. But G2 and G3 are the most important ablations.

---

# Final Note

Round G is not another architecture refactor.

Round G should answer:

```text
Is D3 direct B3 low-LR robust?
Does RDT actually contribute under low LR?
Is the 2×128 branch necessary?
How sensitive is D3 to branch auxiliary weight?
Does focal loss become useful again under low LR?
```

The current mainline hypothesis is:

```text
RDT on + 4-scale + top-k=2 + BCE + branch_aux=0.1 + low LR is the best current recipe.
```

Round G exists to validate or falsify that hypothesis.
