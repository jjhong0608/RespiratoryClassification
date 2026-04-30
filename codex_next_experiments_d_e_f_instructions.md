# Codex Instructions: Next Experiment Plan After C0–C5

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

The goal of this task is to prepare the next round of experiments after C0–C5 by:

1. adding new D/E/F experiment configs,
2. adding only the code changes required for the new experimental knobs,
3. keeping the existing training/evaluation/CV/checkpoint pipeline intact,
4. improving evidence-selection diagnostics and configurability.

---

## Current Branch State to Preserve

The current README says the active encoder type is:

```text
multiscale_rdt_ast
```

The active architecture is:

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

Default active schema uses:

```json
"rdt": {
  "enabled": true,
  "steps": 3,
  "top_tokens_per_branch": 2,
  "gated_residual": true,
  "layerscale_init": 0.01
}
```

and branch auxiliary loss under:

```json
"train": {
  "loss": {
    "branch_auxiliary": {
      "enabled": true,
      "weight": 0.3,
      "aggregation": "mean"
    }
  }
}
```

Keep these semantics:

```text
RDT off == rdt.enabled = false
branch auxiliary settings live under train.loss.branch_auxiliary
classifier.pooling remains latent_mean
legacy latent_query_count is rejected
```

---

# Background: What C0–C5 Showed

The new C0–C5 results suggest:

1. **C3 Stage 2 is the current best candidate.**
   - C3 Stage 2 improved ROC-AUC, PR-AUC, Brier, F1@0.5, and balanced accuracy relative to the direct BCE/RDT runs.
   - However, C3 Stage 2 changed several things at once:
     - warm-start from C3 Stage 1,
     - RDT on,
     - lower encoder LR,
     - different head/RDT LR,
     - lower auxiliary weight.

2. **C1, C4-4scale, and C5-top2 are effectively duplicate configs.**
   - Avoid redundant future runs.

3. **C2 and C3-stage1 are effectively duplicate configs.**
   - Avoid redundant future runs.

4. **top_tokens_per_branch = 4 performed poorly.**
   - Do not keep expanding top-k unless evidence selection is improved.

5. **Evidence scores based on normalized attention weights appear too uniform.**
   - The next key code change is to support evidence selection by:
     - raw attention logits,
     - instance logits,
     - not only normalized attention weights.

6. **The 2×128 branch is not clearly bad.**
   - It is unstable, but it may carry useful ranking signal.
   - Test it under staged training before removing it permanently.

---

# Overall Next Plan

Create and support three groups of experiments:

## D-Series: C3 Causal/Ablation Experiments

Mostly config-only.

Purpose:

```text
Separate the effect of staged warm-start, low learning rate, RDT, top-k, and 3-scale/4-scale design.
```

## E-Series: Evidence Selection Improvements

Requires code changes.

Purpose:

```text
Improve top-k evidence selection by selecting evidence tokens using attention logits or instance logits instead of nearly uniform attention weights.
```

## F-Series: Branch 4 Control

Requires code changes.

Purpose:

```text
Control the unstable 2×128 branch using branch-specific auxiliary weights or by excluding it from RDT evidence selection while keeping it in fusion.
```

---

# Part A — Add D-Series Configs

Create these config files:

```text
configs/training_d1_c3_3scale_stage1.json
configs/training_d1_c3_3scale_stage2.json

configs/training_d2_stage2_no_rdt_from_stage1.json

configs/training_d3_direct_b3_low_lr.json

configs/training_d4_c3_top1_stage2.json
configs/training_d4_c3_top3_stage2.json

configs/training_d5_c3_stage1_seed0.json
configs/training_d5_c3_stage2_seed0.json
configs/training_d5_c3_stage1_seed1.json
configs/training_d5_c3_stage2_seed1.json
configs/training_d5_c3_stage1_seed2.json
configs/training_d5_c3_stage2_seed2.json
```

If the repo’s current config naming conventions require shorter names, keep the same semantic meaning.

Always copy dataset paths, label maps, preprocessing settings, output root, and other local paths from existing working configs. Do **not** invent paths.

---

## D1 — C3 Staged 3-Scale

Purpose:

```text
Check whether the 2×128 branch is beneficial under staged training.
```

### D1 Stage 1

File:

```text
configs/training_d1_c3_3scale_stage1.json
```

Settings:

```json
"experiment": {
  "name": "respiratory_d1_c3_3scale_stage1"
}
```

```json
"model": {
  "encoder": {
    "architecture": {
      "patch_branches": [
        {"patch_size": [16, 16], "stride": [8, 16]},
        {"patch_size": [8, 32], "stride": [4, 32]},
        {"patch_size": [4, 64], "stride": [2, 64]}
      ],
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

```json
"train": {
  "epochs": 40,
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
  }
}
```

Expected dynamic shape:

```text
H_ctx length = 127 + 255 + 511 = 893
top_tokens_per_branch = 2
selected evidence length = 3 × 2 = 6
```

### D1 Stage 2

File:

```text
configs/training_d1_c3_3scale_stage2.json
```

Settings:

```json
"experiment": {
  "name": "respiratory_d1_c3_3scale_stage2"
}
```

Same 3-scale `patch_branches`.

```json
"model": {
  "encoder": {
    "architecture": {
      "rdt": {
        "enabled": true,
        "steps": 3,
        "top_tokens_per_branch": 2,
        "gated_residual": true,
        "layerscale_init": 0.01
      }
    }
  }
}
```

Use warm-start:

```json
"train": {
  "epochs": 30,
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
      "weight": 0.05,
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
    "load_model_state": true,
    "strict": false,
    "load_optimizer_state": false
  }
}
```

`checkpoint_path` is intentionally `null` in the template. The user will fill it with the D1 Stage 1 `best_loss_*.pt`.

---

## D2 — C3 Stage 2 Warm-Start + Low LR, But RDT Off

Purpose:

```text
Determine whether C3 Stage 2 improvement comes from RDT or from warm-start + low-LR schedule.
```

File:

```text
configs/training_d2_stage2_no_rdt_from_stage1.json
```

Use the normal 4-scale branches.

Settings:

```json
"experiment": {
  "name": "respiratory_d2_stage2_no_rdt_from_stage1"
}
```

```json
"model": {
  "encoder": {
    "architecture": {
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

```json
"train": {
  "epochs": 30,
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
      "weight": 0.05,
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
    "load_model_state": true,
    "strict": false,
    "load_optimizer_state": false
  }
}
```

The user will fill `checkpoint_path` with the C3 Stage 1 4-scale `best_loss_*.pt`.

Interpretation:

```text
If D2 ~= C3 Stage 2, improvement is mostly warm-start + LR.
If D2 < C3 Stage 2, RDT likely contributes.
```

---

## D3 — Direct B3 with C3 Stage 2 Low LR, No Warm-Start

Purpose:

```text
Determine whether C3 Stage 2 improvement comes from low learning rate alone.
```

File:

```text
configs/training_d3_direct_b3_low_lr.json
```

Settings:

```json
"experiment": {
  "name": "respiratory_d3_direct_b3_low_lr"
}
```

```json
"model": {
  "encoder": {
    "architecture": {
      "rdt": {
        "enabled": true,
        "steps": 3,
        "top_tokens_per_branch": 2,
        "gated_residual": true,
        "layerscale_init": 0.01
      }
    }
  }
}
```

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

Interpretation:

```text
If D3 ~= C3 Stage 2, low LR is the main factor.
If D3 < C3 Stage 2, warm-start is important.
```

---

## D4 — C3 Staged top-k Sweep

Purpose:

```text
Test whether the RDT evidence bottleneck should be narrower or slightly wider than top-2.
```

Do not repeat top-k=4 yet; C5 top-4 was poor.

### D4 top-k=1 Stage 2

File:

```text
configs/training_d4_c3_top1_stage2.json
```

Use C3 Stage 1 checkpoint.

Settings:

```json
"experiment": {
  "name": "respiratory_d4_c3_top1_stage2"
}
```

```json
"rdt": {
  "enabled": true,
  "steps": 3,
  "top_tokens_per_branch": 1,
  "gated_residual": true,
  "layerscale_init": 0.01
}
```

Expected 4-scale shape:

```text
selected evidence length = 4 × 1 = 4
```

Use C3 Stage 2 optimizer/loss/warm-start settings:

```text
encoder_lr = 1e-5
head_lr = 3e-4
branch_auxiliary.weight = 0.05
strict = false
```

### D4 top-k=3 Stage 2

File:

```text
configs/training_d4_c3_top3_stage2.json
```

Settings:

```json
"experiment": {
  "name": "respiratory_d4_c3_top3_stage2"
}
```

```json
"rdt": {
  "enabled": true,
  "steps": 3,
  "top_tokens_per_branch": 3,
  "gated_residual": true,
  "layerscale_init": 0.01
}
```

Expected 4-scale shape:

```text
selected evidence length = 4 × 3 = 12
```

Use same Stage 2 warm-start settings.

Interpretation:

```text
top-k=1 better than top-k=2 => selector is noisy; narrower is better.
top-k=3 better than top-k=2 => top-2 was too restrictive.
```

---

## D5 — C3 Staged Repeat Seeds

Purpose:

```text
Check reliability of C3 Stage 2 under multiple seeds.
```

Create configs:

```text
configs/training_d5_c3_stage1_seed0.json
configs/training_d5_c3_stage2_seed0.json
configs/training_d5_c3_stage1_seed1.json
configs/training_d5_c3_stage2_seed1.json
configs/training_d5_c3_stage1_seed2.json
configs/training_d5_c3_stage2_seed2.json
```

Use the same config as C3 Stage 1 / Stage 2, but set seed fields consistently according to the repo’s existing seed mechanism.

If the repo has a single seed field, set:

```json
"experiment": {
  "seed": 0
}
```

or the correct current field.

If the current seed field is elsewhere, use the existing schema.

Do not invent an unsupported seed field; inspect existing configs first.

Stage 2 checkpoint paths should remain `null` templates and be filled manually after each Stage 1 run.

---

# Part B — Add E-Series Code Support and Configs

The E-Series requires new code because current top-k selection uses normalized attention weights.

Goal:

```text
Support configurable evidence-score source for top-k evidence selection.
```

Add config field:

```json
"model": {
  "encoder": {
    "architecture": {
      "rdt": {
        "evidence_score_source": "attention_weight"
      }
    }
  }
}
```

Supported values:

```text
attention_weight
attention_logit
instance_logit
```

Default:

```text
attention_weight
```

This preserves current behavior.

---

## E-Series Configs to Add

Create:

```text
configs/training_e1_c3_attention_logit_stage2.json
configs/training_e2_c3_instance_logit_stage2.json
configs/training_e3_c3_attention_temp05_stage2.json
configs/training_e4_c3_entropy001_stage2.json
```

All E-series configs should use C3 Stage 2 staged warm-start settings:

```text
RDT enabled
steps = 3
top_tokens_per_branch = 2
BCE
branch_auxiliary.weight = 0.05
encoder_lr = 1e-5
head_lr = 3e-4
initialization.checkpoint_path = null template
strict = false
```

The user will fill the Stage 1 checkpoint path.

---

## E1 — top-k by raw attention logits

File:

```text
configs/training_e1_c3_attention_logit_stage2.json
```

Config:

```json
"rdt": {
  "enabled": true,
  "steps": 3,
  "top_tokens_per_branch": 2,
  "gated_residual": true,
  "layerscale_init": 0.01,
  "evidence_score_source": "attention_logit"
}
```

### Code requirements

Branch MIL head must expose pre-softmax attention logits.

Add to branch MIL output:

```python
attention_logits: Tensor  # [B, T_s]
```

Top-k selection uses:

```python
scores = mil_output.attention_logits
```

---

## E2 — top-k by instance logits

File:

```text
configs/training_e2_c3_instance_logit_stage2.json
```

Config:

```json
"rdt": {
  "enabled": true,
  "steps": 3,
  "top_tokens_per_branch": 2,
  "gated_residual": true,
  "layerscale_init": 0.01,
  "evidence_score_source": "instance_logit"
}
```

### Code requirements

Add token-level instance logit head to the branch MIL head.

For binary classification:

```text
instance_logits: [B, T_s]
```

For multiclass classification:

```text
instance_logits: [B, T_s, C]
```

For binary top-k evidence:

```python
scores = mil_output.instance_logits
```

For multiclass:

Use a clear policy. Recommended:

```text
instance evidence score = max over non-normal class logits
```

If the current label schema does not identify a normal class reliably, either:

1. use max over all class logits, or
2. support `positive_class_index` in config, or
3. raise a clear error for multiclass instance-logit evidence selection until implemented.

Since current priority is binary respiratory experiments, binary support is required; multiclass behavior can be conservative but must be explicit.

Add to branch MIL output:

```python
instance_logits: Tensor | None
```

Interpretation:

```text
This is the most important evidence-selection improvement.
Top-k evidence tokens should be selected by class evidence strength, not by nearly uniform attention weights.
```

---

## E3 — attention temperature

File:

```text
configs/training_e3_c3_attention_temp05_stage2.json
```

Add MIL config:

```json
"mil": {
  "attention_temperature": 0.5
}
```

If there is no current `mil` sub-config, add:

```json
"architecture": {
  "mil": {
    "attention_temperature": 1.0
  }
}
```

Default:

```text
attention_temperature = 1.0
```

Code:

```python
attention_weights = softmax(attention_logits / temperature, dim=1)
```

Validation:

```text
attention_temperature > 0
```

Run E3 with:

```text
attention_temperature = 0.5
evidence_score_source = attention_weight
```

---

## E4 — attention entropy regularization

File:

```text
configs/training_e4_c3_entropy001_stage2.json
```

Add loss config:

```json
"train": {
  "loss": {
    "attention_entropy": {
      "enabled": true,
      "weight": 0.001
    }
  }
}
```

Default:

```json
"attention_entropy": {
  "enabled": false,
  "weight": 0.0
}
```

Trainer behavior:

```python
entropy = -(attention * log(attention + eps)).sum(dim=1).mean()
total_loss += weight * entropy
```

This penalizes high entropy and encourages sharper attention.

Validation:

```text
if attention_entropy.enabled: weight > 0
```

Make sure this can work with multiple branches:

```text
attention_entropy_loss = mean over branches
```

---

# Part C — Add F-Series Code Support and Configs

The F-series controls branch 4, the 2×128 branch.

Add configs:

```text
configs/training_f1_c3_branch_aux_weights_stage2.json
configs/training_f2_c3_exclude_branch4_evidence_stage2.json
```

Both are C3 Stage 2 style warm-start configs.

---

## F1 — branch-specific auxiliary weights

Purpose:

```text
Reduce auxiliary supervision pressure on unstable branch 4 while keeping the branch in the model.
```

Config:

```json
"train": {
  "loss": {
    "branch_auxiliary": {
      "enabled": true,
      "weight": 0.1,
      "weights": [0.1, 0.1, 0.1, 0.03],
      "aggregation": "mean"
    }
  }
}
```

### Code requirements

Extend `BranchAuxiliaryLossConfig` with optional:

```python
weights: tuple[float, ...] | None = None
```

Validation:

```text
if weights is not None:
  all weights > 0
  length must equal number of patch branches
```

Trainer behavior:

If `weights is None`, preserve current scalar behavior:

```python
aux_loss = weight * mean(branch_losses)
```

If `weights` is provided:

```python
weighted = sum_i weights[i] * branch_loss_i / sum_i weights[i]
total_loss = final_loss + weighted
```

Important:

```text
Do not multiply by both scalar weight and weights unless explicitly intended.
```

Recommended semantics:

- `weight` is used only when `weights` is absent.
- `weights` directly encode branch-wise auxiliary contribution.

Alternative acceptable semantics:

- scalar `weight` is global multiplier and `weights` are relative branch weights.

If using the alternative, document it clearly in README and tests.

Preferred simple semantics:

```text
weights overrides scalar weight.
```

---

## F2 — exclude branch 4 from RDT evidence, keep it in fusion

Purpose:

```text
Keep 2×128 branch signals in branch logits/embeddings/fusion but prevent its noisy top-k tokens from entering the RDT state.
```

Config:

```json
"model": {
  "encoder": {
    "architecture": {
      "rdt": {
        "exclude_branches_from_evidence": [3]
      }
    }
  }
}
```

Default:

```json
"exclude_branches_from_evidence": []
```

Code behavior:

- Branch 4 still produces:
  - event tokens,
  - branch embedding,
  - branch logits,
  - branch auxiliary loss if enabled.
- Branch 4 is excluded only from selected evidence tokens used to build `U0`.
- Branch 4 event tokens may either:
  - remain in `H_ctx`, or
  - be excluded from `H_ctx`.

Preferred behavior:

```text
Exclude from U0 but keep in H_ctx.
```

Rationale:

```text
The RDT can still attend to branch 4 context, but branch 4 does not occupy initial evidence slots.
```

If implementation is simpler to exclude from both U0 and H_ctx, document it explicitly.

Validation:

```text
all branch indices in exclude_branches_from_evidence must be valid
```

Expected 4-scale, top-2, exclude branch 3:

```text
U0 length = 3 branches × 2 = 6
H_ctx length = 1916 if keeping all branches in context
```

---

# Part D — Diagnostics Updates

The current README says diagnostics include `branch_logits` and selected evidence token embeddings. Extend diagnostics so experiment analysis is easier.

## Required model output fields

Ensure `AstModelOutput` contains:

```python
selected_evidence_indices: Tensor | None
selected_evidence_scores: Tensor | None
selected_evidence_branch_ids: Tensor | None
evidence_score_source: str | None
```

If not already present, add them.

Recommended shapes:

```text
selected_evidence_indices:    [B, K_selected]
selected_evidence_scores:     [B, K_selected]
selected_evidence_branch_ids: [B, K_selected]
```

Dynamic `K_selected`:

```text
4-scale top-2 -> 8
3-scale top-2 -> 6
4-scale top-1 -> 4
4-scale top-3 -> 12
4-scale top-4 -> 16
4-scale top-2 excluding branch 3 -> 6
```

## Diagnostics writer behavior

Update diagnostics JSONL to include:

```json
"selected_evidence_indices": [...]
"selected_evidence_scores": [...]
"selected_evidence_branch_ids": [...]
"evidence_score_source": "attention_weight"
```

Always write indices/scores/branch IDs when present.

Only write `selected_evidence_tokens` embeddings when:

```json
"analysis.outputs.save_embeddings": true
```

Rationale:

```text
indices/scores/branch IDs are small and essential.
selected token embeddings are large.
```

## Additional optional diagnostics

If easy, add:

```json
"selected_evidence_time_positions": [...]
```

This can be identical to `selected_evidence_indices` for temporal event tokens.

Do not dump full attention maps by default.

---

# Part E — README Updates

Update README to add:

1. A section: `Next Experiment Series: D/E/F`.
2. A table mapping configs to purpose.
3. Explanation that:
   - D-series decomposes C3 improvement.
   - E-series improves evidence selection.
   - F-series controls the unstable 2×128 branch.
4. Explanation that `best_loss` is the primary checkpoint for comparison.
5. Warning that `best_f1` can overfit validation-threshold tuning.
6. Explanation of dynamic selected evidence length:
   - 3-scale top-2 -> 6
   - 4-scale top-1 -> 4
   - 4-scale top-2 -> 8
   - 4-scale top-3 -> 12
   - 4-scale top-4 -> 16
7. Explanation of new config fields:
   - `rdt.evidence_score_source`
   - `rdt.exclude_branches_from_evidence`
   - `architecture.mil.attention_temperature`
   - `train.loss.attention_entropy`
   - `train.loss.branch_auxiliary.weights`
8. C3/D/E/F staged workflow:
   - run Stage 1,
   - copy Stage 1 `best_loss_*.pt` into Stage 2 `train.initialization.checkpoint_path`,
   - run Stage 2.

---

# Part F — Tests to Add or Update

Add tests for config parsing, model shape, trainer loss, and diagnostics.

## 1. Config parsing tests

Ensure these configs parse:

```text
D1 stage1/stage2
D2
D3
D4 top1/top3
D5 seed configs
E1/E2/E3/E4
F1/F2
```

Test validation errors:

```text
invalid evidence_score_source
attention_temperature <= 0
attention_entropy.enabled=true with weight <= 0
branch_auxiliary.weights length != number of branches
branch_auxiliary.weights contains non-positive value
exclude_branches_from_evidence includes invalid index
top_tokens_per_branch > temporal length of any included evidence branch
```

## 2. Dynamic shape tests

Test:

```text
4-scale top-1 -> selected evidence length 4
4-scale top-2 -> selected evidence length 8
4-scale top-3 -> selected evidence length 12
3-scale top-2 -> selected evidence length 6
4-scale top-2 exclude branch 3 -> selected evidence length 6
```

Also verify context length:

```text
4-scale H_ctx = 1916
3-scale H_ctx = 893
```

If F2 keeps branch 4 in context:

```text
exclude branch 3 from evidence, H_ctx still 1916
```

## 3. Evidence score source tests

For each:

```text
attention_weight
attention_logit
instance_logit
```

Assert model forward succeeds and selected evidence scores come from the requested source.

For unit tests, construct a tiny fake branch MIL output where sources produce different top-k order, then verify the correct source is used.

## 4. Branch auxiliary weights tests

Cases:

```text
scalar weight only
branch weights list
invalid length
invalid non-positive value
```

Verify weighted auxiliary loss calculation.

## 5. Attention entropy loss tests

Cases:

```text
disabled -> no entropy term
enabled -> total loss includes entropy term
```

Verify sign:

```text
total_loss += weight * entropy
```

because minimizing total loss encourages lower entropy.

## 6. Diagnostics tests

Verify JSONL contains:

```text
selected_evidence_indices
selected_evidence_scores
selected_evidence_branch_ids
evidence_score_source
```

Verify embeddings are only saved if `save_embeddings=true`.

## 7. Staged warm-start tests

If the branch already has warm-start support from prior work, keep and test it.

Cases:

```text
strict=false loads checkpoint with missing RDT keys
load_optimizer_state=false does not load optimizer state
checkpoint_path=null is no-op
```

---

# Part G — Acceptance Criteria

The task is complete when:

1. D1/D2/D3/D4/D5 configs exist.
2. E1/E2/E3/E4 configs exist.
3. F1/F2 configs exist.
4. Existing C0–C5 configs are not broken.
5. Current event-MIL architecture remains intact.
6. No latent-query pooling is reintroduced.
7. `evidence_score_source` supports:
   - `attention_weight`
   - `attention_logit`
   - `instance_logit`
8. Branch MIL output includes whatever logits are required for evidence selection:
   - attention weights
   - attention logits
   - instance logits
9. Attention temperature is configurable.
10. Optional attention entropy regularization is supported.
11. Optional branch-specific auxiliary weights are supported.
12. Optional `exclude_branches_from_evidence` is supported.
13. Model handles dynamic branch count.
14. Model handles dynamic selected evidence length.
15. Diagnostics include selected evidence indices/scores/branch IDs/source.
16. README documents the D/E/F experiments and new config fields.
17. Tests pass.

---

# Part H — Run Commands

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

- Do not rewrite the model from scratch.
- Do not change the default patch geometry except in the explicit 3-scale configs.
- Do not reintroduce latent-query pooling.
- Do not use `rdt.steps = 0` as the main RDT-off switch.
- Do not move branch auxiliary settings outside `train.loss.branch_auxiliary`.
- Do not assume 4 branches.
- Do not assume selected evidence length is always 8.
- Do not assume top-k is always 2.
- Do not dump selected evidence token embeddings unless `save_embeddings=true`.
- Do not use `best_f1` as the primary checkpoint-selection criterion.
- Do not make F2 remove branch 4 entirely unless the config explicitly changes patch branches.
- Do not silently ignore invalid branch indices.
- Do not silently fall back to attention weights when `evidence_score_source` is invalid.
- Do not break binary logits shape `[B]`.

---

# Suggested Implementation Notes

## Branch MIL output

Add or update the output dataclass:

```python
@dataclass(frozen=True)
class BranchMilOutput:
    logits: Tensor
    embedding: Tensor
    attention_weights: Tensor
    attention_logits: Tensor
    instance_logits: Tensor | None = None
```

## RDT config

Add fields:

```python
@dataclass(frozen=True)
class RdtConfig:
    enabled: bool = False
    steps: int = 3
    top_tokens_per_branch: int = 2
    gated_residual: bool = True
    layerscale_init: float = 0.01
    evidence_score_source: Literal[
        "attention_weight",
        "attention_logit",
        "instance_logit",
    ] = "attention_weight"
    exclude_branches_from_evidence: tuple[int, ...] = ()
```

## MIL config

Add:

```python
@dataclass(frozen=True)
class MilConfig:
    attention_temperature: float = 1.0
```

Place it under:

```text
model.encoder.architecture.mil
```

## Loss config

Add:

```python
@dataclass(frozen=True)
class AttentionEntropyLossConfig:
    enabled: bool = False
    weight: float = 0.0
```

Add optional branch weights:

```python
@dataclass(frozen=True)
class BranchAuxiliaryLossConfig:
    enabled: bool = False
    weight: float = 0.3
    weights: tuple[float, ...] | None = None
    aggregation: Literal["mean"] = "mean"
```

## Score selection

Suggested helper:

```python
def get_evidence_scores(
    mil_output: BranchMilOutput,
    *,
    source: str,
    task_type: str,
    positive_class_index: int | None = None,
) -> Tensor:
    if source == "attention_weight":
        return mil_output.attention_weights

    if source == "attention_logit":
        return mil_output.attention_logits

    if source == "instance_logit":
        if mil_output.instance_logits is None:
            raise ValueError("instance_logit evidence selection requires instance_logits")
        logits = mil_output.instance_logits
        if logits.ndim == 2:
            return logits
        if logits.ndim == 3:
            # Implement explicit multiclass policy.
            # Prefer max over non-normal classes if normal class is known.
            ...
        raise ValueError(...)

    raise ValueError(f"Unsupported evidence_score_source: {source}")
```

## Evidence selection output

```python
@dataclass(frozen=True)
class SelectedEvidence:
    tokens: Tensor
    indices: Tensor
    scores: Tensor
    branch_ids: Tensor
```

## Entropy loss

Use:

```python
def attention_entropy_loss(attention_weights: Sequence[Tensor]) -> Tensor:
    losses = []
    for attn in attention_weights:
        entropy = -(attn * (attn + 1e-8).log()).sum(dim=1).mean()
        losses.append(entropy)
    return torch.stack(losses).mean()
```

Trainer:

```python
if cfg.loss.attention_entropy.enabled:
    if output.branch_attention_weights is None:
        raise ValueError(...)
    total_loss = total_loss + cfg.loss.attention_entropy.weight * attention_entropy_loss(
        output.branch_attention_weights
    )
```

---

# Recommended Execution Order After Implementation

After Codex completes the changes, run experiments in this order:

## Round 1: D-series cause decomposition

```text
D2: staged warm-start + low LR + RDT off
D3: direct B3 + low LR + no warm-start
D4 top-k=1
D4 top-k=3
D1 staged 3-scale
```

## Round 2: E-series evidence selection

```text
E1: attention_logit
E2: instance_logit
```

Then optional:

```text
E3: attention_temperature=0.5
E4: entropy=0.001
```

## Round 3: F-series branch 4 control

```text
F1: branch-specific auxiliary weights
F2: exclude branch 4 from RDT evidence
```

## Round 4: reliability

Best candidate:

```text
3 seeds
or CV
```

Use `best_loss` checkpoint for primary comparison.

---

# Final Note

The central hypothesis for the next round is:

```text
C3 staged training is the current best path,
but its improvement must be decomposed into warm-start, LR schedule, and RDT effects.
The next structural bottleneck is evidence selection, because normalized attention weights appear too uniform.
```

The most important new code feature is:

```text
rdt.evidence_score_source = instance_logit
```

The most important config-only experiments are:

```text
D2 and D3
```
