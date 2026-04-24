# Codex Instructions: Add C0–C5 Experiment Configs, Checkpoint Warm-Start, and Evidence Diagnostics

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

Do **not** reintroduce the Hugging Face `ASTModel` path. Do **not** revert to latent-query pooling. Do **not** rewrite the model architecture from scratch.

The current branch is already event-MIL-centered and already supports:

- one encoder type: `multiscale_rdt_ast`
- local AST-style fbank preprocessing
- 4-scale patch tokenization
- learned patch position + scale embeddings
- shared branch-wise transformer stem
- scale-specific transformer adapters
- learned frequency-attention pooling
- branch MIL heads
- top-k evidence token selection
- optional RDT refinement controlled by `rdt.enabled`
- branch auxiliary loss controlled by `train.loss.branch_auxiliary`
- config-driven training, evaluation, CV, checkpointing, diagnostics, and threshold optimization

Your job is to:

1. add C0–C5 training configs,
2. add minimal code support for C3 staged training via checkpoint warm-start,
3. improve diagnostics so C4/C5 can be interpreted,
4. update README/tests as needed.

---

## Current Branch State to Preserve

The current README describes the active architecture as:

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

Keep this event-MIL-first design.

---

## Fixed Patch Geometry

The default patch geometry must remain unchanged unless a config explicitly tests the 3-scale ablation.

Default 4-scale branch geometry:

| Branch | Patch size `(time, freq)` | Stride `(time, freq)` | Patch tokens | Temporal event tokens |
|---:|---:|---:|---:|---:|
| 1 | `(16, 16)` | `(8, 16)` | `1016` | `127` |
| 2 | `(8, 32)` | `(4, 32)` | `1020` | `255` |
| 3 | `(4, 64)` | `(2, 64)` | `1022` | `511` |
| 4 | `(2, 128)` | `(1, 128)` | `1023` | `1023` |

Default context length:

```text
H_ctx length = 127 + 255 + 511 + 1023 = 1916
```

Default top-k:

```text
top_tokens_per_branch = 2
```

Default evidence state:

```text
U0: [B, 8, D]
```

because:

```text
4 branches × 2 top evidence tokens per branch = 8 tokens
```

---

## Current Config Semantics to Keep

The current branch already uses:

```json
"model": {
  "encoder": {
    "type": "multiscale_rdt_ast",
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

and:

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

Keep the rule:

```text
RDT off == rdt.enabled = false
```

Do **not** use `steps = 0` as the main representation of RDT off.

Keep branch auxiliary loss under:

```text
train.loss.branch_auxiliary
```

---

# Part A — Add C0–C5 Training Configs

Create the following canonical config files:

```text
configs/training_c0_b3_focal_patience8.json
configs/training_c1_b3_bce_aux01_patience8.json
configs/training_c2_b1_bce_aux01_patience8.json

configs/training_c3_stage1_b1_bce_aux01.json
configs/training_c3_stage2_b3_from_stage1.json

configs/training_c4_3scale_bce_aux01_rdt3.json
configs/training_c4_4scale_bce_aux01_rdt3.json

configs/training_c5_top2_bce_aux01_rdt3.json
configs/training_c5_top4_bce_aux01_rdt3.json
```

Use the current canonical binary config style as the base. Preserve dataset paths, task, labels, preprocessing, sampler, analysis outputs, and optimizer defaults unless explicitly stated below.

The existing config paths and local dataset paths are user-specific. Do not invent new dataset paths. Copy them from the existing canonical configs in the branch.

---

## Global Settings for C0–C5

Unless a specific config says otherwise, use:

```json
"train": {
  "epochs": 40,
  "early_stopping": {
    "enabled": true,
    "monitor": "val_loss",
    "patience": 8,
    "min_delta": 0.00001
  }
}
```

Use `best_loss` checkpoint as the primary model-selection checkpoint for analysis and evaluation. Keep `best_f1` checkpointing if already implemented, but README should warn that `best_f1` can overfit validation threshold tuning.

---

## C0 — B3 Reproduction with Shorter Early Stopping

Purpose:

```text
Re-run the existing B3 structure but reduce overfitting by using shorter training and shorter early-stopping patience.
```

Config filename:

```text
configs/training_c0_b3_focal_patience8.json
```

Required settings:

```json
"experiment": {
  "name": "respiratory_c0_b3_focal_patience8"
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
  "loss": {
    "type": "focal",
    "auto_pos_weight": false,
    "pos_weight": null,
    "gamma": 2.0,
    "branch_auxiliary": {
      "enabled": true,
      "weight": 0.3,
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

---

## C1 — B3 with BCE and Lower Branch Auxiliary Weight

Purpose:

```text
Test whether BCE improves calibration and reduces threshold instability compared with focal loss.
```

Config filename:

```text
configs/training_c1_b3_bce_aux01_patience8.json
```

Required settings:

```json
"experiment": {
  "name": "respiratory_c1_b3_bce_aux01_patience8"
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

Note:

```text
gamma is ignored for BCE if the current loss implementation behaves that way. It is acceptable to keep it in the config for schema compatibility.
```

---

## C2 — B1 with BCE and Lower Branch Auxiliary Weight

Purpose:

```text
Establish a strong event-MIL baseline without RDT under the same BCE/auxiliary/early-stopping settings as C1.
```

Config filename:

```text
configs/training_c2_b1_bce_aux01_patience8.json
```

Required settings:

```json
"experiment": {
  "name": "respiratory_c2_b1_bce_aux01_patience8"
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

---

## C3 — Staged Training: B1 → B3

Purpose:

```text
Train event detectors first without RDT, then warm-start a B3-style RDT model from the event-MIL checkpoint.
```

This requires code support for checkpoint warm-start. See Part B.

---

### C3 Stage 1

Config filename:

```text
configs/training_c3_stage1_b1_bce_aux01.json
```

Recommended settings:

```json
"experiment": {
  "name": "respiratory_c3_stage1_b1_bce_aux01"
}
```

Use the same core settings as C2:

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

---

### C3 Stage 2

Config filename:

```text
configs/training_c3_stage2_b3_from_stage1.json
```

Required settings:

```json
"experiment": {
  "name": "respiratory_c3_stage2_b3_from_stage1"
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

Important:

```text
checkpoint_path is intentionally null in the template.
The user will fill it with the Stage 1 best_loss checkpoint path.
```

Example after Stage 1:

```json
"initialization": {
  "checkpoint_path": "checkpoints/respiratory_c3_stage1_b1_bce_aux01/best_loss_0.123456.pt",
  "load_model_state": true,
  "strict": false,
  "load_optimizer_state": false
}
```

Use `strict = false` because Stage 1 has RDT disabled and Stage 2 has RDT enabled, so RDT parameters may be missing in the warm-start checkpoint.

---

## C4 — 3-Scale vs 4-Scale

Purpose:

```text
Test whether the 2×128 branch helps or hurts.
```

Use BCE, branch auxiliary weight 0.1, RDT on with T=3 by default. This makes C4 comparable to C1.

---

### C4 4-Scale

Config filename:

```text
configs/training_c4_4scale_bce_aux01_rdt3.json
```

Use the default 4 branches:

```json
"patch_branches": [
  { "patch_size": [16, 16], "stride": [8, 16] },
  { "patch_size": [8, 32], "stride": [4, 32] },
  { "patch_size": [4, 64], "stride": [2, 64] },
  { "patch_size": [2, 128], "stride": [1, 128] }
]
```

Use:

```json
"rdt": {
  "enabled": true,
  "steps": 3,
  "top_tokens_per_branch": 2,
  "gated_residual": true,
  "layerscale_init": 0.01
}
```

Use:

```json
"loss": {
  "type": "bce",
  "branch_auxiliary": {
    "enabled": true,
    "weight": 0.1,
    "aggregation": "mean"
  }
}
```

---

### C4 3-Scale

Config filename:

```text
configs/training_c4_3scale_bce_aux01_rdt3.json
```

Remove the `2×128` branch:

```json
"patch_branches": [
  { "patch_size": [16, 16], "stride": [8, 16] },
  { "patch_size": [8, 32], "stride": [4, 32] },
  { "patch_size": [4, 64], "stride": [2, 64] }
]
```

Expected dynamic shapes:

```text
temporal context length = 127 + 255 + 511 = 893
top_tokens_per_branch = 2
RDT state length = 3 branches × 2 = 6
U0: [B, 6, D]
```

The code must not assume exactly 4 branches or exactly 8 selected evidence tokens.

---

## C5 — top evidence tokens: 2 vs 4

Purpose:

```text
Test whether the RDT evidence bottleneck is too narrow.
```

Use BCE, branch auxiliary weight 0.1, RDT on with T=3 by default.

---

### C5 top-2

Config filename:

```text
configs/training_c5_top2_bce_aux01_rdt3.json
```

Use:

```json
"rdt": {
  "enabled": true,
  "steps": 3,
  "top_tokens_per_branch": 2,
  "gated_residual": true,
  "layerscale_init": 0.01
}
```

Expected with 4 branches:

```text
U0: [B, 8, D]
```

---

### C5 top-4

Config filename:

```text
configs/training_c5_top4_bce_aux01_rdt3.json
```

Use:

```json
"rdt": {
  "enabled": true,
  "steps": 3,
  "top_tokens_per_branch": 4,
  "gated_residual": true,
  "layerscale_init": 0.01
}
```

Expected with 4 branches:

```text
U0: [B, 16, D]
```

The RDT block must handle dynamic evidence-state length. It should not assume a fixed length of 8.

---

# Part B — Add Checkpoint Warm-Start for C3

C3 Stage 2 requires training initialization from a Stage 1 checkpoint.

Currently, the training CLI builds a fresh model and starts training. Add optional checkpoint initialization.

---

## Add Config Schema

Add this under `train`:

```json
"initialization": {
  "checkpoint_path": null,
  "load_model_state": true,
  "strict": false,
  "load_optimizer_state": false
}
```

Suggested dataclass:

```python
@dataclass(frozen=True)
class TrainingInitializationConfig:
    checkpoint_path: str | None = None
    load_model_state: bool = True
    strict: bool = False
    load_optimizer_state: bool = False
```

Add it to `TrainConfig`:

```python
@dataclass(frozen=True)
class TrainConfig:
    ...
    initialization: TrainingInitializationConfig = field(
        default_factory=TrainingInitializationConfig
    )
```

---

## Validation Rules

- `checkpoint_path` may be `null`.
- If `checkpoint_path` is not null:
  - path must be a string
  - `load_model_state` must be a boolean
  - `strict` must be a boolean
  - `load_optimizer_state` must be a boolean
- If `load_optimizer_state = true`, document that the optimizer architecture and parameter groups should match. For C3 Stage 2, default must be `false`.

---

## Training CLI Behavior

Modify:

```text
src/cli/training.py
```

After model creation and adaptation but before training starts, load checkpoint if configured.

Recommended order:

1. build datasets
2. build model
3. apply encoder adaptation
4. if `train.initialization.checkpoint_path` is not null:
   - load checkpoint
   - load model state dict
   - optionally load optimizer state after optimizer creation
5. build optimizer
6. train

However, loading optimizer state requires optimizer to exist. Therefore a clean implementation is:

1. build model
2. apply adaptation
3. build optimizer
4. if initialization checkpoint is set:
   - load model state if requested
   - load optimizer state if requested
5. train

Make sure model state is loaded before the first forward/backward step.

---

## Model State Loading

Pseudo-code:

```python
init_cfg = cfg.train.initialization

if init_cfg.checkpoint_path is not None and init_cfg.load_model_state:
    checkpoint = torch.load(init_cfg.checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    load_result = model.load_state_dict(
        state_dict,
        strict=init_cfg.strict,
    )

    logger.info(
        "Initialized model from checkpoint=%s | strict=%s | missing_keys=%s | unexpected_keys=%s",
        init_cfg.checkpoint_path,
        init_cfg.strict,
        list(load_result.missing_keys),
        list(load_result.unexpected_keys),
    )
```

If using older PyTorch where `load_state_dict` returns a tuple, handle that as well.

For C3 Stage 2, use:

```text
strict = false
```

because the Stage 1 checkpoint may not contain RDT parameters while Stage 2 includes RDT parameters.

---

## Optimizer State Loading

If:

```json
"load_optimizer_state": true
```

then load:

```python
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

Default should remain:

```json
"load_optimizer_state": false
```

For C3 Stage 2, optimizer state should not be loaded.

Reason:

```text
Stage 2 changes RDT enabled state and learning rates, so old optimizer state is not appropriate.
```

---

## Checkpoint Metadata

When saving checkpoints, include initialization metadata in `extra_state` or run config automatically.

The saved checkpoint already stores `run_config`. It should now include:

```text
train.initialization
```

No special additional field is required if `asdict(cfg)` is already stored.

---

# Part C — Improve Evidence Diagnostics

The current diagnostics include:

- logits
- probabilities
- pooled embedding if enabled
- branch logits
- selected evidence token embeddings if enabled

This is not enough to interpret C4/C5. Add selected evidence metadata.

---

## Required New Model Output Fields

Extend `AstModelOutput` to include optional evidence metadata.

Recommended:

```python
@dataclass(frozen=True)
class AstModelOutput:
    logits: Tensor
    pooled_embedding: Tensor
    branch_logits: Tensor | None = None
    branch_attention_weights: tuple[Tensor, ...] | None = None
    selected_evidence_tokens: Tensor | None = None
    selected_evidence_indices: Tensor | None = None
    selected_evidence_scores: Tensor | None = None
    selected_evidence_branch_ids: Tensor | None = None
```

Expected shapes for 4-scale, top-2:

```text
selected_evidence_tokens:     [B, 8, D]
selected_evidence_indices:    [B, 8]
selected_evidence_scores:     [B, 8]
selected_evidence_branch_ids: [B, 8]
```

For 3-scale, top-2:

```text
selected_evidence_tokens:     [B, 6, D]
selected_evidence_indices:    [B, 6]
selected_evidence_scores:     [B, 6]
selected_evidence_branch_ids: [B, 6]
```

For 4-scale, top-4:

```text
selected_evidence_tokens:     [B, 16, D]
selected_evidence_indices:    [B, 16]
selected_evidence_scores:     [B, 16]
selected_evidence_branch_ids: [B, 16]
```

---

## Modify Top-k Selection

The current helper likely returns only selected token embeddings.

Change it to return tokens, indices, and scores.

Suggested dataclass:

```python
@dataclass(frozen=True)
class SelectedEvidence:
    tokens: Tensor
    indices: Tensor
    scores: Tensor
```

Function:

```python
def select_top_tokens(
    tokens: Tensor,
    scores: Tensor,
    *,
    top_k: int,
) -> SelectedEvidence:
    ...
    top = scores.topk(top_k, dim=1)
    indices = top.indices          # [B, top_k]
    selected_scores = top.values   # [B, top_k]
    selected_tokens = tokens.gather(
        1,
        indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
    )
    return SelectedEvidence(
        tokens=selected_tokens,
        indices=indices,
        scores=selected_scores,
    )
```

When concatenating across branches, also create branch IDs:

```python
branch_ids = torch.full(
    (batch_size, top_k),
    fill_value=branch_index,
    device=tokens.device,
    dtype=torch.long,
)
```

Then:

```python
selected_evidence_tokens = torch.cat(selected_token_list, dim=1)
selected_evidence_indices = torch.cat(selected_index_list, dim=1)
selected_evidence_scores = torch.cat(selected_score_list, dim=1)
selected_evidence_branch_ids = torch.cat(selected_branch_id_list, dim=1)
```

---

## Update Diagnostics Writer

Modify:

```text
src/evaluation/diagnostics.py
```

When diagnostics are enabled, include:

```json
"selected_evidence_indices": [...]
"selected_evidence_scores": [...]
"selected_evidence_branch_ids": [...]
```

Use `.detach().cpu().tolist()`.

Current diagnostics already conditionally writes selected evidence token embeddings when `save_embeddings` is true. Keep that behavior, but indices/scores/branch IDs are small and should be written whenever diagnostics are enabled, or at least whenever `save_logits` or `save_clip_metadata` is enabled.

Recommended behavior:

```text
Always include selected_evidence_indices, selected_evidence_scores, and selected_evidence_branch_ids if present in the model output.
Only include selected_evidence_tokens when save_embeddings is true.
```

Reason:

```text
selected_evidence_tokens are large; indices/scores/branch IDs are small and critical for interpretation.
```

---

# Part D — Evaluation and Checkpoint-Selection Guidance

Update README and/or add notes in config comments if applicable.

Required message:

```text
Use best_loss checkpoints as the primary model-selection criterion for C0–C5.
```

Reason:

```text
best_f1 checkpoints can overfit validation threshold tuning, especially in binary tasks.
```

Keep `best_f1` checkpointing if already implemented, but document it as a diagnostic/reference checkpoint rather than the default model-selection checkpoint.

If practical, add a helper note or script to evaluate `best_loss_*.pt` checkpoints first.

No large code change is required here unless you want to automate checkpoint selection.

---

# Part E — Tests to Add or Update

## 1. Config Tests

Add tests that the following configs parse:

```text
configs/training_c0_b3_focal_patience8.json
configs/training_c1_b3_bce_aux01_patience8.json
configs/training_c2_b1_bce_aux01_patience8.json
configs/training_c3_stage1_b1_bce_aux01.json
configs/training_c3_stage2_b3_from_stage1.json
configs/training_c4_3scale_bce_aux01_rdt3.json
configs/training_c4_4scale_bce_aux01_rdt3.json
configs/training_c5_top2_bce_aux01_rdt3.json
configs/training_c5_top4_bce_aux01_rdt3.json
```

Test new initialization schema:

```text
checkpoint_path = null
checkpoint_path = "some/path.pt"
strict = false
load_optimizer_state = false
```

---

## 2. Model Shape Tests

Test dynamic branch/top-k behavior.

### 4-scale, top-2

Expected:

```text
context length = 1916
selected evidence length = 8
```

### 3-scale, top-2

Expected:

```text
context length = 893
selected evidence length = 6
```

### 4-scale, top-4

Expected:

```text
context length = 1916
selected evidence length = 16
```

The model must not assume exactly 4 branches or exactly 8 selected evidence tokens.

---

## 3. RDT Path Tests

Add or update tests for:

```text
rdt.enabled = false
rdt.enabled = true, steps = 2
rdt.enabled = true, steps = 3
```

For RDT disabled:

```text
model.rdt_block may be None.
forward still succeeds.
selected evidence tokens are still produced.
```

For RDT enabled:

```text
RDT is applied steps times.
forward succeeds.
output shape is correct.
```

---

## 4. Branch Auxiliary Loss Tests

Trainer loss composition should be tested.

Required cases:

```text
branch_auxiliary.enabled = false -> total loss == final loss
branch_auxiliary.enabled = true -> total loss == final loss + weight * aux_loss
branch_auxiliary.enabled = true but branch_logits missing -> raise clear ValueError
```

---

## 5. Checkpoint Warm-Start Tests

Add tests for:

```text
load checkpoint with strict=true
load checkpoint with strict=false
missing/unexpected keys logged or returned
load_optimizer_state=false does not call optimizer.load_state_dict
```

If testing actual training CLI is too heavy, unit-test a helper function that applies initialization.

Suggested helper:

```python
initialize_from_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    cfg: TrainingInitializationConfig,
) -> InitializationSummary
```

Test this helper directly.

---

## 6. Diagnostics Tests

Add tests that diagnostics include:

```text
selected_evidence_indices
selected_evidence_scores
selected_evidence_branch_ids
```

when the model output contains them.

Also verify:

```text
selected_evidence_tokens are only dumped when save_embeddings = true
```

if you implement the recommended behavior.

---

# Part F — README Updates

Update README to include:

1. C0–C5 experiment descriptions.
2. A table mapping each experiment to config file.
3. A warning that `best_loss` should be the primary checkpoint for comparison.
4. Explanation of C3 staged training:
   - run Stage 1
   - fill Stage 2 `train.initialization.checkpoint_path` with Stage 1 `best_loss_*.pt`
   - run Stage 2
5. Updated diagnostics fields:
   - selected evidence indices
   - selected evidence scores
   - selected evidence branch IDs
6. Note that:
   - 3-scale C4 uses selected evidence length 6 with top-2
   - top-4 C5 uses selected evidence length 16 with 4 branches

---

# Acceptance Criteria

The implementation is acceptable when all of these hold:

1. The branch remains based on `MIL-RDT-AST`.
2. The event-MIL-first architecture remains intact.
3. No Hugging Face `ASTModel` path is reintroduced.
4. No latent-query pooling is reintroduced.
5. C0–C5 config files exist.
6. C0, C1, C2 can be launched without code edits.
7. C4 3-scale and 4-scale configs can be launched without code edits.
8. C5 top-2 and top-4 configs can be launched without code edits.
9. C3 Stage 2 supports checkpoint warm-start through `train.initialization`.
10. `rdt.enabled=false` is still the switch for RDT off.
11. `train.loss.branch_auxiliary` remains the location for auxiliary-loss settings.
12. The code supports dynamic branch count.
13. The code supports dynamic selected evidence length.
14. The code supports `top_tokens_per_branch=4`.
15. The model output includes selected evidence indices, scores, and branch IDs.
16. Diagnostics writes selected evidence indices, scores, and branch IDs.
17. The README explains that `best_loss` is the primary checkpoint-selection criterion.
18. Tests pass.

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
- exact error
- what remains unverified
```

---

# Avoid These Mistakes

- Do not change the patch geometry except in the explicit C4 3-scale config.
- Do not remove the current event-MIL architecture.
- Do not reintroduce latent-query pooling.
- Do not use `rdt.steps = 0` as the main RDT-off mechanism.
- Do not move branch auxiliary settings out of `train.loss.branch_auxiliary`.
- Do not assume 4 branches in the model, trainer, diagnostics, or tests.
- Do not assume selected evidence length is always 8.
- Do not dump large selected evidence token embeddings unless `save_embeddings=true`.
- Do not use `best_f1` as the default model-selection criterion in documentation.
- Do not load optimizer state in C3 Stage 2 by default.
- Do not use `strict=true` for C3 Stage 2 warm-start by default.
- Do not change local dataset paths in the canonical configs; copy them from existing configs.

---

# Suggested Helper: Training Initialization

Implement a helper if it reduces duplication.

```python
@dataclass(frozen=True)
class InitializationSummary:
    checkpoint_path: str | None
    loaded_model_state: bool
    loaded_optimizer_state: bool
    strict: bool
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()

def initialize_from_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    cfg: TrainingInitializationConfig,
    map_location: str | torch.device = "cpu",
) -> InitializationSummary:
    if cfg.checkpoint_path is None:
        return InitializationSummary(
            checkpoint_path=None,
            loaded_model_state=False,
            loaded_optimizer_state=False,
            strict=cfg.strict,
        )

    checkpoint = torch.load(cfg.checkpoint_path, map_location=map_location)

    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()

    if cfg.load_model_state:
        result = model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=cfg.strict,
        )
        missing_keys = tuple(result.missing_keys)
        unexpected_keys = tuple(result.unexpected_keys)

    if cfg.load_optimizer_state:
        if optimizer is None:
            raise ValueError("load_optimizer_state=true but optimizer is None")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return InitializationSummary(
        checkpoint_path=cfg.checkpoint_path,
        loaded_model_state=cfg.load_model_state,
        loaded_optimizer_state=cfg.load_optimizer_state,
        strict=cfg.strict,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )
```

Log the summary in `src/cli/training.py`.

---

# Suggested Helper: Evidence Selection

If current `select_top_tokens` only returns tokens, update it.

```python
@dataclass(frozen=True)
class SelectedEvidence:
    tokens: Tensor
    indices: Tensor
    scores: Tensor

def select_top_tokens(tokens: Tensor, scores: Tensor, *, top_k: int) -> SelectedEvidence:
    if tokens.ndim != 3:
        raise ValueError(...)
    if scores.ndim != 2:
        raise ValueError(...)
    if tokens.shape[:2] != scores.shape:
        raise ValueError(...)
    if top_k <= 0:
        raise ValueError(...)
    if top_k > int(tokens.shape[1]):
        raise ValueError(...)

    top = scores.topk(top_k, dim=1)
    indices = top.indices
    selected_scores = top.values
    selected_tokens = tokens.gather(
        1,
        indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
    )
    return SelectedEvidence(
        tokens=selected_tokens,
        indices=indices,
        scores=selected_scores,
    )
```

In model forward:

```python
selected_tokens_list = []
selected_indices_list = []
selected_scores_list = []
selected_branch_id_list = []

for branch_index, ...:
    selected = select_top_tokens(...)
    selected_tokens_list.append(selected.tokens)
    selected_indices_list.append(selected.indices)
    selected_scores_list.append(selected.scores)
    selected_branch_id_list.append(
        torch.full_like(selected.indices, fill_value=branch_index)
    )

selected_evidence_tokens = torch.cat(selected_tokens_list, dim=1)
selected_evidence_indices = torch.cat(selected_indices_list, dim=1)
selected_evidence_scores = torch.cat(selected_scores_list, dim=1)
selected_evidence_branch_ids = torch.cat(selected_branch_id_list, dim=1)
```

Return them in `AstModelOutput`.

---

# Suggested Evaluation Workflow for C0–C5

After each run, evaluate the best-loss checkpoint first.

Example:

```bash
python -m src.cli.training --config configs/training_c1_b3_bce_aux01_patience8.json
```

Then edit or generate an eval config pointing to:

```text
checkpoints/respiratory_c1_b3_bce_aux01_patience8/best_loss_*.pt
```

and run:

```bash
python -m src.cli.evaluate --config configs/eval_multiscale_rdt.json
```

If you add a helper script for best-loss checkpoint discovery, document it in README.

---

# Final Note

The goal of this task is not another architectural rewrite.

The goal is to make the current event-MIL model experimentally productive:

```text
C0: B3 + focal + shorter early stopping
C1: B3 + BCE + aux0.1 + shorter early stopping
C2: B1 + BCE + aux0.1 + shorter early stopping
C3: staged B1 -> B3 using checkpoint warm-start
C4: 3-scale vs 4-scale
C5: top-k evidence tokens 2 vs 4
```

Use config files for all experiment variants. Add code only where the current branch lacks the necessary support, primarily:

```text
- checkpoint warm-start for C3
- selected evidence indices/scores/branch IDs for diagnostics
- tests and README updates
```
