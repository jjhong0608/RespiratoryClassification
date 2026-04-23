# Codex Implementation Instructions: Refactor `MIL-RDT-AST` into Event-MIL with Optional RDT and Optional Branch Auxiliary Loss

## Target Repository and Branch

Repository:

```text
https://github.com/jjhong0608/RespiratoryClassification.git
```

Base branch:

```text
MIL-RDT-AST
```

This is a **refactor of the current `MIL-RDT-AST` branch**, not a greenfield implementation.

The current branch already has:

- local AST-style fbank preprocessing
- a single supported encoder type: `multiscale_rdt_ast`
- 4 fixed patch branches
- a config-driven training / evaluation / cross-validation pipeline
- checkpoint / diagnostics / threshold-optimization flow
- a default architecture based on a `[B, 4081, D]` token bank, `latent_query_count = 8`, and `rdt_steps = 3`

Your job is to **replace the current latent-query-centered design with an event-MIL-centered design** while keeping the pipeline config-driven and allowing all experimental variants to be controlled by config. Do **not** reintroduce the Hugging Face `ASTModel` path.

---

## What the Current Branch Looks Like

The current `MIL-RDT-AST` branch README describes this default architecture:

```text
input_values: [B, 1024, 128]
  -> unsqueeze channel
[B, 1, 1024, 128]
  -> four patch tokenizers
  -> learned position + scale embeddings
  -> shared branch-wise transformer stem
  -> scale-specific transformer adapters
  -> concat token bank
[B, 4081, D]
  -> latent query pooling
[B, 8, D]
  -> 3 recurrent RDT refinement steps
[B, 8, D]
  -> latent mean pooling
[B, D]
  -> classifier
```

The current branch also uses these default patch branches:

```text
(16, 16) patch with (8, 16) stride -> 1016 tokens
( 8, 32) patch with (4, 32) stride -> 1020 tokens
( 4, 64) patch with (2, 64) stride -> 1022 tokens
( 2,128) patch with (1,128) stride -> 1023 tokens
```

And current default architecture settings include:

```text
hidden_size = 384
num_attention_heads = 6
mlp_ratio = 4.0
shared_stem_depth = 2
adapter_depth = 1
latent_query_count = 8
rdt_steps = 3
classifier pooling = latent_mean
```

This refactor should preserve the config-driven workflow, but **the active model architecture should no longer depend on learned global latent queries as the primary bottleneck**.

---

## Refactor Goal

Refactor the current branch so the model is **event-MIL-centered**, with **RDT as an optional refinement module**.

### New design principles

1. **The model must work with RDT turned off.**
2. **The model must work with branch auxiliary loss turned on or off.**
3. **The number of RDT steps must be configurable.**
4. **The patch geometry must stay the same.**
5. **The entire experiment family B0/B1/B2/B3 must be runnable by changing config only.**

---

## Required Experimental Modes

Make these four experiments possible through config only.

| Experiment | `model.encoder.architecture.rdt.enabled` | `model.encoder.architecture.rdt.steps` | `train.loss.branch_auxiliary.enabled` |
|---|---:|---:|---:|
| B0 | `false` | ignored | `false` |
| B1 | `false` | ignored | `true` |
| B2 | `true` | `2` | `true` |
| B3 | `true` | `3` | `true` |

### Required semantic rule

RDT off must be represented as:

```json
"rdt": {
  "enabled": false
}
```

Do **not** use `steps = 0` as the main representation for “RDT off”.

It is fine if `steps` still exists in the config when `enabled = false`; in that case the code should ignore it.

---

## High-Level New Architecture

### Old active path

```text
patch token bank
[B, 4081, D]
  -> latent query pooling (K=8)
[B, 8, D]
  -> RDT
  -> classifier
```

### New active path

```text
4-scale patch tokenizers
  -> preserve branch grid structure
  -> build temporal event tokens per branch
  -> branch-level MIL heads
  -> branch logits
  -> top evidence token selection
  -> optional RDT refinement
  -> final fusion/classifier
```

### Key architectural change

The model should no longer rely on:

```text
LatentQueryPooler(H_all) -> [B, 8, D]
```

Instead, the model should:

1. build **temporal event token sequences** per branch,
2. compute **branch MIL attention / branch logits**,
3. select **top evidence tokens per branch**,
4. optionally refine those evidence tokens through RDT,
5. combine the refined evidence representation with branch-level signals for the final classifier.

---

## Fixed Patch Branches (Do Not Change)

Keep the existing 4 patch branches exactly:

| Branch | Patch size `(time, freq)` | Stride `(time, freq)` | Patch token grid | Patch token count |
|---:|---:|---:|---:|---:|
| 1 | `(16, 16)` | `(8, 16)` | `127 × 8` | `1016` |
| 2 | `(8, 32)` | `(4, 32)` | `255 × 4` | `1020` |
| 3 | `(4, 64)` | `(2, 64)` | `511 × 2` | `1022` |
| 4 | `(2, 128)` | `(1, 128)` | `1023 × 1` | `1023` |

The input must remain:

```text
input_values: [B, 1024, 128]
```

Internally:

```python
x = input_values.unsqueeze(1)  # [B, 1, 1024, 128]
```

---

## Required New Model Flow

## Step 1: Patch tokenization (keep)

For each branch, tokenize as before.

But **do not immediately flatten into a single global token bank for the main learning path**.

Preserve branch token grids conceptually:

```text
Branch 1: [B, 127, 8, D]
Branch 2: [B, 255, 4, D]
Branch 3: [B, 511, 2, D]
Branch 4: [B, 1023, 1, D]
```

Implementation can still use flattened tokens internally if easier, but the model must retain enough structure to produce a **time-axis event sequence** per branch.

---

## Step 2: Branch-local encoding

The current branch uses:

- learned position embeddings
- learned scale embeddings
- shared branch-wise transformer stem
- scale-specific transformer adapters

You may reuse those building blocks, but the encoder should now serve a different purpose:

```text
produce branch-local event features,
not global latent-query-ready token banks
```

### Recommended simplification

Reduce default encoder width from the current `D=384` to a more stable scratch-training default.

Recommended new default:

```text
hidden_size = 192
num_attention_heads = 4
mlp_ratio = 2.0
shared_stem_depth = 2
adapter_depth = 1
```

If you decide not to change the code defaults directly, at least update the canonical configs to use these smaller values.

Reason:

- current branch is scratch-trained
- the current default transformer-heavy setup is likely too large for the dataset
- the branch should first learn local acoustic event evidence, not global reasoning

---

## Step 3: Frequency pooling to temporal event tokens

For each branch, convert branch token grids into a **time-axis event token sequence**.

Target shapes:

```text
E1: [B, 127, D]
E2: [B, 255, D]
E3: [B, 511, D]
E4: [B, 1023, D]
```

### Required behavior

For branch `s`, the frequency dimension should be aggregated before MIL.

You may implement this as:

- learned frequency attention pooling (**preferred**)
- mean pooling over frequency
- max pooling over frequency

Preferred implementation:

```python
a = softmax(score(h_t_f), dim=freq)
e_t = sum_f a_t_f * h_t_f
```

Do not hardcode only mean pooling if a small learned pooling layer is easy to implement.

If implementation complexity becomes an issue, mean pooling is acceptable as a fallback, but learned pooling is preferred.

---

## Step 4: Branch MIL heads

Each branch must have its own MIL head operating on the temporal event tokens.

Given:

```text
E_s: [B, T_s, D]
```

Each branch should produce at least:

```text
branch attention weights a_s
branch embedding g_s
branch logit z_s
```

### Acceptable MIL choices

Any of the following are acceptable:

- attention MIL
- gated attention MIL
- logsumexp pooling with learnable temperature
- top-k pooling with learned scoring

Preferred default:

```text
attention MIL
```

Expected outputs per branch:

```python
branch_attention_weights_s: [B, T_s]
branch_embedding_s:         [B, D]
branch_logit_s:             [B]      # binary
or [B, C]                   # multiclass
```

Concatenate branch logits later for final fusion.

---

## Step 5: Optional branch auxiliary loss

The model itself should expose branch logits, but **auxiliary loss composition belongs in the training config / trainer**, not inside the model forward.

### Required config placement

Place the option under:

```json
"train": {
  "loss": {
    "branch_auxiliary": {
      "enabled": false,
      "weight": 0.3,
      "aggregation": "mean"
    }
  }
}
```

This placement is intentional and should be kept.

### Required trainer behavior

The trainer should compute:

```python
final_loss = criterion(output.logits, labels)

if branch_auxiliary.enabled and output.branch_logits is not None:
    aux_loss = aggregate_branch_losses(output.branch_logits, labels)
    total_loss = final_loss + weight * aux_loss
else:
    total_loss = final_loss
```

Recommended aggregation:

```text
mean over branch losses
```

### Output contract change

Extend the model output so the trainer can access branch logits.

Preferred:

```python
@dataclass(frozen=True)
class AstModelOutput:
    logits: Tensor
    pooled_embedding: Tensor
    branch_logits: Tensor | None = None
    branch_attention_weights: tuple[Tensor, ...] | None = None
    selected_evidence_tokens: Tensor | None = None
```

If you need a different output dataclass, update all consumers consistently.

---

## Step 6: Top evidence token selection

The RDT state should no longer come from latent queries.

Instead, select the top evidence tokens from each branch using branch MIL scores.

### Required default

Use:

```json
"rdt": {
  "top_tokens_per_branch": 2
}
```

Then:

```text
4 branches × 2 evidence tokens = 8 RDT state tokens
```

This preserves the old `[B, 8, D]` RDT state shape, but gives it a much more meaningful origin.

If a branch has event tokens:

```text
E_s: [B, T_s, D]
```

and branch attention / evidence score:

```text
a_s: [B, T_s]
```

then select the top `M=2` time indices per branch and gather the corresponding event tokens.

Expected:

```text
U0: [B, 8, D]
```

---

## Step 7: RDT as optional evidence refinement

RDT is no longer the primary discovery mechanism. It is now an optional refinement module.

### Required config placement

Put RDT options under:

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

### Required behavior

If:

```json
"rdt": { "enabled": false }
```

then:

- do not call the RDT block
- do not depend on RDT modules in forward
- still return a valid final classifier output

If:

```json
"rdt": { "enabled": true }
```

then:

- build `U0` from top evidence tokens
- build `H_ctx` from concatenated temporal event tokens
- apply the same RDT block repeatedly `steps` times

### RDT shapes

Initial state:

```text
U0: [B, 8, D]
```

Context memory:

```text
H_ctx = concat(E1, E2, E3, E4)
H_ctx: [B, 1916, D]
```

because:

```text
127 + 255 + 511 + 1023 = 1916
```

Refinement:

```text
U1 = RDTBlock(U0, H_ctx) -> [B, 8, D]
U2 = RDTBlock(U1, H_ctx) -> [B, 8, D]
...
```

For `steps = 2`:

```text
U2: [B, 8, D]
```

For `steps = 3`:

```text
U3: [B, 8, D]
```

### RDT block internals

Keep the existing style:

```text
1. self-attention over U
2. cross-attention from U to H_ctx
3. feed-forward
```

But add a stabilizer.

Preferred:

```text
LayerScale or gated residual
```

Recommended:

```python
u = u + alpha * delta
```

with

```text
alpha initialized to 1e-2
```

If you implement a gate instead, that is also acceptable.

### Required semantic rule

If `rdt.enabled = false`, `steps` is ignored.

---

## Step 8: Final fusion and classifier

The final classifier should not depend only on the RDT output.

It should also incorporate branch-level signals.

### Recommended fusion

Let:

```text
r = mean(U_T, dim=1)      # [B, D] if RDT on
or
r = mean(U0, dim=1)       # [B, D] if RDT off and top evidence tokens are still selected

z_branch = stack(branch_logits over 4 branches)
z_branch: [B, 4]          # binary
or [B, 4 * C]            # multiclass if needed
```

Then fuse:

```python
final_input = concat([r, z_branch], dim=-1)
```

and classify with a small final head.

### Minimal acceptable alternative

If you want to keep the existing classifier interface simpler, you may instead:

- use branch embeddings and branch logits to produce a fused `[B, D]`
- then apply the existing classifier head

But the final path must explicitly preserve branch-level signal.

Do **not** rely only on the global pooled event tokens without any branch logit path.

---

## Configuration Schema Changes

## Keep these top-level sections

Keep the current config-driven workflow shape:

```json
{
  "experiment": ...,
  "data": ...,
  "model": ...,
  "train": ...,
  "evaluation": ...
}
```

---

## Model architecture schema changes

Replace the current architecture config notion of:

```json
"latent_query_count": 8,
"rdt_steps": 3
```

with a nested RDT block:

```json
"architecture": {
  "hidden_size": 192,
  "num_attention_heads": 4,
  "mlp_ratio": 2.0,
  "hidden_dropout_prob": 0.1,
  "attention_probs_dropout_prob": 0.1,
  "layer_norm_eps": 1e-6,
  "shared_stem_depth": 2,
  "adapter_depth": 1,
  "patch_branches": [
    {"patch_size": [16, 16], "stride": [8, 16]},
    {"patch_size": [8, 32], "stride": [4, 32]},
    {"patch_size": [4, 64], "stride": [2, 64]},
    {"patch_size": [2, 128], "stride": [1, 128]}
    ],
  "rdt": {
    "enabled": false,
    "steps": 3,
    "top_tokens_per_branch": 2,
    "gated_residual": true,
    "layerscale_init": 0.01
  }
}
```

### Validation requirements

Update config validation so that:

- `rdt.enabled` is a boolean
- `rdt.steps > 0` if `rdt.enabled = true`
- `rdt.top_tokens_per_branch > 0`
- `hidden_size % num_attention_heads == 0`
- patch branches still fit the input dimensions
- `shared_stem_depth > 0`
- `adapter_depth > 0`

Optional strictness:

- reject `latent_query_count` as deprecated in this refactor
- reject top tokens per branch that would exceed practical branch lengths

Recommended behavior:

```text
Reject latent_query_count in new configs.
```

---

## Train loss schema changes

Add under:

```json
"train": {
  "loss": {
    "type": "bce",
    "auto_pos_weight": true,
    "branch_auxiliary": {
      "enabled": false,
      "weight": 0.3,
      "aggregation": "mean"
    }
  }
}
```

This placement is required.

### Validation requirements

- `branch_auxiliary.enabled` is boolean
- if `enabled = true`, `weight > 0`
- `aggregation` currently support `"mean"` only (it is fine to keep it future-proof, but you only need to implement `"mean"`)

---

## File-Level Refactor Plan

The current branch already contains model/config/training/setup files. Refactor those rather than starting over.

### Likely key files

Inspect and modify as needed:

```text
src/models/multiscale_rdt_ast.py
src/utils/config.py
src/training/trainer.py
src/training/ast_setup.py
src/cli/training.py
src/cli/evaluate.py
src/cli/cv.py
src/utils/checkpoint.py
README.md
configs/*.json
tests/*
```

---

## Detailed file-level expectations

## `src/models/multiscale_rdt_ast.py`

This is the main refactor target.

### Remove or deactivate

- `LatentQueryPooler` from the active forward path

You may keep the class in the file temporarily, but it must not be used by the active model.

### Add or repurpose modules

Add or refactor modules to support:

- branch-local encoding
- frequency pooling to temporal event tokens
- branch MIL heads
- top evidence token selection
- optional RDT refinement
- final fusion

### Required encoder output behavior

The encoder / model internals should now expose enough information to support:

- final logits
- pooled embedding
- branch logits
- branch attention / evidence weights (optional but recommended)
- selected evidence tokens (optional but recommended for diagnostics)

---

## `src/utils/config.py`

Refactor the schema to support:

- `model.encoder.architecture.rdt.enabled`
- `model.encoder.architecture.rdt.steps`
- `model.encoder.architecture.rdt.top_tokens_per_branch`
- `model.encoder.architecture.rdt.gated_residual`
- `model.encoder.architecture.rdt.layerscale_init`
- `train.loss.branch_auxiliary.enabled`
- `train.loss.branch_auxiliary.weight`
- `train.loss.branch_auxiliary.aggregation`

The new config should still parse cleanly from the JSON loader and remain checkpoint-serializable.

---

## `src/training/trainer.py`

Refactor loss computation.

Current behavior is single-loss based.

Required new behavior:

```python
final_loss = criterion(output.logits, labels)
total_loss = final_loss

if train.loss.branch_auxiliary.enabled:
    if output.branch_logits is None:
        raise ValueError("branch auxiliary loss enabled but model did not return branch_logits")
    aux_loss = compute_branch_auxiliary_loss(output.branch_logits, labels)
    total_loss = total_loss + weight * aux_loss
```

Also:

- keep binary and multiclass support
- keep diagnostics flow working
- if branch attention maps or selected evidence tokens are returned, optionally add them to diagnostics payload

---

## `src/training/ast_setup.py`

Refactor optimizer grouping and architecture summary logic.

### Current assumption to replace

The current branch documents optimizer grouping as:

```text
encoder parameters -> encoder_lr
latent pooler + RDT + classifier -> head_lr
```

This is no longer correct.

### New grouping

Use:

```text
encoder parameters:
    patch tokenizers
    shared stem
    adapters
    frequency pooling layers
    branch MIL heads

head parameters:
    optional RDT
    final fusion / classifier
```

If RDT is disabled:

```text
head parameters = final fusion / classifier only
```

### Adaptation rules

Keep the same top-level adaptation contract if possible:

- `mode = "full"`: train everything
- `mode = "frozen"`: freeze encoder, train head
- `mode = "partial"`: reject
- `num_layers` must be 0

But update the meaning of “head” so it no longer mentions latent pooling.

Old phrasing like:

```text
latent pooler + RDT + classifier
```

should become:

```text
optional RDT + final classifier/fusion
```

---

## `src/cli/training.py`, `src/cli/cv.py`, `src/cli/evaluate.py`

These should continue to work with config-driven model construction.

You may keep function names such as `build_ast_model(...)` if that avoids wider churn, but they should now build the refactored event-MIL model.

Evaluation and CV must still reconstruct the model from checkpoint config.

---

## `src/utils/checkpoint.py`

Update checkpoint parsing to support the new config fields:

- `rdt.enabled`
- `rdt.steps`
- `rdt.top_tokens_per_branch`
- `rdt.gated_residual`
- `rdt.layerscale_init`
- `train.loss.branch_auxiliary.*` if train config is stored

Reject deprecated `latent_query_count` if needed, or document migration clearly.

Preferred behavior:

```text
Checkpoint/configs using the old latent-query architecture are considered incompatible with this new refactor branch.
```

---

## Canonical Configs to Add or Update

Create or update canonical configs so the four experimental modes can be run directly.

Recommended file set:

```text
configs/training_event_mil_b0.json
configs/training_event_mil_b1.json
configs/training_event_mil_b2.json
configs/training_event_mil_b3.json
```

You may also update:

```text
configs/cv_multiscale_rdt.json
configs/eval_multiscale_rdt.json
configs/training_multiclass.json
```

if those should reflect the new architecture.

### Required differences

#### B0

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
},
"train": {
  "loss": {
    "branch_auxiliary": {
      "enabled": false,
      "weight": 0.3,
      "aggregation": "mean"
    }
  }
}
```

#### B1

Same as B0, but:

```json
"branch_auxiliary": {
  "enabled": true,
  "weight": 0.3,
  "aggregation": "mean"
}
```

#### B2

Same as B1, but:

```json
"rdt": {
  "enabled": true,
  "steps": 2,
  "top_tokens_per_branch": 2,
  "gated_residual": true,
  "layerscale_init": 0.01
}
```

#### B3

Same as B1, but:

```json
"rdt": {
  "enabled": true,
  "steps": 3,
  "top_tokens_per_branch": 2,
  "gated_residual": true,
  "layerscale_init": 0.01
}
```

---

## Tests to Add or Update

Update the test suite so the refactor is validated.

### 1. Config parsing tests

Add tests for:

- `rdt.enabled = false`
- `rdt.enabled = true`
- `rdt.steps = 2`
- `rdt.steps = 3`
- `branch_auxiliary.enabled = false`
- `branch_auxiliary.enabled = true`
- invalid `rdt.steps <= 0` when enabled
- invalid `branch_auxiliary.weight <= 0` when enabled
- deprecated `latent_query_count` rejected

---

### 2. Branch length / shape tests

Verify branch temporal lengths after frequency pooling:

```text
Branch 1 -> 127
Branch 2 -> 255
Branch 3 -> 511
Branch 4 -> 1023
total temporal context length -> 1916
```

Verify:

```text
top_tokens_per_branch = 2
num_branches = 4
U0 shape = [B, 8, D]
H_ctx shape = [B, 1916, D]
```

---

### 3. Model forward tests

#### B0-like forward

- RDT disabled
- branch auxiliary disabled

Expected:

- model forward succeeds
- logits shape correct
- pooled embedding shape correct
- no dependency on RDT path

#### B1-like forward

- RDT disabled
- branch auxiliary enabled

Expected:

- model forward returns branch logits
- trainer can compute final + auxiliary loss

#### B2/B3-like forward

- RDT enabled with `steps=2` or `steps=3`

Expected:

- selected evidence tokens shape `[B, 8, D]`
- RDT context shape `[B, 1916, D]`
- RDT output shape `[B, 8, D]`

---

### 4. Trainer loss composition tests

Add tests that assert:

- with branch auxiliary disabled, total loss == final loss
- with branch auxiliary enabled, total loss == final loss + λ * auxiliary loss
- enabling branch auxiliary without `branch_logits` returned raises a clear error

---

### 5. Optimizer grouping tests

Verify:

- encoder param group contains encoder-side modules
- head param group contains optional RDT + classifier/fusion
- if RDT disabled, the head group still exists and does not fail

---

## Acceptance Criteria

The implementation is acceptable when all of these are true:

1. The base branch is `MIL-RDT-AST`.
2. The active model is no longer latent-query-centered.
3. The patch geometry remains unchanged.
4. The model can run with:

   - RDT disabled
   - branch auxiliary disabled

5. The model can run with:

   - RDT disabled
   - branch auxiliary enabled

6. The model can run with:

   - RDT enabled
   - `steps = 2`
   - branch auxiliary enabled

7. The model can run with:

   - RDT enabled
   - `steps = 3`
   - branch auxiliary enabled

8. RDT off is controlled by:

```json
"rdt": { "enabled": false }
```

9. branch auxiliary loss is controlled under:

```json
"train.loss.branch_auxiliary"
```

10. The trainer composes auxiliary loss only when enabled.
11. The model returns enough outputs for branch auxiliary loss and diagnostics.
12. Evaluation and CV still reconstruct the model from config/checkpoint.
13. README describes the new event-MIL-centered architecture and the B0/B1/B2/B3 config knobs.
14. Canonical configs for B0/B1/B2/B3 exist.
15. Tests pass.

---

## Commands to Run

Run at least:

```bash
pytest
```

Also run if configured:

```bash
ruff check src tests
```

And if practical:

```bash
mypy src tests
```

If any command cannot run because of environment limitations, document:

- the exact command
- the exact failure
- what remains unverified

---

## Avoid These Mistakes

- Do not keep `LatentQueryPooler` in the active model path.
- Do not use `steps = 0` as the primary semantic for “RDT off”.
- Do not place branch auxiliary config under `model`.
- Do not place RDT enable/disable config under `train.loss`.
- Do not change the patch geometry.
- Do not keep the old `[B, 4081, D] -> latent query -> [B, 8, D]` path as the main design.
- Do not make RDT responsible for discovering evidence from scratch.
- Do not use only the final classifier without branch-level supervision support when branch auxiliary is enabled.
- Do not break binary logits shape `[B]`.
- Do not reintroduce Hugging Face `ASTModel`.
- Do not make B0/B1/B2/B3 require code edits instead of config changes.

---

## Suggested Minimal Semantic Pseudocode

```python
class EventMilRdtAstModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = MultiScaleEventEncoder(cfg.encoder)
        self.branch_mil_heads = nn.ModuleList(...)
        self.rdt_block = OptionalRdtBlock(cfg.encoder.architecture.rdt)  # or direct block
        self.final_fusion = FinalFusionHead(...)

    def forward(self, input_values: Tensor) -> AstModelOutput:
        x = input_values.unsqueeze(1)  # [B, 1, 1024, 128]

        branch_event_tokens = self.encoder(x)
        # tuple/list:
        # E1 [B,127,D], E2 [B,255,D], E3 [B,511,D], E4 [B,1023,D]

        branch_logits = []
        branch_embeddings = []
        branch_scores = []
        selected_tokens = []

        for tokens, mil_head in zip(branch_event_tokens, self.branch_mil_heads):
            mil_out = mil_head(tokens)
            branch_logits.append(mil_out.logits)
            branch_embeddings.append(mil_out.embedding)
            branch_scores.append(mil_out.attention_weights)
            selected_tokens.append(select_top_tokens(tokens, mil_out.attention_weights, top_k=2))

        branch_logits = stack_branch_logits(branch_logits)  # [B, 4] binary or adapted shape
        branch_embeddings = stack_branch_embeddings(branch_embeddings)  # [B, 4, D]

        u0 = torch.cat(selected_tokens, dim=1)  # [B, 8, D]
        h_ctx = torch.cat(branch_event_tokens, dim=1)  # [B, 1916, D]

        if self.cfg.encoder.architecture.rdt.enabled:
            u = u0
            for _ in range(self.cfg.encoder.architecture.rdt.steps):
                u = self.rdt_block(u, h_ctx)
        else:
            u = u0

        evidence_embedding = u.mean(dim=1)  # [B, D]
        pooled_embedding = fuse(evidence_embedding, branch_embeddings, branch_logits)  # [B, D] or fuse later
        logits = self.final_fusion(pooled_embedding, branch_logits)

        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)

        return AstModelOutput(
            logits=logits,
            pooled_embedding=pooled_embedding,
            branch_logits=branch_logits,
            branch_attention_weights=tuple(branch_scores),
            selected_evidence_tokens=u0,
        )
```

---

## Final Note

This refactor should produce a single codebase that supports:

- **B0**: 4-scale temporal MIL, no RDT
- **B1**: 4-scale temporal MIL + branch auxiliary loss
- **B2**: B1 + top-2 evidence RDT, `T=2`
- **B3**: B1 + top-2 evidence RDT, `T=3`

by changing config only.

The architectural center of gravity should move from:

```text
latent-query bottleneck + RDT
```

to:

```text
event-MIL first, optional RDT later
```

That is the intended design of this refactor.
