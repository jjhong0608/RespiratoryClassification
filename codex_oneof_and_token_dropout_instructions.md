# Codex Instructions: Add One-of Augmentation, Branch Event Token Dropout, and Selected Evidence Dropout

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

The current branch is already an event-MIL-first respiratory sound classifier. Do **not** rewrite the model architecture from scratch. Do **not** reintroduce Hugging Face `ASTModel`. Do **not** reintroduce latent-query pooling.

The goal is to add three configurable regularization/augmentation mechanisms while preserving the current **H0 branch-aware gated pooling** model:

1. **One-of input augmentation**
2. **Branch event token dropout**
3. **Selected evidence dropout**

All three mechanisms must default to **off** unless explicitly enabled in config.

---

## Current Model Assumption: H0 Branch-Aware Gated Model

All new configs in this task should use the H0 branch-aware gated model.

H0 means:

```text
4-scale Event-MIL-RDT
evidence_pooling.type = "branch_gated"
RDT enabled
RDT steps = 3
top_tokens_per_branch = 2
BCE loss
branch auxiliary loss enabled
branch auxiliary weight = 0.1
encoder_lr = 1e-5
head_lr = 3e-4
no warm-start
```

The current active model flow should remain:

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
  -> branch-aware gated evidence pooling
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
branch MIL heads except adding optional token_mask support
top-k evidence selection logic except optional selected evidence dropout after selection
branch-aware gated evidence pooling
final fusion inputs
branch auxiliary loss behavior
classifier output shape
```

---

## Fixed Default Geometry

Keep the default 4-scale patch geometry for all configs created in this task:

| Branch | Patch size `(time, freq)` | Stride `(time, freq)` | Temporal event tokens |
|---:|---:|---:|---:|
| 0 | `(16, 16)` | `(8, 16)` | `127` |
| 1 | `(8, 32)` | `(4, 32)` | `255` |
| 2 | `(4, 64)` | `(2, 64)` | `511` |
| 3 | `(2, 128)` | `(1, 128)` | `1023` |

Default context/evidence shapes:

```text
H_ctx length = 127 + 255 + 511 + 1023 = 1916
top_tokens_per_branch = 2
selected evidence length = 4 branches × 2 = 8
U0 = [B, 8, D]
```

---

# Part A — One-of Input Augmentation

## Objective

Add a **one-of augmentation policy** so waveform and fbank augmentation can both be used during training without always stacking both on the same sample.

The policy should sample exactly one choice per training example:

```text
none
waveform
fbank
both_light
```

Recommended probabilities:

```text
none:       0.10
waveform:   0.35
fbank:      0.35
both_light: 0.20
```

The motivation:

```text
AUG3 waveform+fbank was somewhat strong.
One-of policy keeps augmentation diversity while reducing over-augmentation per sample.
```

---

## Data Pipeline Placement

The current data pipeline is approximately:

```text
WaveformLoader.load(path)
  -> WaveformPreprocessor.prepare(waveform)
  -> AstLikeFbank(clip_waveform)
  -> transpose(0, 1).contiguous()
  -> ClipSample(input_values=feature_map, ...)
```

Add one-of augmentation like this:

```text
waveform = WaveformLoader.load(path)
clip_waveform = WaveformPreprocessor.prepare(waveform)

choice = augmentation_policy.sample()

if choice in {"waveform", "both_light"}:
    clip_waveform = waveform_augmenter(clip_waveform)

feature_map = AstLikeFbank(clip_waveform).transpose(0, 1).contiguous()

if choice in {"fbank", "both_light"}:
    feature_map = fbank_augmenter(feature_map)

return ClipSample(input_values=feature_map, ...)
```

### Train-only rule

Apply input augmentation only to training data.

Required behavior:

```text
split == "train" and data.augmentation.enabled == true
  -> input augmentation may be applied

split in {"val", "eval", "test"}
  -> input augmentation always disabled
```

This must also work under cross-validation:

```text
CV train fold -> augmentation may be active
CV validation fold -> augmentation off
```

---

## Config Schema for One-of Augmentation

Extend the existing `data.augmentation` config.

Recommended schema:

```json
"data": {
  "augmentation": {
    "enabled": true,
    "policy": {
      "type": "one_of",
      "choices": [
        {"name": "none", "probability": 0.10},
        {"name": "waveform", "probability": 0.35},
        {"name": "fbank", "probability": 0.35},
        {"name": "both_light", "probability": 0.20}
      ]
    },
    "waveform": {
      "enabled": true,
      "probability": 1.0,
      "gain": {
        "enabled": true,
        "probability": 0.4,
        "min_db": -2.0,
        "max_db": 2.0
      },
      "noise": {
        "enabled": true,
        "probability": 0.2,
        "snr_db_min": 20.0,
        "snr_db_max": 35.0
      },
      "time_shift": {
        "enabled": true,
        "probability": 0.3,
        "max_shift_fraction": 0.025,
        "mode": "zero_pad"
      }
    },
    "fbank": {
      "enabled": true,
      "probability": 1.0,
      "time_mask": {
        "enabled": true,
        "num_masks": 1,
        "max_width": 16
      },
      "freq_mask": {
        "enabled": true,
        "num_masks": 1,
        "max_width": 4
      },
      "mask_value": 0.0
    }
  }
}
```

### Default behavior

If `data.augmentation` is omitted:

```text
augmentation disabled
```

If `data.augmentation.enabled = false`:

```text
no waveform or fbank augmentation
```

If `policy.type = "one_of"`:

```text
sample exactly one choice according to configured probabilities
```

If an earlier independent waveform/fbank augmentation mode already exists, preserve it. Add `one_of` as an additional policy type. If no policy is provided but `waveform.enabled` or `fbank.enabled` exists, keep current behavior for backward compatibility.

---

## Validation Rules for One-of Policy

Add validation:

```text
data.augmentation.policy.type in {"independent", "one_of"} or supported current values
choice names must be from {"none", "waveform", "fbank", "both_light"}
choice probabilities must be >= 0
sum(choice probabilities) must be approximately 1.0
at least one choice must have probability > 0
```

Also validate existing waveform/fbank fields:

```text
all probabilities in [0, 1]
gain.min_db <= gain.max_db
noise.snr_db_min > 0
noise.snr_db_min <= noise.snr_db_max
time_shift.max_shift_fraction >= 0
time_shift.max_shift_fraction < 1
time_shift.mode in {"zero_pad", "roll"}
mask.num_masks >= 0
mask.max_width >= 0
```

---

## Implementation Location

Likely files:

```text
src/utils/config.py
src/data/augmentation.py
src/data/dataset.py
src/data/loaders.py
```

If `src/data/augmentation.py` already exists from AUG0–AUG3 work, extend it. Otherwise create it.

Recommended classes:

```python
class AugmentationPolicy:
    def __init__(self, cfg): ...
    def sample(self) -> str: ...


class AugmentationPipeline:
    def __init__(self, cfg, sample_rate: int): ...
    def sample_choice(self) -> str: ...
    def apply_waveform(self, waveform: Tensor, choice: str) -> Tensor: ...
    def apply_fbank(self, feature: Tensor, choice: str) -> Tensor: ...
```

Use PyTorch RNG where possible:

```python
torch.rand
torch.multinomial
torch.randint
torch.randn_like
```

Avoid Python `random` or NumPy unless worker seeding is explicitly handled.

---

## One-of Diagnostics / Logging

Since input augmentation is train-only, evaluation diagnostics usually cannot include sample augmentation choice. At minimum, log epoch-level counts during training:

```text
augmentation_choice_counts:
  none
  waveform
  fbank
  both_light
```

If adding full train diagnostics is too invasive, keep a lightweight counter in the dataset or training loop and log counts per epoch.

This is recommended, not required for model correctness.

---

# Part B — Branch Event Token Dropout

## Objective

Add model-internal dropout after frequency-attention pooling and before branch MIL.

Current branch flow:

```text
branch grid tokens: [B, T_s, F_s, D]
  -> FrequencyAttentionPooler
branch event tokens E_s: [B, T_s, D]
  -> BranchMilHead
```

New flow when enabled:

```text
branch event tokens E_s: [B, T_s, D]
  -> BranchEventTokenDropout
E_s_drop: [B, T_s, D]
token_mask_s: [B, T_s]
  -> BranchMilHead(E_s_drop, token_mask=token_mask_s)
```

Purpose:

```text
Prevent branch MIL from over-relying on a small number of temporal event tokens.
Improve robustness of branch-level event detection.
```

---

## Recommended Dropout Method

Use:

```text
zero token + attention mask
```

Meaning:

```text
1. Dropped event token feature is set to zero.
2. Dropped token is masked out from BranchMilHead attention.
```

Do **not** only zero the token without masking, because the MIL attention head might still select zeroed tokens.

---

## Config Schema

Add under:

```text
model.encoder.architecture.token_augmentation.branch_event_dropout
```

Recommended:

```json
"token_augmentation": {
  "branch_event_dropout": {
    "enabled": false,
    "probability": 0.05,
    "mode": "zero_mask",
    "min_keep_tokens": 1
  }
}
```

Defaults:

```text
enabled = false
probability = 0.0 or 0.05
mode = "zero_mask"
min_keep_tokens = 1
```

Validation:

```text
probability in [0, 1)
mode in {"zero_mask"}
min_keep_tokens >= 1
```

At runtime:

```text
min_keep_tokens must not exceed T_s.
If it does, clamp to T_s.
```

---

## BranchEventTokenDropout Module

Add a module, likely in:

```text
src/models/multiscale_rdt_ast.py
```

Suggested name:

```python
BranchEventTokenDropout
```

Input:

```python
tokens: Tensor
# [B, T_s, D]
```

Output:

```python
dropped_tokens: Tensor
# [B, T_s, D]

token_mask: Tensor
# [B, T_s], dtype bool
# True = keep
# False = dropped
```

Behavior:

```text
if not training or not enabled:
    return tokens, all_true_mask
```

During training:

```text
Sample independent Bernoulli keep/drop mask per token.
Ensure at least min_keep_tokens per sample.
Set dropped tokens to zero.
Return token mask to BranchMilHead.
```

Important:

```text
Do not drop all tokens in a branch for a sample.
```

Pseudo-code:

```python
keep_mask = torch.rand(B, T, device=tokens.device) >= probability

for each sample b:
    if keep_mask[b].sum() < min_keep:
        re-enable random tokens until min_keep is satisfied

dropped_tokens = tokens * keep_mask.unsqueeze(-1).to(tokens.dtype)
return dropped_tokens, keep_mask
```

---

## BranchMilHead Mask Support

Modify `BranchMilHead.forward(...)` to accept an optional mask:

```python
def forward(
    self,
    tokens: Tensor,
    token_mask: Tensor | None = None,
) -> BranchMilOutput:
    ...
```

Apply mask before softmax:

```python
attention_logits = ...
if token_mask is not None:
    attention_logits = attention_logits.masked_fill(~token_mask, -1e4)
attention_weights = torch.softmax(attention_logits, dim=1)
```

The mask must not break instance-logit or branch logit computation.

If `token_mask` is all true, behavior should be identical to previous behavior.

---

## Branch Event Dropout Diagnostics

Add optional train-time summary if practical:

```text
branch_event_dropout_keep_ratio per branch
branch_event_dropout_drop_ratio per branch
```

This can be logged per epoch. It does not need to be in eval JSONL because dropout is disabled at eval time.

---

# Part C — Selected Evidence Dropout

## Objective

Add dropout after top-k evidence selection and before RDT.

Current flow:

```text
Branch MIL
  -> top-k evidence tokens per branch
U0: [B, K, D]
selected_evidence_branch_ids: [B, K]
  -> RDT
```

New flow when enabled:

```text
U0: [B, K, D]
selected_evidence_branch_ids: [B, K]
  -> SelectedEvidenceDropout
U0_drop: [B, K, D]
selected_evidence_dropout_mask: [B, K]
  -> RDT
```

Purpose:

```text
Prevent RDT and branch-aware gated pooling from over-relying on one selected evidence token or one branch's selected token.
```

Important:

```text
Selected evidence dropout affects U0 only.
Do not modify H_ctx.
```

RDT context `H_ctx` must remain unchanged.

---

## Recommended Dropout Method

Use:

```text
branch-constrained zero dropout
```

Default H0:

```text
4 branches
top_tokens_per_branch = 2
K = 8
```

Within each branch, drop selected tokens with probability `p`, but ensure:

```text
min_keep_per_branch = 1
```

So for H0 top-2, at least one selected evidence token per branch remains.

---

## Config Schema

Add under:

```text
model.encoder.architecture.token_augmentation.selected_evidence_dropout
```

Recommended:

```json
"token_augmentation": {
  "selected_evidence_dropout": {
    "enabled": false,
    "probability": 0.05,
    "mode": "zero",
    "min_keep_per_branch": 1
  }
}
```

Defaults:

```text
enabled = false
probability = 0.0 or 0.05
mode = "zero"
min_keep_per_branch = 1
```

Validation:

```text
probability in [0, 1)
mode in {"zero"}
min_keep_per_branch >= 1
```

At runtime:

```text
min_keep_per_branch must not exceed number of selected evidence tokens for that branch.
If it does, clamp to available tokens.
```

---

## SelectedEvidenceDropout Module

Add a module, likely in:

```text
src/models/multiscale_rdt_ast.py
```

Suggested name:

```python
SelectedEvidenceDropout
```

Input:

```python
selected_tokens: Tensor
# [B, K, D]

selected_branch_ids: Tensor
# [B, K]
```

Output:

```python
dropped_selected_tokens: Tensor
# [B, K, D]

dropout_keep_mask: Tensor
# [B, K], dtype bool
```

Behavior:

```text
if not training or not enabled:
    return selected_tokens, all_true_mask
```

During training:

1. For each sample and each branch present in `selected_branch_ids`, sample token keep/drop mask.
2. Ensure at least `min_keep_per_branch` selected tokens are kept per branch.
3. Set dropped selected tokens to zero.
4. Return mask.

Pseudo-code:

```python
keep_mask = torch.ones(B, K, dtype=torch.bool, device=selected_tokens.device)

for each sample b:
    for each unique branch_id in selected_branch_ids[b]:
        positions = torch.where(selected_branch_ids[b] == branch_id)[0]
        branch_keep = torch.rand(len(positions), device=device) >= probability
        if branch_keep.sum() < min_keep:
            re-enable random positions until min_keep is satisfied
        keep_mask[b, positions] = branch_keep

dropped = selected_tokens * keep_mask.unsqueeze(-1).to(selected_tokens.dtype)
return dropped, keep_mask
```

Important:

```text
selected_evidence_indices
selected_evidence_scores
selected_evidence_branch_ids
must remain unchanged.
```

Only selected token embeddings are zeroed for the RDT path.

---

## Selected Evidence Dropout Diagnostics

Add optional model output fields if practical:

```python
selected_evidence_dropout_mask: Tensor | None
selected_evidence_keep_ratio: Tensor | None
```

Shapes:

```text
selected_evidence_dropout_mask: [B, K]
selected_evidence_keep_ratio: [B] or scalar
```

Since dropout is train-only, eval diagnostics will usually show all keep or None.

At minimum, expose/dropout summary in training logs:

```text
selected_evidence_dropout_keep_ratio
```

---

# Part D — Integrate Token Augmentation into Model Forward

Locate the current model forward in:

```text
src/models/multiscale_rdt_ast.py
```

The integration should be approximately:

## Branch event dropout integration

Current likely logic:

```python
branch_event_tokens = freq_pooler(branch_grid_tokens)
mil_output = branch_mil_head(branch_event_tokens)
```

Change to:

```python
branch_event_tokens = freq_pooler(branch_grid_tokens)

if self.branch_event_dropout is not None:
    branch_event_tokens, token_mask = self.branch_event_dropout(branch_event_tokens)
else:
    token_mask = None

mil_output = branch_mil_head(branch_event_tokens, token_mask=token_mask)
```

Ensure branch event dropout only applies when:

```python
self.training and config.enabled
```

## Selected evidence dropout integration

Current likely logic:

```python
selected_tokens = torch.cat(selected_tokens_list, dim=1)
selected_indices = torch.cat(...)
selected_scores = torch.cat(...)
selected_branch_ids = torch.cat(...)

u = selected_tokens

if rdt.enabled:
    for _ in range(rdt.steps):
        u = rdt_block(u, h_ctx)
```

Change to:

```python
selected_tokens = torch.cat(selected_tokens_list, dim=1)
selected_indices = torch.cat(...)
selected_scores = torch.cat(...)
selected_branch_ids = torch.cat(...)

u = selected_tokens

selected_dropout_mask = None
if self.selected_evidence_dropout is not None:
    u, selected_dropout_mask = self.selected_evidence_dropout(
        u,
        selected_branch_ids,
    )

if rdt.enabled:
    for _ in range(rdt.steps):
        u = rdt_block(u, h_ctx)
```

Keep:

```text
H_ctx unchanged
selected_evidence_indices unchanged
selected_evidence_scores unchanged
selected_evidence_branch_ids unchanged
```

---

# Part E — Configs to Add

Create configs for the first controlled experiments using H0 branch-aware gated pooling.

All configs should keep:

```text
4-scale
RDT on
steps = 3
top_tokens_per_branch = 2
BCE
branch_auxiliary.enabled = true
branch_auxiliary.weight = 0.1
encoder_lr = 1e-5
head_lr = 3e-4
no warm-start
evidence_pooling.type = branch_gated
```

## 1. One-of only

File:

```text
configs/training_aug4_h0_gated_oneof.json
```

Purpose:

```text
Use one-of augmentation without internal token dropout.
```

Config:

```json
"data": {
  "augmentation": {
    "enabled": true,
    "policy": {
      "type": "one_of",
      "choices": [
        {"name": "none", "probability": 0.10},
        {"name": "waveform", "probability": 0.35},
        {"name": "fbank", "probability": 0.35},
        {"name": "both_light", "probability": 0.20}
      ]
    },
    "waveform": {
      "enabled": true,
      "probability": 1.0,
      "gain": {
        "enabled": true,
        "probability": 0.4,
        "min_db": -2.0,
        "max_db": 2.0
      },
      "noise": {
        "enabled": true,
        "probability": 0.2,
        "snr_db_min": 20.0,
        "snr_db_max": 35.0
      },
      "time_shift": {
        "enabled": true,
        "probability": 0.3,
        "max_shift_fraction": 0.025,
        "mode": "zero_pad"
      }
    },
    "fbank": {
      "enabled": true,
      "probability": 1.0,
      "time_mask": {
        "enabled": true,
        "num_masks": 1,
        "max_width": 16
      },
      "freq_mask": {
        "enabled": true,
        "num_masks": 1,
        "max_width": 4
      },
      "mask_value": 0.0
    }
  }
}
```

Model token augmentation:

```json
"token_augmentation": {
  "branch_event_dropout": {
    "enabled": false,
    "probability": 0.05,
    "mode": "zero_mask",
    "min_keep_tokens": 1
  },
  "selected_evidence_dropout": {
    "enabled": false,
    "probability": 0.05,
    "mode": "zero",
    "min_keep_per_branch": 1
  }
}
```

---

## 2. One-of + branch event token dropout

File:

```text
configs/training_pt1_h0_gated_oneof_branch_event_dropout.json
```

Purpose:

```text
Add branch event token dropout to one-of augmentation.
```

Same as `training_aug4_h0_gated_oneof.json`, but:

```json
"token_augmentation": {
  "branch_event_dropout": {
    "enabled": true,
    "probability": 0.05,
    "mode": "zero_mask",
    "min_keep_tokens": 1
  },
  "selected_evidence_dropout": {
    "enabled": false,
    "probability": 0.05,
    "mode": "zero",
    "min_keep_per_branch": 1
  }
}
```

---

## 3. One-of + selected evidence dropout

File:

```text
configs/training_pt2_h0_gated_oneof_selected_evidence_dropout.json
```

Purpose:

```text
Add selected evidence dropout to one-of augmentation.
```

Same as AUG4 one-of, but:

```json
"token_augmentation": {
  "branch_event_dropout": {
    "enabled": false,
    "probability": 0.05,
    "mode": "zero_mask",
    "min_keep_tokens": 1
  },
  "selected_evidence_dropout": {
    "enabled": true,
    "probability": 0.05,
    "mode": "zero",
    "min_keep_per_branch": 1
  }
}
```

---

## 4. One-of + both token dropouts

File:

```text
configs/training_pt3_h0_gated_oneof_both_token_dropouts.json
```

Purpose:

```text
Use one-of augmentation plus both model-internal token dropouts.
```

Config:

```json
"token_augmentation": {
  "branch_event_dropout": {
    "enabled": true,
    "probability": 0.05,
    "mode": "zero_mask",
    "min_keep_tokens": 1
  },
  "selected_evidence_dropout": {
    "enabled": true,
    "probability": 0.05,
    "mode": "zero",
    "min_keep_per_branch": 1
  }
}
```

---

# Part F — README Updates

Update README with a section:

```text
One-of and Token-Level Augmentation
```

Document:

## One-of augmentation

```text
The one-of policy samples exactly one augmentation recipe per training sample:
none, waveform, fbank, or both_light.
This avoids overly strong waveform+fbank stacking while still exposing the model to both augmentation families.
```

## Branch event token dropout

```text
Applied after frequency-attention pooling and before branch MIL.
Drops temporal branch event tokens with a mask so branch MIL cannot attend to dropped tokens.
```

## Selected evidence dropout

```text
Applied after top-k evidence selection and before RDT.
Drops selected evidence state tokens U0 while keeping H_ctx unchanged.
```

Add config table:

| Config | Purpose |
|---|---|
| `configs/training_aug4_h0_gated_oneof.json` | one-of augmentation only |
| `configs/training_pt1_h0_gated_oneof_branch_event_dropout.json` | one-of + branch event token dropout |
| `configs/training_pt2_h0_gated_oneof_selected_evidence_dropout.json` | one-of + selected evidence dropout |
| `configs/training_pt3_h0_gated_oneof_both_token_dropouts.json` | one-of + both token dropouts |

Add warning:

```text
All these configs use the same H0 branch-aware gated model.
Only augmentation/regularization settings differ.
```

---

# Part G — Tests

Add or update tests.

## 1. Config parsing tests

Test valid configs:

```text
training_aug4_h0_gated_oneof.json
training_pt1_h0_gated_oneof_branch_event_dropout.json
training_pt2_h0_gated_oneof_selected_evidence_dropout.json
training_pt3_h0_gated_oneof_both_token_dropouts.json
```

Test invalid configs:

```text
one_of probabilities do not sum to 1
invalid one_of choice name
negative probability
branch_event_dropout probability outside [0,1)
selected_evidence_dropout probability outside [0,1)
min_keep_tokens < 1
min_keep_per_branch < 1
unsupported dropout mode
```

---

## 2. One-of policy tests

Test:

```text
policy samples only allowed choices
probability vector is respected approximately in large sample simulation
none choice applies no augmentation
waveform choice applies waveform only
fbank choice applies fbank only
both_light applies waveform and fbank
```

If stochastic approximate tests are flaky, only test deterministic behavior by forcing probabilities:

```text
none=1.0
waveform=1.0
fbank=1.0
both_light=1.0
```

---

## 3. Train-only augmentation tests

Test:

```text
train split can use one-of augmentation
val split does not use augmentation
eval split does not use augmentation
```

If full fbank extraction is heavy, test dataset attributes or use monkeypatched augmenters.

---

## 4. Branch event token dropout tests

Input:

```text
tokens: [B, T, D]
```

Test:

```text
disabled -> identical tokens, all true mask
enabled -> shape preserved
enabled -> some tokens zeroed when probability=1 except min_keep
min_keep_tokens guaranteed
mask dtype bool
BranchMilHead respects token_mask
```

For `probability=1.0` in tests, if validation forbids 1.0, instantiate module directly or use high probability and deterministic mask. Runtime config should use probability < 1.0.

---

## 5. Selected evidence dropout tests

Input:

```text
selected_tokens: [B, K, D]
selected_branch_ids: [B, K]
```

Test:

```text
disabled -> identical tokens, all true mask
enabled -> shape preserved
branch-constrained min_keep_per_branch guaranteed
only selected tokens changed
branch_ids/indices/scores unchanged
H_ctx not modified in model integration
```

Use default H0 case:

```text
branch_ids = [0,0,1,1,2,2,3,3]
min_keep_per_branch = 1
```

Also test dynamic 3-branch case:

```text
branch_ids = [0,0,1,1,2,2]
```

---

## 6. Full model forward tests

Test four modes:

```text
no token augmentation
branch_event_dropout enabled
selected_evidence_dropout enabled
both token dropouts enabled
```

Use training mode for dropout:

```python
model.train()
```

Assert:

```text
forward succeeds
logits shape correct
pooled_embedding shape correct
selected evidence metadata still present
```

Use eval mode to assert dropout is disabled:

```python
model.eval()
```

---

# Part H — Acceptance Criteria

The task is complete when:

1. `data.augmentation.policy.type = "one_of"` is supported.
2. One-of policy supports `none`, `waveform`, `fbank`, and `both_light`.
3. One-of augmentation is train-only.
4. Waveform and fbank augmenters are reused or implemented cleanly.
5. `model.encoder.architecture.token_augmentation.branch_event_dropout` is supported.
6. Branch event token dropout is applied after frequency pooling and before branch MIL.
7. Branch event token dropout uses zeroing plus MIL attention mask.
8. Branch event token dropout guarantees at least `min_keep_tokens` per sample/branch.
9. `model.encoder.architecture.token_augmentation.selected_evidence_dropout` is supported.
10. Selected evidence dropout is applied after top-k evidence selection and before RDT.
11. Selected evidence dropout zeros only U0 selected evidence tokens.
12. Selected evidence dropout does not modify H_ctx.
13. Selected evidence dropout guarantees at least `min_keep_per_branch`.
14. All new features default to off.
15. H0 branch-aware gated architecture is preserved.
16. New configs exist:
    - `training_aug4_h0_gated_oneof.json`
    - `training_pt1_h0_gated_oneof_branch_event_dropout.json`
    - `training_pt2_h0_gated_oneof_selected_evidence_dropout.json`
    - `training_pt3_h0_gated_oneof_both_token_dropouts.json`
17. README documents the new options.
18. Tests pass.

---

# Part I — Avoid These Mistakes

- Do not apply augmentation to validation/evaluation/test data.
- Do not change H0 branch-aware gated architecture.
- Do not change RDT settings.
- Do not change top-k evidence selection before dropout.
- Do not modify H_ctx in selected evidence dropout.
- Do not allow all branch event tokens to be dropped.
- Do not allow all selected evidence tokens from a branch to be dropped.
- Do not drop branch logits or mean branch embedding from final fusion.
- Do not reintroduce latent-query pooling.
- Do not change patch geometry.
- Do not use strong time stretch or pitch shift in this task.
- Do not make one-of augmentation enabled by default.
- Do not make token dropouts enabled by default.
- Do not rely on Python random if PyTorch RNG can be used.
- Do not hard-code 4 branches or 8 selected evidence tokens.

---

# Part J — Suggested Execution Order After Implementation

Run tests:

```bash
pytest
```

Then train:

```bash
python -m src.cli.training --config configs/training_aug4_h0_gated_oneof.json
python -m src.cli.training --config configs/training_pt1_h0_gated_oneof_branch_event_dropout.json
python -m src.cli.training --config configs/training_pt2_h0_gated_oneof_selected_evidence_dropout.json
python -m src.cli.training --config configs/training_pt3_h0_gated_oneof_both_token_dropouts.json
```

Compare with prior:

```text
AUG0
AUG2
AUG3
H0 branch-gated no augmentation
```

Use:

```text
best_loss checkpoints
F1@0.5
F1@opt
ROC-AUC
PR-AUC
Brier
balanced accuracy
optimal threshold
gate entropy
gate weights
selected evidence branch distribution
```

---

# Part K — Self-Review Checklist for Implementability

Before finishing, verify this instruction can be implemented using only the information above.

## One-of augmentation

- [ ] Config path is clear: `data.augmentation.policy`.
- [ ] Allowed choices are clear.
- [ ] Probabilities are clear.
- [ ] Placement in data pipeline is clear.
- [ ] Train-only rule is clear.

## Branch event token dropout

- [ ] Placement is clear: after frequency pooling, before branch MIL.
- [ ] Mask semantics are clear.
- [ ] `BranchMilHead` optional `token_mask` change is clear.
- [ ] `min_keep_tokens` rule is clear.

## Selected evidence dropout

- [ ] Placement is clear: after top-k selection, before RDT.
- [ ] U0-only dropout is clear.
- [ ] H_ctx must remain unchanged.
- [ ] Branch-constrained `min_keep_per_branch` rule is clear.

## H0 preservation

- [ ] It is clear that H0 branch-aware gated architecture must remain unchanged.
- [ ] It is clear that branch auxiliary remains enabled in the new configs.
- [ ] It is clear that RDT remains enabled in the new configs.

## Tests

- [ ] Config tests are specified.
- [ ] Policy tests are specified.
- [ ] Dropout mask tests are specified.
- [ ] Full forward tests are specified.
- [ ] Train-only augmentation tests are specified.
