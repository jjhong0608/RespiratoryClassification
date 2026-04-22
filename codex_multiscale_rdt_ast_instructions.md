# Codex Implementation Instructions: Multi-Scale MIL + RDT AST Replacement

## Objective

Modify the `AST` branch of:

```text
https://github.com/jjhong0608/RespiratoryClassification.git
```

to replace the current Hugging Face `ASTModel` classifier with a new **Multi-Scale Patch + Shared Stem + Scale-Specific Adapter + Latent Query MIL + RDT Refinement** classifier.

This work is for a **new experimental branch**. You do **not** need to preserve the old AST model implementation or backward compatibility with the old Hugging Face `ASTModel` path. Preserve the **training/evaluation/CV pipeline shape** and the **config-driven experiment structure**, but replace the model path with the new architecture.

---

## Critical Intent

Do **not** treat this as an incremental extension of the old AST model.

The old AST encoder can be removed, rewritten, or made unreachable if that simplifies the implementation. What must remain usable:

- local fbank preprocessing pipeline
- dataset / dataloader structure
- training CLI
- evaluation CLI
- CV CLI
- config-driven experiment management
- optimizer / loss / metric / checkpoint / diagnostics flow
- model output contract used by the trainer and diagnostics

What does **not** need to remain compatible:

- Hugging Face `ASTModel`
- pretrained AST loading
- old single-patch AST architecture config
- old AST partial-unfreezing logic
- old AST tests that assume a Hugging Face encoder
- old `model.encoder.type == "ast"` constraint

---

## Base Code Facts to Check First

Before editing, inspect the current AST branch files. The current repo has these important characteristics:

```text
README.md
src/data/audio.py
src/data/dataset.py
src/data/loaders.py
src/models/model.py
src/models/classifier.py
src/training/ast_setup.py
src/training/trainer.py
src/cli/training.py
src/cli/evaluate.py
src/cli/cv.py
src/utils/config.py
src/utils/checkpoint.py
src/evaluation/diagnostics.py
tests/
configs/
```

The current pipeline uses local AST-style fbank extraction. The dataset ultimately provides:

```text
input_values: [B, max_length, num_mel_bins]
```

For the intended experiment, the default input is:

```text
input_values: [B, 1024, 128]
```

The new model should internally convert this to:

```text
x = input_values.unsqueeze(1)
x: [B, 1, 1024, 128]
```

The existing trainer expects the model output to be an `AstModelOutput`-like object with at least:

```python
logits: Tensor
pooled_embedding: Tensor
```

Keep this output contract unless you also update every consumer consistently.

---

## Target Architecture

Implement this exact high-level flow:

```text
Input fbank
[B, 1024, 128]
    ↓
unsqueeze channel
[B, 1, 1024, 128]
    ↓
4 multi-scale patch tokenizers
    - 16×16 stride 8×16
    -  8×32 stride 4×32
    -  4×64 stride 2×64
    -  2×128 stride 1×128
    ↓
learned positional embeddings + learned scale embeddings
    ↓
shared stem: 2 Transformer blocks
    - applied branch-wise
    - same weights for all branches
    ↓
scale-specific tiny Transformer adapter
    - 1 adapter block per scale
    ↓
concatenate all branch tokens
[B, 4081, D]
    ↓
latent query MIL pooling
    - K = 8 learned latent queries
[B, 8, D]
    ↓
RDT refinement
    - T = 3 recurrent steps
    - same RDT block reused at each step
    - recurrence only over the 8 latent summary tokens
    - the full token bank is fixed context memory
[B, 8, D]
    ↓
mean pool over K latent summaries
[B, D]
    ↓
classifier
binary: [B]
multi-class: [B, C]
```

---

## Fixed Patch Geometry and Token Counts

Use time-only 50% overlap and no frequency overlap.

Input shape convention:

```text
time = 1024
frequency = 128
```

Branch definitions:

| Branch | Patch size `(time, freq)` | Stride `(time, freq)` | Token grid | Token count |
|---:|---:|---:|---:|---:|
| 1 | `(16, 16)` | `(8, 16)` | `127 × 8` | `1016` |
| 2 | `(8, 32)` | `(4, 32)` | `255 × 4` | `1020` |
| 3 | `(4, 64)` | `(2, 64)` | `511 × 2` | `1022` |
| 4 | `(2, 128)` | `(1, 128)` | `1023 × 1` | `1023` |

Total token bank:

```text
1016 + 1020 + 1022 + 1023 = 4081
```

Token count formula:

```python
n_t = floor((max_length - patch_t) / stride_t) + 1
n_f = floor((num_mel_bins - patch_f) / stride_f) + 1
n_tokens = n_t * n_f
```

---

## Recommended Concrete Hyperparameters

Use these defaults unless the config overrides them:

```text
hidden_size D = 384
num_attention_heads = 6
shared_stem_depth = 2
adapter_depth = 1
latent_query_count K = 8
rdt_steps T = 3
mlp_ratio = 4.0
dropout = 0.1
attention_dropout = 0.1
layer_norm_eps = 1e-6
classifier.type = "linear" or "mlp"
classifier.pooling = "latent_mean" or remove pooling entirely and always use latent mean
```

Make sure `hidden_size % num_attention_heads == 0`.

---

## Required Model Components

You may organize these in `src/models/model.py` or a new file such as:

```text
src/models/multiscale_rdt_ast.py
```

Prefer a clean new file if it reduces clutter.

### 1. Output Dataclass

Keep or recreate:

```python
@dataclass(frozen=True)
class AstModelOutput:
    logits: Tensor
    pooled_embedding: Tensor
```

The trainer and diagnostics currently consume this shape. Keeping the name minimizes changes.

---

### 2. Patch Branch Config

Add config/dataclass support for branch patch geometry.

Example:

```python
@dataclass(frozen=True)
class PatchBranchConfig:
    patch_size: tuple[int, int]
    stride: tuple[int, int]
```

JSON will provide lists, so convert JSON lists to tuples in parsing.

Default branches:

```python
[
    PatchBranchConfig(patch_size=(16, 16), stride=(8, 16)),
    PatchBranchConfig(patch_size=(8, 32), stride=(4, 32)),
    PatchBranchConfig(patch_size=(4, 64), stride=(2, 64)),
    PatchBranchConfig(patch_size=(2, 128), stride=(1, 128)),
]
```

---

### 3. Multi-Scale Architecture Config

Replace the old AST architecture config with a new architecture config.

Example:

```python
@dataclass(frozen=True)
class MultiScaleRdtAstArchitectureConfig:
    hidden_size: int = 384
    num_attention_heads: int = 6
    mlp_ratio: float = 4.0
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6
    shared_stem_depth: int = 2
    adapter_depth: int = 1
    latent_query_count: int = 8
    rdt_steps: int = 3
    patch_branches: tuple[PatchBranchConfig, ...] = ...
```

You may keep the old class names only if that avoids refactoring, but their meaning should become the new architecture.

---

### 4. Patch Embedding

Implement each patch tokenizer with `nn.Conv2d`.

For branch `s`:

```python
nn.Conv2d(
    in_channels=1,
    out_channels=D,
    kernel_size=(patch_t, patch_f),
    stride=(stride_t, stride_f),
)
```

Forward shape:

```text
x: [B, 1, 1024, 128]
conv output: [B, D, n_t, n_f]
flatten: [B, n_t * n_f, D]
```

Validate input dimensions in the model forward:

```python
if input_values.ndim != 3:
    raise ValueError(...)
if input_values.shape[1] != max_length or input_values.shape[2] != num_mel_bins:
    raise ValueError(...)
```

---

### 5. Positional and Scale Embeddings

For each branch, add:

```text
position_embedding_s: [1, N_s, D]
scale_embedding_s: [1, 1, D]
```

Apply:

```python
z_s = patch_embed_s(x) + pos_embed_s + scale_embed_s
```

Implementation options:

- `nn.ParameterList` for branch positional embeddings
- `nn.Embedding(num_branches, D)` for scale embeddings
- or `nn.Parameter` of shape `[num_branches, D]`

Use learned positional embeddings. No need to interpolate for now; this experiment uses fixed `1024 × 128`.

---

### 6. Transformer Block

Implement a simple PreNorm transformer block with batch-first attention.

Recommended structure:

```python
class TransformerBlock(nn.Module):
    def __init__(...):
        self.norm1 = nn.LayerNorm(D, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=D,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(D, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(D, int(D * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(D * mlp_ratio), D),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x
```

You can avoid recomputing `self.norm1(x)` three times by storing it.

Optional but recommended:

- add a small LayerScale parameter for residual branches, especially in the RDT block
- keep implementation simple and testable

---

### 7. Shared Stem

The shared stem is 2 Transformer blocks.

Important:

- Apply it branch-wise.
- Reuse the same stem module/weights for all branches.
- Do **not** concatenate all 4081 tokens before the stem.

Shape:

```text
Z1: [B, 1016, D] -> shared stem -> S1: [B, 1016, D]
Z2: [B, 1020, D] -> shared stem -> S2: [B, 1020, D]
Z3: [B, 1022, D] -> shared stem -> S3: [B, 1022, D]
Z4: [B, 1023, D] -> shared stem -> S4: [B, 1023, D]
```

Reason:

```text
branch-wise attention cost:
1016² + 1020² + 1022² + 1023²

full concat self-attention cost:
4081²
```

Avoid full concat self-attention at this stage.

---

### 8. Scale-Specific Tiny Adapters

After the shared stem, apply a separate tiny adapter to each scale.

Use 1 Transformer block per branch.

Shape:

```text
S1: [B, 1016, D] -> Adapter_1 -> H1: [B, 1016, D]
S2: [B, 1020, D] -> Adapter_2 -> H2: [B, 1020, D]
S3: [B, 1022, D] -> Adapter_3 -> H3: [B, 1022, D]
S4: [B, 1023, D] -> Adapter_4 -> H4: [B, 1023, D]
```

Then concatenate:

```python
h_all = torch.cat([h1, h2, h3, h4], dim=1)
```

Expected:

```text
H_all: [B, 4081, D]
```

---

### 9. Latent Query MIL Pooling

Implement 8 learnable latent query tokens.

```python
self.latent_queries = nn.Parameter(torch.randn(1, 8, D) * init_std)
```

Batch-expand:

```python
q = self.latent_queries.expand(B, -1, -1)
```

Use cross-attention:

```text
Q: q       [B, 8, D]
K: H_all   [B, 4081, D]
V: H_all   [B, 4081, D]
U0:        [B, 8, D]
```

Recommended structure:

```python
class LatentQueryPooler(nn.Module):
    def forward(self, h_all):
        q = self.latent_queries.expand(h_all.shape[0], -1, -1)
        u, _ = self.cross_attn(
            self.query_norm(q),
            self.context_norm(h_all),
            self.context_norm(h_all),
            need_weights=False,
        )
        u = q + self.dropout(u)
        u = u + self.mlp(self.norm(u))
        return u
```

Keep `H_all` fixed after this point.

---

### 10. RDT Refinement Block

Only refine latent summaries.

Do **not** recurrently update all 4081 context tokens.

Implement:

```python
U_{t+1} = RDTBlock(U_t, H_all)
```

for `T = 3`, with the **same** `RDTBlock` reused each time.

Recommended RDT block:

```text
Input:
    U:     [B, 8, D]
    H_all: [B, 4081, D]

1. latent self-attention
    U <- U + SelfAttn(LN(U))

2. cross-attention to fixed evidence memory
    U <- U + CrossAttn(LN(U), LN(H_all), LN(H_all))

3. feed-forward
    U <- U + MLP(LN(U))

Output:
    U: [B, 8, D]
```

Forward loop:

```python
u = u0
for _ in range(self.rdt_steps):
    u = self.rdt_block(u, h_all)
```

With fixed `rdt_steps=3`, expected flow:

```text
U0: [B, 8, D]
U1: [B, 8, D]
U2: [B, 8, D]
U3: [B, 8, D]
```

---

### 11. Classifier Head

Pool over the 8 latent summaries:

```python
pooled_embedding = u.mean(dim=1)
```

Shape:

```text
pooled_embedding: [B, D]
```

Then classify.

Binary:

```text
logits: [B]
```

Multi-class:

```text
logits: [B, C]
```

You may reuse the existing `LinearClassifier` / `MlpClassifier` in `src/models/classifier.py` if convenient.

Keep the existing binary squeeze behavior:

```python
if logits.ndim == 2 and logits.shape[1] == 1:
    logits = logits.squeeze(1)
```

---

## Full Expected Tensor Flow

For the default experiment:

```text
Input:
    input_values: [B, 1024, 128]

Internal:
    x = input_values.unsqueeze(1)
    x: [B, 1, 1024, 128]

Patch tokenization:
    Z1 = PatchEmbed(16×16, stride 8×16)(x)    -> [B, 1016, D]
    Z2 = PatchEmbed( 8×32, stride 4×32)(x)    -> [B, 1020, D]
    Z3 = PatchEmbed( 4×64, stride 2×64)(x)    -> [B, 1022, D]
    Z4 = PatchEmbed( 2×128, stride 1×128)(x)  -> [B, 1023, D]

Add learned position + scale embeddings:
    Zs -> [B, Ns, D]

Shared stem:
    S1 -> [B, 1016, D]
    S2 -> [B, 1020, D]
    S3 -> [B, 1022, D]
    S4 -> [B, 1023, D]

Scale adapters:
    H1 -> [B, 1016, D]
    H2 -> [B, 1020, D]
    H3 -> [B, 1022, D]
    H4 -> [B, 1023, D]

Concat:
    H_all -> [B, 4081, D]

Latent query MIL:
    U0 -> [B, 8, D]

RDT:
    U1 -> [B, 8, D]
    U2 -> [B, 8, D]
    U3 -> [B, 8, D]

Pool:
    pooled_embedding -> [B, D]

Classifier:
    binary logits -> [B]
    multi-class logits -> [B, C]
```

---

## Suggested Module Layout

Preferred:

```text
src/models/model.py
    - AstModelOutput
    - model config dataclasses if you keep them here
    - import/re-export MultiScaleRdtAstModel if needed

src/models/multiscale_rdt_ast.py
    - PatchBranchConfig
    - MultiScaleRdtAstArchitectureConfig
    - MultiScaleRdtAstEncoderConfig
    - MultiScaleRdtAstModelConfig
    - TransformerBlock
    - CrossAttentionBlock
    - MultiScalePatchEncoder
    - LatentQueryPooler
    - RdtRefinementBlock
    - MultiScaleRdtAstModel
```

Alternative:

```text
Put everything in src/models/model.py
```

That is acceptable if it is cleaner for this branch.

---

## Config Changes

The old `model.encoder.type == "ast"` schema is no longer required.

Use a new model encoder type, or simply make the only supported type:

```json
"type": "multiscale_rdt_ast"
```

Recommended JSON shape:

```json
{
  "model": {
    "encoder": {
      "type": "multiscale_rdt_ast",
      "adaptation": {
        "mode": "full",
        "num_layers": 0
      },
      "architecture": {
        "hidden_size": 384,
        "num_attention_heads": 6,
        "mlp_ratio": 4.0,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "layer_norm_eps": 1e-6,
        "shared_stem_depth": 2,
        "adapter_depth": 1,
        "latent_query_count": 8,
        "rdt_steps": 3,
        "patch_branches": [
          {"patch_size": [16, 16], "stride": [8, 16]},
          {"patch_size": [8, 32], "stride": [4, 32]},
          {"patch_size": [4, 64], "stride": [2, 64]},
          {"patch_size": [2, 128], "stride": [1, 128]}
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
}
```

You may remove `pretrained_name_or_path` and `cache_dir`. If you leave them in the dataclass for compatibility with the config loader structure, they should be ignored or must default to `null`.

### Validation Requirements

Update `src/utils/config.py` so it validates the new schema:

- `model.encoder.type` must be `"multiscale_rdt_ast"` or whatever new single supported type you choose.
- `hidden_size > 0`
- `num_attention_heads > 0`
- `hidden_size % num_attention_heads == 0`
- `mlp_ratio > 0`
- dropouts in `[0, 1)`
- `shared_stem_depth == 2` by default but allow positive int
- `adapter_depth == 1` by default but allow positive int
- `latent_query_count == 8` by default but allow positive int
- `rdt_steps == 3` by default but allow positive int
- each patch branch has positive patch and stride sizes
- each patch branch fits within `data.preprocessing.ast_fbank.max_length` and `num_mel_bins`
- token count for each branch is positive
- classifier pooling is `"latent_mean"` if the field is kept

Remove or disable the old pretrained AST input-dimension validation. It no longer applies.

---

## Training Setup Changes

Modify `src/training/ast_setup.py`.

You can keep the existing function names to reduce CLI changes:

```python
build_ast_model(...)
apply_encoder_adaptation(...)
build_grouped_optimizer(...)
inspect_pretrained_encoder(...)
```

But their behavior should now target the new model.

### `build_ast_model`

Return the new model:

```python
return MultiScaleRdtAstModel(model_cfg)
```

Use `num_mel_bins`, `max_length`, and `num_classes` from the dataset/config.

### `inspect_pretrained_encoder`

No pretrained encoder is used.

Either:

```python
def inspect_pretrained_encoder(...):
    return None
```

or remove its usage from the training and CV CLIs.

### `apply_encoder_adaptation`

The new model is trained from scratch.

Recommended simple behavior:

- `mode == "full"`: all model parameters trainable
- `mode == "frozen"`: freeze `model.encoder`, keep `latent_pooler`, `rdt`, and `classifier` trainable
- `mode == "partial"`: either reject with `ValueError` or treat as `"full"` with a clear log/summary

For the provided config, use:

```json
"adaptation": {
  "mode": "full",
  "num_layers": 0
}
```

### `build_grouped_optimizer`

Keep two groups:

```text
encoder group:
    model.encoder parameters
    learning rate = train.optimizer.encoder_lr

head group:
    latent query pooler
    RDT block
    classifier
    learning rate = train.optimizer.head_lr
```

To make this work, structure the model with:

```python
self.encoder = MultiScalePatchStemAdapterEncoder(...)
self.latent_pooler = LatentQueryPooler(...)
self.rdt_block = RdtRefinementBlock(...)
self.classifier = ...
```

Then existing parameter grouping by `model.encoder.parameters()` remains meaningful.

---

## CLI Changes

Update these files if needed:

```text
src/cli/training.py
src/cli/cv.py
src/cli/evaluate.py
```

### Training and CV

They can still call:

```python
build_ast_model(...)
apply_encoder_adaptation(...)
build_grouped_optimizer(...)
```

but remove old pretrained AST-specific log messaging or make it conditional on `inspect_pretrained_encoder(...) is not None`.

### Evaluation

`src/cli/evaluate.py` currently reconstructs the model from checkpoint `model_cfg`.

Update it to instantiate the new `MultiScaleRdtAstModel`, not the old `RespiratoryAstModel`.

Also update `src/utils/checkpoint.py`:

```python
parse_model_cfg(...)
```

so it can parse the new model config dataclasses from the checkpoint payload.

The checkpoint should still store:

```text
model_cfg
model_state_dict
dims
```

where dims should remain:

```json
{
  "num_mel_bins": 128,
  "max_length": 1024
}
```

or equivalent.

---

## Data Pipeline

Do not change the data pipeline unless necessary.

Current data behavior should remain:

```text
.wav file
→ waveform loading
→ preprocessing
→ AstLikeFbank
→ ClipSample.input_values
→ DataLoader batch
→ input_values: [B, 1024, 128]
```

The model should handle adding the channel dimension internally.

---

## Config Files to Add or Replace

Add a new default training config, for example:

```text
configs/training_multiscale_rdt.json
```

Add corresponding optional configs:

```text
configs/eval_multiscale_rdt.json
configs/cv_multiscale_rdt.json
```

The training config should use:

```json
"data": {
  "preprocessing": {
    "ast_fbank": {
      "num_mel_bins": 128,
      "max_length": 1024,
      "do_normalize": true,
      "mean": -4.2677393,
      "std": 4.5689974
    }
  }
}
```

And model defaults:

```json
"hidden_size": 384,
"num_attention_heads": 6,
"shared_stem_depth": 2,
"adapter_depth": 1,
"latent_query_count": 8,
"rdt_steps": 3
```

---

## Tests

Old tests that explicitly assume Hugging Face `ASTModel` do not need to be preserved. Update or replace them.

### Required Tests

Add or update tests to verify:

#### 1. Token count computation

For input `max_length=1024`, `num_mel_bins=128`:

```text
16×16 stride 8×16   -> 1016 tokens
8×32 stride 4×32    -> 1020 tokens
4×64 stride 2×64    -> 1022 tokens
2×128 stride 1×128  -> 1023 tokens
total               -> 4081 tokens
```

#### 2. Model forward: binary classification

Use a lightweight config for speed if needed.

Example:

```python
model = MultiScaleRdtAstModel(...)
x = torch.randn(2, 1024, 128)
out = model(x)
assert isinstance(out, AstModelOutput)
assert out.logits.shape == (2,)
assert out.pooled_embedding.shape == (2, hidden_size)
```

If the full `1024×128` forward is too slow for unit tests, still include a static token-count test for the full configuration and use a smaller synthetic input with smaller patch branches for a fast forward test.

#### 3. Model forward: multi-class classification

```python
assert out.logits.shape == (B, num_classes)
```

#### 4. Latent query and RDT shapes

Either expose debug attributes in a controlled way or unit test submodules directly:

```text
LatentQueryPooler(H_all [B, N, D]) -> [B, 8, D]
RdtRefinementBlock(U [B, 8, D], H_all [B, N, D]) -> [B, 8, D]
```

#### 5. Config parsing

Verify that `JsonConfigLoader.load_training(...)` accepts the new config schema and rejects invalid values.

Examples to reject:

- `hidden_size` not divisible by `num_attention_heads`
- negative patch size
- patch larger than input dims
- unsupported encoder type
- unsupported classifier pooling

#### 6. Optimizer grouping

Verify:

```text
param group 0 name = "encoder"
param group 1 name = "head"
encoder group has model.encoder params
head group has latent_pooler, rdt_block, classifier params
```

#### 7. Checkpoint config parsing

Verify that `parse_model_cfg(...)` reconstructs the new model config from the checkpoint dict.

---

## Acceptance Criteria

The implementation is acceptable when all of the following hold:

1. `python -m src.cli.training --config configs/training_multiscale_rdt.json` can build the model and start training.
2. `python -m src.cli.cv --config configs/cv_multiscale_rdt.json` can build fold models.
3. `python -m src.cli.evaluate --config configs/eval_multiscale_rdt.json` can reconstruct the model from checkpoint config.
4. The model accepts `input_values: [B, 1024, 128]`.
5. Internally, the default patch branches produce token counts `[1016, 1020, 1022, 1023]`.
6. The concatenated token bank is `[B, 4081, D]`.
7. Latent query pooling produces `[B, 8, D]`.
8. RDT refinement performs exactly 3 recurrent applications of the same block.
9. The returned output is an `AstModelOutput` with:
   - binary logits `[B]`, or
   - multi-class logits `[B, C]`
   - pooled embedding `[B, D]`
10. Existing data preprocessing remains usable.
11. Old Hugging Face AST compatibility is not required.
12. Tests pass.

---

## Commands to Run

Run these before finishing:

```bash
pytest
ruff check src tests
```

If practical, also run:

```bash
mypy src tests
```

If a command fails because of environment limitations, document the exact failure and what remains unverified.

---

## Avoid These Mistakes

- Do not apply self-attention over all 4081 tokens in the shared stem.
- Do not recurrently update all 4081 tokens in the RDT.
- Do not try to load pretrained Hugging Face AST weights into the new rectangular multi-scale patch model.
- Do not preserve old AST partial unfreezing assumptions such as `encoder.encoder.layer`.
- Do not keep validators that reject non-`"ast"` encoder types.
- Do not make `2×128` optional in the default model; include it by default.
- Do not assume input is `[B, 1, 1024, 128]`; the dataloader gives `[B, 1024, 128]`.
- Do not return raw tensors from the model; return `AstModelOutput`.
- Do not silently change binary logits to `[B, 1]`; keep `[B]` unless every downstream consumer is updated.
- Do not let the full token bank shape leak into the classifier; classifier sees only pooled latent summaries.

---

## Suggested Minimal Pseudocode

```python
class MultiScaleRdtAstModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = MultiScalePatchStemAdapterEncoder(cfg.encoder)
        self.latent_pooler = LatentQueryPooler(cfg.encoder.architecture)
        self.rdt_block = RdtRefinementBlock(cfg.encoder.architecture)

        out_dim = 1 if cfg.num_classes == 2 else cfg.num_classes
        self.classifier = build_classifier(
            in_dim=cfg.encoder.architecture.hidden_size,
            out_dim=out_dim,
            classifier_cfg=cfg.classifier,
        )

    def forward(self, input_values: Tensor) -> AstModelOutput:
        if input_values.ndim != 3:
            raise ValueError(...)
        if input_values.shape[1:] != (self.cfg.encoder.feature_dims.max_length,
                                      self.cfg.encoder.feature_dims.num_mel_bins):
            raise ValueError(...)

        x = input_values.unsqueeze(1)          # [B, 1, 1024, 128]
        h_all = self.encoder(x)                # [B, 4081, D]
        u = self.latent_pooler(h_all)          # [B, 8, D]

        for _ in range(self.cfg.encoder.architecture.rdt_steps):
            u = self.rdt_block(u, h_all)       # [B, 8, D]

        pooled = u.mean(dim=1)                 # [B, D]
        logits = self.classifier(pooled)

        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)

        return AstModelOutput(logits=logits, pooled_embedding=pooled)
```

---

## Final Note

The goal is not to keep the old AST model alive. The goal is to reuse the existing respiratory classification experiment pipeline while replacing the model with the new multi-scale MIL + RDT architecture.

Prioritize a clean, testable implementation over backward compatibility.
