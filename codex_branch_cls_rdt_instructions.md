# Codex Implementation Instructions: Replace Latent Query MIL with Branch CLS Summary Tokens

## Target Repository and Branch

Repository:

```text
https://github.com/jjhong0608/RespiratoryClassification.git
```

Base branch:

```text
MIL-RDT-AST
```

This task modifies the existing **Multi-Scale RDT AST** implementation in the `MIL-RDT-AST` branch. Do **not** start from the older `AST` branch. Do **not** reintroduce the Hugging Face `ASTModel` path.

---

## Objective

Replace the current **latent query MIL pooling** mechanism with **branch-local CLS / summary tokens**.

The current model flow is approximately:

```text
input_values: [B, 1024, 128]
    ↓
unsqueeze channel
[B, 1, 1024, 128]
    ↓
4-scale patch tokenizers
    - 16×16 stride 8×16   → 1016 patch tokens
    -  8×32 stride 4×32   → 1020 patch tokens
    -  4×64 stride 2×64   → 1022 patch tokens
    -  2×128 stride 1×128 → 1023 patch tokens
    ↓
position embeddings + scale embeddings
    ↓
shared branch-wise transformer stem
    ↓
scale-specific transformer adapters
    ↓
concat patch token bank
H_all: [B, 4081, D]
    ↓
LatentQueryPooler, K=8
U0: [B, 8, D]
    ↓
RDT refinement, T=3
U3: [B, 8, D]
    ↓
latent mean pooling
[B, D]
    ↓
classifier
```

Change it to:

```text
input_values: [B, 1024, 128]
    ↓
unsqueeze channel
[B, 1, 1024, 128]
    ↓
4-scale patch tokenizers
    - 16×16 stride 8×16   → 1016 patch tokens
    -  8×32 stride 4×32   → 1020 patch tokens
    -  4×64 stride 2×64   → 1022 patch tokens
    -  2×128 stride 1×128 → 1023 patch tokens
    ↓
add 2 learnable branch summary/CLS tokens per scale
    ↓
position embeddings + scale embeddings
    ↓
shared branch-wise transformer stem
    ↓
scale-specific transformer adapters
    ↓
split each branch output into:
    - summary tokens: first 2 tokens
    - patch tokens: remaining N_s tokens
    ↓
concat branch summary tokens
U0: [B, 8, D]
    ↓
concat patch tokens
H_all: [B, 4081, D]
    ↓
RDT refinement, T=3
    U_{t+1} = RDTBlock(U_t, H_all)
U3: [B, 8, D]
    ↓
mean pool over the 8 summary tokens
[B, D]
    ↓
classifier
```

The key idea:

```text
Remove learned global latent queries.
Use 2 branch-conditioned summary/CLS tokens per scale instead.
Keep the full patch token bank H_all as fixed evidence memory for RDT cross-attention.
```

---

## Important Architectural Decision

Use this version:

```text
Branch CLS 8 tokens = 2 summary tokens per branch × 4 branches
```

Do **not** use:

```text
1 CLS token per branch
```

Do **not** discard the patch token bank.

The RDT state must be the 8 branch summary tokens:

```text
U0: [B, 8, D]
```

The RDT context memory must remain the full patch token bank:

```text
H_all: [B, 4081, D]
```

This preserves the current `[B, 8, D]` RDT interface while replacing the semantic origin of those 8 tokens.

---

## Current Code Anchors

Inspect the current branch first. The primary file is expected to be:

```text
src/models/multiscale_rdt_ast.py
```

The current implementation contains these relevant components:

```text
AstFeatureDims
PatchBranchConfig
default_patch_branches
EncoderAdaptationConfig
MultiScaleRdtArchitectureConfig
MultiScaleRdtEncoderConfig
ClassifierConfig
MultiScaleRdtAstModelConfig
AstModelOutput
compute_token_grid
compute_token_count
TransformerBlock
FeedForwardBlock
PatchTokenizer
MultiScalePatchStemAdapterEncoder
LatentQueryPooler
RdtRefinementBlock
MultiScaleRdtAstModel
```

At the time of this instruction, the current architecture config contains:

```python
latent_query_count: int = 8
rdt_steps: int = 3
```

And the current model forward roughly does:

```python
x = input_values.unsqueeze(1)
h_all = self.encoder(x)                       # [B, 4081, D]
latent_summaries = self.latent_pooler(h_all)  # [B, 8, D]

for _ in range(self.cfg.encoder.architecture.rdt_steps):
    latent_summaries = self.rdt_block(latent_summaries, h_all)

pooled_embedding = latent_summaries.mean(dim=1)
logits = self.classifier(pooled_embedding)
```

This must change so the encoder returns both:

```python
summary_tokens: Tensor  # [B, 8, D]
patch_tokens: Tensor    # [B, 4081, D]
```

Then the model forward should do:

```python
x = input_values.unsqueeze(1)
summary_tokens, patch_tokens = self.encoder(x)

latent_summaries = summary_tokens

for _ in range(self.cfg.encoder.architecture.rdt_steps):
    latent_summaries = self.rdt_block(latent_summaries, patch_tokens)

pooled_embedding = latent_summaries.mean(dim=1)
logits = self.classifier(pooled_embedding)
```

---

## Required Tensor Shapes

Default input:

```text
input_values: [B, 1024, 128]
x = input_values.unsqueeze(1): [B, 1, 1024, 128]
```

Default branches:

| Branch | Patch size | Stride | Patch token grid | Patch token count |
|---:|---:|---:|---:|---:|
| 1 | `(16, 16)` | `(8, 16)` | `127 × 8` | `1016` |
| 2 | `(8, 32)` | `(4, 32)` | `255 × 4` | `1020` |
| 3 | `(4, 64)` | `(2, 64)` | `511 × 2` | `1022` |
| 4 | `(2, 128)` | `(1, 128)` | `1023 × 1` | `1023` |

Patch token bank:

```text
1016 + 1020 + 1022 + 1023 = 4081
```

With 2 summary tokens per branch:

| Branch | Patch tokens | Summary tokens | Branch sequence length |
|---:|---:|---:|---:|
| 1 | `1016` | `2` | `1018` |
| 2 | `1020` | `2` | `1022` |
| 3 | `1022` | `2` | `1024` |
| 4 | `1023` | `2` | `1025` |

After shared stem and adapters, split each branch output:

```python
summary_s = branch_output[:, :summary_tokens_per_scale, :]      # [B, 2, D]
patch_s = branch_output[:, summary_tokens_per_scale:, :]        # [B, N_s, D]
```

Concatenate:

```text
U0 = concat(summary_1, summary_2, summary_3, summary_4, dim=1)
U0: [B, 8, D]

H_all = concat(patch_1, patch_2, patch_3, patch_4, dim=1)
H_all: [B, 4081, D]
```

RDT:

```text
U1 = RdtRefinementBlock(U0, H_all)  -> [B, 8, D]
U2 = RdtRefinementBlock(U1, H_all)  -> [B, 8, D]
U3 = RdtRefinementBlock(U2, H_all)  -> [B, 8, D]
```

Classifier:

```text
pooled_embedding = U3.mean(dim=1) -> [B, D]

binary logits:      [B]
multi-class logits: [B, C]
```

---

## Config Changes

### Replace `latent_query_count`

The current config name:

```json
"latent_query_count": 8
```

is no longer semantically correct.

Replace it with:

```json
"summary_tokens_per_scale": 2
```

The total number of RDT summary tokens is computed as:

```text
num_summary_tokens = len(patch_branches) * summary_tokens_per_scale
                   = 4 * 2
                   = 8
```

Keep:

```json
"rdt_steps": 3
```

Keep:

```json
"classifier": {
  "pooling": "latent_mean"
}
```

Even though the tokens are no longer produced by a latent query pooler, `"latent_mean"` can still mean:

```text
mean over the final RDT latent/summary tokens
```

Alternatively, rename it to `"summary_mean"` only if you update every config validator, README, checkpoint parser, and tests consistently. Minimal-change recommendation: keep `"latent_mean"`.

### Updated architecture config example

```json
"architecture": {
  "hidden_size": 384,
  "num_attention_heads": 6,
  "mlp_ratio": 4.0,
  "hidden_dropout_prob": 0.1,
  "attention_probs_dropout_prob": 0.1,
  "layer_norm_eps": 1e-6,
  "shared_stem_depth": 2,
  "adapter_depth": 1,
  "summary_tokens_per_scale": 2,
  "rdt_steps": 3,
  "patch_branches": [
    {"patch_size": [16, 16], "stride": [8, 16]},
    {"patch_size": [8, 32], "stride": [4, 32]},
    {"patch_size": [4, 64], "stride": [2, 64]},
    {"patch_size": [2, 128], "stride": [1, 128]}
  ]
}
```

### Validation requirements

Update config validation so that:

```text
summary_tokens_per_scale is a positive integer.
summary_tokens_per_scale defaults to 2.
latent_query_count is no longer required.
If latent_query_count appears in a config, either reject it or ignore it with a clear deprecation path.
```

Recommended strict behavior:

```text
Reject latent_query_count in new configs.
```

Reason: it avoids ambiguity between the old latent-query architecture and the new branch-summary architecture.

Also continue validating:

```text
hidden_size > 0
num_attention_heads > 0
hidden_size % num_attention_heads == 0
mlp_ratio > 0
dropout values in [0, 1)
shared_stem_depth > 0
adapter_depth > 0
rdt_steps > 0
patch branches fit within max_length and num_mel_bins
computed patch token count > 0
```

---

## Model Implementation Changes

### 1. Remove or bypass `LatentQueryPooler`

The old `LatentQueryPooler` is no longer needed.

You may:

1. delete the class,
2. leave it unused,
3. or keep it for reference but remove it from `MultiScaleRdtAstModel`.

Preferred:

```text
Remove it from the active model path.
```

`MultiScaleRdtAstModel` should no longer define:

```python
self.latent_pooler = LatentQueryPooler(...)
```

And forward should no longer call:

```python
latent_summaries = self.latent_pooler(h_all)
```

---

### 2. Update `MultiScaleRdtArchitectureConfig`

Change:

```python
latent_query_count: int = 8
```

to:

```python
summary_tokens_per_scale: int = 2
```

Keep:

```python
rdt_steps: int = 3
```

---

### 3. Update the encoder return type

Current encoder:

```python
class MultiScalePatchStemAdapterEncoder(nn.Module):
    ...
    def forward(self, x: Tensor) -> Tensor:
        ...
        h_all = torch.cat(branch_outputs, dim=1)
        return self.output_norm(h_all)
```

Change it to return two tensors.

Recommended:

```python
@dataclass(frozen=True)
class MultiScaleEncoderOutput:
    summary_tokens: Tensor  # [B, num_branches * summary_tokens_per_scale, D]
    patch_tokens: Tensor    # [B, total_patch_token_count, D]
```

Then:

```python
def forward(self, x: Tensor) -> MultiScaleEncoderOutput:
    ...
    return MultiScaleEncoderOutput(
        summary_tokens=summary_tokens,
        patch_tokens=patch_tokens,
    )
```

If you prefer tuple return:

```python
return summary_tokens, patch_tokens
```

That is acceptable, but a dataclass is clearer and easier to test.

---

### 4. Add branch summary tokens

Inside `MultiScalePatchStemAdapterEncoder.__init__`, create learnable summary tokens.

Recommended shape:

```python
self.summary_tokens = nn.Parameter(
    torch.empty(
        len(architecture.patch_branches),
        architecture.summary_tokens_per_scale,
        hidden_size,
    )
)
```

Initialize:

```python
nn.init.normal_(self.summary_tokens, std=PATCH_INIT_STD)
```

These are branch-specific learnable summary tokens.

During forward, for branch `branch_index`:

```python
patch_tokens = tokenizer(x)  # [B, N_s, D]
summary_tokens = self.summary_tokens[branch_index].unsqueeze(0).expand(
    patch_tokens.shape[0], -1, -1
)  # [B, 2, D]
branch_tokens = torch.cat([summary_tokens, patch_tokens], dim=1)
```

Then add embeddings.

---

### 5. Positional embedding details

The old branch positional embeddings were shaped:

```python
[1, token_count, D]
```

Now each branch sequence length includes summary tokens:

```text
branch_sequence_length_s = summary_tokens_per_scale + patch_token_count_s
```

Use either of these options.

#### Preferred option: separate summary and patch positional embeddings

Keep patch positional embeddings only for patch tokens:

```python
self.position_embeddings = nn.ParameterList(
    nn.Parameter(torch.empty(1, token_count, hidden_size))
    for token_count in self.branch_token_counts
)
```

Add patch positions only to patch tokens:

```python
patch_tokens = patch_tokens + self.position_embeddings[branch_index]
```

Do **not** add patch position embeddings to summary tokens.

Then concatenate:

```python
summary_tokens = summary_tokens + self.scale_embeddings[branch_index].view(1, 1, -1)
patch_tokens = patch_tokens + self.scale_embeddings[branch_index].view(1, 1, -1)
branch_tokens = torch.cat([summary_tokens, patch_tokens], dim=1)
```

This is the cleanest design.

#### Alternative option: full sequence positional embeddings

Make positional embeddings include summary slots:

```python
self.position_embeddings = nn.ParameterList(
    nn.Parameter(torch.empty(1, summary_tokens_per_scale + token_count, hidden_size))
    for token_count in self.branch_token_counts
)
```

Then add to the full sequence.

This is acceptable but less semantically clean.

Preferred: **separate summary tokens from patch positional embeddings**.

---

### 6. Scale embedding details

Scale embeddings should be added to both summary and patch tokens.

For branch `s`:

```python
scale = self.scale_embeddings[s].view(1, 1, -1)
summary_tokens = summary_tokens + scale
patch_tokens = patch_tokens + scale
```

Then concatenate.

---

### 7. Shared stem and adapter processing

After concatenating summary + patch tokens for a branch:

```python
branch_tokens: [B, 2 + N_s, D]
```

Apply the existing shared stem branch-wise:

```python
for block in self.shared_stem:
    branch_tokens = block(branch_tokens)
```

Then apply the scale-specific adapter:

```python
for block in self.adapters[branch_index]:
    branch_tokens = block(branch_tokens)
```

Then split:

```python
summary_s = branch_tokens[:, :summary_tokens_per_scale, :]
patch_s = branch_tokens[:, summary_tokens_per_scale:, :]
```

Collect:

```python
branch_summary_outputs.append(summary_s)
branch_patch_outputs.append(patch_s)
```

After all branches:

```python
summary_tokens = torch.cat(branch_summary_outputs, dim=1)  # [B, 8, D]
patch_tokens = torch.cat(branch_patch_outputs, dim=1)      # [B, 4081, D]
```

Apply output norm carefully.

Recommended:

```python
summary_tokens = self.output_norm(summary_tokens)
patch_tokens = self.output_norm(patch_tokens)
```

Using the same `LayerNorm` instance is acceptable because it normalizes only the hidden dimension.

---

### 8. Update `MultiScaleRdtAstModel`

Current active path:

```python
self.encoder = MultiScalePatchStemAdapterEncoder(cfg.encoder)
self.latent_pooler = LatentQueryPooler(architecture)
self.rdt_block = RdtRefinementBlock(architecture)
```

Change to:

```python
self.encoder = MultiScalePatchStemAdapterEncoder(cfg.encoder)
self.rdt_block = RdtRefinementBlock(architecture)
```

Remove:

```python
self.latent_pooler
```

Forward:

```python
encoder_output = self.encoder(x)

if isinstance(encoder_output, MultiScaleEncoderOutput):
    latent_summaries = encoder_output.summary_tokens
    evidence_memory = encoder_output.patch_tokens
else:
    latent_summaries, evidence_memory = encoder_output

for _ in range(self.cfg.encoder.architecture.rdt_steps):
    latent_summaries = self.rdt_block(latent_summaries, evidence_memory)

pooled_embedding = latent_summaries.mean(dim=1)
logits = self.classifier(pooled_embedding)
```

Expected:

```text
latent_summaries before RDT: [B, 8, D]
evidence_memory:             [B, 4081, D]
latent_summaries after RDT:  [B, 8, D]
pooled_embedding:            [B, D]
```

---

## Optimizer Grouping

The current README says the optimizer has two groups:

```text
encoder parameters -> train.optimizer.encoder_lr
latent pooler + RDT + classifier parameters -> train.optimizer.head_lr
```

After this change, there is no latent pooler.

Update optimizer grouping and README language to:

```text
encoder parameters -> train.optimizer.encoder_lr
RDT + classifier parameters -> train.optimizer.head_lr
```

Branch summary tokens live inside the encoder and should be part of the encoder parameter group.

This is intentional because they participate in branch representation construction through the shared stem and adapters.

---

## Adaptation Rules

Current adaptation behavior can remain:

```text
mode: "full" trains encoder and head together.
mode: "frozen" freezes encoder and trains only RDT + classifier.
mode: "partial" is rejected.
num_layers must be 0.
```

Update any text that says `"latent pooling"` under frozen mode.

Old text:

```text
freezes the encoder and trains only latent pooling, RDT, and classifier layers
```

New text:

```text
freezes the encoder and trains only RDT and classifier layers
```

If frozen mode is used, branch summary tokens are inside the frozen encoder and will be frozen too. That is acceptable. For this experiment, default config should remain:

```json
"adaptation": {
  "mode": "full",
  "num_layers": 0
}
```

---

## README Updates

Update the Architecture section.

Replace:

```text
concat token bank
[B, 4081, D]
  -> latent query pooling
[B, 8, D]
  -> 3 recurrent RDT refinement steps
[B, 8, D]
```

with:

```text
add 2 branch summary/CLS tokens per scale
  -> shared branch-wise transformer stem
  -> scale-specific transformer adapters
  -> split:
       summary tokens [B, 8, D]
       patch token bank [B, 4081, D]
  -> 3 recurrent RDT refinement steps
       state:   [B, 8, D]
       context: [B, 4081, D]
  -> summary mean pooling
[B, D]
  -> classifier
```

Update default encoder hyperparameters.

Replace:

```text
latent_query_count = 8
```

with:

```text
summary_tokens_per_scale = 2
```

Update config schema examples accordingly.

Update optimizer layout language:

```text
latent pooler + RDT + classifier
```

to:

```text
RDT + classifier
```

---

## Config Files to Update

Update all canonical configs that contain:

```json
"latent_query_count": 8
```

Replace with:

```json
"summary_tokens_per_scale": 2
```

Likely files:

```text
configs/training_multiscale_rdt.json
configs/cv_multiscale_rdt.json
configs/eval_multiscale_rdt.json
configs/training_multiclass.json
```

Inspect the branch and update any additional configs containing `latent_query_count`.

---

## Checkpoint and Config Parsing

Update checkpoint config parsing if it reconstructs `MultiScaleRdtArchitectureConfig`.

Anywhere that currently expects:

```python
latent_query_count
```

must expect:

```python
summary_tokens_per_scale
```

Recommended migration behavior:

- If a checkpoint/config contains `latent_query_count`, reject it as incompatible with this new branch.
- Do not silently reinterpret `latent_query_count` as `summary_tokens_per_scale`; the semantics are different.

Error message example:

```text
"latent_query_count is deprecated in the branch-summary-token architecture. Use summary_tokens_per_scale instead."
```

---

## Tests to Update or Add

Update the test suite to reflect the new architecture.

### 1. Config tests

Add tests that valid configs include:

```json
"summary_tokens_per_scale": 2
```

and do not require:

```json
"latent_query_count"
```

Add invalid config tests:

```text
summary_tokens_per_scale <= 0 should be rejected.
latent_query_count present in architecture should be rejected.
```

### 2. Token count tests

Keep existing patch token count expectations:

```text
16×16 stride 8×16   -> 1016 patch tokens
8×32 stride 4×32    -> 1020 patch tokens
4×64 stride 2×64    -> 1022 patch tokens
2×128 stride 1×128  -> 1023 patch tokens
total patch tokens  -> 4081
```

Add summary-token expectations:

```text
summary_tokens_per_scale = 2
num_branches = 4
total summary tokens = 8
```

Add branch sequence length expectations:

```text
Branch 1 sequence length = 1016 + 2 = 1018
Branch 2 sequence length = 1020 + 2 = 1022
Branch 3 sequence length = 1022 + 2 = 1024
Branch 4 sequence length = 1023 + 2 = 1025
```

### 3. Encoder forward shape test

Test `MultiScalePatchStemAdapterEncoder` directly.

Expected output:

```python
encoder_output.summary_tokens.shape == (B, 8, D)
encoder_output.patch_tokens.shape == (B, 4081, D)
```

If tuple return is used:

```python
summary_tokens, patch_tokens = encoder(x)
assert summary_tokens.shape == (B, 8, D)
assert patch_tokens.shape == (B, 4081, D)
```

### 4. RDT shape test

Keep or add:

```python
u = torch.randn(B, 8, D)
h_all = torch.randn(B, 4081, D)
out = rdt_block(u, h_all)
assert out.shape == (B, 8, D)
```

### 5. Full model forward test

Binary:

```python
x = torch.randn(B, 1024, 128)
out = model(x)
assert isinstance(out, AstModelOutput)
assert out.logits.shape == (B,)
assert out.pooled_embedding.shape == (B, D)
```

Multi-class:

```python
assert out.logits.shape == (B, C)
```

### 6. No LatentQueryPooler active path

Add a test or assertion that the model does not expose/use the old latent pooler.

Example:

```python
assert not hasattr(model, "latent_pooler")
```

or, if the class remains in the module for backward reference:

```python
assert "latent_pooler" not in dict(model.named_modules())
```

### 7. Optimizer grouping test

Update optimizer grouping tests.

Expected:

```text
encoder group includes model.encoder parameters.
head group includes model.rdt_block and model.classifier parameters.
head group does not include latent_pooler because it no longer exists.
```

---

## Acceptance Criteria

Implementation is acceptable when all of these are true:

1. The base branch is `MIL-RDT-AST`.
2. The model no longer uses `LatentQueryPooler` in the active forward path.
3. The config no longer requires `latent_query_count`.
4. The config uses `summary_tokens_per_scale = 2`.
5. Each branch gets 2 learnable summary/CLS tokens.
6. These summary tokens are prepended to the branch token sequence before the shared stem.
7. Patch position embeddings are applied to patch tokens.
8. Scale embeddings are applied to both summary and patch tokens.
9. The shared stem is still branch-wise and weight-shared.
10. Scale-specific adapters are still branch-specific.
11. After adapters, branch outputs are split into summary tokens and patch tokens.
12. Concatenated summary tokens have shape `[B, 8, D]`.
13. Concatenated patch tokens have shape `[B, 4081, D]`.
14. RDT runs exactly `rdt_steps = 3` times by default.
15. The same RDT block is reused for each recurrent step.
16. RDT state is `[B, 8, D]`.
17. RDT context memory is `[B, 4081, D]`.
18. Classifier receives `pooled_embedding = U3.mean(dim=1)` with shape `[B, D]`.
19. Binary logits remain `[B]`.
20. Multi-class logits remain `[B, C]`.
21. Training, evaluation, and CV CLIs still build the model from config.
22. Checkpoint config parsing works with `summary_tokens_per_scale`.
23. README architecture and config examples match the new design.
24. Tests pass.

---

## Commands to Run

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

If any command cannot be run due to environment limitations, document:

```text
- the exact command
- the exact error
- what remains unverified
```

---

## Avoid These Mistakes

- Do not keep using `LatentQueryPooler` in the model forward.
- Do not produce `U0` by cross-attending learned latent queries to `H_all`.
- Do not reduce to only 1 CLS token per branch.
- Do not discard the patch token bank.
- Do not feed only `[B, 8, D]` into RDT without cross-attending to `[B, 4081, D]`.
- Do not concatenate summary tokens into `H_all`; `H_all` should contain patch tokens only.
- Do not count summary tokens as patch tokens.
- Do not change the patch geometry.
- Do not change the input format `[B, 1024, 128]`.
- Do not reintroduce the Hugging Face `ASTModel`.
- Do not preserve `latent_query_count` in active configs.
- Do not silently reinterpret `latent_query_count = 8` as `summary_tokens_per_scale = 8`.
- Do not put summary tokens in the head parameter group if they are implemented inside the encoder.
- Do not apply patch positional embeddings to summary tokens unless you intentionally implement a separate summary positional embedding scheme.
- Do not break binary logits shape `[B]`.

---

## Suggested Pseudocode Patch

### Encoder output dataclass

```python
@dataclass(frozen=True)
class MultiScaleEncoderOutput:
    summary_tokens: Tensor
    patch_tokens: Tensor
```

### Architecture config

```python
@dataclass(frozen=True)
class MultiScaleRdtArchitectureConfig:
    hidden_size: int = 384
    num_attention_heads: int = 6
    mlp_ratio: float = 4.0
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6
    shared_stem_depth: int = 2
    adapter_depth: int = 1
    summary_tokens_per_scale: int = 2
    rdt_steps: int = 3
    patch_branches: tuple[PatchBranchConfig, ...] = field(
        default_factory=default_patch_branches
    )
```

### Encoder init sketch

```python
self.branch_token_counts = tuple(
    compute_token_count(feature_dims=cfg.feature_dims, patch_branch=branch)
    for branch in architecture.patch_branches
)
self.total_token_count = sum(self.branch_token_counts)

self.summary_tokens_per_scale = architecture.summary_tokens_per_scale
self.total_summary_token_count = (
    len(architecture.patch_branches) * architecture.summary_tokens_per_scale
)

self.branch_summary_tokens = nn.Parameter(
    torch.empty(
        len(architecture.patch_branches),
        architecture.summary_tokens_per_scale,
        hidden_size,
    )
)

self.position_embeddings = nn.ParameterList(
    nn.Parameter(torch.empty(1, token_count, hidden_size))
    for token_count in self.branch_token_counts
)
```

### Encoder reset sketch

```python
for position_embedding in self.position_embeddings:
    nn.init.normal_(position_embedding, std=PATCH_INIT_STD)

nn.init.normal_(self.scale_embeddings, std=PATCH_INIT_STD)
nn.init.normal_(self.branch_summary_tokens, std=PATCH_INIT_STD)
```

### Encoder forward sketch

```python
def forward(self, x: Tensor) -> MultiScaleEncoderOutput:
    branch_summary_outputs: list[Tensor] = []
    branch_patch_outputs: list[Tensor] = []

    for branch_index, tokenizer in enumerate(self.patch_tokenizers):
        patch_tokens = tokenizer(x)  # [B, N_s, D]
        batch_size = patch_tokens.shape[0]

        patch_tokens = patch_tokens + self.position_embeddings[branch_index]

        scale_embedding = self.scale_embeddings[branch_index].view(1, 1, -1)

        summary_tokens = self.branch_summary_tokens[branch_index].unsqueeze(0)
        summary_tokens = summary_tokens.expand(batch_size, -1, -1)

        summary_tokens = summary_tokens + scale_embedding
        patch_tokens = patch_tokens + scale_embedding

        branch_tokens = torch.cat([summary_tokens, patch_tokens], dim=1)
        # [B, 2 + N_s, D]

        for block in self.shared_stem:
            branch_tokens = block(branch_tokens)

        branch_adapter = cast(nn.ModuleList, self.adapters[branch_index])
        for block in branch_adapter:
            branch_tokens = block(branch_tokens)

        summary_s = branch_tokens[:, : self.summary_tokens_per_scale, :]
        patch_s = branch_tokens[:, self.summary_tokens_per_scale :, :]

        branch_summary_outputs.append(summary_s)
        branch_patch_outputs.append(patch_s)

    summary_tokens = torch.cat(branch_summary_outputs, dim=1)
    patch_tokens = torch.cat(branch_patch_outputs, dim=1)

    summary_tokens = self.output_norm(summary_tokens)
    patch_tokens = self.output_norm(patch_tokens)

    return MultiScaleEncoderOutput(
        summary_tokens=summary_tokens,
        patch_tokens=patch_tokens,
    )
```

### Model forward sketch

```python
def forward(self, input_values: Tensor) -> AstModelOutput:
    if input_values.ndim != 3:
        raise ValueError(...)

    expected_shape = (
        self.cfg.encoder.feature_dims.max_length,
        self.cfg.encoder.feature_dims.num_mel_bins,
    )
    actual_shape = (int(input_values.shape[1]), int(input_values.shape[2]))
    if actual_shape != expected_shape:
        raise ValueError(...)

    x = input_values.unsqueeze(1)

    encoder_output = self.encoder(x)
    latent_summaries = encoder_output.summary_tokens
    evidence_memory = encoder_output.patch_tokens

    for _ in range(self.cfg.encoder.architecture.rdt_steps):
        latent_summaries = self.rdt_block(latent_summaries, evidence_memory)

    pooled_embedding = latent_summaries.mean(dim=1)
    logits = self.classifier(pooled_embedding)

    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(1)

    return AstModelOutput(logits=logits, pooled_embedding=pooled_embedding)
```

---

## Final Expected Default Flow

```text
input_values: [B, 1024, 128]
    ↓
x: [B, 1, 1024, 128]
    ↓
branch 1 patch tokens: [B, 1016, D] + 2 summary tokens -> [B, 1018, D]
branch 2 patch tokens: [B, 1020, D] + 2 summary tokens -> [B, 1022, D]
branch 3 patch tokens: [B, 1022, D] + 2 summary tokens -> [B, 1024, D]
branch 4 patch tokens: [B, 1023, D] + 2 summary tokens -> [B, 1025, D]
    ↓
shared branch-wise stem
    ↓
scale-specific adapters
    ↓
split
    summary tokens: [B, 8, D]
    patch memory:   [B, 4081, D]
    ↓
RDT, T=3:
    state:   [B, 8, D]
    context: [B, 4081, D]
    ↓
U3: [B, 8, D]
    ↓
mean over 8 summary tokens
pooled_embedding: [B, D]
    ↓
classifier
binary logits: [B]
multi-class logits: [B, C]
```

---

## Final Note

This is not a new architecture from scratch. It is a targeted modification of the existing `MIL-RDT-AST` branch:

```text
Old:
    H_all [B, 4081, D]
      -> LatentQueryPooler with K=8
      -> U0 [B, 8, D]
      -> RDT

New:
    branch summary tokens, 2 per scale
      -> U0 [B, 8, D]
    patch tokens
      -> H_all [B, 4081, D]
    RDT:
      U_{t+1} = RDTBlock(U_t, H_all)
```

The active model should no longer rely on learned global latent queries. The 8 RDT state tokens should come from branch-local summary/CLS tokens processed together with each scale's patch tokens through the shared stem and scale-specific adapter.
