# Codex Instructions: H0 Branch-Aware Gated Evidence Pooling

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

Do **not** rewrite the architecture from scratch. Do **not** reintroduce Hugging Face `ASTModel`. Do **not** reintroduce latent-query pooling. Do **not** modify the patch geometry, branch MIL heads, top-k evidence selection, RDT block, or final fusion inputs except for replacing the evidence readout.

---

## Objective

Implement **branch-aware gated evidence pooling** as an alternative to the current evidence mean pooling.

The target model is **H0**:

```text
4-scale Event-MIL-RDT
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

The current H0 path is:

```text
RDT output
U_T: [B, 8, D]
    -> mean over evidence tokens
evidence_embedding: [B, D]
    -> concat with:
         mean_branch_embedding: [B, D]
         branch_logits: [B, 4]
    -> fusion projector
    -> classifier
```

Change only the evidence readout:

```text
RDT output
U_T: [B, 8, D]
    -> branch-aware gated evidence pooling
gated_evidence_embedding: [B, D]
    -> concat with:
         mean_branch_embedding: [B, D]
         branch_logits: [B, 4]
    -> fusion projector
    -> classifier
```

The final fusion must remain:

```text
[gated_evidence_embedding, mean_branch_embedding, branch_logits]
```

---

## Current Architecture to Preserve

The current README describes the active event-MIL-first flow as:

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

Keep all of that, except make the evidence pooling configurable:

```text
evidence mean pooling
```

becomes:

```text
evidence pooling: "mean" or "branch_gated"
```

---

## Non-Negotiable Design Decisions

These are fixed by the experiment design.

### 1. Gate input

The branch gate must be computed **only from branch evidence summaries**.

Allowed gate input:

```text
branch_evidence_summary: [B, S, D]
```

Not allowed as gate input:

```text
branch_logits
mean_branch_embedding
final logits
labels
```

### 2. Final fusion inputs

Final fusion must still use:

```text
pooled_evidence_embedding
mean_branch_embedding
branch_logits
```

For `branch_gated`, the only change is:

```text
pooled_evidence_embedding = gated_evidence_embedding
```

Do not remove `branch_logits`.

Do not remove `mean_branch_embedding`.

### 3. H0 training objective

The H0 gated configs must keep:

```json
"branch_auxiliary": {
  "enabled": true,
  "weight": 0.1,
  "aggregation": "mean"
}
```

Do not switch to H2 / aux-off.

### 4. RDT

Keep RDT enabled:

```json
"rdt": {
  "enabled": true,
  "steps": 3,
  "top_tokens_per_branch": 2,
  "gated_residual": true,
  "layerscale_init": 0.01
}
```

Do not modify the RDT block.

### 5. Patch branches

Keep the 4-scale H0 geometry:

```json
"patch_branches": [
  { "patch_size": [16, 16], "stride": [8, 16] },
  { "patch_size": [8, 32], "stride": [4, 32] },
  { "patch_size": [4, 64], "stride": [2, 64] },
  { "patch_size": [2, 128], "stride": [1, 128] }
]
```

---

# Part A — Implement Evidence Pooling Config

Add a config block under:

```text
model.encoder.architecture.evidence_pooling
```

Recommended schema:

```json
"evidence_pooling": {
  "type": "mean",
  "gate_hidden_size": null,
  "dropout": 0.1,
  "temperature": 1.0
}
```

Supported values:

```text
mean
branch_gated
```

Default:

```text
mean
```

The default should preserve the current behavior.

For H0 gated configs, use:

```json
"evidence_pooling": {
  "type": "branch_gated",
  "gate_hidden_size": null,
  "dropout": 0.1,
  "temperature": 1.0
}
```

### Validation rules

Add validation:

```text
evidence_pooling.type in {"mean", "branch_gated"}
evidence_pooling.temperature > 0
0 <= evidence_pooling.dropout < 1
gate_hidden_size is null or positive integer
```

Do not confuse this with `classifier.pooling`.

`classifier.pooling` should remain:

```json
"pooling": "latent_mean"
```

The branch-gated module changes the meaning of the evidence readout before the existing fusion projector, not the classifier config interface.

---

# Part B — Implement BranchAwareGatedEvidencePooling

Add a module, likely in:

```text
src/models/multiscale_rdt_ast.py
```

Suggested name:

```python
BranchAwareGatedEvidencePooling
```

## Inputs

```python
evidence_tokens: Tensor
# [B, K, D]

branch_ids: Tensor
# [B, K]
```

For default H0:

```text
S = 4 branches
M = top_tokens_per_branch = 2
K = S * M = 8

evidence_tokens: [B, 8, D]
branch_ids:      [B, 8]
```

Expected branch IDs for H0 are usually equivalent to:

```text
[0, 0, 1, 1, 2, 2, 3, 3]
```

but do not rely only on position if `selected_evidence_branch_ids` already exists. Prefer using branch IDs.

## Outputs

Return a structured output.

Suggested dataclass:

```python
@dataclass(frozen=True)
class EvidencePoolingOutput:
    pooled_embedding: Tensor
    gate_weights: Tensor | None = None
    gate_entropy: Tensor | None = None
    branch_evidence_summary: Tensor | None = None
    branch_evidence_norms: Tensor | None = None
```

For `branch_gated`:

```text
pooled_embedding:         [B, D]
gate_weights:             [B, S]
gate_entropy:             [B]
branch_evidence_summary:  [B, S, D]
branch_evidence_norms:    [B, S]
```

For `mean`:

```text
pooled_embedding: [B, D]
gate_weights: None
gate_entropy: None
branch_evidence_summary: None
branch_evidence_norms: None
```

If adding a dataclass is too invasive, return a tuple plus optional metadata, but the model output must expose the diagnostics fields listed later.

---

## Branch grouping

For each branch `s`, group evidence tokens whose `branch_ids == s`.

Basic logic:

```python
unique_branches = sorted unique branch ids
for each branch s:
    mask = branch_ids == s
    tokens_s = evidence_tokens[mask]
    summary_s = mean(tokens_s over evidence-token dimension)
```

Because batching makes boolean grouping awkward, implement this carefully.

Simpler preferred implementation if selected evidence is always assembled branch-by-branch:

- Keep branch evidence tensors as a list before concatenation.
- Let the pooling module receive `branch_evidence_tokens: list[Tensor]`, where each item has shape `[B, M_s, D]`.
- Compute branch summaries from that list.
- Still return gate diagnostics and maintain selected evidence metadata.

However, if the current model already only has concatenated `selected_evidence_tokens` and `selected_evidence_branch_ids`, use the branch IDs.

### Robust behavior for dynamic branch count

Do not hard-code:

```text
S = 4
K = 8
```

The module must also work for:

```text
3-scale top-2 -> K = 6, S = 3
4-scale top-1 -> K = 4, S = 4
4-scale top-3 -> K = 12, S = 4
```

Even though H0 gated uses 4-scale top-2, the codebase already supports dynamic branch count and selected evidence length. Do not break that.

### Handling empty branches

For H0 gated, no branch should be empty.

But if previous ablation code supports `exclude_branches_from_evidence`, a branch could be absent from selected evidence. In that case:

- exclude the empty branch from the gate computation, or
- assign it a zero summary and mask its gate logit to `-inf`.

Preferred:

```text
Only include branches present in selected evidence.
```

Document this behavior in code comments and tests.

---

## Within-branch summary

For each branch:

```python
branch_summary_s = evidence_tokens_s.mean(dim=1)
```

This is deliberately simple.

Do not use branch logits.

Do not use mean branch embedding.

Do not add another top-k or attention inside each branch for the first implementation.

---

## Gate computation

Given:

```text
branch_evidence_summary: [B, S, D]
```

Compute:

```python
gate_logits = gate_mlp(branch_evidence_summary).squeeze(-1)
gate_weights = softmax(gate_logits / temperature, dim=1)
```

### Gate MLP

If `gate_hidden_size` is `null`, use a simple linear gate:

```python
nn.Linear(D, 1)
```

If `gate_hidden_size` is an integer, use:

```python
LayerNorm(D)
Linear(D, gate_hidden_size)
GELU
Dropout(dropout)
Linear(gate_hidden_size, 1)
```

Either style is acceptable as long as:

```text
gate input = branch evidence summary only
```

Recommended initial implementation:

```python
LayerNorm(D)
Linear(D, max(D // 2, 1))
GELU
Dropout(dropout)
Linear(max(D // 2, 1), 1)
```

if `gate_hidden_size` is null.

But keep it simple and testable.

### Gate entropy

Compute:

```python
gate_entropy = -(gate_weights * (gate_weights + eps).log()).sum(dim=1)
```

Shape:

```text
[B]
```

### Gated evidence embedding

Compute:

```python
pooled_embedding = torch.sum(
    gate_weights.unsqueeze(-1) * branch_evidence_summary,
    dim=1,
)
```

Shape:

```text
[B, D]
```

### Branch evidence norms

Optional but recommended:

```python
branch_evidence_norms = branch_evidence_summary.norm(dim=-1)
```

Shape:

```text
[B, S]
```

---

# Part C — Integrate into Model Forward

Find the current model forward path where it does something equivalent to:

```python
evidence_embedding = refined_evidence_tokens.mean(dim=1)
```

Replace that with:

```python
pooling_output = self.evidence_pooler(
    evidence_tokens=refined_evidence_tokens,
    branch_ids=selected_evidence_branch_ids,
)

evidence_embedding = pooling_output.pooled_embedding
```

Then preserve the existing fusion:

```python
fusion_input = torch.cat(
    [
        evidence_embedding,
        mean_branch_embedding,
        branch_logits_or_flattened_branch_logits,
    ],
    dim=-1,
)
```

Do not change the meaning of `mean_branch_embedding`.

Do not remove branch logits.

Do not alter classifier output shape.

Binary logits must remain:

```text
[B]
```

Multiclass logits must remain:

```text
[B, C]
```

---

# Part D — Extend AstModelOutput and Diagnostics

## Model output fields

Extend `AstModelOutput` or the equivalent model output dataclass with optional fields:

```python
evidence_pooling_type: str | None = None
evidence_gate_weights: Tensor | None = None
evidence_gate_entropy: Tensor | None = None
branch_evidence_norms: Tensor | None = None
```

For `branch_gated`:

```text
evidence_pooling_type = "branch_gated"
evidence_gate_weights: [B, S]
evidence_gate_entropy: [B]
branch_evidence_norms: [B, S]
```

For `mean`:

```text
evidence_pooling_type = "mean"
evidence_gate_weights = None
evidence_gate_entropy = None
branch_evidence_norms = None
```

Keep existing fields such as:

```text
branch_logits
branch_attention_weights
selected_evidence_tokens
selected_evidence_indices
selected_evidence_scores
selected_evidence_branch_ids
```

Do not remove them.

---

## Diagnostics JSONL

Update diagnostics writer, likely:

```text
src/evaluation/diagnostics.py
```

When model output contains gate metadata, write:

```json
"evidence_pooling_type": "branch_gated",
"evidence_gate_weights": [...],
"evidence_gate_entropy": ...,
"branch_evidence_norms": [...]
```

Recommended behavior:

- Always write `evidence_pooling_type` if present.
- Always write `evidence_gate_weights` and `evidence_gate_entropy` if present.
- Always write `branch_evidence_norms` if present.
- Continue to write selected evidence indices/scores/branch IDs as before.
- Continue to write selected evidence token embeddings only when `save_embeddings = true`.

Do not dump large hidden states by default.

---

# Part E — Add H0 Branch-Gated Training Configs

Create 5 seed configs:

```text
configs/training_h0_branch_gated_seed0.json
configs/training_h0_branch_gated_seed1.json
configs/training_h0_branch_gated_seed2.json
configs/training_h0_branch_gated_seed42.json
configs/training_h0_branch_gated_seed43.json
```

These should match the H0/G1 mainline except for:

```json
"evidence_pooling": {
  "type": "branch_gated",
  "gate_hidden_size": null,
  "dropout": 0.1,
  "temperature": 1.0
}
```

Use H0 settings:

```json
"rdt": {
  "enabled": true,
  "steps": 3,
  "top_tokens_per_branch": 2,
  "gated_residual": true,
  "layerscale_init": 0.01
}
```

```json
"branch_auxiliary": {
  "enabled": true,
  "weight": 0.1,
  "aggregation": "mean"
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

Use seeds:

```text
0, 1, 2, 42, 43
```

Use the repository's current seed field. Do not invent an unsupported seed field.

Preserve dataset paths and all local configuration from existing working configs.

Do not warm-start.

---

# Part F — README Updates

Update README with a section:

```text
Evidence Pooling
```

Describe:

## mean

```text
The default legacy behavior.
RDT-refined selected evidence tokens are averaged uniformly.
```

## branch_gated

```text
RDT-refined selected evidence tokens are grouped by source branch.
Each branch's selected evidence tokens are averaged into a branch evidence summary.
A gate is computed from branch evidence summaries only.
The gated weighted sum becomes the evidence embedding.
Final fusion still receives evidence embedding, mean branch embedding, and branch logits.
```

Add warning:

```text
branch_logits are not used as gate input.
branch_logits remain part of the final fusion input.
```

Update architecture diagram text:

Old:

```text
-> RDT refinement
-> evidence mean pooling
-> fuse evidence embedding, mean branch embedding, and branch logits
```

New:

```text
-> RDT refinement
-> evidence pooling: mean or branch-aware gated
-> fuse evidence embedding, mean branch embedding, and branch logits
```

Add a table of new H0 branch-gated configs:

```text
training_h0_branch_gated_seed0.json
training_h0_branch_gated_seed1.json
training_h0_branch_gated_seed2.json
training_h0_branch_gated_seed42.json
training_h0_branch_gated_seed43.json
```

Document comparison target:

```text
Compare H0 mean pooling vs H0 branch-gated pooling using best_loss checkpoints.
Report mean, std, min, max, and worst seed.
Do not choose by a single best seed.
```

---

# Part G — Tests

Add or update tests.

## 1. Config parsing tests

Test valid configs:

```text
evidence_pooling.type = "mean"
evidence_pooling.type = "branch_gated"
```

Test invalid configs:

```text
evidence_pooling.type = "invalid"
evidence_pooling.temperature <= 0
evidence_pooling.dropout < 0
evidence_pooling.dropout >= 1
gate_hidden_size <= 0 when not null
```

Test new 5 seed configs parse.

---

## 2. Pooling module shape tests

### H0 default shape

Input:

```text
evidence_tokens: [B, 8, D]
branch_ids: [B, 8]
```

Expected:

```text
pooled_embedding: [B, D]
gate_weights: [B, 4]
gate_entropy: [B]
branch_evidence_summary: [B, 4, D]
branch_evidence_norms: [B, 4]
```

### Dynamic 3-scale shape

Input:

```text
evidence_tokens: [B, 6, D]
branch_ids: [B, 6]
```

Expected:

```text
pooled_embedding: [B, D]
gate_weights: [B, 3]
gate_entropy: [B]
branch_evidence_summary: [B, 3, D]
```

### Gate sanity

Assert:

```python
torch.allclose(gate_weights.sum(dim=1), torch.ones(B), atol=1e-5)
```

Assert:

```text
gate weights are non-negative.
```

---

## 3. Model forward tests

Test:

```text
evidence_pooling.type = "mean"
evidence_pooling.type = "branch_gated"
```

For both:

```text
forward succeeds
binary logits shape is [B]
pooled_embedding shape is [B, D]
```

For `branch_gated`:

```text
output.evidence_gate_weights is not None
output.evidence_gate_entropy is not None
```

For `mean`:

```text
output.evidence_gate_weights is None
or omitted according to dataclass convention
```

---

## 4. Diagnostics tests

For branch-gated output, verify JSONL contains:

```text
evidence_pooling_type
evidence_gate_weights
evidence_gate_entropy
branch_evidence_norms
```

For mean output, verify the writer does not crash and either omits gate fields or writes them as null.

---

## 5. Regression / no-break tests

Ensure existing configs still parse and run basic forward tests:

```text
training_multiscale_rdt.json
training_event_mil_b0.json
training_event_mil_b1.json
training_event_mil_b2.json
training_event_mil_b3.json
```

At minimum, ensure the new schema has a default `evidence_pooling.type = "mean"` when old configs do not specify it.

---

# Part H — Acceptance Criteria

The task is complete when:

1. `evidence_pooling` config exists under `model.encoder.architecture`.
2. `evidence_pooling.type = "mean"` preserves the current evidence mean pooling behavior.
3. `evidence_pooling.type = "branch_gated"` uses branch-aware gated evidence pooling.
4. Gate is computed only from branch evidence summaries.
5. Branch logits are not used as gate input.
6. Mean branch embedding is not used as gate input.
7. Final fusion still includes:
   - evidence embedding,
   - mean branch embedding,
   - branch logits.
8. RDT block is not modified.
9. Top-k evidence selection is not modified.
10. Patch geometry is not modified.
11. Branch auxiliary loss behavior is not modified.
12. Dynamic branch count works.
13. Dynamic selected evidence length works.
14. Gate weights and entropy are exposed in model output.
15. Gate weights and entropy are written to diagnostics.
16. H0 branch-gated 5-seed configs exist.
17. README documents mean vs branch_gated pooling.
18. Tests pass.

---

# Part I — Avoid These Mistakes

- Do not replace the whole fusion module.
- Do not remove `mean_branch_embedding` from final fusion.
- Do not remove `branch_logits` from final fusion.
- Do not use `branch_logits` as gate input.
- Do not use `mean_branch_embedding` as gate input.
- Do not use labels as gate input.
- Do not hard-code 4 branches.
- Do not hard-code 8 evidence tokens.
- Do not break the old mean pooling path.
- Do not reintroduce latent-query pooling.
- Do not alter the RDT block.
- Do not alter branch MIL heads.
- Do not alter top-k evidence selection.
- Do not alter the patch branch geometry.
- Do not switch H0 gated configs to aux-off.
- Do not warm-start H0 gated configs.
- Do not use `best_f1` as the primary comparison criterion.

---

# Part J — Suggested Implementation Sketch

This is a sketch. Adapt it to the actual code structure.

```python
@dataclass(frozen=True)
class EvidencePoolingConfig:
    type: Literal["mean", "branch_gated"] = "mean"
    gate_hidden_size: int | None = None
    dropout: float = 0.1
    temperature: float = 1.0
```

```python
@dataclass(frozen=True)
class EvidencePoolingOutput:
    pooled_embedding: Tensor
    gate_weights: Tensor | None = None
    gate_entropy: Tensor | None = None
    branch_evidence_summary: Tensor | None = None
    branch_evidence_norms: Tensor | None = None
```

```python
class BranchAwareGatedEvidencePooling(nn.Module):
    def __init__(self, hidden_size: int, cfg: EvidencePoolingConfig) -> None:
        super().__init__()
        self.temperature = cfg.temperature
        hidden = cfg.gate_hidden_size or max(hidden_size // 2, 1)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, evidence_tokens: Tensor, branch_ids: Tensor) -> EvidencePoolingOutput:
        # evidence_tokens: [B, K, D]
        # branch_ids: [B, K]
        # Implementation should support dynamic S and K.
        ...
```

Potential simpler helper:

```python
def summarize_by_branch(
    evidence_tokens: Tensor,
    branch_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    # returns:
    # branch_summaries: [B, S, D]
    # branch_id_values: [S]
    ...
```

Then:

```python
branch_summaries, _ = summarize_by_branch(evidence_tokens, branch_ids)
gate_logits = self.gate(branch_summaries).squeeze(-1)
gate_weights = torch.softmax(gate_logits / self.temperature, dim=1)
pooled = torch.sum(gate_weights.unsqueeze(-1) * branch_summaries, dim=1)
entropy = -(gate_weights * (gate_weights + 1e-8).log()).sum(dim=1)
norms = branch_summaries.norm(dim=-1)
return EvidencePoolingOutput(
    pooled_embedding=pooled,
    gate_weights=gate_weights,
    gate_entropy=entropy,
    branch_evidence_summary=branch_summaries,
    branch_evidence_norms=norms,
)
```

Mean pooling path:

```python
class MeanEvidencePooling(nn.Module):
    def forward(self, evidence_tokens: Tensor, branch_ids: Tensor | None = None) -> EvidencePoolingOutput:
        return EvidencePoolingOutput(
            pooled_embedding=evidence_tokens.mean(dim=1),
        )
```

Model forward:

```python
pooling_out = self.evidence_pooler(
    refined_evidence_tokens,
    selected_evidence_branch_ids,
)

evidence_embedding = pooling_out.pooled_embedding

fusion_input = torch.cat(
    [
        evidence_embedding,
        mean_branch_embedding,
        branch_logits_for_fusion,
    ],
    dim=-1,
)
```

Model output:

```python
return AstModelOutput(
    logits=logits,
    pooled_embedding=pooled_embedding,
    branch_logits=branch_logits,
    branch_attention_weights=branch_attention_weights,
    selected_evidence_tokens=selected_evidence_tokens,
    selected_evidence_indices=selected_evidence_indices,
    selected_evidence_scores=selected_evidence_scores,
    selected_evidence_branch_ids=selected_evidence_branch_ids,
    evidence_pooling_type=self.cfg.encoder.architecture.evidence_pooling.type,
    evidence_gate_weights=pooling_out.gate_weights,
    evidence_gate_entropy=pooling_out.gate_entropy,
    branch_evidence_norms=pooling_out.branch_evidence_norms,
)
```

---

# Part K — Self-Review Checklist for Codex-Only Implementability

Before finishing, verify this instruction can be implemented using only the contents above.

## Scope clarity

- [ ] It is clear that only the evidence readout changes.
- [ ] It is clear that RDT, branch MIL, top-k, patch geometry, and final fusion structure are preserved.
- [ ] It is clear that `branch_logits` and `mean_branch_embedding` stay in final fusion.

## Gate definition

- [ ] It is clear that gate input is only `branch_evidence_summary`.
- [ ] It is clear that `branch_logits` must not be used as gate input.
- [ ] It is clear that `mean_branch_embedding` must not be used as gate input.

## Tensor shapes

- [ ] H0 shape `[B, 8, D] -> [B, 4, 2, D] -> [B, 4, D] -> [B, D]` is explicit.
- [ ] Dynamic 3-scale shape `[B, 6, D] -> [B, 3, D] -> [B, D]` is covered.
- [ ] The implementation is not allowed to hard-code 4 branches or 8 evidence tokens.

## Config

- [ ] `model.encoder.architecture.evidence_pooling` schema is specified.
- [ ] `mean` and `branch_gated` are both supported.
- [ ] Old configs without the new block default to `mean`.

## Diagnostics

- [ ] Gate weights and entropy are added to model output.
- [ ] Gate weights and entropy are added to diagnostics JSONL.
- [ ] Large hidden tensors are not dumped by default.

## Tests

- [ ] Config parsing tests are specified.
- [ ] Pooling shape tests are specified.
- [ ] Forward tests for both mean and branch_gated are specified.
- [ ] Diagnostics tests are specified.

If any checklist item is not satisfied, update the implementation or README/tests before marking the task complete.
