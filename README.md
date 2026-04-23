# Multi-Scale Event-MIL Respiratory Classification

This repo trains and evaluates clip-level respiratory sound classifiers with a
single supported encoder type: `multiscale_rdt_ast`.

- One `.wav` file is one training example.
- The frontend still uses local AST-style `fbank` extraction.
- Hugging Face `ASTModel`, latent-query pooling, and branch-summary-token
  initialization are intentionally removed from the active path.
- Training, evaluation, cross-validation, checkpointing, diagnostics, and
  threshold optimization remain config-driven.

## Setup

```bash
mamba activate respiratory
pip install -r requirements.txt
```

## Canonical Configs

- Binary B0: `configs/training_event_mil_b0.json`
- Binary B1: `configs/training_event_mil_b1.json`
- Binary B2: `configs/training_event_mil_b2.json`
- Binary B3: `configs/training_event_mil_b3.json`
- Binary training example: `configs/training_multiscale_rdt.json`
- Binary CV example: `configs/cv_multiscale_rdt.json`
- Binary evaluation: `configs/eval_multiscale_rdt.json`
- Multiclass training example: `configs/training_multiclass.json`

`training_multiscale_rdt.json`, `cv_multiscale_rdt.json`, and
`training_multiclass.json` are the "full" event-MIL examples with branch
auxiliary supervision enabled and RDT refinement active.

## Train

Binary example:

```bash
python -m src.cli.training --config configs/training_multiscale_rdt.json
```

Multiclass example:

```bash
python -m src.cli.training --config configs/training_multiclass.json
```

Artifacts are written under `experiment.output_dir/experiment.name/`.

Training keeps:

- `last.pt`
- `best_loss_*.pt`
- `best_f1_*.pt`

## Evaluate

```bash
python -m src.cli.evaluate --config configs/eval_multiscale_rdt.json
```

Evaluation writes:

- `eval_metrics.json`
- `eval_predictions.csv`
- `eval_diagnostics.jsonl` when diagnostics are enabled

Binary evaluation keeps fixed-threshold (`0.5`) metrics at the top level and
also stores:

- `decision_threshold`
- `threshold_optimization`
- `optimized_metrics`

Threshold optimization is checkpoint-driven and only applies to binary
classification. For multiclass evaluation it is reported as disabled with an
explicit reason.

## Cross-Validation

```bash
python -m src.cli.cv --config configs/cv_multiscale_rdt.json
```

Each fold is trained independently under
`experiment.output_dir/experiment.name/fold_x/`.

## Architecture

The default model consumes:

```text
input_values: [B, 1024, 128]
```

It applies this event-MIL-first flow:

```text
[B, 1024, 128]
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

Default patch token geometry:

- `(16, 16)` patch with `(8, 16)` stride -> `1016` patch tokens -> `127`
  temporal event tokens
- `(8, 32)` patch with `(4, 32)` stride -> `1020` patch tokens -> `255`
  temporal event tokens
- `(4, 64)` patch with `(2, 64)` stride -> `1022` patch tokens -> `511`
  temporal event tokens
- `(2, 128)` patch with `(1, 128)` stride -> `1023` patch tokens -> `1023`
  temporal event tokens

The concatenated event context length is `1916`. With the default
`top_tokens_per_branch = 2`, the initial evidence state is `U0: [B, 8, D]`.

Default encoder hyperparameters:

- `hidden_size = 192`
- `num_attention_heads = 4`
- `mlp_ratio = 2.0`
- `hidden_dropout_prob = 0.1`
- `attention_probs_dropout_prob = 0.1`
- `layer_norm_eps = 1e-6`
- `shared_stem_depth = 2`
- `adapter_depth = 1`
- `rdt.enabled = false`
- `rdt.steps = 3`
- `rdt.top_tokens_per_branch = 2`
- `rdt.gated_residual = true`
- `rdt.layerscale_init = 0.01`

Classifier pooling remains config-visible as `latent_mean`, but it now means
mean pooling over the selected or RDT-refined evidence tokens.

## B0 / B1 / B2 / B3 Modes

The experiment family is controlled by config only:

- `B0`: `rdt.enabled = false`, `branch_auxiliary.enabled = false`
- `B1`: `rdt.enabled = false`, `branch_auxiliary.enabled = true`
- `B2`: `rdt.enabled = true`, `rdt.steps = 2`, `branch_auxiliary.enabled = true`
- `B3`: `rdt.enabled = true`, `rdt.steps = 3`, `branch_auxiliary.enabled = true`

`rdt.enabled` is the switch that disables refinement. `steps` is still stored
in the config when RDT is off, but it is ignored by the forward path.

## Config Shape

The active schema is:

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
          "layerscale_init": 0.01
        },
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
  },
  "train": {
    "loss": {
      "type": "focal",
      "branch_auxiliary": {
        "enabled": true,
        "weight": 0.3,
        "aggregation": "mean"
      }
    }
  }
}
```

Strict compatibility rules:

- `model.encoder.type` must stay `multiscale_rdt_ast`
- `adaptation.mode` supports only `full` and `frozen`
- `adaptation.num_layers` must be `0`
- `classifier.pooling` must be `latent_mean`
- legacy `latent_query_count`, `summary_tokens_per_scale`, and flat `rdt_steps`
  are rejected

## Loss Rules

Loss behavior still depends on the number of classes:

- Binary (`len(label_to_index) == 2`)
  - supported losses: `bce`, `focal`
  - optional `auto_pos_weight` / `pos_weight`
  - optional branch auxiliary loss uses the same binary criterion on each
    branch logit
- Multi-class (`len(label_to_index) > 2`)
  - required loss: `cross_entropy`
  - binary-only weighting options are rejected
  - branch auxiliary loss applies cross-entropy to each branch head and then
    averages across branches

## Optimizer Layout

Training keeps two AdamW parameter groups:

- encoder parameters -> patch tokenizers, shared stem, adapters,
  frequency-attention poolers, branch MIL heads -> `train.optimizer.encoder_lr`
- head parameters -> optional RDT, fusion projector, final classifier ->
  `train.optimizer.head_lr`

If `adaptation.mode = "frozen"`, the encoder group is frozen and only the head
parameters are trainable.

## Diagnostics

Diagnostics are controlled by `analysis.outputs`:

```json
"analysis": {
  "outputs": {
    "save_logits": true,
    "save_probabilities": true,
    "save_embeddings": false,
    "save_clip_metadata": true
  }
}
```

Training saves validation diagnostics under:

```text
<run_dir>/diagnostics/val_epoch_XXX.jsonl
```

Evaluation writes:

```text
<checkpoint_dir>/eval_diagnostics.jsonl
```

When enabled, diagnostics now include:

- `branch_logits` with the saved logit payload
- `selected_evidence_tokens` with the saved embedding payload

Full branch attention maps stay in the model output for training and tests, but
they are not dumped into JSONL by default because they are large.

## Notes

- `src.cli.plot_mels` remains available as a utility.
- `src.cli.pretrained_info` exits immediately because pretrained HF AST
  inspection is no longer part of this branch.
- `configs/eval_multiscale_rdt.json` remains evaluation-only and reconstructs
  the model from checkpoint `model_cfg`.
