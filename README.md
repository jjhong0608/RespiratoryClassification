# Multi-Scale RDT AST Respiratory Classification

This repo trains and evaluates clip-level respiratory sound classifiers with a
single supported encoder: `multiscale_rdt_ast`.

- One `.wav` file is one training example.
- The data pipeline still uses local AST-style `fbank` extraction.
- The Hugging Face `ASTModel` path has been removed on purpose.
- The training, evaluation, cross-validation, checkpoint, diagnostics, and
  threshold-optimization flow remain config-driven.

## Setup

```bash
mamba activate respiratory
pip install -r requirements.txt
```

## Canonical Configs

- Binary training: `configs/training_multiscale_rdt.json`
- Binary CV: `configs/cv_multiscale_rdt.json`
- Binary evaluation: `configs/eval_multiscale_rdt.json`
- Multiclass training example: `configs/training_multiclass.json`

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

It applies this fixed flow:

```text
[B, 1024, 128]
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

Default branch geometry:

- `(16, 16)` patch with `(8, 16)` stride -> `1016` tokens
- `(8, 32)` patch with `(4, 32)` stride -> `1020` tokens
- `(4, 64)` patch with `(2, 64)` stride -> `1022` tokens
- `(2, 128)` patch with `(1, 128)` stride -> `1023` tokens

Default encoder hyperparameters:

- `hidden_size = 384`
- `num_attention_heads = 6`
- `mlp_ratio = 4.0`
- `hidden_dropout_prob = 0.1`
- `attention_probs_dropout_prob = 0.1`
- `layer_norm_eps = 1e-6`
- `shared_stem_depth = 2`
- `adapter_depth = 1`
- `latent_query_count = 8`
- `rdt_steps = 3`

Classifier pooling is fixed to `latent_mean`.

## Config Shape

The active schema is:

```json
{
  "experiment": {
    "mode": "clip"
  },
  "data": {
    "audio": {
      "sample_rate": 16000,
      "clip_duration_sec": 30.0
    },
    "preprocessing": {
      "source_type": "original",
      "bandpass": {
        "enabled": false
      },
      "ast_fbank": {
        "num_mel_bins": 128,
        "max_length": 1024,
        "do_normalize": true,
        "mean": -4.2677393,
        "std": 4.5689974
      }
    }
  },
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

Adaptation rules:

- `mode: "full"` trains the encoder and head together.
- `mode: "frozen"` freezes the encoder and trains only latent pooling, RDT, and
  classifier layers.
- `mode: "partial"` is rejected.
- `num_layers` must be `0`.

## Loss Rules

Loss behavior still depends on the number of classes:

- Binary (`len(label_to_index) == 2`)
  - supported losses: `bce`, `focal`
  - optional `auto_pos_weight` / `pos_weight`
- Multi-class (`len(label_to_index) > 2`)
  - required loss: `cross_entropy`
  - binary-only weighting options are rejected

## Optimizer Layout

Training keeps two AdamW parameter groups:

- encoder parameters -> `train.optimizer.encoder_lr`
- latent pooler + RDT + classifier parameters -> `train.optimizer.head_lr`

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

These files are intended for false positive / false negative clip inspection.

## Notes

- `src.cli.plot_mels` remains available as a utility.
- `src.cli.pretrained_info` now exits immediately because pretrained HF AST
  inspection is no longer part of this branch.
