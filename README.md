# AST Respiratory Classification

This branch trains and evaluates **clip-level respiratory sound classifiers** with an
**Audio Spectrogram Transformer (AST)** backbone.

- One `.wav` file is one training example.
- The active training/evaluation pipeline uses local AST-style `fbank` extraction.
- The encoder is Hugging Face `ASTModel`.
- The classification head stays in-repo so classifier type, optimizer grouping, and
  logging follow the existing project style.
- This is a **clean break** from the old Whisper + MIL path. Old MIL configs and
  checkpoints are not supported.

## Setup

```bash
mamba activate respiratory
pip install -r requirements.txt
```

## Train

Binary example:

```bash
python -m src.cli.training --config configs/training.json
```

Multi-class example:

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
python -m src.cli.evaluate --config configs/eval.json
```

Evaluation writes:

- `eval_metrics.json`
- `eval_predictions.csv`
- `eval_diagnostics.jsonl` when diagnostics are enabled

Binary evaluation keeps fixed-threshold (`0.5`) metrics at the top level and also
stores:

- `decision_threshold`
- `threshold_optimization`
- `optimized_metrics`

Threshold optimization is checkpoint-driven and only applies to **binary**
classification. For multi-class evaluation it is reported as disabled with an
explicit reason.

## Cross-Validation

```bash
python -m src.cli.cv --config configs/cv_run.json
```

Each fold is trained independently under
`experiment.output_dir/experiment.name/fold_x/`.

## Config Shape

The main pipeline now uses an AST-only schema:

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
      "type": "ast",
      "pretrained_name_or_path": "MIT/ast-finetuned-audioset-10-10-0.4593",
      "adaptation": {
        "mode": "partial",
        "num_layers": 1
      }
    },
    "classifier": {
      "type": "linear",
      "hidden_dim": 256,
      "dropout": 0.1,
      "pooling": "cls"
    }
  }
}
```

## Encoder Adaptation

Encoder adaptation is configured under `model.encoder.adaptation`:

- `mode: "frozen" | "partial" | "full"`
- `num_layers`

Partial unfreezing keeps embeddings and early blocks frozen, and unfreezes:

- the last `num_layers` transformer blocks
- the final encoder layer norm

## Loss Rules

Loss behavior depends on the number of classes:

- Binary (`len(label_to_index) == 2`)
  - supported losses: `bce`, `focal`
  - optional `auto_pos_weight` / `pos_weight`
- Multi-class (`len(label_to_index) > 2`)
  - required loss: `cross_entropy`
  - binary-only weighting options are rejected

## Optimizer Layout

Training uses two AdamW parameter groups:

- encoder trainable parameters -> `train.optimizer.encoder_lr`
- classifier trainable parameters -> `train.optimizer.head_lr`

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

## AST Info

Inspect a pretrained AST config:

```bash
python -m src.cli.pretrained_info --name_or_path MIT/ast-finetuned-audioset-10-10-0.4593
```

## Plot Features

`src.cli.plot_mels` remains available as a utility. It still supports both
`log_mel` and `ast_fbank` feature plotting, but the main train/eval/CV pipeline is
AST-only.
