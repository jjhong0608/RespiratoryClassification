# AST+MIL Respiratory Classification

This branch trains and evaluates **recording-level weakly supervised respiratory sound classifiers** with an **Audio Spectrogram Transformer (AST)** instance encoder and a **Multiple Instance Learning (MIL)** aggregator.

- One `.wav` file is one **recording bag**
- One bag contains multiple overlapping instances
- Default segmentation is **2.0 second windows** with **1.0 second hop**
- Labels are recording-level only
- Supported MIL aggregators are:
  - `gated_attention` (default)
  - `linear_softmax`

## Setup

```bash
mamba activate respiratory
pip install -r requirements.txt
```

Use the requested Python interpreter for project commands:

```bash
/Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python
```

## Data layout

The v1 pipeline keeps directory-based discovery. Each `.wav` under a split root is treated as one recording bag, and the parent directory name is used as the label.

```text
datasets/
  wheeze/
    train/
      normal/*.wav
      wheeze/*.wav
    val/
      normal/*.wav
      wheeze/*.wav
    test/
      normal/*.wav
      wheeze/*.wav
```

Configure these roots under `data.metadata.splits.train/val/eval.roots`.

## Train

Binary example with default `gated_attention`:

```bash
/Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python -m src.cli.training --config configs/training.json
```

Multi-class example with `linear_softmax`:

```bash
/Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python -m src.cli.training --config configs/training_multiclass.json
```

Artifacts are written under `experiment.output_dir/experiment.name/`.

Training keeps:

- `last.pt`
- `best_loss_*.pt`
- `best_f1_*.pt`

## Evaluate

```bash
/Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python -m src.cli.evaluate --config configs/eval.json
```

Evaluation writes:

- `eval_metrics.json`
- `eval_predictions.csv`
- `eval_diagnostics.jsonl` when diagnostics are enabled

Binary evaluation reports fixed-threshold (`0.5`) metrics and checkpoint-derived optimized-threshold metrics. Threshold optimization is only applied to binary classification.

## Cross-validation

```bash
/Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python -m src.cli.cv --config configs/cv_run.json
```

Each fold is trained independently under `experiment.output_dir/experiment.name/fold_x/`.

## Config shape

The AST+MIL pipeline uses a nested recording-level schema:

```json
{
  "experiment": {
    "mode": "recording_mil"
  },
  "data": {
    "metadata": {
      "label_to_index": {
        "normal": 0,
        "wheeze": 1
      },
      "splits": {
        "train": {
          "roots": ["datasets/wheeze/train"]
        },
        "val": {
          "roots": ["datasets/wheeze/val"]
        },
        "eval": {
          "roots": ["datasets/wheeze/test"]
        }
      }
    },
    "audio": {
      "sample_rate": 16000
    },
    "instance": {
      "window_sec": 2.0,
      "hop_sec": 1.0,
      "tail_policy": "cover_end"
    },
    "features": {
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
    },
    "loader": {
      "num_workers": 0
    }
  },
  "model": {
    "encoder": {
      "type": "ast",
      "pretrained_name_or_path": "MIT/ast-finetuned-audioset-10-10-0.4593",
      "pooling": "cls",
      "adaptation": {
        "mode": "partial",
        "num_layers": 1
      }
    },
    "instance_head": {
      "projection_dim": 256,
      "dropout": 0.1,
      "normalize": true
    },
    "mil": {
      "type": "gated_attention"
    },
    "classifier": {
      "type": "linear",
      "hidden_dim": 256,
      "dropout": 0.1
    }
  },
  "train": {
    "batch_size": 4
  },
  "eval": {
    "batch_size": 4
  },
  "logging": {
    "diagnostics": {
      "top_k_instances": 5
    }
  }
}
```

## Segmentation

All train/validation/evaluation/inference paths share the same segmentation logic:

- `window_sec = 2.0`
- `hop_sec = 1.0`
- short recordings are padded to produce at least one instance
- tails are covered by appending a final end-aligned window when needed

Per-instance metadata is retained in the bag pipeline:

- `instance_index`
- `instance_start_sec`
- `instance_end_sec`

## MIL aggregators

### `gated_attention`

- Default aggregator
- Uses AST instance embeddings
- Applies mask-aware gated attention over valid instances
- Produces attention weights for diagnostics

### `linear_softmax`

- Score-based mask-aware MIL pooling
- Applies the classifier to each instance first
- Pools instance probabilities with linear-softmax pooling
- Supports both binary and multi-class classification

## Loss rules

- Binary (`len(label_to_index) == 2`)
  - supported losses: `bce`, `focal`
  - optional `auto_pos_weight` / `pos_weight`
- Multi-class (`len(label_to_index) > 2`)
  - required loss: `cross_entropy`

## Diagnostics

Diagnostics are controlled by `logging.diagnostics` and are saved for validation and evaluation.

Each recording-level diagnostics row can include:

- `recording_id`
- `recording_path`
- `true_label`
- `predicted_label`
- `bag_logits`
- `bag_probabilities`
- `num_instances`
- `instance_start_sec`
- `instance_end_sec`
- `instance_logits`
- `instance_probabilities`
- `attention_weights` for `gated_attention`
- `top_k_instances`

Validation diagnostics are written under:

```text
<run_dir>/diagnostics/val_epoch_XXX.jsonl
```

Evaluation diagnostics are written under:

```text
<checkpoint_dir>/eval_diagnostics.jsonl
```

## AST info

Inspect a pretrained AST config:

```bash
/Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python -m src.cli.pretrained_info --name_or_path MIT/ast-finetuned-audioset-10-10-0.4593
```

## Plot features

`src.cli.plot_mels` remains available as a utility for `log_mel` and `ast_fbank` feature plotting. The main training/evaluation/CV pipeline is AST+MIL recording-level only.
