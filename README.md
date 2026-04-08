# Wheeze MIL Experiments

This branch is a MIL-first refactor for weakly supervised wheeze detection.

- One audio file is one bag.
- Shorter audio segments cut from that file are the instances.
- Training and evaluation are bag-level binary classification (`normal` vs `wheeze`).
- Segmentation mode and MIL aggregation mode are independent config choices.

## Setup

```bash
mamba activate respiratory
pip install -r requirements.txt
```

## Train

Use the default MIL training config:

```bash
python -m src.cli.training --config configs/training.json
```

Ready-to-run example configs:

- `configs/mil_sliding_max.json`
- `configs/mil_sliding_topk.json`
- `configs/mil_sliding_attention.json`
- `configs/mil_nonoverlap_mean.json`

Artifacts are written under `experiment.output_dir/experiment.name/`.

Training keeps:

- `last.pt`
- `best_loss_*.pt`
- `best_f1_*.pt`

Class-imbalance handling lives under `train.loss`:

- `auto_pos_weight: true`
  - computes `negatives / positives` from the active training split
- `pos_weight`
  - optional manual override when `auto_pos_weight` is `false`
- `type: "bce" | "focal"`
- `gamma`
  - focal-loss focusing parameter

## Evaluate

```bash
python -m src.cli.evaluate --config configs/eval.json
```

This writes:

- `eval_metrics.json`
- `eval_predictions.csv`
- `eval_diagnostics.jsonl` when diagnostics are enabled

`eval_metrics.json` keeps the fixed-threshold (`0.5`) metrics at the top level and
also includes:

- `decision_threshold`
- `threshold_optimization`
- `optimized_metrics`

Standard evaluation does not tune thresholds on eval or test labels.
When threshold optimization is enabled in the eval config, evaluation loads the
validation-derived threshold saved in the checkpoint and applies that threshold
to the eval probabilities. If checkpoint threshold metadata is unavailable,
evaluation falls back safely to `0.5`.

Enable threshold tuning in eval configs with:

```json
"threshold_optimization": {
  "enabled": true,
  "metric": "f1"
}
```

This block selects which validation-derived checkpoint threshold is applied
during evaluation. It does not trigger threshold fitting on the eval split.

## Cross-Validation

```bash
python -m src.cli.cv --config configs/cv_run.json
```

Each fold is trained independently under `experiment.output_dir/experiment.name/fold_x/`.

## Bag And Instance Definition

The MIL pipeline uses the dataset layout:

```text
<root>/
  normal/
    *.wav
  wheeze/
    *.wav
```

`label_to_index` must stay binary and contiguous:

```json
{
  "normal": 0,
  "wheeze": 1
}
```

Bag formation works like this:

1. Load one full waveform.
2. Resample to `data.audio.sample_rate`.
3. Trim to at most `data.audio.clip_duration_sec`.
4. Apply waveform preprocessing such as band-pass and `source_type`.
5. Segment the waveform into instances.
6. Convert each segment into the Whisper-like log-mel tensor seen by the encoder.

## Segmentation Modes

`data.segment.mode` controls how bags are split:

- `full_clip`
  - One bag contains one instance spanning the full trimmed clip.
- `non_overlap`
  - The bag is cut into equal non-overlapping windows.
- `sliding_window`
  - The bag is cut with `length_sec` and `stride_sec`.

Important segmentation fields:

- `data.audio.clip_duration_sec`
  - Maximum waveform duration used for each bag.
- `data.segment.length_sec`
  - Segment/window length for `non_overlap` and `sliding_window`.
- `data.segment.stride_sec`
  - Window stride for `sliding_window`.
- `data.segment.pad_last`
  - Keep an incomplete tail window and pad it during feature extraction.
- `data.segment.drop_last`
  - Drop the incomplete tail window.

`pad_last` and `drop_last` are mutually exclusive.

## MIL Aggregators

`model.mil.aggregator` selects the bag-level reducer independently of segmentation.

Supported aggregators:

- `max`
- `mean`
- `topk`
- `attention`
- `logsumexp`
- `softmax_weighted`
- `noisy_or`

Aggregator-specific nested config:

- `model.mil.topk.k`
- `model.mil.attention.hidden_dim`
- `model.mil.attention.dropout`
- `model.mil.attention.gated`
- `model.mil.logsumexp.temperature`
- `model.mil.softmax_weighted.temperature`
- `model.mil.noisy_or.clamp_eps`

This means combinations such as:

- sliding window + max
- sliding window + top-k
- sliding window + attention
- non-overlap + mean

are selected only by config edits.

## Model Structure

The MIL model is explicit:

1. Segment encoder
   - Whisper audio encoder applied to each segment.
2. Instance head
   - Produces one per-segment logit from the segment embedding.
3. Bag aggregator
   - Reduces instance logits into one bag-level logit.

The current segment encoder is Whisper-based and configured under `model.encoder`.

## Diagnostics

Diagnostics are controlled by `analysis.outputs`:

```json
"analysis": {
  "outputs": {
    "save_segment_scores": true,
    "save_attention_weights": true,
    "save_topk_indices": true,
    "save_bag_metadata": true
  }
}
```

Available diagnostics:

- per-segment scores
- attention weights when using attention MIL
- selected top-k indices when using top-k MIL
- bag metadata including segment start/end mapping

Training saves validation diagnostics under:

```text
<run_dir>/diagnostics/val_epoch_XXX.jsonl
```

Evaluation writes:

```text
<checkpoint_dir>/eval_diagnostics.jsonl
```

These files are intended for false positive / false negative inspection.

## Nested Config Structure

The experiment interface is nested JSON:

```json
{
  "experiment": {
    "name": "wheeze_mil_sliding_attention",
    "task": "normal_vs_wheeze",
    "mode": "mil",
    "seed": 42,
    "device": "cpu",
    "output_dir": "checkpoints"
  },
  "checkpoint_path": "checkpoints/wheeze_mil_sliding_attention/last.pt",
  "threshold_optimization": {
    "enabled": true,
    "metric": "f1"
  },
  "data": {
    "train_dirs": ["datasets/wheeze/train"],
    "val_dirs": ["datasets/wheeze/val"],
    "eval_dirs": ["datasets/wheeze/test"],
    "label_to_index": {
      "normal": 0,
      "wheeze": 1
    },
    "batch_size": 4,
    "num_workers": 0,
    "audio": {
      "sample_rate": 16000,
      "clip_duration_sec": 30.0
    },
    "preprocessing": {
      "feature_type": "log_mel",
      "source_type": "original",
      "n_mels": 80,
      "bandpass": {
        "enabled": false,
        "low_hz": 250.0,
        "high_hz": 1000.0,
        "q": 0.707
      }
    },
    "segment": {
      "mode": "sliding_window",
      "length_sec": 5.0,
      "stride_sec": 2.5,
      "pad_last": true,
      "drop_last": false
    }
  },
  "model": {
    "encoder": {
      "type": "whisper",
      "backbone": "tiny",
      "pretrained_name_or_path": "tiny",
      "freeze": false,
      "strict": true,
      "download_root": null,
      "n_audio_state": 384,
      "n_audio_head": 6,
      "n_audio_layer": 4
    },
    "instance_head": {
      "type": "mlp",
      "hidden_dim": 256,
      "dropout": 0.1
    },
    "mil": {
      "aggregator": "attention",
      "return_instance_scores": true,
      "topk": {
        "k": 3
      },
      "attention": {
        "hidden_dim": 128,
        "dropout": 0.1,
        "gated": true
      },
      "logsumexp": {
        "temperature": 1.0
      },
      "softmax_weighted": {
        "temperature": 1.0
      },
      "noisy_or": {
        "clamp_eps": 1e-6
      }
    }
  },
  "train": {
    "epochs": 20,
    "top_k": 3,
    "warmup_ratio": 0.05,
    "max_grad_norm": 1.0,
    "optimizer": {
      "lr": 0.0001,
      "weight_decay": 0.01
    },
    "loss": {
      "type": "focal",
      "auto_pos_weight": true,
      "pos_weight": null
    },
    "sampler": {
      "weighted_random": false
    }
  },
  "analysis": {
    "outputs": {
      "save_segment_scores": true,
      "save_attention_weights": true,
      "save_topk_indices": false,
      "save_bag_metadata": true
    }
  }
}
```

## Notes

- `model.encoder.pretrained_name_or_path` can point to an official OpenAI Whisper checkpoint name such as `tiny`, `base`, or `small`, or to a local MIL checkpoint.
- MIL segment length controls the encoder context length automatically; it no longer has to match Whisper’s original 30-second context.
- `python -m src.cli.pretrained_info --name_or_path tiny` prints the encoder dimensions for a given Whisper checkpoint.
