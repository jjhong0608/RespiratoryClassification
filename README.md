# Wheeze MIL Experiments

This branch uses a final-model-oriented MIL architecture for weakly supervised wheeze detection.

- One audio file is one bag.
- Shorter audio segments cut from that file are the instances.
- Training and evaluation are bag-level binary classification (`normal` vs `wheeze`).
- Every segment embedding is built with intra-segment attention pooling over Whisper token states.
- The bag-level reducer remains configurable with `attention`, `max`, or `topk`.

## Setup

```bash
mamba activate respiratory
pip install -r requirements.txt
```

## Train

Use the default training config:

```bash
python -m src.cli.training --config configs/training.json
```

Ready-to-run example configs:

- `configs/mil_sliding_attention.json`
- `configs/mil_sliding_max.json`
- `configs/mil_sliding_topk.json`

Artifacts are written under `experiment.output_dir/experiment.name/`.

Training keeps:

- `last.pt`
- `best_loss_*.pt`
- `best_f1_*.pt`

Loose early stopping now runs alongside top-k checkpoint saving:

- monitor: `val_loss`
- default patience: `15`
- default `min_delta`: `1e-4`

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

## Cross-Validation

```bash
python -m src.cli.cv --config configs/cv_run.json
```

Each fold is trained independently under `experiment.output_dir/experiment.name/fold_x/`.

## Plot Features

Plot a single full-file feature map with the same frontend math used by the MIL data pipeline:

```bash
python -m src.cli.plot_mels --input path/to/audio.wav --out plots/audio_feature --formats html,png
```

Select the frontend with `--feature-type`:

- `log_mel`
  - uses the repo's Whisper-style log-mel frontend
- `ast_fbank`
  - uses local Kaldi `fbank` extraction plus AST normalization constants
  - normalizes real `fbank` frames before padding, then zero-fills any padded tail

AST plotting remains full-file plotting, not MIL segment-bag plotting.

Waveform preprocessing still runs before feature extraction for both frontends:

- `--source-type original|harmonic|percussive`
- `--bandpass-enabled`
- `--bandpass-low-freq`
- `--bandpass-high-freq`
- `--bandpass-q`

Example AST-style plot:

```bash
python -m src.cli.plot_mels \
  --input path/to/audio.wav \
  --out plots/audio_ast \
  --feature-type ast_fbank \
  --ast-num-mel-bins 128 \
  --ast-max-length 1024 \
  --source-type harmonic \
  --bandpass-enabled \
  --formats html
```

For `ast_fbank`, choose `--ast-max-length` close to the natural frame count for your segment duration to avoid extra compute:

- about `198` for `2 s`
- about `498-500` for `5 s`

## Model Structure

The MIL model is explicitly hierarchical:

1. Segment waveform/features
2. Whisper audio encoder token states
3. Intra-segment attention pooling
4. Segment embedding
5. Instance head -> segment logit
6. Inter-segment reducer
7. Bag logit

### Intra-segment attention

The segment encoder always uses token-level attention pooling. It is configured under:

- `model.segment_encoder.pooling.type`
- `model.segment_encoder.pooling.hidden_dim`
- `model.segment_encoder.pooling.dropout`
- `model.segment_encoder.pooling.gated`

`model.segment_encoder.pooling.type` must be `attention`.

### Inter-segment reducers

`model.mil.aggregator` selects the bag-level reducer.

Supported reducers:

- `attention`
- `max`
- `topk`

Reducer-specific config:

- `model.mil.attention.hidden_dim`
- `model.mil.attention.dropout`
- `model.mil.attention.gated`
- `model.mil.topk.k`

Retired paths:

- `mean`
- `logsumexp`
- `softmax_weighted`
- `noisy_or`
- `nonoverlap_mean`

## Encoder Adaptation

Encoder adaptation is configured under `model.segment_encoder.adaptation`:

- `mode: "frozen" | "partial" | "full"`
- `num_layers`

Primary configs use:

- `mode: "partial"`
- `num_layers: 1`

Partial unfreezing keeps most of the Whisper encoder frozen and only unfreezes:

- the last `num_layers` encoder blocks
- the final encoder layer norm

The convolutional front-end remains frozen in partial mode.

## Optimizer Layout

Training uses two AdamW parameter groups:

- encoder trainable parameters -> `train.optimizer.encoder_lr`
- non-encoder trainable parameters -> `train.optimizer.head_lr`

Shared optimizer fields:

- `train.optimizer.weight_decay`
- `train.scheduler.warmup_ratio`

Primary configs use a smaller encoder LR than head LR so the pretrained encoder adapts conservatively.

## Class Imbalance Handling

Class-imbalance handling lives under `train.loss`:

- `auto_pos_weight: true`
  - computes `negatives / positives` from the active training split
- `pos_weight`
  - optional manual override when `auto_pos_weight` is `false`
- `type: "bce" | "focal"`
- `gamma`
  - focal-loss focusing parameter

Weighted sampling remains configurable under:

- `train.sampler.weighted_random`

## Diagnostics

Diagnostics are controlled by `analysis.outputs`:

```json
"analysis": {
  "outputs": {
    "save_segment_scores": true,
    "save_instance_logits": true,
    "save_intra_attention_weights": true,
    "save_inter_attention_weights": true,
    "save_topk_indices": true,
    "save_bag_metadata": true
  }
}
```

Available diagnostics include:

- per-segment scores
- per-segment logits
- intra-segment token attention weights
- inter-segment attention weights when using attention MIL
- selected top-k segment indices when using top-k MIL
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

## Config Shape

The branch now uses a nested config schema only.

Example training shape:

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
      "bandpass": {
        "enabled": false,
        "low_hz": 250.0,
        "high_hz": 1000.0,
        "q": 0.707
      },
      "log_mel": {
        "n_fft": 400,
        "hop_length": 160,
        "win_length": 400,
        "n_mels": 80
      },
      "ast_fbank": {
        "num_mel_bins": 128,
        "max_length": 1024,
        "do_normalize": true,
        "mean": -4.2677393,
        "std": 4.5689974
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
    "segment_encoder": {
      "type": "whisper",
      "backbone": "tiny",
      "pretrained_name_or_path": "tiny",
      "strict": true,
      "download_root": null,
      "dims": {
        "n_audio_state": 384,
        "n_audio_head": 6,
        "n_audio_layer": 4
      },
      "pooling": {
        "type": "attention",
        "hidden_dim": 128,
        "dropout": 0.1,
        "gated": true
      },
      "adaptation": {
        "mode": "partial",
        "num_layers": 1
      }
    },
    "instance_head": {
      "type": "mlp",
      "hidden_dim": 256,
      "dropout": 0.1
    },
    "mil": {
      "aggregator": "attention",
      "attention": {
        "hidden_dim": 128,
        "dropout": 0.1,
        "gated": true
      },
      "topk": {
        "k": 3
      }
    }
  },
  "train": {
    "epochs": 20,
    "top_k": 3,
    "max_grad_norm": 1.0,
    "optimizer": {
      "encoder_lr": 0.00002,
      "head_lr": 0.0001,
      "weight_decay": 0.01
    },
    "scheduler": {
      "warmup_ratio": 0.05
    },
    "loss": {
      "type": "focal",
      "auto_pos_weight": true,
      "pos_weight": null
    },
    "sampler": {
      "weighted_random": false
    },
    "early_stopping": {
      "enabled": true,
      "monitor": "val_loss",
      "patience": 15,
      "min_delta": 0.0001
    }
  },
  "analysis": {
    "outputs": {
      "save_segment_scores": true,
      "save_instance_logits": true,
      "save_intra_attention_weights": true,
      "save_inter_attention_weights": true,
      "save_topk_indices": false,
      "save_bag_metadata": true
    }
  }
}
```

## Notes

- `model.segment_encoder.pretrained_name_or_path` can point to an official OpenAI Whisper checkpoint name such as `tiny`, `base`, or `small`, or to a local MIL checkpoint.
- `data.preprocessing.feature_type` selects the segment feature frontend. `log_mel` uses the repo's Whisper-style log-mel frontend, and `ast_fbank` uses a local Kaldi fbank frontend with AST normalization constants.
- `data.preprocessing.source_type` and `data.preprocessing.bandpass` remain waveform-level preprocessing and are applied before MIL segmentation for both frontends.
- `ast_fbank` is intended for scratch training or matching local checkpoints. Official OpenAI Whisper checkpoint names are only supported with `feature_type="log_mel"`.
- MIL segment length controls the encoder context length automatically; it no longer has to match Whisper’s original 30-second context.
- `python -m src.cli.pretrained_info --name_or_path tiny` prints the encoder dimensions for a given Whisper checkpoint.
- `configs/training_ast_fbank.json` is included as an AST-style scratch-training example.
