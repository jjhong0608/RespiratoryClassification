# Whisper Respiratory Sound Classification

This repo adds a **Whisper-encoder-based** audio classifier (training, evaluation, and k-fold CV) on top of the vendored upstream code in `whisper_origin/`.

## Setup

```bash
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Train

```bash
python -m src.cli.training --config configs/training.json
```

Artifacts are written under `checkpoints/<run_name>/` (models, logs, copied config, plots).

## Evaluate

```bash
python -m src.cli.evaluate --config configs/eval.json
```

This writes `eval_metrics.json` and `eval_predictions.csv` under the checkpoint directory.
The CSV includes each evaluation sample's `filename`, `predicted_probability`, `predicted_label`, and `predicted_index`.

## Cross-Validation

```bash
python -m src.cli.cv --config configs/cv_run.json
```

## Fine-Tune (Unfrozen Encoder)

```bash
python -m src.cli.finetune --config configs/finetune.json
```

## Supervised Contrastive Pretrain

```bash
python -m src.cli.supcon_pretrain --config configs/supcon_pretrain.json
```

This stage trains `encoder + projection head` with supervised contrastive loss and saves an encoder-only checkpoint. The projection head is discarded after pretraining, so the saved `last.pt` / `best_loss_*.pt` files are meant for downstream transfer rather than exact resume of the pretraining stage.

For stage-2 classification, set `pretrained.name_or_path` in `configs/training.json` to the saved SupCon checkpoint:

- `freeze_encoder: true` trains only the downstream classifier on top of the pretrained encoder
- `freeze_encoder: false` initializes from the pretrained encoder and fine-tunes the classifier model end to end

## Repair Dataset Symlinks

If a symlink-based dataset was generated with the wrong relative path depth, you can repair broken `.wav` links in place:

```bash
python -m src.cli.repair_dataset_symlinks --root datasets/crackle_wheeze_datasets
```

This CLI scans `*.wav` symlinks under the given root, keeps valid links unchanged, and rewrites broken links that should resolve under the sibling `DataProcessing/` directory.

## Plot Results

If you have aggregated fold results in `results/results.json` (format: `model -> fold -> metrics`), you can generate an interactive Plotly report and static exports:

```bash
python -m src.cli.plot_results --input results/results.json --out plots/results_metrics
```

This writes `plots/results_metrics.html`, `plots/results_metrics.png`, and `plots/results_metrics.pdf` (PNG/PDF require `kaleido`).

Note: `kaleido` v1+ also requires Google Chrome for static export; if PNG/PDF export fails, run `plotly_get_chrome` and retry.

Alternatively, you can let the CLI attempt installing Chrome:

```bash
python -m src.cli.plot_results --input results/results.json --out plots/results_metrics --install_chrome
```

## Plot Single Result

If you have fold-wise metrics for one result (format: `fold -> metrics`), you can plot a single-panel bar chart with error bars across folds:

```bash
python -m src.cli.plot_single_results --input results/single_results.json --out plots/single_results_metrics
```

This writes `plots/single_results_metrics.html`, `plots/single_results_metrics.png`, and `plots/single_results_metrics.pdf` (PNG/PDF require `kaleido`).
The figure uses automatic y-axis ranges for both sides (not fixed to zero), with a separate right y-axis for `brier_score`; that right axis is inverted so smaller values appear higher.

## Plot Mel Spectrograms

To visualize the exact mel-spectrogram preprocessing used by the model for one `.wav` file:

```bash
python -m src.cli.plot_mels --input datasets/example.wav --out plots/example_mel --formats html,png,pdf
```

To process a directory recursively and save one figure per `.wav` while preserving relative paths under the output directory:

```bash
python -m src.cli.plot_mels --input datasets/for_wheeze/remove_crackle/test --out plots/mels --formats html
```

This CLI uses the project preprocessing path (`resample -> bandpass -> source_type -> Whisper-like log-mel`), so the figure reflects what the model actually sees. It supports `original`, `harmonic`, and `percussive` via `--source-type`, and static `png` / `pdf` export uses the same Plotly + kaleido path as the other plotting CLIs.

## Configuration Files

This repo uses JSON configs under `configs/`.

### Dataset Layout

Each root directory is expected to be organized as:

```
<root>/
  <label_name>/
    *.wav
```

`<label_name>` must exist in `label_to_index`, and `label_to_index` must be contiguous `0..N-1`.

### Common Audio Settings

Configs specify `sample_rate`, `clip_seconds`, `source_type`, and optional `bandpass`; audio is resampled (if needed), can be band-pass filtered, optionally decomposed (HPSS), and then padded/trimmed to a fixed length.

`source_type` controls which waveform is used before mel-spectrogram extraction:

- `original`: raw waveform (default)
- `harmonic`: harmonic component from HPSS
- `percussive`: percussive component from HPSS

`bandpass` optionally applies waveform-domain noise reduction before HPSS / mel extraction. The current recommended example range for respiratory-sound experiments in this repo is `250-1000 Hz`.

HPSS verification note: the repo now includes a librosa-backed regression test that compares the project HPSS output against `librosa.effects.hpss` on deterministic synthetic signals. This is intended to validate the harmonic/percussive separation step itself, not downstream classifier accuracy.

Important: `model.n_audio_ctx` must match the derived context length for the chosen audio settings:

- `n_samples = sample_rate * clip_seconds`
- `n_frames = n_samples / hop_length` (Whisper-like defaults use `hop_length=160`)
- `n_audio_ctx = n_frames / 2` (encoder conv stride=2)

With `sample_rate=16000` and `clip_seconds=30`, this is `n_audio_ctx=1500`.

### `configs/training.json`

Top-level fields:

| Key | Type | Meaning |
| --- | --- | --- |
| `run_name` | string | Output subdir name under `output_dir/` |
| `seed` | int | Random seed |
| `device` | string | Torch device (e.g. `cpu`, `mps`, `cuda`) |
| `output_dir` | string | Root output directory (default pattern: `checkpoints/`) |
| `top_k` | int | Keep top-k best checkpoints by validation loss |
| `pretrained` | object/null | Optional pretrained Whisper encoder loader |
| `data` | object | Data loading settings |
| `model` | object | Encoder + Hugging Face-style audio-classification head settings |
| `training` | object | Optimizer/schedule settings |
| `imbalance` | object | Optional binary-class imbalance settings |
| `threshold_optimization` | object | Optional validation-threshold optimization settings for binary tasks |

`pretrained` fields (optional):

| Key | Type | Meaning |
| --- | --- | --- |
| `name_or_path` | string | Local `.pt` path or OpenAI model name (e.g. `tiny`, `base`, `small`, `medium`, `large-v3`) |
| `load_encoder_only` | bool | Only loads the Whisper `encoder.*` weights (must be `true`) |
| `strict` | bool | Strict key matching when loading encoder weights |
| `freeze_encoder` | bool | If `true`, freezes encoder parameters (`requires_grad=false`) |
| `download_root` | string/null | Cache directory for official model downloads (defaults to `~/.cache/whisper`) |

Note: using an official model name downloads the checkpoint on first run and reuses the cached file on later runs.
The same field also accepts local encoder-only checkpoints produced by `src.cli.supcon_pretrain`.

To discover the correct encoder dims for a model name, run:

```bash
python -m src.cli.pretrained_info --name_or_path tiny
```

To list supported official names:

```bash
python -m src.cli.pretrained_info --list_models
```

`data` fields:

| Key | Type | Meaning |
| --- | --- | --- |
| `train_dirs` | list[string] | One or more dataset roots for training |
| `val_dirs` | list[string] | One or more dataset roots for validation |
| `label_to_index` | object | Label name → class index |
| `sample_rate` | int | Target sample rate |
| `clip_seconds` | number | Fixed clip duration |
| `source_type` | string | Audio source for mel (`original`, `harmonic`, `percussive`) |
| `bandpass` | object | Optional waveform band-pass filter (`enabled`, `low_freq`, `high_freq`, `q`) |
| `batch_size` | int | Batch size |
| `num_workers` | int | DataLoader workers |

`data.bandpass` fields:

| Key | Type | Meaning |
| --- | --- | --- |
| `enabled` | bool | Apply band-pass filtering before HPSS / mel extraction |
| `low_freq` | number/null | Lower cutoff frequency in Hz |
| `high_freq` | number/null | Upper cutoff frequency in Hz |
| `q` | number | Biquad Q value (default `0.707`) |

Validation:

- If `enabled=true`, both cutoff frequencies must be set.
- The config must satisfy `0 < low_freq < high_freq < sample_rate / 2`.

`model` fields:

| Key | Type | Meaning |
| --- | --- | --- |
| `n_mels` | int | Mel bins (80 typical) |
| `n_audio_ctx` | int | Encoder input context length (must match audio settings) |
| `n_audio_state` | int | Encoder embedding dim |
| `n_audio_head` | int | Attention heads |
| `n_audio_layer` | int | Encoder layers |
| `pooling` | string | Readout mode: `mean` or `cls` |
| `use_weighted_layer_sum` | bool | If `true`, learn a weighted sum over the encoder input embedding state plus every transformer block output before projection |
| `classifier_proj_size` | int | Projection size before the final readout and linear classifier |

Notes:

- New training runs use a Hugging Face-style Whisper audio-classification head: optional weighted layer sum -> token-wise projection -> readout (`mean` or `cls`) -> linear classifier.
- `pooling="cls"` uses a transfer-friendly CLS token design: audio tokens keep their original positional encoding, and a separate learnable CLS token plus CLS positional parameter are prepended before the transformer blocks.
- Older checkpoints with legacy `linear` / `mlp` heads still load for evaluation and fine-tuning.

`training` fields:

| Key | Type | Meaning |
| --- | --- | --- |
| `epochs` | int | Training epochs |
| `learning_rate` | number | AdamW LR |
| `weight_decay` | number | AdamW weight decay |
| `warmup_ratio` | number | Warmup steps ratio (of total steps) |
| `max_grad_norm` | number | Gradient clipping |

`imbalance` fields (optional, binary tasks only):

| Key | Type | Meaning |
| --- | --- | --- |
| `auto_pos_weight` | bool | If `true`, sets BCE `pos_weight = negatives / positives` from the training set |
| `pos_weight` | number/null | Manual BCE positive-class weight; do not set together with `auto_pos_weight` |
| `sampler` | string | Train loader sampler: `none` or `weighted_random` |

Notes:

- `pos_weight` changes the binary BCE loss only during training.
- `weighted_random` uses train-set inverse-frequency sampling and is never applied to validation or evaluation loaders.
- Imbalance options are currently supported only for `num_classes = 2`.

`threshold_optimization` fields (optional, binary tasks only):

| Key | Type | Meaning |
| --- | --- | --- |
| `enabled` | bool | If `true`, optimize the binary decision threshold on each validation epoch |
| `metric` | string | Threshold objective: `f1`, `balanced_accuracy`, or `youden_j` |

Notes:

- Threshold optimization uses validation probabilities only; it is never fit on evaluation/test data.
- The selected threshold is saved into each checkpoint and reused automatically by `src.cli.evaluate`.
- Checkpoint ranking remains based on validation loss, not the threshold objective.
- In binary classification, `balanced_accuracy` and `youden_j` select the same optimum up to tie-breaking; both names are supported for clarity.

### `configs/supcon_pretrain.json`

Top-level fields:

| Key | Type | Meaning |
| --- | --- | --- |
| `run_name` | string | Output subdir name under `output_dir/` |
| `seed` | int | Random seed |
| `device` | string | Torch device |
| `output_dir` | string | Root output directory |
| `top_k` | int | Keep top-k best checkpoints by validation SupCon loss |
| `pretrained` | object/null | Optional encoder initializer (same schema as training, but `freeze_encoder` must be `false`) |
| `data` | object | Same audio/data settings used by standard training |
| `encoder` | object | Whisper encoder dimensions for the pretraining backbone |
| `training` | object | Optimizer/schedule settings |
| `supervised_contrastive` | object | SupCon loss, projection-head, and augmentation settings |

`encoder` fields:

| Key | Type | Meaning |
| --- | --- | --- |
| `n_mels` | int | Mel bins |
| `n_audio_ctx` | int | Encoder input context length (must match audio settings) |
| `n_audio_state` | int | Encoder embedding dim |
| `n_audio_head` | int | Attention heads |
| `n_audio_layer` | int | Encoder layers |

`supervised_contrastive` fields:

| Key | Type | Meaning |
| --- | --- | --- |
| `temperature` | number | SupCon temperature |
| `normalize` | bool | L2-normalize projected embeddings before similarity |
| `projection_head` | object | Temporary MLP projection head used only during pretraining |
| `augmentation` | object | Two-view spectrogram augmentation settings |

`supervised_contrastive.projection_head` fields:

| Key | Type | Meaning |
| --- | --- | --- |
| `hidden_dim` | int | Hidden size of the temporary projection MLP |
| `output_dim` | int | Output embedding size used by the SupCon loss |

`supervised_contrastive.augmentation` fields:

| Key | Type | Meaning |
| --- | --- | --- |
| `time_mask_param` | int | Maximum width of each time mask |
| `time_mask_count` | int | Number of time masks per view |
| `freq_mask_param` | int | Maximum width of each frequency mask |
| `freq_mask_count` | int | Number of frequency masks per view |
| `gaussian_noise_std` | number | Standard deviation of additive Gaussian noise on log-mel features |

Notes:

- Pretraining uses two augmented views per sample and applies supervised contrastive loss with label-based positives.
- The projection head is not saved in the resulting checkpoint; only the encoder is saved for downstream transfer.
- Downstream stage-2 training should use `src.cli.training` with the saved checkpoint wired through `pretrained.name_or_path`.

### `configs/eval.json`

| Key | Type | Meaning |
| --- | --- | --- |
| `device` | string | Torch device |
| `checkpoint_path` | string | Path to `*.pt` produced by this repo’s trainer |
| `data` | object | Evaluation dataset settings (`eval_dirs`, `label_to_index`, `sample_rate`, `clip_seconds`, `source_type`, `bandpass`, `batch_size`, `num_workers`) |
| `threshold` | object | Optional manual binary decision-threshold override |
| `unsafe_pickle_load` | bool | Optional: allow `torch.load(weights_only=False)` for legacy checkpoints (only if you trust the checkpoint) |

Note: evaluation reconstructs the model from `model_cfg` stored inside the checkpoint (saved by `src.cli.training` / `src.cli.cv`).

`threshold` fields:

| Key | Type | Meaning |
| --- | --- | --- |
| `manual` | number/null | Manual binary threshold override in `[0.0, 1.0]` |

Threshold resolution priority for binary evaluation:

- If `threshold.manual` is set, evaluation uses that value.
- Otherwise, if the checkpoint contains `threshold_optimization_result`, evaluation uses the stored validation-selected threshold.
- Otherwise, evaluation falls back to the default `0.5` threshold.

Evaluation writes both `decision_threshold` and `decision_threshold_source` into `eval_metrics.json`. The source is one of `manual`, `checkpoint`, or `default`.

If you have older checkpoints created before this repo stored configs as plain dicts, evaluation may fail under PyTorch’s default `weights_only=True`. In that case, set `unsafe_pickle_load=true` in `configs/eval.json` only if you trust the checkpoint.

### `configs/cv_run.json`

Cross-validation is a thin wrapper that runs training per fold.

| Key | Type | Meaning |
| --- | --- | --- |
| `run_name` | string | Output subdir name under `output_dir/` |
| `seed` | int | Random seed |
| `device` | string | Torch device |
| `output_dir` | string | Root output directory |
| `top_k` | int | Keep top-k per-fold best checkpoints |
| `pretrained` | object/null | Optional pretrained Whisper encoder loader (same schema as training) |
| `folds` | list[object] | Fold specs (`name`, `train_dirs`, `val_dirs`) |
| `label_to_index` | object | Label name → class index |
| `sample_rate` | int | Target sample rate |
| `clip_seconds` | number | Fixed clip duration |
| `source_type` | string | Audio source for mel (`original`, `harmonic`, `percussive`) |
| `bandpass` | object | Optional waveform band-pass filter (`enabled`, `low_freq`, `high_freq`, `q`) |
| `batch_size` | int | Batch size |
| `num_workers` | int | DataLoader workers |
| `model` | object | Same as training config |
| `training` | object | Same as training config |
| `imbalance` | object | Optional binary-class imbalance settings (same schema as training) |
| `threshold_optimization` | object | Optional validation-threshold optimization settings (same schema as training) |
