# Codex Instructions: Add AUG0–AUG3 Data Augmentation for H0 Branch-Gated Model

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

The current branch is already an event-MIL-first respiratory sound classifier. Do **not** rewrite the model architecture. Do **not** reintroduce Hugging Face `ASTModel`. Do **not** reintroduce latent-query pooling.

This task adds **train-only data augmentation** so that the following four experiments can be run:

| Experiment | Waveform augmentation | Fbank augmentation |
|---|---:|---:|
| **AUG0** | off | off |
| **AUG1** | on | off |
| **AUG2** | off | on |
| **AUG3** | on | on |

The model for all four configs must be the **H0 branch-aware gated pooling model**:

```text
4-scale Event-MIL-RDT
evidence_pooling.type = "branch_gated"
RDT enabled
RDT steps = 3
top_tokens_per_branch = 2
BCE loss
branch auxiliary enabled
branch auxiliary weight = 0.1
encoder_lr = 1e-5
head_lr = 3e-4
no warm-start
```

If `branch_gated` evidence pooling is already implemented, use it. If it is not implemented in the current branch, do **not** silently fall back to mean pooling. Either implement it according to the prior H0 branch-gated pooling instructions or fail loudly.

---

## Current Data Pipeline to Preserve

The current data pipeline is approximately:

```text
path
  -> WaveformLoader.load(path)
  -> WaveformPreprocessor.prepare(waveform)
  -> AstLikeFbank(clip_waveform)
  -> transpose(0, 1).contiguous()
  -> ClipSample(input_values=feature_map, ...)
```

Important current behavior:

- `WaveformLoader` loads `.wav`, converts multichannel audio to mono, and resamples to the configured sample rate.
- `WaveformPreprocessor.prepare(...)` trims audio, applies optional bandpass filtering, and applies source type selection.
- `AstLikeFbank(...)` computes Kaldi fbank features.
- `AstLikeFbank` normalizes internally when `do_normalize = true`.
- `AstLikeFbank` returns `[F, T]`.
- `dataset.py` then applies `.transpose(0, 1).contiguous()` to produce `[T, F]`.
- Model input remains:

```text
input_values: [B, 1024, 128]
```

Do not change the model input shape.

---

## Augmentation Placement

### Waveform-level augmentation

Apply waveform augmentation **after** waveform preprocessing and **before** fbank extraction:

```text
waveform = WaveformLoader.load(path)
clip_waveform = WaveformPreprocessor.prepare(waveform)
clip_waveform = waveform_augmenter(clip_waveform)     # train only
feature = AstLikeFbank(clip_waveform)
```

Reason:

```text
This keeps mono conversion, resampling, trimming, optional bandpass, and source-type preprocessing deterministic, then applies train-time acoustic variation before fbank extraction.
```

### Fbank-level augmentation

Apply fbank augmentation **after** fbank extraction and after dataset transpose to `[T, F]`:

```text
feature_map = AstLikeFbank(clip_waveform).transpose(0, 1).contiguous()
feature_map = fbank_augmenter(feature_map)            # train only
```

Reason:

```text
AstLikeFbank normalizes internally. Applying fbank masks after transpose means augmentation sees the exact model input shape [1024, 128].
```

Use:

```text
mask_value = 0.0
```

as the default because the features are already normalized and zero is a neutral normalized value.

---

## Train-Only Rule

Augmentation must be applied **only** to the training split.

Required behavior:

```text
split == "train" and data.augmentation.enabled == true
  -> augmentation may be applied according to sub-configs

split == "val"
  -> no augmentation

split == "eval"
  -> no augmentation
```

This must also hold for cross-validation:

```text
CV train fold -> augmentation may be active
CV validation fold -> augmentation off
```

Do not augment validation, evaluation, or test data.

---

# Part A — Add Augmentation Config Schema

Modify:

```text
src/utils/config.py
```

Add config dataclasses for data augmentation.

Recommended schema:

```python
@dataclass(frozen=True)
class RandomGainConfig:
    enabled: bool = False
    probability: float = 0.5
    min_db: float = -3.0
    max_db: float = 3.0


@dataclass(frozen=True)
class AdditiveNoiseConfig:
    enabled: bool = False
    probability: float = 0.3
    snr_db_min: float = 15.0
    snr_db_max: float = 30.0


@dataclass(frozen=True)
class TimeShiftConfig:
    enabled: bool = False
    probability: float = 0.5
    max_shift_fraction: float = 0.05
    mode: Literal["zero_pad", "roll"] = "zero_pad"


@dataclass(frozen=True)
class WaveformAugmentationConfig:
    enabled: bool = False
    probability: float = 1.0
    gain: RandomGainConfig = field(default_factory=RandomGainConfig)
    noise: AdditiveNoiseConfig = field(default_factory=AdditiveNoiseConfig)
    time_shift: TimeShiftConfig = field(default_factory=TimeShiftConfig)


@dataclass(frozen=True)
class MaskConfig:
    enabled: bool = False
    num_masks: int = 1
    max_width: int = 32


@dataclass(frozen=True)
class FbankAugmentationConfig:
    enabled: bool = False
    probability: float = 0.5
    time_mask: MaskConfig = field(default_factory=MaskConfig)
    freq_mask: MaskConfig = field(default_factory=MaskConfig)
    mask_value: float = 0.0


@dataclass(frozen=True)
class DataAugmentationConfig:
    enabled: bool = False
    waveform: WaveformAugmentationConfig = field(default_factory=WaveformAugmentationConfig)
    fbank: FbankAugmentationConfig = field(default_factory=FbankAugmentationConfig)
```

Add to `DataConfig`:

```python
augmentation: DataAugmentationConfig = field(default_factory=DataAugmentationConfig)
```

Parsing rules:

- If `"augmentation"` is omitted, default to disabled.
- If `"augmentation.enabled" = false`, both waveform and fbank augmentation are globally disabled.
- If `"augmentation.enabled" = true`, each sub-augmentation still obeys its own `enabled` flag.

### Validation rules

Add validation:

```text
all probabilities in [0, 1]
gain.min_db <= gain.max_db
noise.snr_db_min > 0
noise.snr_db_min <= noise.snr_db_max
time_shift.max_shift_fraction >= 0
time_shift.max_shift_fraction < 1
time_shift.mode in {"zero_pad", "roll"}
mask.num_masks >= 0
mask.max_width >= 0
fbank.probability in [0, 1]
```

Do not reject `max_width = 0`; it can represent a no-op.

If `max_width` is larger than feature dimensions, either reject during validation if dimensions are available, or clamp safely at runtime.

Preferred runtime behavior:

```text
width = min(sampled_width, feature_dim)
```

---

# Part B — Add `src/data/augmentation.py`

Create a new file:

```text
src/data/augmentation.py
```

Implement:

```python
class WaveformAugmenter:
    def __init__(self, cfg: DataAugmentationConfig, sample_rate: int) -> None: ...
    def __call__(self, waveform: Tensor) -> Tensor: ...


class FbankAugmenter:
    def __init__(self, cfg: DataAugmentationConfig) -> None: ...
    def __call__(self, feature: Tensor) -> Tensor: ...
```

## General requirements

All augmentation functions must:

```text
preserve shape
preserve dtype
preserve device
avoid NaN/Inf
be train-time stochastic
be no-op when disabled
```

Use PyTorch random operations where possible:

```python
torch.rand
torch.randint
torch.empty(...).uniform_
torch.randn_like
```

Reason:

```text
This works more naturally with PyTorch/DataLoader seeding than mixing Python random or NumPy.
```

---

## WaveformAugmenter

Input:

```text
waveform: [T]
```

Output:

```text
waveform: [T]
```

Apply sub-transforms in this order:

```text
1. random gain
2. additive noise
3. time shift
```

Each sub-transform should apply only when:

```text
global augmentation enabled
waveform enabled
global waveform probability passed
sub-transform enabled
sub-transform probability passed
```

Recommended logic:

```python
if not cfg.augmentation.enabled or not cfg.augmentation.waveform.enabled:
    return waveform

if torch.rand(()) > cfg.augmentation.waveform.probability:
    return waveform

x = waveform
if gain enabled and probability passes:
    x = apply_gain(x)

if noise enabled and probability passes:
    x = apply_noise_snr(x)

if time_shift enabled and probability passes:
    x = apply_time_shift(x)

return x
```

### Random gain

Use dB scale:

```python
gain_db = uniform(min_db, max_db)
gain = 10 ** (gain_db / 20)
waveform = waveform * gain
```

Do not clip by default. If clipping is introduced, document it.

### Additive noise by SNR

Use:

```python
signal_power = waveform.pow(2).mean().clamp_min(1e-12)
snr_db = uniform(snr_db_min, snr_db_max)
noise_power = signal_power / (10 ** (snr_db / 10))
noise = torch.randn_like(waveform) * noise_power.sqrt()
waveform = waveform + noise
```

### Time shift

Default mode:

```text
zero_pad
```

`roll` is supported but should not be the default.

For `zero_pad`:

```text
positive shift: move waveform later, pad left with zeros
negative shift: move waveform earlier, pad right with zeros
```

Preserve length exactly.

`max_shift_fraction` defines:

```python
max_shift = int(round(max_shift_fraction * waveform.numel()))
```

If `max_shift == 0`, return waveform unchanged.

---

## FbankAugmenter

Input:

```text
feature: [T, F]
```

Default H0:

```text
feature: [1024, 128]
```

Output:

```text
feature: [T, F]
```

Apply:

```text
time masks
frequency masks
```

Each transform should apply only when:

```text
global augmentation enabled
fbank enabled
fbank probability passed
mask enabled
```

### Time masking

For each mask:

```python
width = randint(0, max_width + 1)
start = randint(0, T - width + 1)
feature[start:start + width, :] = mask_value
```

If `width == 0`, no-op.

### Frequency masking

For each mask:

```python
width = randint(0, max_width + 1)
start = randint(0, F - width + 1)
feature[:, start:start + width] = mask_value
```

If `width == 0`, no-op.

### Mutation safety

Avoid modifying cached tensors in-place unless the feature is freshly created per `__getitem__`.

The current feature map is freshly created, so in-place masking is acceptable, but clone defensively if needed:

```python
feature = feature.clone()
```

---

# Part C — Modify Dataset

Modify:

```text
src/data/dataset.py
```

Current relevant path:

```text
waveform = self._waveform_loader.load(path)
clip_waveform = self._preprocessor.prepare(waveform)
feature_map = self._feature_extractor(clip_waveform).transpose(0, 1).contiguous()
```

Change to:

```python
waveform = self._waveform_loader.load(path)
clip_waveform = self._preprocessor.prepare(waveform)

if self._waveform_augmenter is not None:
    clip_waveform = self._waveform_augmenter(clip_waveform)

feature_map = self._feature_extractor(clip_waveform).transpose(0, 1).contiguous()

if self._fbank_augmenter is not None:
    feature_map = self._fbank_augmenter(feature_map)
```

Update constructors.

Recommended:

```python
class _RespiratoryClipRootDataset(...):
    def __init__(
        self,
        cfg: DataConfig,
        root: str,
        *,
        split: str = "train",
        apply_augmentation: bool = False,
    ):
        ...
        self.split = split
        self.apply_augmentation = bool(apply_augmentation)
        self._waveform_augmenter = (
            WaveformAugmenter(cfg.augmentation, sample_rate=cfg.audio.sample_rate)
            if self.apply_augmentation and cfg.augmentation.enabled and cfg.augmentation.waveform.enabled
            else None
        )
        self._fbank_augmenter = (
            FbankAugmenter(cfg.augmentation)
            if self.apply_augmentation and cfg.augmentation.enabled and cfg.augmentation.fbank.enabled
            else None
        )
```

For `RespiratoryClipDataset`:

```python
class RespiratoryClipDataset(...):
    def __init__(
        self,
        cfg: DataConfig,
        roots: Sequence[str],
        *,
        split: str = "train",
        apply_augmentation: bool = False,
    ):
        ...
        self._datasets = [
            _RespiratoryClipRootDataset(
                cfg,
                root,
                split=split,
                apply_augmentation=apply_augmentation,
            )
            for root in self.roots
            if Path(root).exists()
        ]
```

---

# Part D — Modify Dataset Builder

Modify:

```text
src/data/loaders.py
```

Current:

```python
dataset = RespiratoryClipDataset(cfg, split_to_roots[split])
```

Change to:

```python
apply_augmentation = split == "train" and cfg.augmentation.enabled

dataset = RespiratoryClipDataset(
    cfg,
    split_to_roots[split],
    split=split,
    apply_augmentation=apply_augmentation,
)
```

This ensures:

```text
train split -> augmentation controlled by config
val split -> no augmentation
eval split -> no augmentation
```

Cross-validation should automatically inherit this because the CV path calls `build_dataset(..., split="train")` and `build_dataset(..., split="val")`.

---

# Part E — CLI Logging

No major changes should be required in:

```text
src/cli/training.py
src/cli/cv.py
src/cli/evaluate.py
```

But add logging if convenient.

Recommended logging during training/CV setup:

```text
augmentation.enabled
augmentation.waveform.enabled
augmentation.fbank.enabled
waveform gain/noise/time_shift settings
fbank time_mask/freq_mask settings
```

Evaluation should never augment data.

Do not add augmentation to evaluation CLI.

---

# Part F — Add AUG0–AUG3 Configs

Create these configs:

```text
configs/training_aug0_h0_gated_no_aug.json
configs/training_aug1_h0_gated_waveform_aug.json
configs/training_aug2_h0_gated_fbank_aug.json
configs/training_aug3_h0_gated_waveform_fbank_aug.json
```

Use an existing working H0 branch-gated config as the base.

If no H0 branch-gated config exists yet, create one using the H0 model settings plus:

```json
"model": {
  "encoder": {
    "architecture": {
      "evidence_pooling": {
        "type": "branch_gated",
        "gate_hidden_size": null,
        "dropout": 0.1,
        "temperature": 1.0
      }
    }
  }
}
```

## Common model settings for AUG0–AUG3

All AUG configs must use the same model/training settings.

```json
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
      "evidence_pooling": {
        "type": "branch_gated",
        "gate_hidden_size": null,
        "dropout": 0.1,
        "temperature": 1.0
      },
      "rdt": {
        "enabled": true,
        "steps": 3,
        "top_tokens_per_branch": 2,
        "gated_residual": true,
        "layerscale_init": 0.01
      },
      "patch_branches": [
        { "patch_size": [16, 16], "stride": [8, 16] },
        { "patch_size": [8, 32], "stride": [4, 32] },
        { "patch_size": [4, 64], "stride": [2, 64] },
        { "patch_size": [2, 128], "stride": [1, 128] }
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
```

Training settings:

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
    "gamma": 2.0,
    "branch_auxiliary": {
      "enabled": true,
      "weight": 0.1,
      "aggregation": "mean"
    }
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

Keep dataset paths and local settings copied from current working configs.

Do not invent local paths.

---

## AUG0 — No Augmentation

File:

```text
configs/training_aug0_h0_gated_no_aug.json
```

Experiment name:

```json
"experiment": {
  "name": "respiratory_aug0_h0_gated_no_aug"
}
```

Augmentation config:

```json
"data": {
  "augmentation": {
    "enabled": false
  }
}
```

AUG0 should reproduce H0 branch-gated without data augmentation.

---

## AUG1 — Waveform Augmentation Only

File:

```text
configs/training_aug1_h0_gated_waveform_aug.json
```

Experiment name:

```json
"experiment": {
  "name": "respiratory_aug1_h0_gated_waveform_aug"
}
```

Augmentation config:

```json
"data": {
  "augmentation": {
    "enabled": true,
    "waveform": {
      "enabled": true,
      "probability": 0.8,
      "gain": {
        "enabled": true,
        "probability": 0.5,
        "min_db": -3.0,
        "max_db": 3.0
      },
      "noise": {
        "enabled": true,
        "probability": 0.3,
        "snr_db_min": 15.0,
        "snr_db_max": 30.0
      },
      "time_shift": {
        "enabled": true,
        "probability": 0.5,
        "max_shift_fraction": 0.05,
        "mode": "zero_pad"
      }
    },
    "fbank": {
      "enabled": false
    }
  }
}
```

Do not add time stretch or pitch shift in this first round.

Do not add EQ/filter perturbation in this first round unless already implemented cleanly.

---

## AUG2 — Fbank Augmentation Only

File:

```text
configs/training_aug2_h0_gated_fbank_aug.json
```

Experiment name:

```json
"experiment": {
  "name": "respiratory_aug2_h0_gated_fbank_aug"
}
```

Augmentation config:

```json
"data": {
  "augmentation": {
    "enabled": true,
    "waveform": {
      "enabled": false
    },
    "fbank": {
      "enabled": true,
      "probability": 0.5,
      "time_mask": {
        "enabled": true,
        "num_masks": 1,
        "max_width": 32
      },
      "freq_mask": {
        "enabled": true,
        "num_masks": 1,
        "max_width": 8
      },
      "mask_value": 0.0
    }
  }
}
```

No time warping.

Keep masks weak.

---

## AUG3 — Waveform + Fbank Augmentation

File:

```text
configs/training_aug3_h0_gated_waveform_fbank_aug.json
```

Experiment name:

```json
"experiment": {
  "name": "respiratory_aug3_h0_gated_waveform_fbank_aug"
}
```

Use waveform settings from AUG1 and fbank settings from AUG2:

```json
"data": {
  "augmentation": {
    "enabled": true,
    "waveform": {
      "enabled": true,
      "probability": 0.8,
      "gain": {
        "enabled": true,
        "probability": 0.5,
        "min_db": -3.0,
        "max_db": 3.0
      },
      "noise": {
        "enabled": true,
        "probability": 0.3,
        "snr_db_min": 15.0,
        "snr_db_max": 30.0
      },
      "time_shift": {
        "enabled": true,
        "probability": 0.5,
        "max_shift_fraction": 0.05,
        "mode": "zero_pad"
      }
    },
    "fbank": {
      "enabled": true,
      "probability": 0.5,
      "time_mask": {
        "enabled": true,
        "num_masks": 1,
        "max_width": 32
      },
      "freq_mask": {
        "enabled": true,
        "num_masks": 1,
        "max_width": 8
      },
      "mask_value": 0.0
    }
  }
}
```

---

# Part G — README Updates

Update README with a section:

```text
Data Augmentation
```

Document:

```text
AUG0: no augmentation
AUG1: waveform augmentation only
AUG2: fbank augmentation only
AUG3: waveform + fbank augmentation
```

Explain placement:

```text
Waveform augmentation is applied after waveform preprocessing and before AST fbank extraction.
Fbank augmentation is applied after normalized fbank extraction and transpose to [1024,128].
Augmentation is applied only to train split.
Validation, evaluation, and test splits are never augmented.
```

Add config table:

| Study | Config | Purpose |
|---|---|---|
| AUG0 | `configs/training_aug0_h0_gated_no_aug.json` | H0 gated baseline without augmentation |
| AUG1 | `configs/training_aug1_h0_gated_waveform_aug.json` | waveform augmentation only |
| AUG2 | `configs/training_aug2_h0_gated_fbank_aug.json` | fbank augmentation only |
| AUG3 | `configs/training_aug3_h0_gated_waveform_fbank_aug.json` | combined waveform + fbank augmentation |

Add warning:

```text
AUG0–AUG3 are intended to isolate augmentation effects. Keep model, optimizer, loss, early stopping, and seed policy fixed across the four configs.
```

---

# Part H — Tests

Add or update tests.

## 1. Config parsing tests

Ensure these configs parse:

```text
configs/training_aug0_h0_gated_no_aug.json
configs/training_aug1_h0_gated_waveform_aug.json
configs/training_aug2_h0_gated_fbank_aug.json
configs/training_aug3_h0_gated_waveform_fbank_aug.json
```

Test default behavior:

```text
omitting data.augmentation -> augmentation disabled
```

Test invalid config rejection:

```text
probability < 0
probability > 1
gain.min_db > gain.max_db
noise.snr_db_min <= 0
noise.snr_db_min > noise.snr_db_max
time_shift.max_shift_fraction < 0
time_shift.max_shift_fraction >= 1
time_shift.mode invalid
mask.num_masks < 0
mask.max_width < 0
```

---

## 2. WaveformAugmenter tests

Test:

```text
disabled augmenter returns identical waveform
output shape equals input shape
dtype preserved
device preserved
random gain changes amplitude when forced probability=1
noise adds finite values when forced probability=1
time shift zero_pad preserves length
no NaN/Inf
```

Use small synthetic waveforms.

---

## 3. FbankAugmenter tests

Test:

```text
disabled augmenter returns identical feature
output shape equals input shape [T,F]
time mask with probability=1 masks time rows
freq mask with probability=1 masks frequency columns
mask_value respected
no NaN/Inf
```

Use small synthetic `[T, F]` tensors to avoid heavy tests.

---

## 4. Train-only dataset tests

Test that augmentation is only applied to train split.

Suggested approach:

- Use a small temporary fake `.wav` dataset if existing tests already do this.
- Or monkeypatch/dummy augmenters to make changes deterministic.

Assertions:

```text
build_dataset(cfg, split="train") uses augmentation when enabled
build_dataset(cfg, split="val") does not use augmentation
build_dataset(cfg, split="eval") does not use augmentation
```

If direct waveform comparison is hard due to fbank extraction, test dataset attributes:

```text
dataset.apply_augmentation == True for train
dataset.apply_augmentation == False for val/eval
```

or expose a minimal property only for tests if acceptable.

---

## 5. AUG0 regression test

When:

```json
"augmentation": {
  "enabled": false
}
```

the dataset path should behave as before.

At minimum:

```text
feature shape remains [1024,128]
model forward still succeeds with AUG0 config
```

---

# Part I — Acceptance Criteria

The task is complete when:

1. `data.augmentation` config schema exists.
2. Missing `data.augmentation` defaults to disabled.
3. Waveform augmentation is implemented.
4. Fbank augmentation is implemented.
5. Waveform augmentation runs after waveform preprocessing and before fbank extraction.
6. Fbank augmentation runs after fbank extraction and transpose to `[T,F]`.
7. Augmentation applies only to train split.
8. Val/eval/test data are never augmented.
9. AUG0–AUG3 configs exist.
10. AUG0 uses H0 branch-gated model with augmentation off.
11. AUG1 uses H0 branch-gated model with waveform augmentation only.
12. AUG2 uses H0 branch-gated model with fbank augmentation only.
13. AUG3 uses H0 branch-gated model with waveform + fbank augmentation.
14. All AUG configs keep model/loss/optimizer/early-stopping settings fixed except augmentation.
15. README documents augmentation placement and AUG0–AUG3.
16. Tests pass.

---

# Part J — Avoid These Mistakes

- Do not augment validation/evaluation/test data.
- Do not apply augmentation inside the model.
- Do not alter H0 model architecture.
- Do not switch from branch-gated pooling to mean pooling in AUG configs.
- Do not change RDT settings.
- Do not change branch auxiliary settings.
- Do not change patch geometry.
- Do not add time stretch or pitch shift in the first augmentation round.
- Do not add strong SpecAugment masks.
- Do not use `roll` time shift as default.
- Do not use Python random or NumPy random unless worker seeding is explicitly handled.
- Do not change dataset paths.
- Do not make augmentation enabled by default.
- Do not break existing configs that omit `data.augmentation`.

---

# Part K — Suggested Execution Order After Implementation

Run:

```bash
pytest
```

Then train:

```bash
python -m src.cli.training --config configs/training_aug0_h0_gated_no_aug.json
python -m src.cli.training --config configs/training_aug1_h0_gated_waveform_aug.json
python -m src.cli.training --config configs/training_aug2_h0_gated_fbank_aug.json
python -m src.cli.training --config configs/training_aug3_h0_gated_waveform_fbank_aug.json
```

Compare using:

```text
best_loss checkpoints
F1@0.5
F1@opt
ROC-AUC
PR-AUC
Brier
balanced accuracy
optimal threshold
```

Do not choose the winner from a single best seed if seed replicates are later added. This AUG0–AUG3 stage is the first augmentation screening round.

---

# Final Note

The goal of this task is not to find the final augmentation policy immediately.

The goal is to add a clean train-only augmentation framework and run the first controlled comparison:

```text
AUG0: none
AUG1: waveform only
AUG2: fbank only
AUG3: waveform + fbank
```

All four must use the same H0 branch-aware gated pooling model so the effect of augmentation is isolated.
