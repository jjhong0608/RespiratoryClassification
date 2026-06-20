# Project Memory

This file records durable project context that should be checked before each
task in this repository.

## Project Purpose

This repository supports experiments and manuscript writing for respiratory
disease-group prediction from auscultation audio. The target disease groups are:

- Normal Group
- Lung Parenchymal Disease Group
- Airway Disease Group

Two classification strategies are in scope:

- Direct 3-class classification: one model predicts Normal, Airway, or
  Lung_Parenchymal.
- Cascade classification: stage 1 predicts Normal vs Abnormal, then stage 2
  predicts Airway vs Lung_Parenchymal for clips routed as Abnormal.

The model families under consideration are AST, Whisper, and ResNet50. The
current runnable training and evaluation pipeline supports all three backbones
under the same disease-group experiment contract.

## Current Implementation State

- The active training pipeline is clip-level classification with a shared
  model output contract. AST uses Hugging Face `ASTModel` and local AST-style
  Kaldi fbank extraction. Whisper uses a local Whisper-like encoder and log-mel
  extraction. ResNet50 uses torchvision ResNet50 with a spectrogram-image
  frontend.
- Because the study uses transfer learning, AST and Whisper frontends should
  remain backbone-native rather than being forced into one shared spectrogram
  representation. ResNet50 likewise uses its own image-style frontend. Fair
  comparison is enforced at the split, label, checkpoint, metric, and reporting
  contract levels.
- ResNet50 preprocessing uses 15.0 s clips, 128-bin log-mel power maps, HPSS
  full/harmonic/percussive channels, bilinear resize to 224 x 224, and ImageNet
  normalization. The default ResNet50 encoder adaptation is frozen, with
  partial and full fine-tuning available through config.
- Whisper encoder adaptation is also config-driven. Existing Whisper configs
  without an explicit adaptation field default to frozen adaptation. Partial
  Whisper fine-tuning with `num_layers=1` unfreezes only the final Whisper
  encoder block after pretrained checkpoint loading.
- Binary tasks use one positive-class logit across backbones for BCE/focal
  loss, threshold optimization, evaluation, and cascade routing.
- Config files use dataclass-backed JSON loading in `src/utils/config.py`.
- Dataset loading supports multiple roots through `ConcatDataset`, and label
  names are mapped from config `label_to_index` values.
- Binary tasks support `bce` and `focal` losses. Multi-class tasks require
  `cross_entropy`.
- Evaluation computes accuracy, precision, recall, specificity, balanced
  accuracy, F1 score, ROC AUC, PR AUC, brier score, and confusion matrix.
- Binary evaluation can reuse checkpoint validation thresholds for optimized
  metrics. Multi-class evaluation reports threshold optimization as disabled.

## Disease-Group Experiment Contracts

The disease-group tasks use these names:

- Direct: `Normal_vs_Airway_vs_LungParenchymal`
- Cascade stage 1: `Normal_vs_Abnormal`
- Cascade stage 2: `Airway_vs_LungParenchymal`

The direct 3-class config is `configs/cv_run_direct_3class.json`. The preferred
cascade configs are `configs/cv_run_cascade_stage1_normal_vs_abnormal.json` for
stage 1 and `configs/cv_run_cascade_stage2_airway_vs_lung_parenchymal.json` for
stage 2. The older `configs/cv_run.json` and `configs/cv_run2.json` files are
retained as legacy aliases. All disease-group CV configs use the same disease
5-fold root under:

```text
/Users/jjhong0608/Documents/AudioData/RespiratoryClassification/DATA/DISEASE_CNUH_DATA/5_Folds/
```

Cascade comparison expects evaluated fold checkpoints and matching
`eval_metrics__best_f1_*.json` files under a task root. AST direct and cascade
configs use a family-specific root while preserving the canonical task names:

```text
Disease_Group_Results/AST/<task>/fold_<n>/
```

Whisper configs use a family-specific root:

```text
Disease_Group_Results/Whisper/<task>/fold_<n>/
```

ResNet50 configs use a family-specific root:

```text
Disease_Group_Results/ResNet50/<task>/fold_<n>/
```

Whisper and ResNet50 partial L1 configs use separate roots so they do not
overwrite frozen baseline outputs:

```text
Disease_Group_Results/Whisper_Partial_L1/<task>/fold_<n>/
Disease_Group_Results/ResNet50_Partial_L1/<task>/fold_<n>/
```

On 2026-06-19, the AST direct 3-class 5-fold CV was observed running with:

```text
PYTHONPATH=. /Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python -m src.cli.cv --config configs/cv_run_direct_3class.json
```

Do not start a duplicate process against the same AST direct result root while
that job is still active, because concurrent fold writers can overwrite
`last.pt`, `best_loss_*.pt`, `best_f1_*.pt`, and `run.log`.

On 2026-06-20, the ResNet50 direct-to-cascade 5-fold CV queue was started as a
detached background driver with:

```text
PYTHONPATH=. /Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python -m src.cli.cv --config configs/resnet50/cv_run_direct_3class.json
PYTHONPATH=. /Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python -m src.cli.cv --config configs/resnet50/cv_run_cascade_stage1_normal_vs_abnormal.json
PYTHONPATH=. /Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python -m src.cli.cv --config configs/resnet50/cv_run_cascade_stage2_airway_vs_lung_parenchymal.json
```

The first ResNet50 direct fold created `run.log`, `last.pt`,
`best_loss_*.pt`, `best_f1_*.pt`, and diagnostics. Do not start a duplicate
ResNet50 queue against `Disease_Group_Results/ResNet50/` while that driver is
still active.

The split-local `Abnormal` test directory is an overlay for
`Airway + Lung_Parenchymal` and should not be treated as a fourth final label.

## Repository Restoration Notes

This checkout was restored from a previous AST respiratory classification repo.
The repository was missing `src.plots.export`, even though multiple Plotly CLI
modules imported `PlotlyExportMixin`. The helper has been restored from the
neighboring AST repository at:

```text
/Users/jjhong0608/Documents/AudioData/RespiratoryClassification/AST/src/plots/export.py
```

The restored helper supports `html`, `png`, and `pdf` output formats.

The full pytest suite also expects `scripts/analyze_cv_results.py`. That script
has been restored from the neighboring AST repository and provides lightweight
CV metric parsing plus Markdown report helpers for evaluated fold outputs.

## Documentation Notes

- `README.md`, this file, and `docs/paper/main.tex` should remain in English.
- `README.md` should describe the current operational workflow, not historical
  experiments that are no longer runnable.
- `docs/paper/main.tex` is an initial manuscript scaffold. It should be updated
  as dataset counts, experimental results, and figures are finalized.
- Whisper and ResNet50 are implemented comparison arms in code and configs, but
  final paper numbers still need to be generated from completed experiments.
- Manuscript wording should describe AST, Whisper, and ResNet50 as pretrained
  backbone pipelines with native acoustic frontends. It should not claim that
  their spectrogram inputs were unified.
- Manuscript and README wording should distinguish frozen baseline runs from
  partial L1 runs for Whisper and ResNet50.

## Known Issues and Constraints

- `AGENTS.md` is ignored by `.gitignore`, so it is local guidance rather than a
  tracked project artifact in this checkout.
- ResNet50 uses torchvision ImageNet weights by default. Network-restricted
  environments should switch `model.encoder.weights` to `none` for smoke tests
  or pre-stage the torchvision weights cache before full training.
- ResNet50 HPSS is computed online in the frontend. If preprocessing becomes a
  bottleneck, add a feature cache as a separate optimization rather than
  changing the experiment contract.
- Whisper uses OpenAI Whisper `tiny` as the default encoder checkpoint in the
  disease-group configs. Downloaded checkpoints are resolved through the local
  Whisper loader and are not tracked in git.
- Whisper `pretrained.freeze_encoder=true` remains enabled in configs. The
  training setup applies the final adaptation policy after loading pretrained
  weights, so partial L1 can reopen the final encoder block while preserving a
  frozen default for legacy configs.
- `configs/eval.json` is a shared example. For each task, update
  `checkpoint_path`, `label_to_index`, and `eval_dirs` before running
  checkpoint evaluation.
- Result directories such as `Disease_Group_Results/`, `checkpoints/`, and
  `reports/` are ignored and should not be restored as tracked artifacts.
