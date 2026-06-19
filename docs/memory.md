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
current runnable training and evaluation pipeline implements AST only.
Whisper-based and ResNet50-based disease-group experiments are planned
extensions, not active implementations in this checkout.

## Current Implementation State

- The active training pipeline is clip-level AST classification using
  Hugging Face `ASTModel` and local AST-style Kaldi fbank extraction.
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

The disease-group AST tasks use these names:

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
`eval_metrics__best_f1_*.json` files under:

```text
Disease_Group_Results/<task>/fold_<n>/
```

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

## Known Issues and Constraints

- `AGENTS.md` is ignored by `.gitignore`, so it is local guidance rather than a
  tracked project artifact in this checkout.
- Whisper and ResNet50 disease-group pipelines are not implemented yet.
- `configs/eval.json` is a shared example. For each task, update
  `checkpoint_path`, `label_to_index`, and `eval_dirs` before running
  checkpoint evaluation.
- Result directories such as `Disease_Group_Results/`, `checkpoints/`, and
  `reports/` are ignored and should not be restored as tracked artifacts.
