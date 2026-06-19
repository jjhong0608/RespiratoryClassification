# Respiratory Disease-Group Classification

This repository supports experiments and manuscript preparation for respiratory
disease-group prediction from auscultation audio.

The final disease groups are:

- `Normal`
- `Airway`
- `Lung_Parenchymal`

Two prediction strategies are the current focus:

- **Direct 3-class classification**: one model predicts the final label directly.
- **Cascade classification**: stage 1 predicts `Normal` vs `Abnormal`; stage 2
  predicts `Airway` vs `Lung_Parenchymal` for clips routed as `Abnormal`.

The runnable implementation in this checkout is AST-based. Whisper-based and
ResNet50-based disease-group pipelines are planned extensions and are not yet
implemented.

## Setup

Use the respiratory mamba environment:

```bash
mamba activate respiratory
pip install -r requirements.txt
```

The project instructions expect Python commands to run through:

```text
/Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python
```

Most commands can also be run as `python -m ...` after activating the same
environment.

## Repository Layout

- `src/`: core package code.
- `src/cli/`: training, evaluation, cross-validation, catalog, and plotting CLIs.
- `configs/`: JSON configs for training, evaluation, and cross-validation.
- `scripts/analyze_cv_results.py`: lightweight CV metrics summary helper.
- `docs/memory.md`: durable project notes that should be checked before work.
- `docs/paper/main.tex`: initial manuscript scaffold.
- `tests/`: pytest tests.
- `checkpoints/` and `Disease_Group_Results/`: ignored runtime outputs.

## Current Model Scope

The active pipeline is clip-level Audio Spectrogram Transformer (AST)
classification:

- One `.wav` file is one training example.
- Audio is loaded as mono, resampled to 16 kHz when needed, and converted to
  AST-style Kaldi fbank features.
- The encoder is Hugging Face `ASTModel`.
- The classifier head is implemented in this repository and supports `linear`
  and `mlp` heads.
- Encoder adaptation is controlled by `model.encoder.adaptation` with
  `frozen`, `partial`, or `full` modes.

Whisper and ResNet50 are not active training backbones in this checkout. Add
them later under the same dataset, metric, and reporting contract after the AST
workflow is stable.

## Dataset Contract

The disease-group data is expected under:

```text
/Users/jjhong0608/Documents/AudioData/RespiratoryClassification/DATA/DISEASE_CNUH_DATA/
```

Cross-validation configs use:

```text
/Users/jjhong0608/Documents/AudioData/RespiratoryClassification/DATA/DISEASE_CNUH_DATA/5_Folds/
```

The final labels are `Normal`, `Airway`, and `Lung_Parenchymal`. The split-local
`Abnormal` directory is an overlay for `Airway + Lung_Parenchymal` and is used
for the cascade stage 1 binary task. It is not a fourth final class.

Dataset loading supports multiple root directories through `ConcatDataset`.
Within each root, wav files are discovered recursively and mapped by parent
directory name through the config `data.label_to_index`.

## Cross-Validation Experiments

Run the direct 3-class disease-group AST experiment:

```bash
PYTHONPATH=. python -m src.cli.cv --config configs/cv_run_direct_3class.json
```

Run cascade stage 1:

```bash
PYTHONPATH=. python -m src.cli.cv --config configs/cv_run_cascade_stage1_normal_vs_abnormal.json
```

Run cascade stage 2:

```bash
PYTHONPATH=. python -m src.cli.cv --config configs/cv_run_cascade_stage2_airway_vs_lung_parenchymal.json
```

Each fold is trained independently under:

```text
Disease_Group_Results/<experiment.name>/fold_<n>/
```

Training writes:

- `last.pt`
- `best_loss_*.pt`
- `best_f1_*.pt`
- `run.log`
- `diagnostics/val_epoch_*.jsonl` when diagnostics are enabled

## Checkpoint Evaluation

Evaluate one checkpoint with:

```bash
PYTHONPATH=. python -m src.cli.evaluate --config configs/eval.json
```

`configs/eval.json` is a shared example. Before evaluating a specific task,
update:

- `checkpoint_path`
- `data.eval_dirs`
- `data.label_to_index`

Evaluation writes files next to the checkpoint:

- `eval_metrics.json`
- `eval_predictions.csv`
- `eval_diagnostics.jsonl` when diagnostics are enabled

Evaluate many checkpoints under one result root with:

```bash
PYTHONPATH=. python -m src.cli.evaluate_all \
  --config configs/eval.json \
  --root Disease_Group_Results/Normal_vs_Airway_vs_LungParenchymal
```

The cascade comparison expects `eval_metrics__best_f1_*.json` files next to the
matching `best_f1_*.pt` checkpoints for each direct, stage 1, and stage 2 fold.
The older `configs/cv_run.json` and `configs/cv_run2.json` files are retained
as legacy aliases for the two cascade stages, but the explicit cascade config
filenames above are the preferred entrypoints.

For a lightweight Markdown summary of evaluated fold metrics, use
`scripts/analyze_cv_results.py` as the current report helper API.

## Direct vs Cascade Test Comparison

After evaluating fold checkpoints for all three disease-group tasks, compare
direct 3-class inference against the two-stage cascade:

```bash
PYTHONPATH=. python -m src.cli.disease_group_cascade_eval
```

The comparison uses:

- direct: `Normal_vs_Airway_vs_LungParenchymal`
- stage 1: `Normal_vs_Abnormal`
- stage 2: `Airway_vs_LungParenchymal`

For each `fold_0` through `fold_4`, the script selects the evaluated checkpoint
with the highest saved `optimized_metrics.f1_score` among
`eval_metrics__best_f1_*.json` files. It evaluates the same held-out
`Normal`, `Airway`, and `Lung_Parenchymal` clips for the direct and cascade
paths, and validates that the `Abnormal` overlay matches
`Airway + Lung_Parenchymal` by filename.

Default outputs are written under:

```text
Disease_Group_Results/reports/cascade_test_comparison/
```

Key artifacts:

- `cascade_test_predictions.csv`
- `cascade_test_fold_metrics.csv`
- `cascade_test_summary.json`
- `cascade_test_summary.md`
- `direct_confusion_matrix.*`
- `cascade_confusion_matrix.*`
- `cascade_stage_flow.*`

## Visual Summaries and Figures

Build a richer Plotly summary bundle from cascade comparison outputs and CV
summary context:

```bash
PYTHONPATH=. python -m src.cli.plot_disease_group_visual_summary
```

Default output:

```text
Disease_Group_Results/reports/visual_summary/
```

Build a one-page structure schematic for direct 3-class versus cascade routing:

```bash
PYTHONPATH=. python -m src.cli.build_direct_vs_cascade_structure_svg
```

Default output:

```text
Disease_Group_Results/reports/structure_comparison/
```

The structure figure is about decision routing, not performance. It should
remain suitable for manuscript figures that explain direct 3-class inference,
cascade stage 1, cascade stage 2, and the shared final label space.

Visualize AST attention for a single wav and checkpoint:

```bash
PYTHONPATH=. python -m src.cli.plot_ast_attention \
  --checkpoint Disease_Group_Results/Normal_vs_Airway_vs_LungParenchymal/fold_0/best_f1_0.844886.pt \
  --wav /path/to/sample.wav \
  --out-dir Disease_Group_Results/reports/ast_attention_visualization \
  --attention-method last_cls_patch_head_mean \
  --visualization both \
  --top-k 10 \
  --formats html,png,pdf
```

## Disease Dataset Catalog

Build a file-level catalog joining disease-group wav inventory, Excel
annotations, and 5-fold/test membership:

```bash
PYTHONPATH=. python -m src.cli.build_disease_dataset_catalog
```

Default output:

```text
/Users/jjhong0608/Documents/AudioData/RespiratoryClassification/DATA/DISEASE_CNUH_DATA/disease_dataset_catalog/
```

Key artifacts:

- `disease_dataset_catalog.csv`
- `disease_dataset_catalog_summary.json`
- `disease_dataset_catalog_warnings.csv`
- `disease_dataset_catalog_run.log`

Build Plotly figures from the generated catalog:

```bash
PYTHONPATH=. python -m src.cli.plot_disease_dataset_catalog
```

## Config Rules

The AST config schema is JSON and loaded through dataclasses.

Core audio and feature settings:

```json
{
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
  }
}
```

Loss behavior depends on class count:

- Binary tasks (`len(label_to_index) == 2`): `bce` or `focal`.
- Multi-class tasks (`len(label_to_index) > 2`): `cross_entropy`.

Training uses AdamW with separate parameter groups:

- encoder trainable parameters: `train.optimizer.encoder_lr`
- classifier trainable parameters: `train.optimizer.head_lr`

The scheduler is cosine annealing with linear warmup, controlled by
`train.scheduler.warmup_ratio`.

## Metrics and Diagnostics

Evaluation computes:

- accuracy
- precision
- recall
- specificity
- balanced accuracy
- F1 score
- ROC AUC
- PR AUC
- brier score
- confusion matrix

Diagnostics are controlled by `analysis.outputs`:

```json
{
  "analysis": {
    "outputs": {
      "save_logits": true,
      "save_probabilities": true,
      "save_embeddings": false,
      "save_clip_metadata": true
    }
  }
}
```

Training diagnostics are saved under:

```text
<run_dir>/diagnostics/val_epoch_XXX.jsonl
```

Evaluation diagnostics are saved under:

```text
<checkpoint_dir>/eval_diagnostics.jsonl
```

## Development Checks

Run the project checks with the respiratory Python:

```bash
/Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python -m pytest -q
/Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python -m ruff check src
/Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python -m ruff format src
/Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python -m mypy src
```

For plotting and cascade import smoke tests:

```bash
/Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python -m pytest \
  tests/test_plot_mels.py \
  tests/test_plot_ast_attention.py \
  tests/test_plot_disease_group_visual_summary.py \
  tests/test_plot_disease_dataset_catalog.py \
  tests/test_disease_group_cascade_eval.py \
  -q
```

## Known Issues

- `AGENTS.md` is currently ignored by `.gitignore`, so it is local guidance in
  this checkout rather than a tracked project artifact.
- Whisper and ResNet50 disease-group experiments are documented as planned work
  but are not implemented.
- Runtime output directories such as `Disease_Group_Results/`, `checkpoints/`,
  `reports/`, and root-level `plots/` are ignored by git.
- `configs/eval.json` is a reusable example rather than a task-specific eval
  config. Adjust it before each checkpoint evaluation.
