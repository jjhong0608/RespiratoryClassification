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

## Disease-Group Cascade Test Comparison

Compare the direct 3-class disease-group classifier against a two-stage cascade
on the held-out disease-group test split:

```bash
PYTHONPATH=. python -m src.cli.disease_group_cascade_eval
```

The script compares:

- direct: `Normal_vs_Airway_vs_LungParenchymal`
- cascade: `Normal_vs_Abnormal` followed by `Airway_vs_LungParenchymal`

For each `fold_0` through `fold_4`, it selects the checkpoint with the highest
saved `optimized_metrics.f1_score` among `eval_metrics__best_f1_*.json` files,
then evaluates the same `Normal`, `Airway`, and `Lung_Parenchymal` test clips.
The split-local `Abnormal` overlay is validated for consistency but is not
counted as a fourth final label.

Outputs are written by default under:

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

Build a richer visual summary bundle from the cascade comparison CSV/JSON files
and the existing disease-group CV summaries:

```bash
PYTHONPATH=. python -m src.cli.plot_disease_group_visual_summary
```

The visual summary is written under:

```text
Disease_Group_Results/reports/visual_summary/
```

It includes metric bars and fold-wise plots, row-normalized confusion matrices,
confusion deltas, per-label outcome charts, cascade flow/error plots,
probability/confidence distributions, CV context plots, and a
`visual_summary_index.md` manifest for browsing the generated figures.

Build a one-page SVG schematic that compares the **structure** of direct
3-class inference and the two-stage cascade without reporting performance
metrics:

```bash
PYTHONPATH=. python -m src.cli.build_direct_vs_cascade_structure_svg
```

The structure figure is written under:

```text
Disease_Group_Results/reports/structure_comparison/
```

Key artifacts:

- `direct_vs_cascade_structure_comparison.svg`
- `direct_vs_cascade_structure_comparison.html`
- `direct_vs_cascade_structure_comparison_metadata.json`
- `build_direct_vs_cascade_structure_svg.log`

PNG/PDF exports are also written when a local SVG converter such as
`rsvg-convert` or `inkscape` is available.

## Disease Dataset Catalog

Build a file-level catalog that joins the disease-group wav inventory, Excel
annotations, and 5-fold/test membership:

```bash
PYTHONPATH=. python -m src.cli.build_disease_dataset_catalog
```

The catalog is written under:

```text
/Users/jjhong0608/Documents/AudioData/RespiratoryClassification/DATA/DISEASE_CNUH_DATA/disease_dataset_catalog/
```

Key artifacts:

- `disease_dataset_catalog.csv`
- `disease_dataset_catalog_summary.json`
- `disease_dataset_catalog_warnings.csv`
- `disease_dataset_catalog_run.log`

The primary row unit is one wav file under `Normal`, `Airway`, or
`Lung_Parenchymal`. `Abnormal` is treated as the derived binary group
`Airway + Lung_Parenchymal` and is validated without adding duplicate rows.

Build a Plotly visual summary bundle from the generated catalog CSV/JSON files:

```bash
PYTHONPATH=. python -m src.cli.plot_disease_dataset_catalog
```

The figure bundle is written under:

```text
/Users/jjhong0608/Documents/AudioData/RespiratoryClassification/DATA/DISEASE_CNUH_DATA/disease_dataset_catalog/visual_summary/
```

It includes label/split/annotation distributions, row-normalized crosstab
heatmaps, fold-vs-test deltas, Sankey flows, metadata coverage, warning
summaries, and an index/manifest for browsing all generated figures.

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

## Plot AST Attention

Visualize AST patch attention for one `.wav` file and one trained checkpoint:

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

Supported attention methods:

- `last_cls_patch`: final-layer CLS-to-patch attention from one head.
- `last_cls_patch_head_mean`: final-layer CLS-to-patch attention averaged over heads.
- `attention_rollout`: residual row-normalized attention rollout across all layers.
- `class_gradient_attention`: target-class `ReLU(attention * gradient)` patch scores.

Supported visualization modes:

- `rectangle`: top-k patch rectangles over the AST fbank.
- `heatmap`: upsampled patch attention overlay over the AST fbank.
- `both`: write both overlays.

The command writes `*_fbank.*`, `*_attention_heatmap_overlay.*`,
`*_attention_rectangle_overlay.*`, `*_top_patches.csv`, and
`*_attention_metadata.json` under `--out-dir`.
