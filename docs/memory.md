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
- Whisper attention visualization is implemented as time-token attribution in
  `src.cli.plot_whisper_attention`. It should not be described as an AST-style
  two-dimensional time-frequency patch explanation. The supported scores are
  `last_time_attention`, `last_time_attention_head_mean`,
  `time_attention_rollout`, and `class_gradient_time_attention`; the default is
  `last_time_attention_head_mean`.
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
configs use a partial L1 family root while preserving the canonical task names:

```text
Disease_Group_Results/AST_Partial_L1/<task>/fold_<n>/
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
Disease_Group_Results/AST_Partial_L1/<task>/fold_<n>/
Disease_Group_Results/Whisper_Partial_L1/<task>/fold_<n>/
Disease_Group_Results/ResNet50_Partial_L1/<task>/fold_<n>/
```

On 2026-06-19, the AST direct 3-class 5-fold CV was observed running with:

```text
PYTHONPATH=. /Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python -m src.cli.cv --config configs/cv_run_direct_3class.json
```

On 2026-06-22, the completed AST partial L1 result root was renamed from
`Disease_Group_Results/AST/` to `Disease_Group_Results/AST_Partial_L1/` so
the held-out evaluation roots match Whisper and ResNet50 partial L1 naming.
The explicit AST CV configs now write to `Disease_Group_Results/AST_Partial_L1/`.

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

On 2026-06-21, the Whisper and ResNet50 partial L1 direct-to-cascade queue was
started as a detached background driver. The driver PID was `64487`, and the
driver log was:

```text
Disease_Group_Results/background_logs/partial_l1_whisper_resnet50_20260621_005335.log
```

The first active command was:

```text
PYTHONPATH=. /Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python -m src.cli.cv --config configs/whisper_partial_l1/cv_run_direct_3class.json
```

The initial fold log confirmed `family=whisper`, `mode=partial`,
`num_layers=1`, and nonzero trainable encoder parameters. On 2026-06-21, this
queue completed with `ALL_DONE partial_l1_whisper_resnet50`.

The current held-out test evaluation targets only partial L1 roots and excludes
frozen Whisper/ResNet50 baselines. The target roots are:

```text
Disease_Group_Results/AST_Partial_L1/
Disease_Group_Results/Whisper_Partial_L1/
Disease_Group_Results/ResNet50_Partial_L1/
```

Each of the three roots has direct, cascade stage 1, and cascade stage 2 task
directories. Each task root has 35 checkpoints: five `last.pt`, fifteen
`best_f1_*.pt`, and fifteen `best_loss_*.pt`. The current evaluation contract
is to evaluate all generated `.pt` checkpoints, not only best-F1 checkpoints,
for a total of 315 checkpoint evaluations.

On 2026-06-22, the partial L1 held-out test evaluation completed successfully
for all 315 checkpoints. The detached driver log is:

```text
Disease_Group_Results/background_logs/partial_l1_test_evaluation_20260622_085245.log
```

The final marker was `ALL_DONE partial_l1_test_evaluation`, and each of the
nine partial L1 task roots had 35 `eval_metrics__*.json` files.

The partial L1 direct-vs-cascade comparison should run as a sequential detached
background driver for `AST_Partial_L1`, `Whisper_Partial_L1`, and
`ResNet50_Partial_L1`. Reports must be separated by model family under:

```text
Disease_Group_Results/reports/cascade_test_comparison/AST_Partial_L1/
Disease_Group_Results/reports/cascade_test_comparison/Whisper_Partial_L1/
Disease_Group_Results/reports/cascade_test_comparison/ResNet50_Partial_L1/
```

Plotly comparison figures should be exported as `html`, `png`, `pdf`, and
`json`; the `json` files preserve the serialized Plotly figures for later
auditing and regeneration.

On 2026-06-22, the partial L1 direct-vs-cascade comparison completed for
`AST_Partial_L1`, `Whisper_Partial_L1`, and `ResNet50_Partial_L1`. These
reports are now reflected in `docs/paper/main.tex` as a held-out test results
table. The test manifest contains 332 clips (`Normal=205`, `Airway=49`,
`Lung_Parenchymal=78`), and the comparison files contain 1660 prediction rows
because the same test manifest is evaluated across five fold-selected
checkpoint sets. The current macro-F1 means are:

```text
AST_Partial_L1: direct=0.858692, cascade=0.839969
Whisper_Partial_L1: direct=0.636406, cascade=0.643897
ResNet50_Partial_L1: direct=0.634061, cascade=0.596285
```

The manuscript should describe the cascade effect as backbone-dependent:
direct classification is stronger for AST and ResNet50, while Whisper shows a
small cascade advantage under the current partial L1 setting.

For qualitative Airway error analysis, `src.cli.plot_ast_vs_whisper_airway_attention`
selects fold-level cases from the AST and Whisper partial L1
`cascade_test_predictions.csv` files where `true_label == "Airway"`, AST
`direct_pred_label == "Airway"`, and Whisper
`direct_pred_label == "Normal"`. The comparison is intentionally based on
direct 3-class predictions, not `cascade_final_pred_label`. The execution unit
is `(fold, audio_path)` because each fold points to a different direct
checkpoint. The default output root is:

```text
Disease_Group_Results/reports/ast_vs_whisper_airway_attention/
```

The same CLI also supports `--selection both_correct_airway` for the shared
success group where `true_label == "Airway"`, AST `direct_pred_label ==
"Airway"`, and Whisper `direct_pred_label == "Airway"`. The current report
contains 95 fold-level cases and 26 unique audio files for this selection. The
recommended separate output root is:

```text
Disease_Group_Results/reports/ast_vs_whisper_airway_attention_both_correct/
```

For broader qualitative inspection, `src.cli.plot_ast_vs_whisper_3class_attention_cases`
selects high-confidence direct three-class cases across `Normal`, `Airway`, and
`Lung_Parenchymal`. The default output root is:

```text
Disease_Group_Results/reports/ast_vs_whisper_3class_attention_cases/
```

The default selection sets are `ast_correct_high_confidence`,
`ast_wrong_high_confidence`, `whisper_correct_high_confidence`, and
`whisper_wrong_high_confidence`. Each selection set chooses five unique audio
files per true label, for 60 cases by default. Correctness is always based on
`direct_pred_label`, not cascade outputs. Confidence is the probability assigned
to the reference model's own direct prediction. If the same audio appears in
multiple folds, the representative row is the fold with the highest reference
confidence, with deterministic tie-breaking by fold and audio path. The selected
row's fold-specific AST and Whisper direct checkpoints are then used to render
both attention outputs for the same audio.

The AST/Whisper Airway attention overlap analysis is implemented as a
post-processing CLI that reads the two qualitative attention output roots and
does not reload models or regenerate attention maps:

```text
PYTHONPATH=. /Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python -m src.cli.analyze_ast_whisper_attention_overlap --failure-root Disease_Group_Results/reports/ast_vs_whisper_airway_attention --both-correct-root Disease_Group_Results/reports/ast_vs_whisper_airway_attention_both_correct --out-dir Disease_Group_Results/reports/ast_vs_whisper_airway_attention_overlap --top-k 10 --formats html,json
```

The analysis projects AST top patches and Whisper top time spans to temporal
intervals, ignores frequency, merges overlapping intervals within each model,
and computes overlap metrics from the unioned top-10 intervals. AST maximum
time is defined as `patch_geometry.max_length * 0.01` seconds from AST
metadata. Whisper interval portions beyond that AST maximum time do not
contribute to overlap, but they remain in the Whisper denominator and are
reported as `whisper_outside_ast_time_sec` and
`whisper_outside_ast_ratio`. On 2026-06-22, the analysis completed for 204
fold-level cases: 109 `ast_correct_whisper_normal` cases and 95
`both_correct_airway` cases. The current mean temporal IoU values are 0.016028
and 0.017066, respectively, with median temporal IoU equal to 0.0 in both
groups.

The auscultation-by-diagnosis held-out accuracy heatmap report is implemented
in `src.cli.plot_model_accuracy_by_auscultation_diagnosis_heatmap`. It joins
the disease dataset catalog test rows to the AST, Whisper, and ResNet50 partial
L1 `cascade_test_predictions.csv` files and writes six model/strategy figures
under:

```text
Disease_Group_Results/reports/model_accuracy_by_auscultation_diagnosis_heatmap/
```

The report uses fold-level prediction rows as the calculation unit. With the
current partial L1 comparison files, each model contributes 1660 prediction
rows, the joined direct-plus-cascade audit table contains 9960 rows, and the
summary table contains 192 cells (`3 models x 2 strategies x 4 auscultation
labels x 8 diagnosis labels`). The y-axis order is `non-specific`, `crackle`,
`rhonchi`, `wheeze`; the x-axis order is `healthy`, `lung cancer`,
`lung nodule`, `pneumonia`, `IPF`, `ILD except IPF`, `COPD`, `asthma`. The
figure adds a diagnosis-group annotation band above the x-axis: `healthy` maps
to `Normal`, `lung cancer` through `ILD except IPF` map to
`Lung Parenchymal`, and `COPD` plus `asthma` map to `Airway`. This group band
is display metadata only; cell accuracy remains computed at the
`auscultation x diagnosis` level. Cells show accuracy plus `correct/total`;
zero-denominator cells are displayed as `N/A<br>0/0`. The current run produced
`html`, `json`, `png`, and `pdf` files for all six heatmaps with no excluded
joined rows.
The diagnosis-group annotation band is aligned as a compact header immediately
above the heatmap grid. The intended paper-domain gap between the heatmap top
and the band bottom is 0.015, so the group labels read as column-group headers
rather than a detached legend. The group-label annotations use explicit center
anchors and a small upward pixel shift so the text is visually centered inside
the band in static PNG/PDF exports.

Task/backbone-specific held-out eval configs are stored under
`configs/eval/{ast,whisper,resnet50}_partial_l1/`. They all use the held-out
test root:

```text
/Users/jjhong0608/Documents/AudioData/RespiratoryClassification/DATA/DISEASE_CNUH_DATA/5_Folds/test/
```

The label mappings are fixed by task:

```text
Direct:  Normal=0, Airway=1, Lung_Parenchymal=2
Stage 1: Normal=0, Abnormal=1
Stage 2: Airway=0, Lung_Parenchymal=1
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

The restored helper supports `html`, `png`, `pdf`, and Plotly `json` output
formats.

The full pytest suite also expects `scripts/analyze_cv_results.py`. That script
has been restored from the neighboring AST repository and provides lightweight
CV metric parsing plus Markdown report helpers for evaluated fold outputs.

## Documentation Notes

- `README.md`, this file, and `docs/paper/main.tex` should remain in English.
- `README.md` should describe the current operational workflow, not historical
  experiments that are no longer runnable.
- `docs/paper/main.tex` is a manuscript scaffold with the current partial L1
  held-out direct-vs-cascade results. It should be updated as patient-level
  dataset counts, error analyses, and final tracked figures are finalized.
- Whisper and ResNet50 are implemented comparison arms in code and configs.
  The current manuscript includes partial L1 held-out direct-vs-cascade
  numbers; frozen-baseline numbers should remain separate unless explicitly
  added as another comparison arm.
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
- `configs/eval.json` is a shared example. The current partial L1 held-out
  evaluation uses the nine task/backbone-specific configs under
  `configs/eval/`.
- Result directories such as `Disease_Group_Results/`, `checkpoints/`, and
  `reports/` are ignored and should not be restored as tracked artifacts.
