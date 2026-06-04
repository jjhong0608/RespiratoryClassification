# Multi-Scale RDT-AST Respiratory Classification

This repository trains and evaluates clip-level respiratory sound classifiers
with a config-driven PyTorch pipeline. The active encoder is
`multiscale_rdt_ast`: a multi-scale AST-style frontend with branch adapters,
RDT refinement, MIL evidence heads, and optional class-aware branch gating.

The current documentation is intentionally limited to the maintained config
surface and model behavior. Historical experiment catalogs are not kept in
`README.md`; experiment outputs belong under `checkpoints/`.

## Setup

Use the project respiratory environment:

```bash
mamba activate respiratory
```

or run commands directly with the known interpreter:

```bash
PYTHONPATH=. /Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python -m pytest tests/test_ast_config.py
```

The dataset paths are provided by JSON config files. One `.wav` file is treated
as one clip-level sample.

## Config Files

Only the active config files are kept under `configs/`.

| File | Role |
|---|---|
| `configs/training_CNUH_new_test_CNUH_3classes.json` | Current CNUH 3-class class-aware training run. |
| `configs/training_CNUH.json` | Baseline CNUH 3-class training config. |
| `configs/cv_multiscale_rdt.json` | Cross-validation config for disease-group experiments. |
| `configs/eval_multiscale_rdt.json` | Evaluation config for a saved checkpoint. |

Training and CV configs share the same main schema: `experiment`, `data`,
`model`, `train`, and `analysis`. CV adds `folds`; evaluation adds
`checkpoint_path` and `threshold_optimization`.

## Running

Run training:

```bash
PYTHONPATH=. /Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python src/cli/training.py --config configs/training_CNUH_new_test_CNUH_3classes.json
```

Run the baseline training config:

```bash
PYTHONPATH=. /Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python src/cli/training.py --config configs/training_CNUH.json
```

Run cross-validation:

```bash
PYTHONPATH=. /Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python src/cli/cv.py --config configs/cv_multiscale_rdt.json
```

Run evaluation:

```bash
PYTHONPATH=. /Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python src/cli/evaluate.py --config configs/eval_multiscale_rdt.json
```

Outputs are written under `experiment.output_dir/experiment.name/` unless the
CLI config specifies a fold-specific subdirectory.

## Training Logs

Each training run keeps three log levels under the run directory:

- `run.log`: human-readable epoch summaries. Each epoch is written as a
  multi-line block with train/validation metrics, active loss components, gate
  state, adaptive branch-objective state, and the validation diagnostics path.
- `logs/metrics_epoch.jsonl`: epoch-level metrics, class-wise recall/F1,
  confusion summaries such as `wheeze->normal`, threshold optimization state,
  and the diagnostics path.
- `logs/loss_components_epoch.jsonl`: raw loss measurements and weighted loss
  contributions for train and validation.
- `logs/adaptive_state_epoch.jsonl`: current label-wise branch margins,
  regret thresholds, train violation rates, train eligible rates, and EMA state.

Inactive components, for example gate entropy outside its configured epoch
window, are hidden from `run.log` to keep the block readable. JSONL payloads,
checkpoint histories, and validation diagnostics keep the numeric state needed
for later analysis.

## Model Architecture

`model.encoder.type` must be `multiscale_rdt_ast`.

The encoder consumes local AST-style fbank tensors shaped by
`data.preprocessing.ast_fbank`. The current path does not use Hugging Face
`ASTModel`; the frontend, branch processing, and evidence pooling are local
modules.

The main components are:

- `patch_branches`: multi-scale patch streams over the fbank input.
- `shared_stem_depth`: common Transformer blocks shared before branch-specific processing.
- `adapter_depth`: branch-specific Transformer depth after the shared stem.
- `rdt`: recurrent refinement over branch evidence tokens.
- `branch MIL heads`: branch-level instance logits, attention, and pooled evidence.
- `branch_binary_auxiliary`: optional normal-vs-abnormal branch supervision.
- `evidence_pooling`: combines branch evidence into model-level evidence.
- `classifier`: in class-aware mode, acts as the global residual classifier.

`model.encoder.adaptation.mode` controls trainability. `full` trains the
encoder; `frozen` freezes the encoder adaptation path while leaving heads and
new task-specific modules trainable according to model construction.

## Evidence Pooling and Gates

`model.encoder.architecture.evidence_pooling.type` selects how branch evidence
is combined.

- `mean`: averages branch evidence without a learned evidence gate.
- `branch_gated`: learns one branch gate per sample, shaped `[B, R]`.
- `class_aware_branch_gated`: learns one gate per sample and class, shaped `[B, C, R]`.

In `class_aware_branch_gated`, the model computes class-specific evidence
embeddings and class evidence logits. The final logits are:

```text
residual_term = global_residual_logits
if global_residual.bounding.enabled:
  residual_term = bound * tanh(global_residual_logits / temperature)

final_logits = class_evidence_logits + effective_residual_scale * residual_term
```

The global residual path reuses `model.classifier`. Its input combines
class-aware gated evidence features and class-gated branch-logit features.

### Class Gate Options

The active class-aware gate config lives under
`model.encoder.architecture.evidence_pooling.class_gate`.

- `mode="query"` uses learnable class queries.
- `scorer="diagonal"` scores each class-specific evidence embedding with a class-specific diagonal scorer.
- `evidence_scorer.type="two_tower_mlp"` scores class evidence with separate embedding and branch-feature towers before fusion.
- `global_residual.enabled` controls whether the residual classifier contributes to final logits.
- `global_residual.warmup` can hold residual contribution near zero early and increase it later.
- `global_residual.bounding` can bound residual logits with `bound * tanh(residual / temperature)` before final-logit fusion.
- `evidence_auxiliary.enabled` adds cross entropy on `class_evidence_logits`.
- `branch_logit_feature.mode="raw"` feeds raw class-gated branch logits to the residual path.
- `branch_logit_feature.mode="hardest_negative_margin"` feeds label-free class margin features.
- `gate_mixing.enabled` mixes uniform gates into learned gates during early epochs.

Gate mixing uses:

```text
used_gate = alpha * uniform_gate + (1 - alpha) * learned_gate
```

Diagnostics keep both the used gate and learned gate when the model produces
both fields.

## Config Shape

### `experiment`

Defines run identity and execution target.

- `name`: output directory name under `output_dir`.
- `task`: task label stored with checkpoints and logs.
- `mode`: currently `clip`.
- `seed`: random seed.
- `device`: `cpu`, `mps`, or CUDA device string.
- `output_dir`: checkpoint and log root.
- `logging.terminal_width`: Rich terminal output width. Use `null` for dynamic
  terminal detection, or a positive integer such as `120` for fixed-width
  terminal rendering. This does not affect `run.log`, JSONL logs, diagnostics,
  checkpoints, or model behavior.

### `data`

Defines splits, labels, frontend preprocessing, and augmentation.

- `train_dirs`, `val_dirs`, `eval_dirs`: one or more directories of `.wav` files.
- `label_to_index`: public label mapping used by loss and metrics.
- `batch_size`, `num_workers`: dataloader settings.
- `audio`: sample rate and clip duration.
- `preprocessing.ast_fbank`: mel-bin count, sequence length, normalization mean, and std.
- `augmentation`: optional waveform and fbank augmentation policy.

CV configs keep `data.train_dirs` and `data.val_dirs` empty at top level and
fill them through `folds`.

### `model`

Defines encoder architecture and classifier head.

The class-aware CNUH config uses a deeper adapter than shared stem so branch
specialization can develop after a small common frontend. `classifier.pooling`
remains `latent_mean`; in class-aware mode this classifier is the residual
classifier rather than the only logits source.

### `train`

Defines epochs, optimizer, scheduler, checkpoint initialization, sampling,
early stopping, and losses.

Use `train.loss.type="cross_entropy"` for multiclass tasks and for
two-label tasks that intentionally use `class_aware_branch_gated`. Use
`bce` or `focal` only for one-logit binary models.

### `analysis`

Controls saved diagnostics:

- `save_logits`
- `save_probabilities`
- `save_embeddings`
- `save_clip_metadata`

## Loss Rules

The trainer computes a main loss and then adds enabled auxiliary or
regularization terms. Disabled terms keep backward-compatible defaults.

### Main Loss

- `cross_entropy`: uses integer class labels and logits shaped `[B, C]`.
- `bce`: uses one-logit binary output and sigmoid probabilities.
- `focal`: uses one-logit binary output with focal weighting.

Threshold optimization is meaningful only for one-logit binary output.
Class-aware two-label CE uses softmax plus argmax.

### Class Weighting and Label Smoothing

`class_weighting` can compute train-derived class weights for CE losses.
`label_smoothing` applies CE label smoothing when enabled.

### Branch Auxiliary Losses

`branch_auxiliary` applies the main task loss to branch logits when enabled.
`branch_binary_auxiliary` applies a separate normal-vs-abnormal branch BCE
target. This binary auxiliary is independent of the main multiclass CE target.

### Evidence Auxiliary Loss

`class_gate.evidence_auxiliary` applies CE directly to `class_evidence_logits`.
It is available for `class_aware_branch_gated` and is meant to improve the
class evidence scorer before the residual classifier dominates.

`class_gate.evidence_scorer.type="two_tower_mlp"` converts class-aware evidence
embeddings into `class_evidence_logits` with independent per-class towers:

```text
embedding_tower: class_evidence_embeddings -> LayerNorm -> Linear -> GELU -> Dropout
branch_feature_tower: tanh(branch_features / temperature) -> Linear -> GELU -> Dropout
fusion: concat(embedding_hidden, branch_hidden) -> LayerNorm -> Linear -> GELU -> Dropout -> Linear
```

The branch tower always receives two label-free features:

- `class_gated_branch_logit_features`: the same class-wise branch-logit feature used by the residual path.
- `top_branch_margin_style`: class-wise `max_r(branch_logit[c] - max_{j != c} branch_logit[j])`.

This path is inference-safe because it does not use true labels. It does not
class-wise normalize branch features; `branch_feature_transform.mode="tanh"`
keeps absolute branch support information while bounding the feature scale.

### Gate Regularization

`gate_entropy_regularization` adds an entropy bonus as a negative loss term:

```text
loss += -weight * mean_entropy
```

Supported targets include aggregate evidence gates, class-aware gates, true
class gates, and learned class-aware gates when the model outputs them.

`class_gate_diversity_regularization` encourages class gate distributions to
be different from each other through pairwise JS divergence. It is a soft
regularizer, not a hard cap on gate weights.

Both regularizers can be bounded by `start_epoch` and `end_epoch`.

### Margin Losses

`class_evidence_margin` applies margin ranking to `class_evidence_logits`.

`class_gated_branch_logit_margin` applies margin ranking to
`class_gated_branch_logits`.

`branch_to_evidence_ranking_consistency` compares a detached branch-side
teacher gap against the same true-vs-hardest-negative gap from
`class_evidence_logits`. The teacher can come from
`class_gated_branch_logits`, or from `source="top_branch_margin"`, which uses
the best branch-level true-vs-hardest-negative margin as the teacher signal.
`teacher_floor_by_label` can set a minimum teacher gap per label, and
`teacher_gap_cap` can prevent overly large branch margins from dominating the
evidence scorer update. `support_weighting` can increase the penalty when
top-branch support is strong, and `hardness_weighting` can increase the penalty
when the evidence gap is currently negative. The support signal is detached so
this loss still updates the evidence path rather than the branch teacher.

`class_evidence_margin.support_weighting` applies the same detached branch
support multiplier to the class evidence margin. This makes evidence ranking
errors matter more when at least one branch already provides a strong signal.

`global_residual_anti_veto` limits label-agnostic residual veto behavior. In
the original `target="global_residual_logits"` mode, it applies only when the
detached evidence gap is above `evidence_confidence_threshold` and penalizes
residual logits below `min_residual_gap`. In `mode="final_gap_preservation"`,
it uses `support_source="top_branch_margin"` to select supported samples and
penalizes `final_logits` when the final true-vs-hardest-negative gap drops more
than `allowed_gap_drop` below the detached `class_evidence_logits` reference
gap.

`gate_weighted_branch_margin` improves branch logits selected by the true-class
gate. The gate is detached, so this loss updates branch heads rather than
directly moving the gate.

`top_branch_margin` asks at least one branch to produce a strong true-vs-hardest
negative margin for each sample. `margin_by_label` can override the scalar
`margin` per label; labels not listed use the scalar value.

`gate_branch_regret` penalizes the gate when it gives too much mass to branches
whose branch margin is worse than the best available branch margin.
`positive_threshold_by_label` can override the scalar threshold per label.
`weight_schedule` can ramp the regret contribution after `warmup_epochs`; the
warmup gate wins, so epochs `<= warmup_epochs` still contribute zero loss even
when a schedule is configured. With `start_epoch=16`, `end_epoch=30`,
`start_multiplier=0.2`, and `end_multiplier=1.0`, the regret loss uses 20% of
its configured weight at epoch 16 and reaches full weight at epoch 30.

`gate_bad_branch_suppression` is the complementary gate-side objective. It
penalizes true-class gate mass left on branches whose detached true-vs-hardest
negative branch margin is below a label-specific bad-margin threshold. Because
the branch margin is detached, this loss updates the gate, not the branch-logit
head. `bad_margin_threshold_by_label` can override the scalar fallback; negative
thresholds are allowed, so normal can be fixed at a permissive value such as
`-0.2`. It uses the same `weight_schedule` shape as `gate_branch_regret`, but
keeps independent values and adaptive state.

`residual_contradiction_regularization` penalizes the raw global residual logits
when class evidence supports the true class but the residual gap points too far
against it. It is label-agnostic and uses the same true-vs-hardest-negative
negative class as the evidence reference. This differs from final-gap anti-veto:
anti-veto preserves the final logits, while residual contradiction limits the
raw residual path.

`auto_margin_by_train_stats` and `auto_positive_threshold_by_train_stats` adjust
these label-wise values from training-epoch statistics only. Validation
diagnostics are not used for online tuning. The margin controller tracks the
train top-branch violation rate, and the regret controller tracks the train
eligible rate. Both update once per epoch and clamp values within the configured
per-label bounds.
`auto_bad_margin_threshold_by_train_stats` also uses training-epoch statistics
only. It tracks label-wise bad gate mass, then raises or lowers the bad-margin
threshold within per-label bounds to keep suppression pressure near the
configured target.
Set `min_*_by_label` equal to `max_*_by_label` to keep a label fixed while other
labels adapt; for example, normal can stay fixed while crackle and wheeze use
separate adaptive ranges.

The margin losses support `reduction="mean"` and
`reduction="class_balanced_violating_mean"`. The class-balanced violating
reduction averages only active margin violations per class, then averages over
classes that actually have violations in the batch.

## Optimizer Layout

The optimizer uses separate learning rates for encoder and head parameters:

- `train.optimizer.encoder_lr`
- `train.optimizer.head_lr`
- `train.optimizer.weight_decay`

The scheduler uses linear warmup followed by cosine annealing, controlled by
`train.scheduler.warmup_ratio`.

## Initialization

`train.initialization` controls warm-start or resume behavior:

- `checkpoint_path`: source checkpoint path or `null`.
- `load_model_state`: whether to load model tensors.
- `strict`: passed to normal `load_state_dict` when shape filtering is disabled.
- `load_optimizer_state`: whether to load optimizer state.
- `skip_mismatched_shapes`: if `true`, loads only matching model-state keys and shapes.

`skip_mismatched_shapes=true` is for warm-start transfer, not exact resume.
Do not combine it with optimizer-state loading.

## Diagnostics

Training and evaluation diagnostics are saved under the run directory when
enabled by `analysis.outputs`.

Important class-aware fields include:

- `class_evidence_gate_weights`: gate actually used for pooling.
- `class_evidence_learned_gate_weights`: learned gate before uniform mixing, when available.
- `true_class_gate_weights`: used gate for the true label.
- `predicted_class_gate_weights`: used gate for the predicted label.
- `class_evidence_gate_mixing_alpha`: uniform-to-learned mixing coefficient.
- `class_evidence_logits`: direct evidence logits.
- `class_gated_branch_logits`: raw class-gated branch logits.
- `class_gated_branch_logit_features`: branch-logit features fed to the residual classifier.
- `global_residual_logits`: residual classifier logits.
- `global_residual_scale`: raw residual scale.
- `global_residual_schedule_multiplier`: epoch schedule multiplier.
- `global_residual_effective_scale`: residual scale actually applied.

Use diagnostics to inspect class-specific gate behavior. For class-aware runs,
prefer per-class gate fields over aggregate gate means.

## Development Checks

Run targeted config tests after schema or config changes:

```bash
PYTHONPATH=. /Users/jjhong0608/.local/share/mamba/envs/respiratory/bin/python -m pytest tests/test_ast_config.py
```

Run static checks after code changes:

```bash
ruff check src tests
mypy src
```
