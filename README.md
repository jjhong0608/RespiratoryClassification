# Multi-Scale Event-MIL Respiratory Classification

This repo trains and evaluates clip-level respiratory sound classifiers with a
single supported encoder type: `multiscale_rdt_ast`.

- One `.wav` file is one training example.
- The frontend still uses local AST-style `fbank` extraction.
- Hugging Face `ASTModel`, latent-query pooling, and branch-summary-token
  initialization are intentionally removed from the active path.
- Training, evaluation, cross-validation, checkpointing, diagnostics, and
  threshold optimization remain config-driven.

## Setup

```bash
mamba activate respiratory
pip install -r requirements.txt
```

## Canonical Configs

- Binary B0: `configs/training_event_mil_b0.json`
- Binary B1: `configs/training_event_mil_b1.json`
- Binary B2: `configs/training_event_mil_b2.json`
- Binary B3: `configs/training_event_mil_b3.json`
- Binary training example: `configs/training_multiscale_rdt.json`
- Binary CV example: `configs/cv_multiscale_rdt.json`
- Binary evaluation: `configs/eval_multiscale_rdt.json`
- Multiclass training example: `configs/training_multiclass.json`

`training_multiscale_rdt.json`, `cv_multiscale_rdt.json`, and
`training_multiclass.json` are the "full" event-MIL examples with branch
auxiliary supervision enabled and RDT refinement active.

## C0-C5 Experiments

Use `best_loss_*.pt` checkpoints as the primary comparison target for C0-C5.
`best_f1_*.pt` checkpoints are still saved, but treat them as diagnostic or
reference artifacts because validation-threshold tuning can overfit binary F1.

| Study | Config | Purpose |
|---|---|---|
| C0 | `configs/training_c0_b3_focal_patience8.json` | B3 reproduction with focal loss and shorter patience |
| C1 | `configs/training_c1_b3_bce_aux01_patience8.json` | B3 with BCE and branch auxiliary weight `0.1` |
| C2 | `configs/training_c2_b1_bce_aux01_patience8.json` | B1 event-MIL baseline without RDT |
| C3 stage 1 | `configs/training_c3_stage1_b1_bce_aux01.json` | Train event detectors without RDT |
| C3 stage 2 | `configs/training_c3_stage2_b3_from_stage1.json` | Warm-start B3 RDT from stage 1 |
| C4 3-scale | `configs/training_c4_3scale_bce_aux01_rdt3.json` | Remove the `(2, 128)` branch |
| C4 4-scale | `configs/training_c4_4scale_bce_aux01_rdt3.json` | Matched 4-scale comparator |
| C5 top-2 | `configs/training_c5_top2_bce_aux01_rdt3.json` | Default top-2 evidence bottleneck |
| C5 top-4 | `configs/training_c5_top4_bce_aux01_rdt3.json` | Wider top-4 evidence bottleneck |

C3 staged training:

1. Run `python -m src.cli.training --config configs/training_c3_stage1_b1_bce_aux01.json`.
2. Find the stage-1 `best_loss_*.pt` checkpoint under `checkpoints/respiratory_c3_stage1_b1_bce_aux01/`.
3. Put that path into `train.initialization.checkpoint_path` in `configs/training_c3_stage2_b3_from_stage1.json`.
4. Run `python -m src.cli.training --config configs/training_c3_stage2_b3_from_stage1.json`.

Stage 2 intentionally uses `strict=false` and `load_optimizer_state=false`
because it turns on RDT and uses different learning rates.

## Next Experiment Series: D/E/F

Use the same checkpoint-selection policy as C0-C5: compare experiments with
`best_loss_*.pt`, and treat `best_f1_*.pt` as a diagnostic or reference
checkpoint because validation-threshold tuning can overfit F1.

| Study | Config | Purpose |
|---|---|---|
| D1 stage 1 | `configs/training_d1_c3_3scale_stage1.json` | Three-scale B1 detector warm-start stage |
| D1 stage 2 | `configs/training_d1_c3_3scale_stage2.json` | Three-scale B3 warm-start from D1 stage 1 |
| D2 | `configs/training_d2_stage2_no_rdt_from_stage1.json` | Warm-started stage 2 with RDT still disabled |
| D3 | `configs/training_d3_direct_b3_low_lr.json` | Direct B3 training with lower learning rates |
| D4 top-1 | `configs/training_d4_c3_top1_stage2.json` | C3 stage 2 with one selected token per branch |
| D4 top-3 | `configs/training_d4_c3_top3_stage2.json` | C3 stage 2 with three selected tokens per branch |
| D5 seed 0 stage 1 | `configs/training_d5_c3_stage1_seed0.json` | C3 stage 1 seed-0 repeat |
| D5 seed 0 stage 2 | `configs/training_d5_c3_stage2_seed0.json` | C3 stage 2 seed-0 repeat |
| D5 seed 1 stage 1 | `configs/training_d5_c3_stage1_seed1.json` | C3 stage 1 seed-1 repeat |
| D5 seed 1 stage 2 | `configs/training_d5_c3_stage2_seed1.json` | C3 stage 2 seed-1 repeat |
| D5 seed 2 stage 1 | `configs/training_d5_c3_stage1_seed2.json` | C3 stage 1 seed-2 repeat |
| D5 seed 2 stage 2 | `configs/training_d5_c3_stage2_seed2.json` | C3 stage 2 seed-2 repeat |
| E1 | `configs/training_e1_c3_attention_logit_stage2.json` | Select evidence with pre-softmax MIL attention logits |
| E2 | `configs/training_e2_c3_instance_logit_stage2.json` | Select evidence with token-level instance logits |
| E3 | `configs/training_e3_c3_attention_temp05_stage2.json` | Use MIL attention temperature `0.5` |
| E4 | `configs/training_e4_c3_entropy001_stage2.json` | Add attention entropy loss with weight `0.001` |
| F1 | `configs/training_f1_c3_branch_aux_weights_stage2.json` | Use branch-specific auxiliary-loss weights |
| F2 | `configs/training_f2_c3_exclude_branch4_evidence_stage2.json` | Exclude the fourth scale from selected evidence only |

Staged D/E/F runs follow the C3 workflow:

1. Run the matching stage-1 config.
2. Find that run's `best_loss_*.pt` checkpoint.
3. Put the checkpoint path into the stage-2 config at `train.initialization.checkpoint_path`.
4. Run the stage-2 config.

Stage-2 templates keep `train.initialization.checkpoint_path = null` until you
choose the stage-1 checkpoint manually.

## Round G Experiments

Round G treats `configs/training_d3_direct_b3_low_lr.json` as the current
mainline: direct low-LR B3 training with RDT on, four scales, top-2 evidence,
BCE loss, branch auxiliary weight `0.1`, and no warm-start.

Use `best_loss_*.pt` as the primary comparison checkpoint. `best_f1_*.pt`
remains useful as a diagnostic/reference checkpoint, but do not use it as the
main model-selection criterion for Round G.

| Study | Config | Purpose |
|---|---|---|
| G1 seed0/1/2/42/43 | `configs/training_g1_d3_seed*.json` | Repeat D3 direct low-LR across seeds |
| G2 | `configs/training_g2_direct_low_lr_no_rdt.json` | Direct no-RDT low-LR control |
| G3 | `configs/training_g3_direct_low_lr_3scale.json` | Direct 3-scale control without the `2x128` branch |
| G4 | `configs/training_g4_direct_low_lr_aux005.json` | Branch auxiliary weight `0.05` |
| G5 | `configs/training_g5_direct_low_lr_no_aux.json` | Disable branch auxiliary loss |
| G6 | `configs/training_g6_direct_low_lr_focal_gamma1.json` | Focal loss with gamma `1.0` |
| G7 | `configs/training_g7_direct_low_lr_focal_gamma2.json` | Focal loss with gamma `2.0` |

Interpretation:

- `G1` checks whether D3 is robust across seeds.
- `G2` tests whether RDT contributes under direct low-LR training.
- `G3` tests whether the `2x128` branch is necessary.
- `G4/G5` test branch auxiliary sensitivity.
- `G6/G7` retest focal loss under the low-LR setting.

Report these metrics from the `best_loss_*.pt` checkpoint:

- `F1@0.5`
- `F1@opt`
- `ROC-AUC`
- `PR-AUC`
- `Brier`
- `balanced accuracy`
- `optimal threshold`

Suggested execution order:

1. `G2`
2. `G3`
3. `G4` and `G5`
4. `G6` and `G7`
5. `G1` seed repeats

## H Round: 5-Seed Final Candidate Ablation

H Round uses the existing G1/D3 5-seed mainline as H0 reference: four scales,
RDT on, top-2 evidence, BCE loss, branch auxiliary weight `0.1`, low learning
rates, no warm-start, and seeds `0`, `1`, `2`, `42`, and `43`.

Use `best_loss_*.pt` as the primary checkpoint for H Round comparisons.
`best_f1_*.pt` remains diagnostic/reference only. Compare mean, standard
deviation, min, max, and worst seed; do not select the final candidate from a
single best seed.

| Round | Configs | Purpose |
|---|---|---|
| H0 gated | `configs/training_h0_branch_gated_seed*.json` | Compare branch-aware gated evidence pooling against H0/G1 mean pooling |
| H1 | `configs/training_h1_no_rdt_seed*.json` | Test RDT necessity |
| H2 | `configs/training_h2_no_aux_seed*.json` | Test branch auxiliary necessity |
| H3 | `configs/training_h3_3scale_seed*.json` | Test `2x128` branch necessity |

Interpretation:

- `H1` vs `H0/G1` decides whether RDT remains in the final candidate.
- `H2` vs `H0/G1` decides whether branch auxiliary loss remains.
- `H3` vs `H0/G1` decides whether the `2x128` branch remains.
- H0 branch-gated vs H0/G1 decides whether branch-aware evidence pooling
  improves the final readout.

Dynamic shapes:

- 4-scale top-2 -> `U0: [B, 8, D]`, `H_ctx` length `1916`
- 3-scale top-2 -> `U0: [B, 6, D]`, `H_ctx` length `893`

## Data Augmentation

AUG0-AUG3 isolate train-time augmentation effects on the same H0 branch-gated
model. Keep model, optimizer, loss, early stopping, and seed policy fixed across
the four configs.

| Study | Config | Purpose |
|---|---|---|
| AUG0 | `configs/training_aug0_h0_gated_no_aug.json` | H0 gated baseline without augmentation |
| AUG1 | `configs/training_aug1_h0_gated_waveform_aug.json` | Waveform augmentation only |
| AUG2 | `configs/training_aug2_h0_gated_fbank_aug.json` | Fbank augmentation only |
| AUG3 | `configs/training_aug3_h0_gated_waveform_fbank_aug.json` | Combined waveform and fbank augmentation |

Placement:

- Waveform augmentation runs after waveform preprocessing and before AST fbank
  extraction.
- Fbank augmentation runs after normalized fbank extraction and transpose to
  `[1024, 128]`.
- Augmentation is train-only. Validation, evaluation, and test splits are never
  augmented.
- CV train folds may augment; CV validation folds do not.

Compare AUG0-AUG3 with `best_loss_*.pt` checkpoints. Report `F1@0.5`,
`F1@opt`, `ROC-AUC`, `PR-AUC`, `Brier`, `balanced accuracy`, and optimal
threshold.

## One-of and Token-Level Augmentation

AUG4/PT1/PT2/PT3 keep the same H0 branch-aware gated model as AUG0-AUG3. Only
input augmentation policy and model-internal token regularization differ.

| Config | Purpose |
|---|---|
| `configs/training_aug4_h0_gated_oneof.json` | One-of augmentation only |
| `configs/training_pt1_h0_gated_oneof_branch_event_dropout.json` | One-of + branch event token dropout |
| `configs/training_pt2_h0_gated_oneof_selected_evidence_dropout.json` | One-of + selected evidence dropout |
| `configs/training_pt3_h0_gated_oneof_both_token_dropouts.json` | One-of + both token dropouts |

The one-of policy samples exactly one recipe per training clip: `none`,
`waveform`, `fbank`, or `both_light`. This keeps augmentation diversity without
always stacking waveform and fbank perturbations on the same sample.

Token-level regularizers are train-only:

- Branch event token dropout runs after frequency-attention pooling and before
  branch MIL. Dropped event tokens are zeroed and masked so branch MIL cannot
  attend to them.
- Selected evidence dropout runs after top-k evidence selection and before RDT.
  It zeros selected evidence state tokens `U0` while leaving `H_ctx`, selected
  indices, scores, and branch IDs unchanged.
- Both dropouts guarantee at least one kept token at their configured branch
  granularity.

## Train

Binary example:

```bash
python -m src.cli.training --config configs/training_multiscale_rdt.json
```

Multiclass example:

```bash
python -m src.cli.training --config configs/training_multiclass.json
```

Artifacts are written under `experiment.output_dir/experiment.name/`.

Training keeps:

- `last.pt`
- `best_loss_*.pt`
- `best_f1_*.pt`

For C0-C5, D/E/F, and Round G analysis and evaluation, start from
`best_loss_*.pt`.

## Evaluate

```bash
python -m src.cli.evaluate --config configs/eval_multiscale_rdt.json
```

Evaluation writes:

- `eval_metrics.json`
- `eval_predictions.csv`
- `eval_diagnostics.jsonl` when diagnostics are enabled

Binary evaluation keeps fixed-threshold (`0.5`) metrics at the top level and
also stores:

- `decision_threshold`
- `threshold_optimization`
- `optimized_metrics`

Threshold optimization is checkpoint-driven and only applies to binary
classification. For multiclass evaluation it is reported as disabled with an
explicit reason.

## Cross-Validation

```bash
python -m src.cli.cv --config configs/cv_multiscale_rdt.json
```

Each fold is trained independently under
`experiment.output_dir/experiment.name/fold_x/`.

## Architecture

The default model consumes:

```text
input_values: [B, 1024, 128]
```

It applies this event-MIL-first flow:

```text
[B, 1024, 128]
  -> unsqueeze channel
[B, 1, 1024, 128]
  -> four patch tokenizers
  -> learned patch position + scale embeddings
  -> shared branch-wise transformer stem
  -> scale-specific transformer adapters
  -> reshape each branch to [B, T_s, F_s, D]
  -> learned frequency-attention pooling
  -> branch event tokens
  -> branch MIL attention heads
  -> top-k evidence tokens per branch
U0: [B, 8, D], H_ctx: [B, 1916, D]
  -> optional recurrent RDT refinement over U using H_ctx
  -> evidence pooling: mean or branch-aware gated
  -> fuse evidence embedding, mean branch embedding, and branch logits
  -> fusion projector
[B, D]
  -> classifier
```

Default patch token geometry:

- `(16, 16)` patch with `(8, 16)` stride -> `1016` patch tokens -> `127`
  temporal event tokens
- `(8, 32)` patch with `(4, 32)` stride -> `1020` patch tokens -> `255`
  temporal event tokens
- `(4, 64)` patch with `(2, 64)` stride -> `1022` patch tokens -> `511`
  temporal event tokens
- `(2, 128)` patch with `(1, 128)` stride -> `1023` patch tokens -> `1023`
  temporal event tokens

The concatenated event context length is `1916`. With the default
`top_tokens_per_branch = 2`, the initial evidence state is `U0: [B, 8, D]`.
The C4 3-scale ablation removes the final branch, so the event context length
becomes `893` and top-2 evidence selection yields `U0: [B, 6, D]`. The C5
top-4 ablation keeps four branches and yields `U0: [B, 16, D]`.

Dynamic selected-evidence lengths:

- 3-scale top-2 -> `6`
- 4-scale top-1 -> `4`
- 4-scale top-2 -> `8`
- 4-scale top-3 -> `12`
- 4-scale top-4 -> `16`
- 4-scale top-2 with `exclude_branches_from_evidence = [3]` -> `6`

`exclude_branches_from_evidence` uses zero-based branch indices. Excluding
branch `3` removes the fourth scale from `U0` only; it still contributes to
`H_ctx`, branch logits, branch embeddings, auxiliary loss, and final fusion.
For that F2 setting, `H_ctx` remains length `1916`.

Default encoder hyperparameters:

- `hidden_size = 192`
- `num_attention_heads = 4`
- `mlp_ratio = 2.0`
- `hidden_dropout_prob = 0.1`
- `attention_probs_dropout_prob = 0.1`
- `layer_norm_eps = 1e-6`
- `shared_stem_depth = 2`
- `adapter_depth = 1`
- `rdt.enabled = false`
- `rdt.steps = 3`
- `rdt.top_tokens_per_branch = 2`
- `rdt.gated_residual = true`
- `rdt.layerscale_init = 0.01`

Classifier pooling remains config-visible as `latent_mean`, but evidence
readout is controlled by `model.encoder.architecture.evidence_pooling`.

## Evidence Pooling

`evidence_pooling.type = "mean"` is the default legacy behavior. Selected
evidence tokens, after optional RDT refinement, are averaged uniformly.

`evidence_pooling.type = "branch_gated"` groups selected evidence tokens by
their source branch, averages each branch's selected tokens into a branch
evidence summary, computes a gate from those branch evidence summaries only,
and uses the gated weighted sum as the evidence embedding.

Branch logits are not used as gate input. Mean branch embeddings are not used as
gate input. Final fusion still receives the evidence embedding, mean branch
embedding, and branch logits.

H0 branch-gated configs:

- `configs/training_h0_branch_gated_seed0.json`
- `configs/training_h0_branch_gated_seed1.json`
- `configs/training_h0_branch_gated_seed2.json`
- `configs/training_h0_branch_gated_seed42.json`
- `configs/training_h0_branch_gated_seed43.json`

Compare H0 mean pooling against H0 branch-gated pooling with `best_loss_*.pt`
checkpoints. Report mean, standard deviation, min, max, and worst seed; do not
choose a final candidate from a single best seed.

## B0 / B1 / B2 / B3 Modes

The experiment family is controlled by config only:

- `B0`: `rdt.enabled = false`, `branch_auxiliary.enabled = false`
- `B1`: `rdt.enabled = false`, `branch_auxiliary.enabled = true`
- `B2`: `rdt.enabled = true`, `rdt.steps = 2`, `branch_auxiliary.enabled = true`
- `B3`: `rdt.enabled = true`, `rdt.steps = 3`, `branch_auxiliary.enabled = true`

`rdt.enabled` is the switch that disables refinement. `steps` is still stored
in the config when RDT is off, but it is ignored by the forward path.

## Config Shape

The active schema is:

```json
{
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
        "rdt": {
          "enabled": true,
          "steps": 3,
          "top_tokens_per_branch": 2,
          "gated_residual": true,
          "layerscale_init": 0.01,
          "evidence_score_source": "attention_weight",
          "exclude_branches_from_evidence": []
        },
        "mil": {
          "attention_temperature": 1.0
        },
        "evidence_pooling": {
          "type": "mean",
          "gate_hidden_size": null,
          "dropout": 0.1,
          "temperature": 1.0
        },
        "patch_branches": [
          {"patch_size": [16, 16], "stride": [8, 16]},
          {"patch_size": [8, 32], "stride": [4, 32]},
          {"patch_size": [4, 64], "stride": [2, 64]},
          {"patch_size": [2, 128], "stride": [1, 128]}
        ]
      }
    },
    "classifier": {
      "type": "linear",
      "hidden_dim": 256,
      "dropout": 0.1,
      "pooling": "latent_mean"
    }
  },
  "train": {
    "loss": {
      "type": "focal",
      "branch_auxiliary": {
        "enabled": true,
        "weight": 0.3,
        "aggregation": "mean",
        "weights": null
      },
      "attention_entropy": {
        "enabled": false,
        "weight": 0.0
      }
    },
    "initialization": {
      "checkpoint_path": null,
      "load_model_state": true,
      "strict": false,
      "load_optimizer_state": false
    }
  }
}
```

Strict compatibility rules:

- `model.encoder.type` must stay `multiscale_rdt_ast`
- `adaptation.mode` supports only `full` and `frozen`
- `adaptation.num_layers` must be `0`
- `classifier.pooling` must be `latent_mean`
- legacy `latent_query_count`, `summary_tokens_per_scale`, and flat `rdt_steps`
  are rejected

Additional experiment knobs:

- `rdt.evidence_score_source` supports `attention_weight`, `attention_logit`,
  and `instance_logit`
- `rdt.exclude_branches_from_evidence` removes zero-based branches only from
  selected evidence `U0`
- `architecture.mil.attention_temperature` must be greater than zero and scales
  branch MIL attention softmax logits
- `architecture.evidence_pooling.type` supports `mean` and `branch_gated`
- `data.augmentation` is disabled by default and applies only to train datasets
- `data.augmentation.policy.type` supports `independent` and `one_of`
- `architecture.token_augmentation` controls branch event and selected evidence
  token dropout
- `train.loss.attention_entropy.enabled = true` adds
  `weight * mean_branch(entropy(attention))`
- `train.loss.branch_auxiliary.weights` overrides scalar
  `branch_auxiliary.weight` with normalized per-branch weighting

## Loss Rules

Loss behavior still depends on the number of classes:

- Binary (`len(label_to_index) == 2`)
  - supported losses: `bce`, `focal`
  - optional `auto_pos_weight` / `pos_weight`
  - optional branch auxiliary loss uses the same binary criterion on each
    branch logit
- Multi-class (`len(label_to_index) > 2`)
  - required loss: `cross_entropy`
  - binary-only weighting options are rejected
  - branch auxiliary loss applies cross-entropy to each branch head and then
    averages across branches

When `train.loss.branch_auxiliary.weights` is absent, the scalar
`branch_auxiliary.weight` is applied to the mean branch loss. When `weights` is
present, the scalar is ignored and the auxiliary term is
`sum(weights[i] * branch_loss[i]) / sum(weights)`.

## Optimizer Layout

Training keeps two AdamW parameter groups:

- encoder parameters -> patch tokenizers, shared stem, adapters,
  frequency-attention poolers, branch MIL heads -> `train.optimizer.encoder_lr`
- head parameters -> optional RDT, fusion projector, final classifier ->
  `train.optimizer.head_lr`

If `adaptation.mode = "frozen"`, the encoder group is frozen and only the head
parameters are trainable.

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

When enabled, diagnostics now include:

- `branch_logits` with the saved logit payload
- `selected_evidence_indices` for the selected event-token positions
- `selected_evidence_scores` from the configured evidence score source
- `selected_evidence_branch_ids` identifying which branch selected each token
- `evidence_score_source` identifying the selection score source
- `evidence_pooling_type` identifying the evidence readout
- `evidence_gate_weights`, `evidence_gate_entropy`, and
  `branch_evidence_norms` when branch-aware gated pooling is active
- `selected_evidence_tokens` with the saved embedding payload only when
  `save_embeddings=true`

Full branch attention maps stay in the model output for training and tests, but
they are not dumped into JSONL by default because they are large.

## Directory Copy Utility

Use `copy_directory_excluding_suffixes.py` to recursively copy one directory's
contents into another directory while skipping one or more file suffixes:

```bash
python copy_directory_excluding_suffixes.py SOURCE_DIR TARGET_DIR .wav .mp3 .npy
```

The source directory must already exist. The target directory is created if
needed. Suffixes may be passed with or without a leading dot, matching is
case-insensitive, and copied files overwrite existing target files with the same
relative path.

## Notes

- `src.cli.plot_mels` remains available as a utility.
- `src.cli.pretrained_info` exits immediately because pretrained HF AST
  inspection is no longer part of this branch.
- `configs/eval_multiscale_rdt.json` remains evaluation-only and reconstructs
  the model from checkpoint `model_cfg`.
