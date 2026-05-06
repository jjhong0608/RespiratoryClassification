# Codex Instructions: FSD50K SSL → FSD50K Supervised → CNUH Transfer Pipeline

## Target Repository and Branch

Repository:

```text
https://github.com/jjhong0608/RespiratoryClassification.git
```

Base branch:

```text
MIL-RDT-AST-SSL
```

This task is based on the latest `MIL-RDT-AST-SSL` branch.

The current branch uses a custom event-MIL-first respiratory sound model with a single active encoder type:

```text
multiscale_rdt_ast
```

Do **not** reintroduce Hugging Face `ASTModel`.  
Do **not** reintroduce latent-query pooling.  
Do **not** rewrite the model architecture from scratch.  
Do **not** break existing CNUH supervised training/evaluation configs.  
Do **not** force SSL behavior into the normal supervised forward path.

---

## Objective

Implement a three-stage pretraining and fine-tuning pipeline:

```text
Stage 1. FSD50K masked fbank SSL pretraining
Stage 2. FSD50K supervised multi-label fine-tuning
Stage 3. CNUH 4-class transfer/fine-tuning support
```

The intended high-level workflow is:

```text
FSD50K SSL
  -> FSD50K supervised multi-label
  -> CNUH 4-class fine-tuning
```

The design goal is to learn broad acoustic representations from FSD50K before adapting the model to CNUH respiratory sound classification.

---

# Part A — Non-Negotiable Constraints

## Preserve existing supervised path

The existing supervised path must remain valid:

```python
model.forward(input_values, ...)
```

Do not break current CNUH binary/4-class training.

Add a separate SSL path:

```python
model.forward_ssl(...)
```

or a wrapper:

```python
MaskedFbankSSLWrapper(base_model)
```

Preferred:

```text
Use a separate SSL wrapper and a public branch-encoding helper on the base model.
```

---

## Do not add branch auxiliary to FSD50K supervised stage

For FSD50K supervised multi-label fine-tuning:

```text
branch_auxiliary.enabled = false
branch_binary_auxiliary.enabled = false
```

Do not force each branch to solve the 200-class multi-label task.

---

## Do not add global binary auxiliary

This pipeline does not require a global binary auxiliary head.

---

## Keep CNUH classifier reset

When transferring from FSD50K supervised checkpoint to CNUH:

```text
discard FSD50K multi-label classifier head
reset CNUH classifier head
```

Use:

```text
strict = false
```

for checkpoint loading.

---

# Part B — FSD50K Dataset Loader

## B1. Add metadata-driven FSD50K dataset

Add a new FSD50K dataset implementation, likely under:

```text
src/data/
```

Suggested filename:

```text
src/data/fsd50k_dataset.py
```

The dataset must be metadata-driven.

Do not infer labels from folder names.

Use:

```text
vocabulary.csv
dev.csv
eval.csv
audio directories
```

Exact official filenames may vary by local download, so support configurable paths.

---

## B2. FSD50K config schema

Add or extend data config to support:

```json
"data": {
  "dataset": "fsd50k",
  "mode": "ssl",
  "root": "/path/to/FSD50K",
  "dev_audio_dir": "/path/to/FSD50K.dev_audio",
  "eval_audio_dir": "/path/to/FSD50K.eval_audio",
  "ground_truth_dir": "/path/to/FSD50K.ground_truth",
  "vocabulary_csv": "/path/to/FSD50K.ground_truth/vocabulary.csv",
  "dev_csv": "/path/to/FSD50K.ground_truth/dev.csv",
  "eval_csv": "/path/to/FSD50K.ground_truth/eval.csv",
  "val_ratio": 0.1,
  "split_seed": 42
}
```

Supported modes:

```text
ssl
supervised
```

---

## B3. SSL mode output

When:

```json
"mode": "ssl"
```

return:

```python
{
    "input_values": fbank,   # Tensor [1024, 128]
    "clip_id": clip_id,
}
```

No label is required.

---

## B4. Supervised mode output

When:

```json
"mode": "supervised"
```

return:

```python
{
    "input_values": fbank,   # Tensor [1024, 128]
    "labels": multi_hot,     # Tensor [200]
    "clip_id": clip_id,
}
```

`labels` must be multi-hot.

---

## B5. Vocabulary parsing

Parse `vocabulary.csv` and build:

```python
label_to_index: dict[str, int]
index_to_label: list[str]
```

The loader must map the exact label keys used in `dev.csv` / `eval.csv` to indices.

If the metadata uses AudioSet MIDs, use MIDs.  
If it uses display labels, use display labels.

Do not silently drop labels unless they are explicitly missing from the vocabulary, in which case raise a clear error or warning depending on config.

---

## B6. Multi-hot target construction

Pseudo-code:

```python
target = torch.zeros(num_classes, dtype=torch.float32)
for label in labels_for_clip:
    target[label_to_index[label]] = 1.0
```

Expected shape:

```text
[200]
```

---

## B7. Split handling

Use this priority:

```text
1. If official split column exists, use it.
2. Else create deterministic train/val split from dev.csv using split_seed and val_ratio.
3. Reserve eval.csv for FSD50K supervised evaluation.
```

For SSL, use train/val split from the development set.

For supervised FSD50K, use the same train/val split and keep eval for final evaluation.

---

## B8. Audio frontend

Use the same frontend conventions as CNUH:

```text
load waveform
mono conversion
resample
crop/pad
fbank extraction
normalization
return [1024, 128]
```

Train split:

```text
random crop
```

Validation/evaluation:

```text
deterministic center crop
```

Do not introduce a different fbank frontend for FSD50K.

---

# Part C — Stage 1: FSD50K Masked Fbank SSL

## C1. Stage 1 summary

Implement masked fbank reconstruction SSL.

```text
Dataset:
  FSD50K SSL mode

Epochs:
  200

Input/target:
  normalized fbank [B, 1024, 128]

Masking:
  branch-specific input-space patch masking
  tokenizer patch grid basis
  random token sampling
  token_mask_ratio = 0.4

Decoder:
  branch-wise separate shallow Conv1d decoder

Loss:
  mean branch masked MSE
```

---

## C2. SSL default config

Add a config file:

```text
configs/fsd50k_ssl_masked_fbank_pretrain.json
```

Recommended structure:

```json
{
  "task": "fsd50k_masked_fbank_ssl",
  "data": {
    "dataset": "fsd50k",
    "mode": "ssl",
    "root": "/path/to/FSD50K",
    "dev_audio_dir": "/path/to/FSD50K.dev_audio",
    "eval_audio_dir": "/path/to/FSD50K.eval_audio",
    "ground_truth_dir": "/path/to/FSD50K.ground_truth",
    "vocabulary_csv": "/path/to/FSD50K.ground_truth/vocabulary.csv",
    "dev_csv": "/path/to/FSD50K.ground_truth/dev.csv",
    "eval_csv": "/path/to/FSD50K.ground_truth/eval.csv",
    "val_ratio": 0.1,
    "split_seed": 42
  },
  "train": {
    "epochs": 200,
    "batch_size": 32,
    "optimizer": {
      "lr": 0.0003,
      "weight_decay": 0.01
    },
    "scheduler": {
      "type": "linear_warmup_cosine",
      "warmup_ratio": 0.05
    }
  },
  "ssl": {
    "target": "normalized_fbank",
    "masking": {
      "type": "branch_specific_token_grid_input_patch",
      "sampling": "random",
      "token_mask_ratio": 0.4,
      "mask_value": 0.0,
      "actual_mask_ratio_warning_threshold": 0.85
    },
    "decoder": {
      "type": "branch_conv1d",
      "sharing": "separate",
      "channels": 256,
      "kernel_size": 5,
      "num_layers": 2,
      "dropout": 0.1,
      "use_layer_norm": true
    },
    "loss": {
      "type": "mean_branch_masked_mse"
    },
    "visualization": {
      "enabled": false,
      "num_examples": 8,
      "save_every_n_epochs": 10
    },
    "return_reconstructions": false
  },
  "checkpointing": {
    "monitors": [
      {
        "name": "val_ssl_loss",
        "mode": "min",
        "keep_top_k": 3,
        "filename_prefix": "best_ssl_loss"
      },
      {
        "name": "last",
        "mode": "last",
        "keep_top_k": 3,
        "filename_prefix": "last"
      }
    ],
    "periodic": {
      "enabled": false
    }
  }
}
```

Batch size must be configurable.  
Masked reconstruction does not require large contrastive batches.

---

# Part D — Branch-Specific Token-Grid Input-Space Masking

## D1. Patch geometry

Use the same branch geometry as the supervised model.

| Branch | Patch size `(time, freq)` | Stride `(time, freq)` |
|---:|---:|---:|
| 0 | `(16, 16)` | `(8, 16)` |
| 1 | `(8, 32)` | `(4, 32)` |
| 2 | `(4, 64)` | `(2, 64)` |
| 3 | `(2, 128)` | `(1, 128)` |

Input shape:

```text
T = 1024
F = 128
```

Patch grid:

\[
N_t = \lfloor (T - p_t) / s_t 
floor + 1
\]

\[
N_f = \lfloor (F - p_f) / s_f 
floor + 1
\]

Expected grids:

```text
branch 0: [127, 8]
branch 1: [255, 4]
branch 2: [511, 2]
branch 3: [1023, 1]
```

---

## D2. Token mask sampling

For each branch:

```text
sample token_mask_ratio fraction of token grid positions uniformly at random
```

Default:

```text
token_mask_ratio = 0.4
```

Token mask shape:

```text
branch 0: [B, 127, 8]
branch 1: [B, 255, 4]
branch 2: [B, 511, 2]
branch 3: [B, 1023, 1]
```

True means:

```text
this token's corresponding input patch is masked
```

---

## D3. Convert token mask to fbank mask

For a selected token at grid position `(i, j)`:

```text
t0 = i * stride_t
f0 = j * stride_f
```

Mask the input-space region:

```text
[t0 : t0 + patch_t, f0 : f0 + patch_f]
```

Patch overlaps are handled by union.

Fbank mask shape:

```text
[B, 1024, 128]
```

True means:

```text
masked fbank bin
```

---

## D4. Pseudo-code

```python
def compute_patch_grid_size(T: int, F: int, patch_size: tuple[int, int], stride: tuple[int, int]):
    pt, pf = patch_size
    st, sf = stride
    nt = (T - pt) // st + 1
    nf = (F - pf) // sf + 1
    return nt, nf


def sample_token_mask(
    batch_size: int,
    nt: int,
    nf: int,
    token_mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    num_tokens = nt * nf
    num_mask = int(round(token_mask_ratio * num_tokens))

    token_mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
    for b in range(batch_size):
        idx = torch.randperm(num_tokens, device=device)[:num_mask]
        token_mask[b, idx] = True

    return token_mask.view(batch_size, nt, nf)


def token_grid_to_fbank_mask(
    token_mask: torch.Tensor,
    *,
    T: int,
    F: int,
    patch_size: tuple[int, int],
    stride: tuple[int, int],
) -> torch.Tensor:
    B, nt, nf = token_mask.shape
    pt, pf = patch_size
    st, sf = stride

    fbank_mask = torch.zeros(B, T, F, dtype=torch.bool, device=token_mask.device)

    for i in range(nt):
        t0 = i * st
        for j in range(nf):
            f0 = j * sf
            selected = token_mask[:, i, j]
            if selected.any():
                fbank_mask[selected, t0:t0 + pt, f0:f0 + pf] = True

    return fbank_mask
```

---

## D5. Masked input

Given original normalized fbank:

```text
x: [B, 1024, 128]
```

and branch fbank mask:

```text
M_s: [B, 1024, 128]
```

masked branch input:

```python
x_s = x.masked_fill(M_s, mask_value)
```

Default:

```text
mask_value = 0.0
```

---

## D6. Logging

Log per branch:

```text
token_mask_ratio
actual_fbank_mask_ratio
```

where:

```python
actual_fbank_mask_ratio = fbank_mask.float().mean()
```

If:

```text
actual_fbank_mask_ratio > actual_mask_ratio_warning_threshold
```

then log a warning.

Do not clip the mask.

Default warning threshold:

```text
0.85
```

---

# Part E — SSL Wrapper and Branch Encoding Helper

## E1. Preferred structure

Add a wrapper:

```python
MaskedFbankSSLWrapper(base_model)
```

Do not modify the normal supervised `forward` behavior.

---

## E2. Base model helper

Add a public SSL helper to the base model if feasible:

```python
def forward_branches_for_ssl(
    self,
    branch_inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    """
    Args:
        branch_inputs:
          list of branch-specific masked fbank inputs,
          each [B, 1024, 128]

    Returns:
        branch_event_tokens_list:
          [
            [B, T_0, D],
            [B, T_1, D],
            [B, T_2, D],
            [B, T_3, D],
          ]
        after tokenizer, embeddings, shared stem/adapters,
        and frequency-attention pooling.
    """
```

Codex may choose a different helper name if necessary, but the semantics must be the same.

---

## E3. Direct internal module access

Direct access from the wrapper to base model internals is acceptable only if adding a helper is impractical.

If direct access is used, it must exactly match the supervised branch path up to frequency-attention pooling.

Do not omit:

```text
patch tokenization
position embedding
scale embedding
shared stem
scale-specific adapter
frequency-attention pooling
```

The SSL branch event tokens must be the same type of representation used by branch MIL in supervised training.

---

# Part F — Branch-wise Conv1d Fbank Decoder

## F1. Decoder design

Use separate decoder per branch.

Input:

```text
E_s: [B, T_s, D]
```

Output:

```text
x_hat_s: [B, 1024, 128]
```

Default config:

```json
"decoder": {
  "type": "branch_conv1d",
  "sharing": "separate",
  "channels": 256,
  "kernel_size": 5,
  "num_layers": 2,
  "dropout": 0.1,
  "use_layer_norm": true
}
```

---

## F2. Pseudo-code

```python
class BranchConv1dFbankDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        freq_bins: int = 128,
        channels: int = 256,
        kernel_size: int = 5,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()

        self.proj = nn.Conv1d(hidden_size, channels, kernel_size=1)

        padding = kernel_size // 2
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Conv1d(channels, channels, kernel_size, padding=padding),
                nn.GELU(),
                nn.Dropout(dropout),
            ])

        self.conv = nn.Sequential(*layers)
        self.out = nn.Conv1d(channels, freq_bins, kernel_size=1)

    def forward(self, tokens: torch.Tensor, output_frames: int = 1024) -> torch.Tensor:
        # tokens: [B, T_s, D]
        x = tokens.transpose(1, 2)  # [B, D, T_s]
        x = F.interpolate(x, size=output_frames, mode="linear", align_corners=False)
        x = x.transpose(1, 2)  # [B, 1024, D]
        x = self.norm(x)
        x = x.transpose(1, 2)  # [B, D, 1024]

        x = self.proj(x)
        x = self.conv(x)
        x = self.out(x)       # [B, 128, 1024]
        return x.transpose(1, 2)  # [B, 1024, 128]
```

---

# Part G — SSL Output Dataclass

Add a dataclass like:

```python
@dataclass
class MaskedFbankSSLOutput:
    loss: torch.Tensor
    branch_losses: torch.Tensor          # [S]
    actual_mask_ratios: torch.Tensor     # [S]
    token_mask_ratios: torch.Tensor      # [S]
    reconstructions: list[torch.Tensor] | None = None
    masks: list[torch.Tensor] | None = None
    token_masks: list[torch.Tensor] | None = None
    reconstructed_average: torch.Tensor | None = None
```

Default behavior:

```text
return_reconstructions = false
```

When `return_reconstructions=false`, do not keep large reconstruction tensors in the output except as needed for loss computation.

When visualization is enabled, populate:

```text
reconstructions
masks
token_masks
reconstructed_average
```

---

# Part H — SSL Loss

Do not use `L_avg + alpha * L_branch`.

Use only mean branch masked MSE.

For each branch:

\[
L_s =
rac{
\sum_{t,f} M_s(t,f)(\hat{x}_s(t,f)-x(t,f))^2
}{
\sum_{t,f} M_s(t,f) + \epsilon
}
\]

Total:

\[
L_{SSL} = rac{1}{S}\sum_s L_s
\]

Pseudo-code:

```python
def masked_mse(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    # recon/target: [B, T, F]
    # mask: [B, T, F], bool
    diff2 = (recon - target).pow(2)
    mask_f = mask.to(diff2.dtype)
    return (diff2 * mask_f).sum() / mask_f.sum().clamp_min(eps)


branch_losses = []
for s in range(num_branches):
    branch_losses.append(masked_mse(recon_s, input_values, fbank_mask_s))

loss = torch.stack(branch_losses).mean()
```

Optional debug:

```python
reconstructed_average = torch.stack(reconstructions, dim=0).mean(dim=0)
```

Do not use `reconstructed_average` in the loss.

---

# Part I — Reconstruction Visualization

Add optional reconstruction visualization.

Default:

```json
"visualization": {
  "enabled": false
}
```

When enabled, save a small number of examples.

Recommended saved panels:

```text
original fbank
masked fbank for selected branch
reconstruction for selected branch
optional average reconstruction
mask image
```

Visualization must be off by default.

---

# Part J — Stage 1 Checkpointing and Metrics

Log:

```text
train_ssl_loss
val_ssl_loss
train_branch_loss_0
train_branch_loss_1
train_branch_loss_2
train_branch_loss_3
val_branch_loss_0
val_branch_loss_1
val_branch_loss_2
val_branch_loss_3
branch_*_token_mask_ratio
branch_*_actual_fbank_mask_ratio
```

Checkpointing:

```json
"checkpointing": {
  "monitors": [
    {
      "name": "val_ssl_loss",
      "mode": "min",
      "keep_top_k": 3,
      "filename_prefix": "best_ssl_loss"
    },
    {
      "name": "last",
      "mode": "last",
      "keep_top_k": 3,
      "filename_prefix": "last"
    }
  ],
  "periodic": {
    "enabled": false
  }
}
```

---

# Part K — Stage 2: FSD50K Supervised Multi-label Fine-tuning

## K1. Add config

Add:

```text
configs/fsd50k_supervised_multilabel_finetune.json
```

Use the same FSD50K dataset in supervised mode.

Recommended config:

```json
{
  "task": "fsd50k_multilabel",
  "data": {
    "dataset": "fsd50k",
    "mode": "supervised",
    "root": "/path/to/FSD50K",
    "dev_audio_dir": "/path/to/FSD50K.dev_audio",
    "eval_audio_dir": "/path/to/FSD50K.eval_audio",
    "ground_truth_dir": "/path/to/FSD50K.ground_truth",
    "vocabulary_csv": "/path/to/FSD50K.ground_truth/vocabulary.csv",
    "dev_csv": "/path/to/FSD50K.ground_truth/dev.csv",
    "eval_csv": "/path/to/FSD50K.ground_truth/eval.csv",
    "val_ratio": 0.1,
    "split_seed": 42
  },
  "model": {
    "encoder": {
      "type": "multiscale_rdt_ast",
      "init_from": "<ssl_checkpoint_path>"
    },
    "classifier": {
      "num_classes": 200
    }
  },
  "train": {
    "epochs": 120,
    "optimizer": {
      "encoder_lr": 0.00001,
      "body_lr": 0.00003,
      "head_lr": 0.0003,
      "weight_decay": 0.01
    },
    "scheduler": {
      "type": "linear_warmup_cosine",
      "warmup_ratio": 0.05
    },
    "gradient_clipping": {
      "enabled": false,
      "max_norm": 1.0
    },
    "loss": {
      "type": "bce_with_logits",
      "pos_weight": {
        "type": "sqrt_negative_over_positive",
        "cap": 10,
        "source": "train",
        "save_to_checkpoint_metadata": true
      },
      "branch_auxiliary": {
        "enabled": false
      },
      "branch_binary_auxiliary": {
        "enabled": false
      }
    }
  },
  "metrics": {
    "primary": "macro_AP",
    "secondary": "micro_AP",
    "f1_threshold": 0.5,
    "save_per_class_AP": true,
    "zero_positive_class_policy": "exclude_with_warning"
  }
}
```

---

## K2. Model

Use H0 full structure:

```text
patch tokenizer
shared stem
scale-specific adapters
frequency-attention pooling
branch MIL
top-k evidence
RDT
branch-aware gated pooling
fusion projector
multi-label classifier head [B, 200]
```

Load Stage 1 SSL checkpoint into compatible encoder/body modules.

Do not load SSL decoders.

---

## K3. Branch auxiliary off

For FSD50K supervised:

```text
branch_auxiliary.enabled = false
branch_binary_auxiliary.enabled = false
```

Do not force every branch to solve the 200-class multi-label task.

---

## K4. BCE pos_weight

Compute from train split:

\[
pos\_weight_c = \min\left(\sqrt{rac{n_c^-}{n_c^+}}, 10
ight)
\]

Pseudo-code:

```python
positive_counts = train_targets.sum(dim=0)  # [C]
negative_counts = len(train_targets) - positive_counts
pos_weight = torch.sqrt(negative_counts / positive_counts.clamp_min(1.0))
pos_weight = torch.clamp(pos_weight, max=10.0)
```

Use:

```python
BCEWithLogitsLoss(pos_weight=pos_weight)
```

Save `pos_weight` in checkpoint metadata.

---

## K5. Optimizer parameter groups

Create three groups:

```text
Group 1. SSL-pretrained encoder, LR = encoder_lr
  - patch tokenizers
  - position / scale embeddings
  - shared stem
  - scale-specific adapters
  - frequency-attention poolers

Group 2. supervised body, LR = body_lr
  - branch MIL
  - top-k evidence modules if parameterized
  - RDT
  - gated pooling
  - fusion projector

Group 3. multi-label head, LR = head_lr
```

Default:

```json
"optimizer": {
  "encoder_lr": 1e-5,
  "body_lr": 3e-5,
  "head_lr": 3e-4,
  "weight_decay": 0.01
}
```

---

## K6. Metrics

Use sigmoid probabilities:

```python
probs = logits.sigmoid()
```

Primary metrics:

```text
val_macro_AP
val_micro_AP
```

Use `average_precision_score` or equivalent.

For macro AP:

```text
exclude classes with zero positives in validation split
log warning
save per-class AP
```

Also log threshold metrics at fixed threshold 0.5:

```text
macro_F1_at_0.5
micro_F1_at_0.5
macro_recall_at_0.5
micro_recall_at_0.5
```

Do not implement class-wise threshold optimization in this task.

Checkpoint monitors:

```json
"checkpointing": {
  "monitors": [
    {
      "name": "val_macro_AP",
      "mode": "max",
      "keep_top_k": 3,
      "filename_prefix": "best_macro_AP"
    },
    {
      "name": "val_micro_AP",
      "mode": "max",
      "keep_top_k": 3,
      "filename_prefix": "best_micro_AP"
    },
    {
      "name": "val_loss",
      "mode": "min",
      "keep_top_k": 3,
      "filename_prefix": "best_loss"
    },
    {
      "name": "last",
      "mode": "last",
      "keep_top_k": 3,
      "filename_prefix": "last"
    }
  ]
}
```

---

# Part L — Stage 3: CNUH Transfer / Fine-tuning Support

## L1. Add template config

Add:

```text
configs/cnuh_4class_from_fsd50k_transfer_template.json
```

The user will later customize the CNUH recipe, so keep this as a transfer template.

Recommended:

```json
{
  "task": "cnuh_4class",
  "transfer": {
    "checkpoint_path": "<fsd50k_supervised_checkpoint>",
    "reset_classifier": true,
    "freeze_encoder": true,
    "strict": false,
    "freeze_modules": [
      "patch_tokenizers",
      "position_embeddings",
      "scale_embeddings",
      "shared_stem",
      "scale_specific_adapters",
      "frequency_attention_poolers"
    ]
  },
  "optimizer": {
    "frozen_encoder_lr": 0.0,
    "body_lr": 0.00001,
    "head_lr": 0.0003
  }
}
```

---

## L2. Transfer rules

Load from FSD50K supervised checkpoint.

Transfer:

```text
patch tokenizers
position / scale embeddings
shared stem
scale-specific adapters
frequency-attention poolers
branch MIL
RDT
branch-aware gated pooling
fusion projector
```

Discard/reset:

```text
FSD50K multi-label classifier head
CNUH classifier head newly initialized
```

Use:

```text
strict = false
```

---

## L3. Encoder freeze default

Default freeze:

```text
patch tokenizers
position / scale embeddings
shared stem
scale-specific adapters
frequency-attention poolers
```

Train:

```text
branch MIL
RDT
branch-aware gated pooling
fusion projector
CNUH classifier
```

Allow config to change this later.

---

# Part M — README Updates

Add a section:

```text
FSD50K SSL and Supervised Pretraining
```

Document the three stages:

```text
Stage 1: FSD50K masked fbank SSL
Stage 2: FSD50K supervised multi-label
Stage 3: CNUH transfer
```

Document:

```text
branch-specific tokenizer-grid input-space masking
branch-wise Conv1d fbank decoder
mean branch masked MSE
Stage 2 branch auxiliary off
Stage 2 macro AP / micro AP monitors
Stage 3 encoder freeze default
```

Mention config files:

```text
configs/fsd50k_ssl_masked_fbank_pretrain.json
configs/fsd50k_supervised_multilabel_finetune.json
configs/cnuh_4class_from_fsd50k_transfer_template.json
```

---

# Part N — Tests

Add or update tests.

---

## N1. FSD50K dataset tests

Test:

```text
ssl mode returns input_values only
supervised mode returns input_values + multi-hot labels [200]
vocabulary parsing works
unknown label handling works
train split uses random crop
val/eval split uses deterministic crop
```

Use small synthetic metadata/audio fixtures if full FSD50K is unavailable.

---

## N2. Mask generator tests

Test expected patch grid sizes:

```text
branch 0: [127, 8]
branch 1: [255, 4]
branch 2: [511, 2]
branch 3: [1023, 1]
```

Test:

```text
token_mask_ratio approximately 0.4
fbank mask shape [B,1024,128]
actual fbank mask ratio computed
overlap union works
warning triggers above threshold
```

---

## N3. SSL wrapper tests

Test:

```text
forward SSL succeeds
loss is scalar
branch_losses shape [4]
actual_mask_ratios shape [4]
token_mask_ratios shape [4]
decoder outputs [B,1024,128] when returned
supervised forward still works
```

---

## N4. Decoder tests

Test:

```text
BranchConv1dFbankDecoder input [B,T_s,D]
output [B,1024,128]
LayerNorm enabled
two Conv1d layers
separate decoder per branch
```

---

## N5. SSL loss tests

Test masked MSE:

```text
loss uses only masked bins
branch losses averaged
reconstructed_average not used in loss
```

---

## N6. FSD50K supervised tests

Test:

```text
multi-label head output [B,200]
BCEWithLogitsLoss with pos_weight
pos_weight formula sqrt(neg/pos), cap 10
pos_weight saved in checkpoint metadata
branch auxiliary disabled
```

---

## N7. AP metric tests

Test:

```text
macro AP
micro AP
per-class AP saved
zero-positive class excluded with warning
F1 at threshold 0.5
```

---

## N8. Transfer tests

Test:

```text
load Stage 1 SSL checkpoint into Stage 2 supervised with strict=false
SSL decoder weights ignored
load Stage 2 supervised checkpoint into CNUH with strict=false
FSD50K classifier reset
freeze_encoder true freezes intended modules
body/head parameter groups created correctly
```

---

# Part O — Acceptance Criteria

The task is complete when:

1. FSD50K dataset supports SSL and supervised mode.
2. SSL mode returns normalized fbank only.
3. Supervised mode returns normalized fbank + multi-hot labels.
4. Branch-specific tokenizer-grid input-space masking is implemented.
5. Token mask ratio default is 0.4.
6. Actual fbank mask ratio is logged per branch.
7. Mask ratio warning is logged above threshold, no clipping.
8. Masked fbank input is branch-specific.
9. Existing supervised forward remains functional.
10. SSL wrapper or SSL forward is implemented.
11. Branch-wise separate Conv1d decoders are implemented.
12. Conv1d decoder has 2 layers, channels 256, kernel 5, dropout 0.1, LayerNorm.
13. SSL loss is mean branch masked MSE only.
14. Average reconstruction is not used in SSL loss.
15. Stage 1 config exists.
16. Stage 2 FSD50K supervised config exists.
17. Stage 2 uses BCEWithLogitsLoss with sqrt neg/pos pos_weight cap 10.
18. Stage 2 branch auxiliaries are disabled.
19. Stage 2 macro AP / micro AP metrics are implemented.
20. Per-class AP is saved.
21. Stage 3 CNUH transfer template exists.
22. Stage 3 transfer resets classifier and freezes encoder by default.
23. Tests pass.

---

# Part P — Avoid These Mistakes

- Do not break existing supervised CNUH forward/training.
- Do not use a single global mask for all branches in SSL.
- Do not use token-space mask after tokenizer for the first implementation.
- Do not use learned/gated branch averaging in SSL.
- Do not use averaged reconstruction in the SSL loss.
- Do not enable branch auxiliary in FSD50K supervised stage.
- Do not add global binary auxiliary.
- Do not implement class-wise threshold optimization for FSD50K F1.
- Do not use periodic checkpointing for SSL.
- Do not load SSL decoder into FSD50K supervised model.
- Do not load FSD50K classifier head into CNUH classifier.
- Do not freeze MIL/RDT/gate/fusion by default in CNUH transfer.
- Do not silently ignore unknown FSD50K labels.
- Do not compute AP over zero-positive classes without warning.

---

# Final Intended Pipeline

```text
Stage 1:
FSD50K SSL
  normalized fbank reconstruction
  branch-specific tokenizer-grid input-space masking
  branch-wise Conv1d decoder
  mean branch masked MSE

Stage 2:
FSD50K supervised multi-label
  initialize from Stage 1
  H0 full structure
  BCEWithLogitsLoss
  pos_weight sqrt(neg/pos), cap 10
  macro AP / micro AP monitors
  branch auxiliary off

Stage 3:
CNUH 4-class transfer
  initialize from Stage 2
  reset CNUH classifier
  freeze encoder by default
  train MIL/RDT/gate/fusion/classifier
```
