from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

import torch
from torch import Tensor, nn

from src.models.classifier import ClassifierDims, LinearClassifier, MlpClassifier

PATCH_INIT_STD = 0.02


@dataclass(frozen=True)
class AstFeatureDims:
    num_mel_bins: int
    max_length: int


@dataclass(frozen=True)
class PatchBranchConfig:
    patch_size: tuple[int, int]
    stride: tuple[int, int]


def default_patch_branches() -> tuple[PatchBranchConfig, ...]:
    return (
        PatchBranchConfig(patch_size=(16, 16), stride=(8, 16)),
        PatchBranchConfig(patch_size=(8, 32), stride=(4, 32)),
        PatchBranchConfig(patch_size=(4, 64), stride=(2, 64)),
        PatchBranchConfig(patch_size=(2, 128), stride=(1, 128)),
    )


@dataclass(frozen=True)
class EncoderAdaptationConfig:
    mode: Literal["frozen", "full", "partial"] = "full"
    num_layers: int = 0


@dataclass(frozen=True)
class RdtConfig:
    enabled: bool = False
    steps: int = 3
    top_tokens_per_branch: int = 2
    gated_residual: bool = True
    layerscale_init: float = 0.01
    evidence_score_source: Literal[
        "attention_weight", "attention_logit", "instance_logit"
    ] = "attention_weight"
    exclude_branches_from_evidence: tuple[int, ...] = ()


@dataclass(frozen=True)
class MilConfig:
    attention_temperature: float = 1.0


@dataclass(frozen=True)
class EvidencePoolingConfig:
    type: Literal["mean", "branch_gated"] = "mean"
    gate_hidden_size: int | None = None
    dropout: float = 0.1
    temperature: float = 1.0


@dataclass(frozen=True)
class BranchEventDropoutConfig:
    enabled: bool = False
    probability: float = 0.0
    mode: Literal["zero_mask"] = "zero_mask"
    min_keep_tokens: int = 1


@dataclass(frozen=True)
class SelectedEvidenceDropoutConfig:
    enabled: bool = False
    probability: float = 0.0
    mode: Literal["zero"] = "zero"
    min_keep_per_branch: int = 1


@dataclass(frozen=True)
class TokenAugmentationConfig:
    branch_event_dropout: BranchEventDropoutConfig = field(
        default_factory=BranchEventDropoutConfig
    )
    selected_evidence_dropout: SelectedEvidenceDropoutConfig = field(
        default_factory=SelectedEvidenceDropoutConfig
    )


@dataclass(frozen=True)
class MultiScaleRdtArchitectureConfig:
    hidden_size: int = 192
    num_attention_heads: int = 4
    mlp_ratio: float = 2.0
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6
    shared_stem_depth: int = 2
    adapter_depth: int = 1
    patch_branches: tuple[PatchBranchConfig, ...] = field(
        default_factory=default_patch_branches
    )
    rdt: RdtConfig = field(default_factory=RdtConfig)
    mil: MilConfig = field(default_factory=MilConfig)
    evidence_pooling: EvidencePoolingConfig = field(
        default_factory=EvidencePoolingConfig
    )
    token_augmentation: TokenAugmentationConfig = field(
        default_factory=TokenAugmentationConfig
    )


@dataclass(frozen=True)
class MultiScaleRdtEncoderConfig:
    feature_dims: AstFeatureDims
    type: Literal["multiscale_rdt_ast"] = "multiscale_rdt_ast"
    adaptation: EncoderAdaptationConfig = field(default_factory=EncoderAdaptationConfig)
    architecture: MultiScaleRdtArchitectureConfig = field(
        default_factory=MultiScaleRdtArchitectureConfig
    )


@dataclass(frozen=True)
class ClassifierConfig:
    type: Literal["linear", "mlp"] = "linear"
    hidden_dim: int = 256
    dropout: float = 0.0
    pooling: Literal["latent_mean"] = "latent_mean"


@dataclass(frozen=True)
class MultiScaleRdtAstModelConfig:
    encoder: MultiScaleRdtEncoderConfig
    classifier: ClassifierConfig
    num_classes: int


@dataclass(frozen=True)
class AstModelOutput:
    logits: Tensor
    pooled_embedding: Tensor
    branch_logits: Tensor | None = None
    branch_binary_logits: Tensor | None = None
    branch_attention_weights: tuple[Tensor, ...] | None = None
    selected_evidence_tokens: Tensor | None = None
    selected_evidence_indices: Tensor | None = None
    selected_evidence_scores: Tensor | None = None
    selected_evidence_branch_ids: Tensor | None = None
    evidence_score_source: str | None = None
    evidence_pooling_type: str | None = None
    evidence_gate_weights: Tensor | None = None
    evidence_gate_entropy: Tensor | None = None
    branch_evidence_norms: Tensor | None = None
    selected_evidence_dropout_mask: Tensor | None = None
    selected_evidence_keep_ratio: Tensor | None = None


@dataclass(frozen=True)
class MultiScaleEncoderOutput:
    branch_event_tokens: tuple[Tensor, ...]
    context_tokens: Tensor


@dataclass(frozen=True)
class BranchMilOutput:
    logits: Tensor
    attention_weights: Tensor
    attention_logits: Tensor
    instance_logits: Tensor | None
    embedding: Tensor


@dataclass(frozen=True)
class SelectedEvidence:
    tokens: Tensor
    indices: Tensor
    scores: Tensor


@dataclass(frozen=True)
class EvidencePoolingOutput:
    pooled_embedding: Tensor
    gate_weights: Tensor | None = None
    gate_entropy: Tensor | None = None
    branch_evidence_summary: Tensor | None = None
    branch_evidence_norms: Tensor | None = None


def compute_token_grid(
    *,
    feature_dims: AstFeatureDims,
    patch_branch: PatchBranchConfig,
) -> tuple[int, int]:
    patch_t, patch_f = patch_branch.patch_size
    stride_t, stride_f = patch_branch.stride
    n_t = ((feature_dims.max_length - patch_t) // stride_t) + 1
    n_f = ((feature_dims.num_mel_bins - patch_f) // stride_f) + 1
    return n_t, n_f


def compute_token_count(
    *,
    feature_dims: AstFeatureDims,
    patch_branch: PatchBranchConfig,
) -> int:
    n_t, n_f = compute_token_grid(
        feature_dims=feature_dims,
        patch_branch=patch_branch,
    )
    return n_t * n_f


def select_top_tokens(
    tokens: Tensor, scores: Tensor, *, top_k: int
) -> SelectedEvidence:
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape (B, T, D), got {tuple(tokens.shape)}")
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (B, T), got {tuple(scores.shape)}")
    if tokens.shape[:2] != scores.shape:
        raise ValueError(
            "scores must align with tokens on batch/time dimensions; "
            f"got tokens={tuple(tokens.shape)} scores={tuple(scores.shape)}"
        )
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if top_k > int(tokens.shape[1]):
        raise ValueError(f"top_k={top_k} exceeds token length {int(tokens.shape[1])}")
    top = scores.topk(top_k, dim=1)
    top_indices = top.indices
    selected_tokens = tokens.gather(
        1,
        top_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
    )
    return SelectedEvidence(
        tokens=selected_tokens,
        indices=top_indices,
        scores=top.values,
    )


def get_evidence_scores(
    mil_output: BranchMilOutput,
    *,
    source: str,
) -> Tensor:
    if source == "attention_weight":
        return mil_output.attention_weights
    if source == "attention_logit":
        return mil_output.attention_logits
    if source == "instance_logit":
        if mil_output.instance_logits is None:
            raise ValueError(
                "instance_logit evidence selection requires instance_logits"
            )
        if mil_output.instance_logits.ndim == 2:
            return mil_output.instance_logits
        if mil_output.instance_logits.ndim == 3:
            return mil_output.instance_logits.max(dim=-1).values
        raise ValueError(
            "instance_logits must have shape (B, T) or (B, T, C), "
            f"got {tuple(mil_output.instance_logits.shape)}"
        )
    raise ValueError(f"Unsupported evidence_score_source: {source}")


class MeanEvidencePooling(nn.Module):
    def forward(
        self,
        evidence_tokens: Tensor,
        branch_ids: Tensor | None = None,
    ) -> EvidencePoolingOutput:
        del branch_ids
        if evidence_tokens.ndim != 3:
            raise ValueError(
                "evidence_tokens must have shape (B, K, D), "
                f"got {tuple(evidence_tokens.shape)}"
            )
        return EvidencePoolingOutput(pooled_embedding=evidence_tokens.mean(dim=1))


class BranchAwareGatedEvidencePooling(nn.Module):
    def __init__(self, *, hidden_size: int, cfg: EvidencePoolingConfig) -> None:
        super().__init__()
        self.temperature = cfg.temperature
        gate_hidden_size = cfg.gate_hidden_size or max(hidden_size // 2, 1)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, gate_hidden_size),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(gate_hidden_size, 1),
        )

    def _summarize_by_branch(
        self,
        evidence_tokens: Tensor,
        branch_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        branch_values = torch.unique(branch_ids.detach().cpu()).sort().values.tolist()
        if not branch_values:
            raise ValueError("branch_ids must contain at least one branch")

        branch_summaries: list[Tensor] = []
        branch_present_masks: list[Tensor] = []
        for branch_value in branch_values:
            branch_mask = branch_ids == int(branch_value)
            branch_counts = branch_mask.sum(dim=1)
            branch_sum = (
                evidence_tokens * branch_mask.unsqueeze(-1).to(evidence_tokens.dtype)
            ).sum(dim=1)
            branch_summaries.append(
                branch_sum / branch_counts.clamp_min(1).unsqueeze(-1)
            )
            branch_present_masks.append(branch_counts > 0)

        return (
            torch.stack(branch_summaries, dim=1),
            torch.stack(branch_present_masks, dim=1),
        )

    def forward(
        self,
        evidence_tokens: Tensor,
        branch_ids: Tensor | None = None,
    ) -> EvidencePoolingOutput:
        if evidence_tokens.ndim != 3:
            raise ValueError(
                "evidence_tokens must have shape (B, K, D), "
                f"got {tuple(evidence_tokens.shape)}"
            )
        if branch_ids is None:
            raise ValueError("branch_gated evidence pooling requires branch_ids")
        if branch_ids.ndim != 2:
            raise ValueError(
                f"branch_ids must have shape (B, K), got {tuple(branch_ids.shape)}"
            )
        if branch_ids.shape != evidence_tokens.shape[:2]:
            raise ValueError(
                "branch_ids must align with evidence_tokens on batch/token "
                f"dimensions; got evidence_tokens={tuple(evidence_tokens.shape)} "
                f"branch_ids={tuple(branch_ids.shape)}"
            )

        branch_evidence_summary, branch_present_mask = self._summarize_by_branch(
            evidence_tokens,
            branch_ids,
        )
        if torch.any(branch_present_mask.sum(dim=1) == 0):
            raise ValueError("Every sample must contain at least one evidence branch")

        gate_logits = self.gate(branch_evidence_summary).squeeze(-1)
        gate_logits = gate_logits.masked_fill(~branch_present_mask, float("-inf"))
        gate_weights = torch.softmax(gate_logits / self.temperature, dim=1)
        gate_entropy = -(
            gate_weights * (gate_weights + torch.finfo(gate_weights.dtype).eps).log()
        ).sum(dim=1)
        pooled_embedding = torch.sum(
            gate_weights.unsqueeze(-1) * branch_evidence_summary,
            dim=1,
        )
        branch_evidence_norms = branch_evidence_summary.norm(dim=-1)
        return EvidencePoolingOutput(
            pooled_embedding=pooled_embedding,
            gate_weights=gate_weights,
            gate_entropy=gate_entropy,
            branch_evidence_summary=branch_evidence_summary,
            branch_evidence_norms=branch_evidence_norms,
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        mlp_ratio: float,
        dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(self.norm(x))


class PatchTokenizer(nn.Module):
    def __init__(self, *, hidden_size: int, branch: PatchBranchConfig) -> None:
        super().__init__()
        self.branch = branch
        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_size,
            kernel_size=branch.patch_size,
            stride=branch.stride,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class FrequencyAttentionPooler(nn.Module):
    def __init__(self, *, hidden_size: int, layer_norm_eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, branch_grid: Tensor) -> Tensor:
        norm_grid = self.norm(branch_grid)
        attention_logits = self.score(norm_grid).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=2)
        return torch.sum(branch_grid * attention_weights.unsqueeze(-1), dim=2)


class BranchEventTokenDropout(nn.Module):
    def __init__(self, cfg: BranchEventDropoutConfig) -> None:
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def _ensure_min_keep(keep_mask: Tensor, min_keep_tokens: int) -> Tensor:
        for batch_index in range(keep_mask.shape[0]):
            keep_count = int(keep_mask[batch_index].sum().item())
            if keep_count >= min_keep_tokens:
                continue
            dropped_positions = torch.where(~keep_mask[batch_index])[0]
            if dropped_positions.numel() == 0:
                continue
            needed = min(min_keep_tokens - keep_count, int(dropped_positions.numel()))
            order = torch.randperm(
                int(dropped_positions.numel()),
                device=keep_mask.device,
            )
            keep_mask[batch_index, dropped_positions[order[:needed]]] = True
        return keep_mask

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 3:
            raise ValueError(
                f"tokens must have shape (B, T, D), got {tuple(tokens.shape)}"
            )
        batch_size, token_count, _ = tokens.shape
        all_true = torch.ones(
            batch_size,
            token_count,
            dtype=torch.bool,
            device=tokens.device,
        )
        if not self.training or not self.cfg.enabled or self.cfg.probability <= 0:
            return tokens, all_true

        keep_mask = torch.rand(batch_size, token_count, device=tokens.device) >= float(
            self.cfg.probability
        )
        min_keep_tokens = min(int(self.cfg.min_keep_tokens), token_count)
        keep_mask = self._ensure_min_keep(keep_mask, min_keep_tokens)
        dropped_tokens = tokens * keep_mask.unsqueeze(-1).to(tokens.dtype)
        return dropped_tokens, keep_mask


class SelectedEvidenceDropout(nn.Module):
    def __init__(self, cfg: SelectedEvidenceDropoutConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        selected_tokens: Tensor,
        selected_branch_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if selected_tokens.ndim != 3:
            raise ValueError(
                "selected_tokens must have shape (B, K, D), "
                f"got {tuple(selected_tokens.shape)}"
            )
        if selected_branch_ids.ndim != 2:
            raise ValueError(
                "selected_branch_ids must have shape (B, K), "
                f"got {tuple(selected_branch_ids.shape)}"
            )
        if selected_branch_ids.shape != selected_tokens.shape[:2]:
            raise ValueError(
                "selected_branch_ids must align with selected_tokens on batch/token "
                f"dimensions; got selected_tokens={tuple(selected_tokens.shape)} "
                f"selected_branch_ids={tuple(selected_branch_ids.shape)}"
            )

        batch_size, selected_count, _ = selected_tokens.shape
        keep_mask = torch.ones(
            batch_size,
            selected_count,
            dtype=torch.bool,
            device=selected_tokens.device,
        )
        if not self.training or not self.cfg.enabled or self.cfg.probability <= 0:
            return selected_tokens, keep_mask

        for batch_index in range(batch_size):
            branch_ids = torch.unique(selected_branch_ids[batch_index])
            for branch_id in branch_ids:
                positions = torch.where(selected_branch_ids[batch_index] == branch_id)[
                    0
                ]
                if positions.numel() == 0:
                    continue
                branch_keep = torch.rand(
                    int(positions.numel()), device=selected_tokens.device
                ) >= float(self.cfg.probability)
                min_keep = min(
                    int(self.cfg.min_keep_per_branch), int(positions.numel())
                )
                keep_count = int(branch_keep.sum().item())
                if keep_count < min_keep:
                    dropped_positions = torch.where(~branch_keep)[0]
                    needed = min(min_keep - keep_count, int(dropped_positions.numel()))
                    order = torch.randperm(
                        int(dropped_positions.numel()),
                        device=selected_tokens.device,
                    )
                    branch_keep[dropped_positions[order[:needed]]] = True
                keep_mask[batch_index, positions] = branch_keep

        dropped_tokens = selected_tokens * keep_mask.unsqueeze(-1).to(
            selected_tokens.dtype
        )
        return dropped_tokens, keep_mask


class BranchMilHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        output_dim: int,
        layer_norm_eps: float,
        attention_temperature: float,
    ) -> None:
        super().__init__()
        self.attention_temperature = attention_temperature
        self.token_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = nn.Linear(hidden_size, 1)
        self.instance_logit_proj = nn.Linear(hidden_size, output_dim)
        self.embedding_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.logit_proj = nn.Linear(hidden_size, output_dim)

    def forward(
        self,
        event_tokens: Tensor,
        token_mask: Tensor | None = None,
    ) -> BranchMilOutput:
        normalized_tokens = self.token_norm(event_tokens)
        attention_logits = self.attention(normalized_tokens).squeeze(-1)
        if token_mask is not None:
            if token_mask.shape != attention_logits.shape:
                raise ValueError(
                    "token_mask must align with event token batch/time dimensions; "
                    f"got token_mask={tuple(token_mask.shape)} "
                    f"event_tokens={tuple(event_tokens.shape)}"
                )
            attention_logits = attention_logits.masked_fill(~token_mask, -1e4)
        attention_weights = torch.softmax(
            attention_logits / self.attention_temperature,
            dim=1,
        )
        embedding = torch.sum(
            event_tokens * attention_weights.unsqueeze(-1),
            dim=1,
        )
        instance_logits = self.instance_logit_proj(normalized_tokens)
        if instance_logits.ndim == 3 and instance_logits.shape[-1] == 1:
            instance_logits = instance_logits.squeeze(-1)
        logits = self.logit_proj(self.embedding_norm(embedding))
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        return BranchMilOutput(
            logits=logits,
            attention_weights=attention_weights,
            attention_logits=attention_logits,
            instance_logits=instance_logits,
            embedding=embedding,
        )


class MultiScalePatchStemAdapterEncoder(nn.Module):
    def __init__(self, cfg: MultiScaleRdtEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        architecture = cfg.architecture
        hidden_size = architecture.hidden_size
        self.branch_token_grids = tuple(
            compute_token_grid(feature_dims=cfg.feature_dims, patch_branch=branch)
            for branch in architecture.patch_branches
        )
        self.branch_time_lengths = tuple(
            time_steps for time_steps, _ in self.branch_token_grids
        )
        self.branch_frequency_lengths = tuple(
            freq_steps for _, freq_steps in self.branch_token_grids
        )
        self.branch_token_counts = tuple(
            time_steps * freq_steps
            for time_steps, freq_steps in self.branch_token_grids
        )
        self.total_token_count = sum(self.branch_token_counts)
        self.total_temporal_length = sum(self.branch_time_lengths)
        self.patch_tokenizers = nn.ModuleList(
            PatchTokenizer(hidden_size=hidden_size, branch=branch)
            for branch in architecture.patch_branches
        )
        self.position_embeddings = nn.ParameterList(
            nn.Parameter(torch.empty(1, token_count, hidden_size))
            for token_count in self.branch_token_counts
        )
        self.scale_embeddings = nn.Parameter(
            torch.empty(len(architecture.patch_branches), hidden_size)
        )
        self.shared_stem = nn.ModuleList(
            TransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=architecture.num_attention_heads,
                mlp_ratio=architecture.mlp_ratio,
                dropout=architecture.hidden_dropout_prob,
                attention_dropout=architecture.attention_probs_dropout_prob,
                layer_norm_eps=architecture.layer_norm_eps,
            )
            for _ in range(architecture.shared_stem_depth)
        )
        self.adapters = nn.ModuleList(
            nn.ModuleList(
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=architecture.num_attention_heads,
                    mlp_ratio=architecture.mlp_ratio,
                    dropout=architecture.hidden_dropout_prob,
                    attention_dropout=architecture.attention_probs_dropout_prob,
                    layer_norm_eps=architecture.layer_norm_eps,
                )
                for _ in range(architecture.adapter_depth)
            )
            for _ in architecture.patch_branches
        )
        self.frequency_poolers = nn.ModuleList(
            FrequencyAttentionPooler(
                hidden_size=hidden_size,
                layer_norm_eps=architecture.layer_norm_eps,
            )
            for _ in architecture.patch_branches
        )
        self.output_norm = nn.LayerNorm(hidden_size, eps=architecture.layer_norm_eps)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for position_embedding in self.position_embeddings:
            nn.init.normal_(position_embedding, std=PATCH_INIT_STD)
        nn.init.normal_(self.scale_embeddings, std=PATCH_INIT_STD)

    def forward(self, x: Tensor) -> MultiScaleEncoderOutput:
        branch_event_outputs: list[Tensor] = []
        batch_size = x.shape[0]
        for branch_index, tokenizer in enumerate(self.patch_tokenizers):
            branch_tokens = tokenizer(x)
            branch_tokens = (
                branch_tokens
                + self.position_embeddings[branch_index]
                + self.scale_embeddings[branch_index].view(1, 1, -1)
            )
            for block in self.shared_stem:
                branch_tokens = block(branch_tokens)
            branch_adapter = cast(nn.ModuleList, self.adapters[branch_index])
            for block in branch_adapter:
                branch_tokens = block(branch_tokens)

            time_steps, freq_steps = self.branch_token_grids[branch_index]
            branch_grid = branch_tokens.reshape(
                batch_size,
                time_steps,
                freq_steps,
                branch_tokens.shape[-1],
            )
            frequency_pooler = cast(
                FrequencyAttentionPooler,
                self.frequency_poolers[branch_index],
            )
            branch_event_tokens = frequency_pooler(branch_grid)
            branch_event_outputs.append(self.output_norm(branch_event_tokens))

        context_tokens = torch.cat(branch_event_outputs, dim=1)
        return MultiScaleEncoderOutput(
            branch_event_tokens=tuple(branch_event_outputs),
            context_tokens=context_tokens,
        )


class RdtRefinementBlock(nn.Module):
    def __init__(self, cfg: MultiScaleRdtArchitectureConfig) -> None:
        super().__init__()
        self.gated_residual = cfg.rdt.gated_residual
        self.latent_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.latent_self_attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            dropout=cfg.attention_probs_dropout_prob,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.context_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            dropout=cfg.attention_probs_dropout_prob,
            batch_first=True,
        )
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.feed_forward = FeedForwardBlock(
            hidden_size=cfg.hidden_size,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.hidden_dropout_prob,
            layer_norm_eps=cfg.layer_norm_eps,
        )
        self.self_attn_scale: nn.Parameter | None
        self.cross_attn_scale: nn.Parameter | None
        self.ffn_scale: nn.Parameter | None
        if self.gated_residual:
            init = float(cfg.rdt.layerscale_init)
            self.self_attn_scale = nn.Parameter(torch.full((cfg.hidden_size,), init))
            self.cross_attn_scale = nn.Parameter(torch.full((cfg.hidden_size,), init))
            self.ffn_scale = nn.Parameter(torch.full((cfg.hidden_size,), init))
        else:
            self.self_attn_scale = None
            self.cross_attn_scale = None
            self.ffn_scale = None

    def _apply_residual(
        self,
        x: Tensor,
        delta: Tensor,
        scale: nn.Parameter | None,
    ) -> Tensor:
        if scale is None:
            return x + delta
        return x + scale.view(1, 1, -1) * delta

    def forward(self, latent_states: Tensor, evidence_memory: Tensor) -> Tensor:
        norm_latent = self.latent_norm(latent_states)
        self_attended, _ = self.latent_self_attn(
            norm_latent,
            norm_latent,
            norm_latent,
            need_weights=False,
        )
        latent_states = self._apply_residual(
            latent_states,
            self.dropout(self_attended),
            self.self_attn_scale,
        )

        norm_queries = self.query_norm(latent_states)
        norm_memory = self.context_norm(evidence_memory)
        cross_attended, _ = self.cross_attn(
            norm_queries,
            norm_memory,
            norm_memory,
            need_weights=False,
        )
        latent_states = self._apply_residual(
            latent_states,
            self.dropout(cross_attended),
            self.cross_attn_scale,
        )

        ff_delta = self.feed_forward(latent_states)
        return self._apply_residual(latent_states, ff_delta, self.ffn_scale)


class MultiScaleRdtAstModel(nn.Module):
    def __init__(self, cfg: MultiScaleRdtAstModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        architecture = cfg.encoder.architecture
        self.encoder = MultiScalePatchStemAdapterEncoder(cfg.encoder)
        output_dim = 1 if cfg.num_classes == 2 else cfg.num_classes
        self.branch_mil_heads = nn.ModuleList(
            BranchMilHead(
                hidden_size=architecture.hidden_size,
                output_dim=output_dim,
                layer_norm_eps=architecture.layer_norm_eps,
                attention_temperature=architecture.mil.attention_temperature,
            )
            for _ in architecture.patch_branches
        )
        self.branch_binary_head = nn.Linear(architecture.hidden_size, 1)
        branch_event_dropout_cfg = architecture.token_augmentation.branch_event_dropout
        self.branch_event_dropout = (
            BranchEventTokenDropout(branch_event_dropout_cfg)
            if branch_event_dropout_cfg.enabled
            else None
        )
        selected_evidence_dropout_cfg = (
            architecture.token_augmentation.selected_evidence_dropout
        )
        self.selected_evidence_dropout = (
            SelectedEvidenceDropout(selected_evidence_dropout_cfg)
            if selected_evidence_dropout_cfg.enabled
            else None
        )
        self.rdt_block = (
            RdtRefinementBlock(architecture) if architecture.rdt.enabled else None
        )
        if architecture.evidence_pooling.type == "mean":
            self.evidence_pooler: nn.Module = MeanEvidencePooling()
        elif architecture.evidence_pooling.type == "branch_gated":
            self.evidence_pooler = BranchAwareGatedEvidencePooling(
                hidden_size=architecture.hidden_size,
                cfg=architecture.evidence_pooling,
            )
        else:
            raise ValueError(
                "Unsupported evidence pooling type: "
                f"{architecture.evidence_pooling.type}"
            )
        fusion_input_dim = (2 * architecture.hidden_size) + (
            len(architecture.patch_branches) * output_dim
        )
        self.fusion_projector = nn.Sequential(
            nn.Linear(fusion_input_dim, architecture.hidden_size),
            nn.GELU(),
            nn.Dropout(cfg.classifier.dropout),
        )
        classifier_dims = ClassifierDims(
            in_dim=architecture.hidden_size,
            num_classes=output_dim,
            hidden_dim=cfg.classifier.hidden_dim,
            dropout=cfg.classifier.dropout,
        )
        if cfg.classifier.type == "linear":
            self.classifier: nn.Module = LinearClassifier(classifier_dims)
        elif cfg.classifier.type == "mlp":
            self.classifier = MlpClassifier(classifier_dims)
        else:
            raise ValueError(f"Unsupported classifier type: {cfg.classifier.type}")

    def encoder_side_modules(self) -> tuple[nn.Module, ...]:
        return (self.encoder, self.branch_mil_heads, self.branch_binary_head)

    def head_side_modules(self) -> tuple[nn.Module, ...]:
        modules: list[nn.Module] = [
            self.evidence_pooler,
            self.fusion_projector,
            self.classifier,
        ]
        if self.rdt_block is not None:
            modules.insert(0, self.rdt_block)
        return tuple(modules)

    def _stack_branch_logits(
        self,
        branch_logits: list[Tensor],
    ) -> Tensor:
        if self.cfg.num_classes == 2:
            return torch.stack(branch_logits, dim=1)
        return torch.stack(branch_logits, dim=1)

    def forward(self, input_values: Tensor) -> AstModelOutput:
        if input_values.ndim != 3:
            raise ValueError(
                "input_values must have shape (B, max_length, num_mel_bins), "
                f"got {tuple(input_values.shape)}"
            )
        expected_shape = (
            self.cfg.encoder.feature_dims.max_length,
            self.cfg.encoder.feature_dims.num_mel_bins,
        )
        actual_shape = (int(input_values.shape[1]), int(input_values.shape[2]))
        if actual_shape != expected_shape:
            raise ValueError(
                "input_values feature dims do not match model feature dims. "
                f"expected {expected_shape}, got {actual_shape}"
            )

        encoder_output = self.encoder(input_values.unsqueeze(1))
        branch_logits: list[Tensor] = []
        branch_binary_logits: list[Tensor] = []
        branch_embeddings: list[Tensor] = []
        branch_attention_weights: list[Tensor] = []
        selected_tokens: list[Tensor] = []
        selected_indices: list[Tensor] = []
        selected_scores: list[Tensor] = []
        selected_branch_ids: list[Tensor] = []
        top_k = self.cfg.encoder.architecture.rdt.top_tokens_per_branch
        evidence_source = self.cfg.encoder.architecture.rdt.evidence_score_source
        excluded_branches = set(
            self.cfg.encoder.architecture.rdt.exclude_branches_from_evidence
        )
        for branch_index, (branch_tokens, branch_head) in enumerate(
            zip(
                encoder_output.branch_event_tokens,
                self.branch_mil_heads,
                strict=True,
            )
        ):
            branch_token_mask = None
            if self.branch_event_dropout is not None:
                branch_tokens, branch_token_mask = self.branch_event_dropout(
                    branch_tokens
                )
            mil_output = cast(
                BranchMilOutput,
                branch_head(branch_tokens, token_mask=branch_token_mask),
            )
            branch_logits.append(mil_output.logits)
            branch_binary_logits.append(
                self.branch_binary_head(mil_output.embedding).squeeze(-1)
            )
            branch_embeddings.append(mil_output.embedding)
            branch_attention_weights.append(mil_output.attention_weights)
            if branch_index in excluded_branches:
                continue
            evidence_scores = get_evidence_scores(
                mil_output,
                source=evidence_source,
            )
            selected = select_top_tokens(
                branch_tokens,
                evidence_scores,
                top_k=top_k,
            )
            selected_tokens.append(selected.tokens)
            selected_indices.append(selected.indices)
            selected_scores.append(selected.scores)
            selected_branch_ids.append(
                torch.full_like(selected.indices, fill_value=branch_index)
            )

        stacked_branch_logits = self._stack_branch_logits(branch_logits)
        stacked_branch_binary_logits = torch.stack(branch_binary_logits, dim=1)
        if not selected_tokens:
            raise ValueError("At least one branch must contribute selected evidence")
        selected_evidence_tokens = torch.cat(selected_tokens, dim=1)
        selected_evidence_indices = torch.cat(selected_indices, dim=1)
        selected_evidence_scores = torch.cat(selected_scores, dim=1)
        selected_evidence_branch_ids = torch.cat(selected_branch_ids, dim=1)
        evidence_tokens = selected_evidence_tokens
        selected_evidence_dropout_mask = None
        selected_evidence_keep_ratio = None
        if self.selected_evidence_dropout is not None:
            evidence_tokens, selected_evidence_dropout_mask = (
                self.selected_evidence_dropout(
                    evidence_tokens,
                    selected_evidence_branch_ids,
                )
            )
            selected_evidence_keep_ratio = selected_evidence_dropout_mask.to(
                evidence_tokens.dtype
            ).mean(dim=1)
        if self.rdt_block is not None:
            for _ in range(self.cfg.encoder.architecture.rdt.steps):
                evidence_tokens = self.rdt_block(
                    evidence_tokens,
                    encoder_output.context_tokens,
                )

        pooling_output = self.evidence_pooler(
            evidence_tokens,
            selected_evidence_branch_ids,
        )
        evidence_embedding = pooling_output.pooled_embedding
        branch_embedding_mean = torch.stack(branch_embeddings, dim=1).mean(dim=1)
        branch_logit_features = stacked_branch_logits.reshape(
            stacked_branch_logits.shape[0],
            -1,
        )
        fusion_input = torch.cat(
            [evidence_embedding, branch_embedding_mean, branch_logit_features],
            dim=1,
        )
        pooled_embedding = self.fusion_projector(fusion_input)
        logits = self.classifier(pooled_embedding)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        return AstModelOutput(
            logits=logits,
            pooled_embedding=pooled_embedding,
            branch_logits=stacked_branch_logits,
            branch_binary_logits=stacked_branch_binary_logits,
            branch_attention_weights=tuple(branch_attention_weights),
            selected_evidence_tokens=selected_evidence_tokens,
            selected_evidence_indices=selected_evidence_indices,
            selected_evidence_scores=selected_evidence_scores,
            selected_evidence_branch_ids=selected_evidence_branch_ids,
            evidence_score_source=evidence_source,
            evidence_pooling_type=self.cfg.encoder.architecture.evidence_pooling.type,
            evidence_gate_weights=pooling_output.gate_weights,
            evidence_gate_entropy=pooling_output.gate_entropy,
            branch_evidence_norms=pooling_output.branch_evidence_norms,
            selected_evidence_dropout_mask=selected_evidence_dropout_mask,
            selected_evidence_keep_ratio=selected_evidence_keep_ratio,
        )
