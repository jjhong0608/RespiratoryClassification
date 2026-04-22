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
class MultiScaleRdtArchitectureConfig:
    hidden_size: int = 384
    num_attention_heads: int = 6
    mlp_ratio: float = 4.0
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6
    shared_stem_depth: int = 2
    adapter_depth: int = 1
    latent_query_count: int = 8
    rdt_steps: int = 3
    patch_branches: tuple[PatchBranchConfig, ...] = field(
        default_factory=default_patch_branches
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
        return x + self.mlp(self.norm(x))


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


class MultiScalePatchStemAdapterEncoder(nn.Module):
    def __init__(self, cfg: MultiScaleRdtEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        architecture = cfg.architecture
        hidden_size = architecture.hidden_size
        self.patch_tokenizers = nn.ModuleList(
            PatchTokenizer(hidden_size=hidden_size, branch=branch)
            for branch in architecture.patch_branches
        )
        self.branch_token_counts = tuple(
            compute_token_count(feature_dims=cfg.feature_dims, patch_branch=branch)
            for branch in architecture.patch_branches
        )
        self.total_token_count = sum(self.branch_token_counts)
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
        self.output_norm = nn.LayerNorm(hidden_size, eps=architecture.layer_norm_eps)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for position_embedding in self.position_embeddings:
            nn.init.normal_(position_embedding, std=PATCH_INIT_STD)
        nn.init.normal_(self.scale_embeddings, std=PATCH_INIT_STD)

    def forward(self, x: Tensor) -> Tensor:
        branch_outputs: list[Tensor] = []
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
            branch_outputs.append(branch_tokens)
        h_all = torch.cat(branch_outputs, dim=1)
        return self.output_norm(h_all)


class LatentQueryPooler(nn.Module):
    def __init__(self, cfg: MultiScaleRdtArchitectureConfig) -> None:
        super().__init__()
        self.latent_query_count = cfg.latent_query_count
        self.latent_queries = nn.Parameter(
            torch.empty(1, cfg.latent_query_count, cfg.hidden_size)
        )
        nn.init.normal_(self.latent_queries, std=PATCH_INIT_STD)
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

    def forward(self, h_all: Tensor) -> Tensor:
        queries = self.latent_queries.expand(h_all.shape[0], -1, -1)
        norm_queries = self.query_norm(queries)
        norm_context = self.context_norm(h_all)
        pooled, _ = self.cross_attn(
            norm_queries,
            norm_context,
            norm_context,
            need_weights=False,
        )
        pooled = queries + self.dropout(pooled)
        return self.feed_forward(pooled)


class RdtRefinementBlock(nn.Module):
    def __init__(self, cfg: MultiScaleRdtArchitectureConfig) -> None:
        super().__init__()
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

    def forward(self, latent_summaries: Tensor, evidence_memory: Tensor) -> Tensor:
        norm_latent = self.latent_norm(latent_summaries)
        self_attended, _ = self.latent_self_attn(
            norm_latent,
            norm_latent,
            norm_latent,
            need_weights=False,
        )
        latent_summaries = latent_summaries + self.dropout(self_attended)
        norm_queries = self.query_norm(latent_summaries)
        norm_memory = self.context_norm(evidence_memory)
        cross_attended, _ = self.cross_attn(
            norm_queries,
            norm_memory,
            norm_memory,
            need_weights=False,
        )
        latent_summaries = latent_summaries + self.dropout(cross_attended)
        return self.feed_forward(latent_summaries)


class MultiScaleRdtAstModel(nn.Module):
    def __init__(self, cfg: MultiScaleRdtAstModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        architecture = cfg.encoder.architecture
        self.encoder = MultiScalePatchStemAdapterEncoder(cfg.encoder)
        self.latent_pooler = LatentQueryPooler(architecture)
        self.rdt_block = RdtRefinementBlock(architecture)
        output_dim = 1 if cfg.num_classes == 2 else cfg.num_classes
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
        x = input_values.unsqueeze(1)
        h_all = self.encoder(x)
        latent_summaries = self.latent_pooler(h_all)
        for _ in range(self.cfg.encoder.architecture.rdt_steps):
            latent_summaries = self.rdt_block(latent_summaries, h_all)
        pooled_embedding = latent_summaries.mean(dim=1)
        logits = self.classifier(pooled_embedding)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        return AstModelOutput(logits=logits, pooled_embedding=pooled_embedding)
