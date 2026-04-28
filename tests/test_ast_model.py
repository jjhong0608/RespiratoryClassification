from __future__ import annotations

from typing import Literal
from unittest import mock

import pytest
import torch
from src.models.model import (
    AstFeatureDims,
    BranchAwareGatedEvidencePooling,
    BranchEventDropoutConfig,
    BranchEventTokenDropout,
    BranchMilHead,
    BranchMilOutput,
    ClassifierConfig,
    EncoderAdaptationConfig,
    EvidencePoolingConfig,
    MeanEvidencePooling,
    MilConfig,
    MultiScaleRdtArchitectureConfig,
    MultiScaleRdtAstModel,
    MultiScaleRdtAstModelConfig,
    MultiScaleRdtEncoderConfig,
    PatchBranchConfig,
    RdtConfig,
    SelectedEvidenceDropout,
    SelectedEvidenceDropoutConfig,
    TokenAugmentationConfig,
    default_patch_branches,
    get_evidence_scores,
)

from conftest import small_patch_branches


def _small_architecture(
    *,
    rdt_enabled: bool = True,
    rdt_steps: int = 3,
    top_tokens_per_branch: int = 2,
    patch_branches: tuple[PatchBranchConfig, ...] | None = None,
    evidence_score_source: Literal[
        "attention_weight", "attention_logit", "instance_logit"
    ] = "attention_weight",
    exclude_branches_from_evidence: tuple[int, ...] = (),
    attention_temperature: float = 1.0,
    evidence_pooling: EvidencePoolingConfig | None = None,
    token_augmentation: TokenAugmentationConfig | None = None,
) -> MultiScaleRdtArchitectureConfig:
    return MultiScaleRdtArchitectureConfig(
        hidden_size=32,
        num_attention_heads=4,
        mlp_ratio=2.0,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-6,
        shared_stem_depth=1,
        adapter_depth=1,
        patch_branches=patch_branches or small_patch_branches(),
        rdt=RdtConfig(
            enabled=rdt_enabled,
            steps=rdt_steps,
            top_tokens_per_branch=top_tokens_per_branch,
            gated_residual=True,
            layerscale_init=0.01,
            evidence_score_source=evidence_score_source,
            exclude_branches_from_evidence=exclude_branches_from_evidence,
        ),
        mil=MilConfig(attention_temperature=attention_temperature),
        evidence_pooling=evidence_pooling or EvidencePoolingConfig(),
        token_augmentation=token_augmentation or TokenAugmentationConfig(),
    )


def _small_model_config(
    *,
    num_classes: int,
    rdt_enabled: bool = True,
    rdt_steps: int = 3,
    top_tokens_per_branch: int = 2,
    patch_branches: tuple[PatchBranchConfig, ...] | None = None,
    evidence_score_source: Literal[
        "attention_weight", "attention_logit", "instance_logit"
    ] = "attention_weight",
    exclude_branches_from_evidence: tuple[int, ...] = (),
    attention_temperature: float = 1.0,
    evidence_pooling: EvidencePoolingConfig | None = None,
    token_augmentation: TokenAugmentationConfig | None = None,
) -> MultiScaleRdtAstModelConfig:
    return MultiScaleRdtAstModelConfig(
        encoder=MultiScaleRdtEncoderConfig(
            feature_dims=AstFeatureDims(num_mel_bins=32, max_length=32),
            adaptation=EncoderAdaptationConfig(mode="full", num_layers=0),
            architecture=_small_architecture(
                rdt_enabled=rdt_enabled,
                rdt_steps=rdt_steps,
                top_tokens_per_branch=top_tokens_per_branch,
                patch_branches=patch_branches,
                evidence_score_source=evidence_score_source,
                exclude_branches_from_evidence=exclude_branches_from_evidence,
                attention_temperature=attention_temperature,
                evidence_pooling=evidence_pooling,
                token_augmentation=token_augmentation,
            ),
        ),
        classifier=ClassifierConfig(
            type="mlp",
            hidden_dim=24,
            dropout=0.1,
            pooling="latent_mean",
        ),
        num_classes=num_classes,
    )


def test_binary_model_returns_expected_event_mil_outputs() -> None:
    model = MultiScaleRdtAstModel(_small_model_config(num_classes=2))
    input_values = torch.randn(2, 32, 32)

    output = model(input_values)

    assert output.logits.shape == (2,)
    assert output.pooled_embedding.shape == (2, 32)
    assert output.branch_logits is not None
    assert output.branch_logits.shape == (2, 4)
    assert output.selected_evidence_tokens is not None
    assert output.selected_evidence_tokens.shape == (2, 8, 32)
    assert output.selected_evidence_indices is not None
    assert output.selected_evidence_indices.shape == (2, 8)
    assert output.selected_evidence_scores is not None
    assert output.selected_evidence_scores.shape == (2, 8)
    assert output.selected_evidence_branch_ids is not None
    assert output.selected_evidence_branch_ids.shape == (2, 8)
    assert output.selected_evidence_branch_ids[0].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert output.evidence_score_source == "attention_weight"
    assert output.evidence_pooling_type == "mean"
    assert output.evidence_gate_weights is None
    assert output.evidence_gate_entropy is None
    assert output.branch_evidence_norms is None
    assert output.branch_attention_weights is not None
    assert tuple(attn.shape[1] for attn in output.branch_attention_weights) == (
        7,
        15,
        31,
        3,
    )


def test_multiclass_model_returns_expected_event_mil_outputs() -> None:
    model = MultiScaleRdtAstModel(_small_model_config(num_classes=3))
    input_values = torch.randn(2, 32, 32)

    output = model(input_values)

    assert output.logits.shape == (2, 3)
    assert output.pooled_embedding.shape == (2, 32)
    assert output.branch_logits is not None
    assert output.branch_logits.shape == (2, 4, 3)
    assert output.selected_evidence_tokens is not None
    assert output.selected_evidence_tokens.shape == (2, 8, 32)
    assert output.selected_evidence_indices is not None
    assert output.selected_evidence_indices.shape == (2, 8)
    assert output.evidence_pooling_type == "mean"


def test_invalid_input_dims_raise_clear_value_error() -> None:
    model = MultiScaleRdtAstModel(_small_model_config(num_classes=2))

    with pytest.raises(ValueError, match="input_values must have shape"):
        model(torch.randn(2, 32))

    with pytest.raises(
        ValueError, match="feature dims do not match model feature dims"
    ):
        model(torch.randn(2, 31, 32))


def test_default_branch_geometry_matches_expected_token_and_time_counts() -> None:
    model = MultiScaleRdtAstModel(
        MultiScaleRdtAstModelConfig(
            encoder=MultiScaleRdtEncoderConfig(
                feature_dims=AstFeatureDims(num_mel_bins=128, max_length=1024),
                adaptation=EncoderAdaptationConfig(mode="full", num_layers=0),
                architecture=MultiScaleRdtArchitectureConfig(
                    hidden_size=48,
                    num_attention_heads=4,
                ),
            ),
            classifier=ClassifierConfig(pooling="latent_mean"),
            num_classes=2,
        )
    )

    assert model.encoder.branch_token_counts == (1016, 1020, 1022, 1023)
    assert model.encoder.branch_time_lengths == (127, 255, 511, 1023)
    assert model.encoder.total_token_count == 4081
    assert model.encoder.total_temporal_length == 1916


def test_encoder_returns_temporal_event_tokens_and_context() -> None:
    model = MultiScaleRdtAstModel(_small_model_config(num_classes=2))
    input_values = torch.randn(2, 32, 32)

    encoder_output = model.encoder(input_values.unsqueeze(1))

    assert tuple(tokens.shape for tokens in encoder_output.branch_event_tokens) == (
        (2, 7, 32),
        (2, 15, 32),
        (2, 31, 32),
        (2, 3, 32),
    )
    assert encoder_output.context_tokens.shape == (2, 56, 32)


def test_mean_evidence_pooling_returns_legacy_embedding_only() -> None:
    pooler = MeanEvidencePooling()
    output = pooler(
        torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3),
        torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
    )

    assert output.pooled_embedding.shape == (2, 3)
    assert output.gate_weights is None
    assert output.gate_entropy is None
    assert output.branch_evidence_summary is None
    assert output.branch_evidence_norms is None


def test_branch_aware_gated_pooling_returns_h0_shapes() -> None:
    pooler = BranchAwareGatedEvidencePooling(
        hidden_size=32,
        cfg=EvidencePoolingConfig(type="branch_gated", dropout=0.0),
    )
    evidence_tokens = torch.randn(2, 8, 32)
    branch_ids = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [0, 0, 1, 1, 2, 2, 3, 3]])

    output = pooler(evidence_tokens, branch_ids)

    assert output.pooled_embedding.shape == (2, 32)
    assert output.gate_weights is not None
    assert output.gate_weights.shape == (2, 4)
    assert output.gate_entropy is not None
    assert output.gate_entropy.shape == (2,)
    assert output.branch_evidence_summary is not None
    assert output.branch_evidence_summary.shape == (2, 4, 32)
    assert output.branch_evidence_norms is not None
    assert output.branch_evidence_norms.shape == (2, 4)
    assert torch.all(output.gate_weights >= 0)
    assert torch.allclose(output.gate_weights.sum(dim=1), torch.ones(2), atol=1e-5)


def test_branch_aware_gated_pooling_supports_dynamic_branch_count() -> None:
    pooler = BranchAwareGatedEvidencePooling(
        hidden_size=16,
        cfg=EvidencePoolingConfig(type="branch_gated", dropout=0.0),
    )
    evidence_tokens = torch.randn(2, 6, 16)
    branch_ids = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]])

    output = pooler(evidence_tokens, branch_ids)

    assert output.pooled_embedding.shape == (2, 16)
    assert output.gate_weights is not None
    assert output.gate_weights.shape == (2, 3)
    assert output.gate_entropy is not None
    assert output.gate_entropy.shape == (2,)
    assert output.branch_evidence_summary is not None
    assert output.branch_evidence_summary.shape == (2, 3, 16)


def test_branch_aware_gated_pooling_uses_only_present_branches() -> None:
    pooler = BranchAwareGatedEvidencePooling(
        hidden_size=16,
        cfg=EvidencePoolingConfig(type="branch_gated", dropout=0.0),
    )
    evidence_tokens = torch.randn(1, 6, 16)
    branch_ids = torch.tensor([[0, 0, 1, 1, 2, 2]])

    output = pooler(evidence_tokens, branch_ids)

    assert output.gate_weights is not None
    assert output.gate_weights.shape == (1, 3)


def test_branch_event_token_dropout_disabled_returns_identity() -> None:
    tokens = torch.randn(2, 5, 4)
    dropout = BranchEventTokenDropout(
        BranchEventDropoutConfig(enabled=False, probability=1.0, min_keep_tokens=2)
    )
    dropout.train()

    output, keep_mask = dropout(tokens)

    assert output is tokens
    assert torch.equal(output, tokens)
    assert keep_mask.dtype == torch.bool
    assert torch.all(keep_mask)


def test_branch_event_token_dropout_zeroes_tokens_and_keeps_minimum() -> None:
    torch.manual_seed(0)
    tokens = torch.ones(2, 5, 4)
    dropout = BranchEventTokenDropout(
        BranchEventDropoutConfig(enabled=True, probability=1.0, min_keep_tokens=2)
    )
    dropout.train()

    output, keep_mask = dropout(tokens)

    assert output.shape == tokens.shape
    assert keep_mask.shape == tokens.shape[:2]
    assert keep_mask.dtype == torch.bool
    assert torch.all(keep_mask.sum(dim=1) == 2)
    assert torch.all(output[~keep_mask] == 0)
    assert torch.all(output[keep_mask] == 1)


def test_branch_mil_head_respects_token_mask() -> None:
    head = BranchMilHead(
        hidden_size=4,
        output_dim=1,
        layer_norm_eps=1e-6,
        attention_temperature=1.0,
    )
    event_tokens = torch.randn(1, 4, 4)
    token_mask = torch.tensor([[True, False, True, False]])

    output = head(event_tokens, token_mask=token_mask)

    assert output.attention_weights.shape == (1, 4)
    assert torch.all(output.attention_weights[~token_mask] < 1e-6)
    assert torch.allclose(output.attention_weights.sum(dim=1), torch.ones(1))


def test_selected_evidence_dropout_disabled_returns_identity() -> None:
    tokens = torch.randn(1, 6, 4)
    branch_ids = torch.tensor([[0, 0, 1, 1, 2, 2]])
    dropout = SelectedEvidenceDropout(
        SelectedEvidenceDropoutConfig(
            enabled=False,
            probability=1.0,
            min_keep_per_branch=1,
        )
    )
    dropout.train()

    output, keep_mask = dropout(tokens, branch_ids)

    assert output is tokens
    assert torch.equal(output, tokens)
    assert keep_mask.dtype == torch.bool
    assert torch.all(keep_mask)


@pytest.mark.parametrize(
    "branch_ids",
    [
        torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]]),
        torch.tensor([[0, 0, 1, 1, 2, 2]]),
    ],
)
def test_selected_evidence_dropout_keeps_minimum_per_branch(
    branch_ids: torch.Tensor,
) -> None:
    torch.manual_seed(0)
    tokens = torch.ones(1, branch_ids.shape[1], 4)
    dropout = SelectedEvidenceDropout(
        SelectedEvidenceDropoutConfig(
            enabled=True,
            probability=1.0,
            min_keep_per_branch=1,
        )
    )
    dropout.train()

    output, keep_mask = dropout(tokens, branch_ids)

    assert output.shape == tokens.shape
    assert keep_mask.shape == branch_ids.shape
    assert keep_mask.dtype == torch.bool
    for branch_id in torch.unique(branch_ids):
        branch_mask = branch_ids == branch_id
        assert int(keep_mask[branch_mask].sum().item()) == 1
    assert torch.all(output[~keep_mask] == 0)
    assert torch.all(output[keep_mask] == 1)


def test_model_runs_without_rdt_when_disabled() -> None:
    model = MultiScaleRdtAstModel(_small_model_config(num_classes=2, rdt_enabled=False))
    output = model(torch.randn(1, 32, 32))

    assert model.rdt_block is None
    assert output.logits.shape == (1,)
    assert output.pooled_embedding.shape == (1, 32)
    assert output.selected_evidence_tokens is not None
    assert output.selected_evidence_tokens.shape == (1, 8, 32)
    assert output.selected_evidence_indices is not None
    assert output.selected_evidence_indices.shape == (1, 8)
    assert output.selected_evidence_scores is not None
    assert output.selected_evidence_scores.shape == (1, 8)
    assert output.selected_evidence_branch_ids is not None
    assert output.selected_evidence_branch_ids.shape == (1, 8)


@pytest.mark.parametrize(
    "token_augmentation",
    [
        TokenAugmentationConfig(),
        TokenAugmentationConfig(
            branch_event_dropout=BranchEventDropoutConfig(
                enabled=True,
                probability=0.5,
                min_keep_tokens=1,
            )
        ),
        TokenAugmentationConfig(
            selected_evidence_dropout=SelectedEvidenceDropoutConfig(
                enabled=True,
                probability=0.5,
                min_keep_per_branch=1,
            )
        ),
        TokenAugmentationConfig(
            branch_event_dropout=BranchEventDropoutConfig(
                enabled=True,
                probability=0.5,
                min_keep_tokens=1,
            ),
            selected_evidence_dropout=SelectedEvidenceDropoutConfig(
                enabled=True,
                probability=0.5,
                min_keep_per_branch=1,
            ),
        ),
    ],
)
def test_model_forward_supports_token_augmentation_modes(
    token_augmentation: TokenAugmentationConfig,
) -> None:
    torch.manual_seed(0)
    model = MultiScaleRdtAstModel(
        _small_model_config(
            num_classes=2,
            token_augmentation=token_augmentation,
        )
    )
    model.train()

    output = model(torch.randn(1, 32, 32))

    assert output.logits.shape == (1,)
    assert output.pooled_embedding.shape == (1, 32)
    assert output.selected_evidence_tokens is not None
    assert output.selected_evidence_indices is not None
    assert output.selected_evidence_scores is not None
    assert output.selected_evidence_branch_ids is not None
    if token_augmentation.selected_evidence_dropout.enabled:
        assert output.selected_evidence_dropout_mask is not None
        assert output.selected_evidence_dropout_mask.shape == (1, 8)
        assert output.selected_evidence_keep_ratio is not None
        assert output.selected_evidence_keep_ratio.shape == (1,)
    else:
        assert output.selected_evidence_dropout_mask is None
        assert output.selected_evidence_keep_ratio is None


def test_model_eval_disables_selected_evidence_dropout() -> None:
    model = MultiScaleRdtAstModel(
        _small_model_config(
            num_classes=2,
            token_augmentation=TokenAugmentationConfig(
                selected_evidence_dropout=SelectedEvidenceDropoutConfig(
                    enabled=True,
                    probability=0.5,
                    min_keep_per_branch=1,
                )
            ),
        )
    )
    model.eval()

    output = model(torch.randn(1, 32, 32))

    assert output.selected_evidence_dropout_mask is not None
    assert torch.all(output.selected_evidence_dropout_mask)
    assert output.selected_evidence_keep_ratio is not None
    assert torch.allclose(output.selected_evidence_keep_ratio, torch.ones(1))


@pytest.mark.parametrize("num_classes", [2, 3])
def test_model_forward_uses_branch_gated_evidence_pooling(num_classes: int) -> None:
    model = MultiScaleRdtAstModel(
        _small_model_config(
            num_classes=num_classes,
            evidence_pooling=EvidencePoolingConfig(
                type="branch_gated",
                dropout=0.0,
            ),
        )
    )

    output = model(torch.randn(2, 32, 32))

    expected_logit_shape = (2,) if num_classes == 2 else (2, num_classes)
    assert output.logits.shape == expected_logit_shape
    assert output.pooled_embedding.shape == (2, 32)
    assert output.evidence_pooling_type == "branch_gated"
    assert output.evidence_gate_weights is not None
    assert output.evidence_gate_weights.shape == (2, 4)
    assert output.evidence_gate_entropy is not None
    assert output.evidence_gate_entropy.shape == (2,)
    assert output.branch_evidence_norms is not None
    assert output.branch_evidence_norms.shape == (2, 4)
    assert torch.allclose(
        output.evidence_gate_weights.sum(dim=1),
        torch.ones(2),
        atol=1e-5,
    )


def test_three_scale_top2_forward_uses_dynamic_selected_evidence_length() -> None:
    model = MultiScaleRdtAstModel(
        _small_model_config(
            num_classes=2,
            patch_branches=small_patch_branches()[:3],
        )
    )

    assert model.encoder.total_temporal_length == 53
    output = model(torch.randn(1, 32, 32))

    assert output.selected_evidence_tokens is not None
    assert output.selected_evidence_tokens.shape == (1, 6, 32)
    assert output.branch_logits is not None
    assert output.branch_logits.shape == (1, 3)
    assert output.selected_evidence_indices is not None
    assert output.selected_evidence_indices.shape == (1, 6)
    assert output.selected_evidence_branch_ids is not None
    assert output.selected_evidence_branch_ids[0].tolist() == [0, 0, 1, 1, 2, 2]


def test_four_scale_top4_forward_uses_dynamic_selected_evidence_length() -> None:
    top4_patch_branches = (
        PatchBranchConfig(patch_size=(8, 8), stride=(4, 8)),
        PatchBranchConfig(patch_size=(4, 16), stride=(2, 16)),
        PatchBranchConfig(patch_size=(2, 32), stride=(1, 32)),
        PatchBranchConfig(patch_size=(8, 4), stride=(4, 4)),
    )
    model = MultiScaleRdtAstModel(
        _small_model_config(
            num_classes=2,
            top_tokens_per_branch=4,
            patch_branches=top4_patch_branches,
        )
    )

    output = model(torch.randn(1, 32, 32))

    assert output.selected_evidence_tokens is not None
    assert output.selected_evidence_tokens.shape == (1, 16, 32)
    assert output.selected_evidence_indices is not None
    assert output.selected_evidence_indices.shape == (1, 16)
    assert output.selected_evidence_branch_ids is not None
    assert output.selected_evidence_branch_ids[0].tolist() == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
    ]


@pytest.mark.parametrize(
    ("top_tokens_per_branch", "expected_length"),
    [(1, 4), (2, 8), (3, 12)],
)
def test_four_scale_topk_forward_uses_dynamic_selected_evidence_length(
    top_tokens_per_branch: int,
    expected_length: int,
) -> None:
    model = MultiScaleRdtAstModel(
        _small_model_config(
            num_classes=2,
            top_tokens_per_branch=top_tokens_per_branch,
        )
    )

    output = model(torch.randn(1, 32, 32))

    assert output.selected_evidence_tokens is not None
    assert output.selected_evidence_tokens.shape == (1, expected_length, 32)
    assert model.encoder.total_temporal_length == 56


def test_excluded_branch_keeps_context_but_not_selected_evidence() -> None:
    model = MultiScaleRdtAstModel(
        _small_model_config(
            num_classes=2,
            exclude_branches_from_evidence=(3,),
        )
    )

    assert model.encoder.total_temporal_length == 56
    output = model(torch.randn(1, 32, 32))

    assert output.selected_evidence_tokens is not None
    assert output.selected_evidence_tokens.shape == (1, 6, 32)
    assert output.selected_evidence_branch_ids is not None
    assert output.selected_evidence_branch_ids[0].tolist() == [0, 0, 1, 1, 2, 2]


def test_three_scale_geometry_and_forward_use_dynamic_evidence_length() -> None:
    full_geometry_model = MultiScaleRdtAstModel(
        MultiScaleRdtAstModelConfig(
            encoder=MultiScaleRdtEncoderConfig(
                feature_dims=AstFeatureDims(num_mel_bins=128, max_length=1024),
                adaptation=EncoderAdaptationConfig(mode="full", num_layers=0),
                architecture=_small_architecture(
                    patch_branches=default_patch_branches()[:3],
                ),
            ),
            classifier=ClassifierConfig(
                type="linear",
                hidden_dim=24,
                dropout=0.1,
                pooling="latent_mean",
            ),
            num_classes=2,
        )
    )
    assert full_geometry_model.encoder.total_temporal_length == 893

    small_model = MultiScaleRdtAstModel(
        _small_model_config(
            num_classes=2,
            patch_branches=small_patch_branches()[:3],
        )
    )
    output = small_model(torch.randn(1, 32, 32))

    assert small_model.encoder.total_temporal_length == 53
    assert output.selected_evidence_tokens is not None
    assert output.selected_evidence_tokens.shape == (1, 6, 32)
    assert output.branch_logits is not None
    assert output.branch_logits.shape == (1, 3)
    assert output.selected_evidence_indices is not None
    assert output.selected_evidence_indices.shape == (1, 6)
    assert output.selected_evidence_branch_ids is not None
    assert output.selected_evidence_branch_ids[0].tolist() == [0, 0, 1, 1, 2, 2]


@pytest.mark.parametrize(
    "source",
    ["attention_weight", "attention_logit", "instance_logit"],
)
def test_evidence_score_sources_forward_success(source: str) -> None:
    model = MultiScaleRdtAstModel(
        _small_model_config(
            num_classes=2,
            evidence_score_source=source,  # type: ignore[arg-type]
        )
    )

    output = model(torch.randn(1, 32, 32))

    assert output.selected_evidence_tokens is not None
    assert output.selected_evidence_tokens.shape == (1, 8, 32)
    assert output.evidence_score_source == source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("attention_weight", [[0.1, 0.9, 0.2]]),
        ("attention_logit", [[-1.0, 2.0, 0.5]]),
        ("instance_logit", [[0.3, -0.2, 1.5]]),
    ],
)
def test_get_evidence_scores_uses_requested_source(
    source: str,
    expected: list[list[float]],
) -> None:
    mil_output = BranchMilOutput(
        logits=torch.tensor([0.0]),
        attention_weights=torch.tensor([[0.1, 0.9, 0.2]]),
        attention_logits=torch.tensor([[-1.0, 2.0, 0.5]]),
        instance_logits=torch.tensor([[0.3, -0.2, 1.5]]),
        embedding=torch.zeros(1, 4),
    )

    scores = get_evidence_scores(mil_output, source=source)

    assert torch.allclose(scores, torch.tensor(expected))


def test_get_evidence_scores_uses_max_class_logit_for_multiclass_instances() -> None:
    mil_output = BranchMilOutput(
        logits=torch.zeros(1, 3),
        attention_weights=torch.zeros(1, 2),
        attention_logits=torch.zeros(1, 2),
        instance_logits=torch.tensor([[[0.1, 0.7, -0.3], [0.4, -0.2, 1.2]]]),
        embedding=torch.zeros(1, 4),
    )

    scores = get_evidence_scores(mil_output, source="instance_logit")

    assert torch.allclose(scores, torch.tensor([[0.7, 1.2]]))


def test_rdt_block_is_reused_for_all_configured_steps() -> None:
    model = MultiScaleRdtAstModel(
        _small_model_config(num_classes=2, rdt_enabled=True, rdt_steps=2)
    )
    input_values = torch.randn(1, 32, 32)
    assert model.rdt_block is not None

    with mock.patch.object(
        model.rdt_block,
        "forward",
        wraps=model.rdt_block.forward,
    ) as wrapped:
        output = model(input_values)

    assert wrapped.call_count == 2
    assert output.pooled_embedding.shape == (1, 32)
