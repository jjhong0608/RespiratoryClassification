from __future__ import annotations

from pathlib import Path

from src.training.ast_setup import (
    apply_encoder_adaptation,
    build_ast_model,
    build_grouped_optimizer,
    inspect_pretrained_encoder,
)
from src.utils.config import (
    AstArchitectureConfig,
    AstEncoderConfig,
    ClassifierConfig,
    EncoderAdaptationConfig,
    GatedAttentionConfig,
    InstanceHeadConfig,
    MilConfig,
    ModelConfig,
)
from transformers import ASTConfig


def _run_model_config(*, pretrained_name_or_path: str | None = None) -> ModelConfig:
    return ModelConfig(
        encoder=AstEncoderConfig(
            pretrained_name_or_path=pretrained_name_or_path,
            cache_dir=None,
            pooling="cls",
            adaptation=EncoderAdaptationConfig(mode="partial", num_layers=1),
            architecture=AstArchitectureConfig(
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=64,
            ),
        ),
        instance_head=InstanceHeadConfig(
            projection_dim=24,
            dropout=0.1,
            normalize=True,
        ),
        mil=MilConfig(
            type="gated_attention",
            gated_attention=GatedAttentionConfig(attention_dim=16, dropout=0.1),
        ),
        classifier=ClassifierConfig(
            type="mlp",
            hidden_dim=24,
            dropout=0.1,
        ),
    )


def test_partial_unfreezing_only_enables_last_block_and_layernorm() -> None:
    model = build_ast_model(
        _run_model_config(),
        num_mel_bins=32,
        max_length=32,
        num_classes=2,
    )

    summary = apply_encoder_adaptation(
        model,
        model.cfg.encoder.adaptation,
    )
    trainable = {
        name
        for name, parameter in model.encoder.named_parameters()
        if parameter.requires_grad
    }

    assert summary.mode == "partial"
    assert summary.trainable_parameters > 0
    assert any(name.startswith("encoder.layer.1.") for name in trainable)
    assert any(name.startswith("layernorm.") for name in trainable)
    assert all(not name.startswith("encoder.layer.0.") for name in trainable)
    assert all(not name.startswith("embeddings.") for name in trainable)


def test_full_unfreezing_enables_all_encoder_parameters() -> None:
    model = build_ast_model(
        _run_model_config(),
        num_mel_bins=32,
        max_length=32,
        num_classes=2,
    )

    apply_encoder_adaptation(
        model,
        EncoderAdaptationConfig(mode="full", num_layers=1),
    )

    assert all(parameter.requires_grad for parameter in model.encoder.parameters())


def test_grouped_optimizer_uses_encoder_and_head_learning_rates() -> None:
    model = build_ast_model(
        _run_model_config(),
        num_mel_bins=32,
        max_length=32,
        num_classes=2,
    )
    apply_encoder_adaptation(model, model.cfg.encoder.adaptation)

    optimizer, summary = build_grouped_optimizer(
        model,
        encoder_lr=1e-5,
        head_lr=1e-4,
        weight_decay=0.01,
    )

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["name"] == "encoder"
    assert optimizer.param_groups[0]["lr"] == 1e-5
    assert optimizer.param_groups[1]["name"] == "head"
    assert optimizer.param_groups[1]["lr"] == 1e-4
    assert summary.encoder_trainable_parameters > 0
    assert summary.head_trainable_parameters > 0


def test_inspect_pretrained_encoder_reads_local_ast_config(tmp_path: Path) -> None:
    pretrained_dir = tmp_path / "ast_pretrained"
    ASTConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        num_mel_bins=32,
        max_length=32,
    ).save_pretrained(pretrained_dir)

    info = inspect_pretrained_encoder(
        _run_model_config(pretrained_name_or_path=str(pretrained_dir))
    )

    assert info is not None
    assert info.num_mel_bins == 32
    assert info.max_length == 32
