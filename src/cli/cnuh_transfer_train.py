from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from torch.optim import Optimizer

from src.data.loaders import RespiratoryClipDataset, build_clip_loader, build_dataset
from src.models.model import MultiScaleRdtAstModel
from src.training.ast_setup import (
    ModelArchitectureSummary,
    build_ast_model,
    summarize_model_architecture,
)
from src.training.imbalance import (
    build_sqrt_inverse_class_sampler,
    build_weighted_sampler,
    collect_targets,
    resolve_imbalance,
    resolve_loss_weights,
)
from src.training.trainer import Trainer, TrainerConfig
from src.training.transfer import (
    FreezeSummary,
    TransferLoadSummary,
    TransferOptimizerSummary,
    apply_cnuh_transfer_freeze,
    build_cnuh_transfer_optimizer,
    load_compatible_model_state,
)
from src.utils.config import (
    CnuhTransferConfig,
    CnuhTransferOptimizerConfig,
    JsonConfigLoader,
    ModelConfig,
)
from src.utils.fs import Fs
from src.utils.logging import configure_terminal_width, enable_file_logging, logger
from src.utils.reproducibility import Reproducibility


@dataclass(frozen=True)
class CnuhTransferSetup:
    model: MultiScaleRdtAstModel
    architecture_summary: ModelArchitectureSummary
    transfer_load_summary: TransferLoadSummary
    freeze_summary: FreezeSummary
    optimizer: Optimizer
    optimizer_summary: TransferOptimizerSummary


def prepare_cnuh_transfer_model_and_optimizer(
    model_cfg: ModelConfig,
    transfer_cfg: CnuhTransferConfig,
    optimizer_cfg: CnuhTransferOptimizerConfig,
    *,
    num_mel_bins: int,
    max_length: int,
    num_classes: int,
    weight_decay: float,
) -> CnuhTransferSetup:
    model = build_ast_model(
        model_cfg,
        num_mel_bins=num_mel_bins,
        max_length=max_length,
        num_classes=num_classes,
    )
    architecture_summary = summarize_model_architecture(model)
    transfer_load_summary = load_compatible_model_state(
        model=model,
        checkpoint_path=transfer_cfg.checkpoint_path,
        reset_classifier=transfer_cfg.reset_classifier,
        strict=transfer_cfg.strict,
        map_location="cpu",
    )
    freeze_summary = apply_cnuh_transfer_freeze(
        model,
        freeze_encoder=transfer_cfg.freeze_encoder,
        freeze_modules=transfer_cfg.freeze_modules,
    )
    optimizer, optimizer_summary = build_cnuh_transfer_optimizer(
        model,
        frozen_encoder_lr=optimizer_cfg.frozen_encoder_lr,
        body_lr=optimizer_cfg.body_lr,
        head_lr=optimizer_cfg.head_lr,
        weight_decay=weight_decay,
    )
    return CnuhTransferSetup(
        model=model,
        architecture_summary=architecture_summary,
        transfer_load_summary=transfer_load_summary,
        freeze_summary=freeze_summary,
        optimizer=optimizer,
        optimizer_summary=optimizer_summary,
    )


def _log_augmentation(train_dataset: RespiratoryClipDataset) -> None:
    augmentation = train_dataset.cfg.augmentation
    logger.info(
        "Data augmentation | enabled=%s | train_apply=%s | waveform=%s | "
        "waveform_probability=%.4f | gain=%s | noise=%s | time_shift=%s | "
        "fbank=%s | fbank_probability=%.4f | time_mask=%s | freq_mask=%s",
        augmentation.enabled,
        train_dataset.apply_augmentation,
        augmentation.waveform.enabled,
        augmentation.waveform.probability,
        augmentation.waveform.gain.enabled,
        augmentation.waveform.noise.enabled,
        augmentation.waveform.time_shift.enabled,
        augmentation.fbank.enabled,
        augmentation.fbank.probability,
        augmentation.fbank.time_mask.enabled,
        augmentation.fbank.freq_mask.enabled,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = JsonConfigLoader.load_training(args.config)
    transfer_template = JsonConfigLoader.load_cnuh_transfer_template(args.config)
    configure_terminal_width(cfg.terminal.width)
    generators = Reproducibility.seed_everything(cfg.experiment.seed)

    run_dir = Fs.ensure_dir(Path(cfg.experiment.output_dir) / cfg.experiment.name)
    Fs.copy_file(args.config, run_dir)
    enable_file_logging(run_dir / "run.log", mode="w")

    train_dataset = build_dataset(cfg.data, split="train")
    val_dataset = build_dataset(cfg.data, split="val")
    _log_augmentation(train_dataset)

    num_classes = len(cfg.data.label_to_index)
    train_targets = collect_targets(train_dataset)
    resolved_loss_weights = resolve_loss_weights(
        targets=train_targets,
        num_classes=num_classes,
        label_to_index=cfg.data.label_to_index,
        loss_cfg=cfg.train.loss,
    )
    imbalance = resolve_imbalance(
        targets=train_targets,
        num_classes=num_classes,
        pos_weight=cfg.train.loss.pos_weight,
        auto_pos_weight=cfg.train.loss.auto_pos_weight,
        weighted_random=cfg.train.sampler.weighted_random,
    )
    logger.info("Train class counts: %s", dict(imbalance.class_counts))
    logger.info(
        "Train loss weights | class_counts=%s | class_weights=%s | "
        "main_to_binary=%s | binary_counts=%s | branch_binary_pos_weight=%s",
        list(resolved_loss_weights.class_counts),
        list(resolved_loss_weights.class_weights)
        if resolved_loss_weights.class_weights is not None
        else None,
        list(resolved_loss_weights.main_index_to_binary_target)
        if resolved_loss_weights.main_index_to_binary_target is not None
        else None,
        dict(resolved_loss_weights.binary_counts)
        if resolved_loss_weights.binary_counts is not None
        else None,
        resolved_loss_weights.branch_binary_pos_weight,
    )
    if imbalance.pos_weight is not None:
        logger.info(
            "Using positive-class weight=%.6f (loss=%s)",
            imbalance.pos_weight,
            cfg.train.loss.type,
        )

    setup = prepare_cnuh_transfer_model_and_optimizer(
        cfg.model,
        transfer_template.transfer,
        transfer_template.optimizer,
        num_mel_bins=train_dataset.num_mel_bins,
        max_length=train_dataset.max_length,
        num_classes=num_classes,
        weight_decay=cfg.train.optimizer.weight_decay,
    )
    logger.info(
        "Model architecture | encoder_type=%s | hidden_size=%d | num_heads=%d | "
        "branch_token_counts=%s | branch_time_lengths=%s | total_patch_tokens=%d | "
        "total_temporal_length=%d | rdt_enabled=%s | rdt_steps=%d | "
        "top_tokens_per_branch=%d | evidence_score_source=%s | "
        "exclude_branches_from_evidence=%s | attention_temperature=%.4f | "
        "evidence_pooling=%s | branch_auxiliary=%s",
        setup.architecture_summary.encoder_type,
        setup.architecture_summary.hidden_size,
        setup.architecture_summary.num_attention_heads,
        list(setup.architecture_summary.branch_token_counts),
        list(setup.architecture_summary.branch_time_lengths),
        setup.architecture_summary.total_patch_token_count,
        setup.architecture_summary.total_temporal_length,
        setup.architecture_summary.rdt_enabled,
        setup.architecture_summary.rdt_steps,
        setup.architecture_summary.rdt_top_tokens_per_branch,
        setup.architecture_summary.rdt_evidence_score_source,
        list(setup.architecture_summary.rdt_exclude_branches_from_evidence),
        setup.architecture_summary.mil_attention_temperature,
        setup.architecture_summary.evidence_pooling_type,
        cfg.train.loss.branch_auxiliary.enabled,
    )
    logger.info(
        "CNUH transfer load | checkpoint=%s | loaded=%d | skipped=%d | "
        "missing=%d | unexpected=%d",
        setup.transfer_load_summary.checkpoint_path,
        len(setup.transfer_load_summary.loaded_keys),
        len(setup.transfer_load_summary.skipped_keys),
        len(setup.transfer_load_summary.missing_keys),
        len(setup.transfer_load_summary.unexpected_keys),
    )
    logger.info(
        "CNUH transfer load keys | skipped=%s | missing=%s | unexpected=%s",
        list(setup.transfer_load_summary.skipped_keys),
        list(setup.transfer_load_summary.missing_keys),
        list(setup.transfer_load_summary.unexpected_keys),
    )
    logger.info(
        "CNUH transfer freeze | freeze_encoder=%s | freeze_modules=%s | "
        "frozen_parameters=%d | trainable_parameters=%d",
        transfer_template.transfer.freeze_encoder,
        list(transfer_template.transfer.freeze_modules),
        len(setup.freeze_summary.frozen_parameter_names),
        len(setup.freeze_summary.trainable_parameter_names),
    )

    sampler = None
    if cfg.train.sampler.enabled:
        sampler, sampler_summary = build_sqrt_inverse_class_sampler(
            train_targets,
            num_classes=num_classes,
            cfg=cfg.train.sampler,
            generator=generators.sampler,
        )
        logger.info(
            "Sqrt-inverse class sampler enabled | type=%s | replacement=%s | "
            "num_samples=%d | source=%s",
            cfg.train.sampler.type,
            cfg.train.sampler.replacement,
            sampler_summary.num_samples,
            cfg.train.sampler.source,
        )
    elif imbalance.weighted_random:
        sampler = build_weighted_sampler(train_targets, generator=generators.sampler)
        logger.info("Using legacy weighted random sampler for clip training")

    train_loader = build_clip_loader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generators.train_loader,
    )
    val_loader = build_clip_loader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        generator=generators.val_loader,
    )
    logger.info("Train clips: %d | Val clips: %d", len(train_dataset), len(val_dataset))
    logger.info(
        "CNUH transfer optimizer groups | frozen_encoder_lr=%.8f | "
        "body_lr=%.8f | head_lr=%.8f | frozen_encoder_params=%d | "
        "body_params=%d | head_params=%d | groups=%d",
        transfer_template.optimizer.frozen_encoder_lr,
        transfer_template.optimizer.body_lr,
        transfer_template.optimizer.head_lr,
        setup.optimizer_summary.frozen_encoder_params,
        setup.optimizer_summary.body_params,
        setup.optimizer_summary.head_params,
        setup.optimizer_summary.param_group_count,
    )

    trainer = Trainer(
        TrainerConfig(
            device=cfg.experiment.device,
            epochs=cfg.train.epochs,
            encoder_lr=transfer_template.optimizer.frozen_encoder_lr,
            head_lr=transfer_template.optimizer.head_lr,
            weight_decay=cfg.train.optimizer.weight_decay,
            warmup_ratio=cfg.train.scheduler.warmup_ratio,
            max_grad_norm=cfg.train.max_grad_norm,
            top_k=cfg.train.top_k,
            run_dir=run_dir,
            num_classes=num_classes,
            loss_type=cfg.train.loss.type,
            gamma=cfg.train.loss.gamma,
            pos_weight=imbalance.pos_weight,
            class_weights=resolved_loss_weights.class_weights,
            label_smoothing=cfg.train.loss.label_smoothing,
            branch_auxiliary=cfg.train.loss.branch_auxiliary,
            branch_binary_auxiliary=cfg.train.loss.branch_binary_auxiliary,
            branch_binary_pos_weight=resolved_loss_weights.branch_binary_pos_weight,
            main_index_to_binary_target=(
                resolved_loss_weights.main_index_to_binary_target
            ),
            attention_entropy=cfg.train.loss.attention_entropy,
            analysis=cfg.analysis,
            early_stopping=cfg.train.early_stopping,
            checkpointing=cfg.checkpointing,
            terminal_width=cfg.terminal.width,
        )
    )
    trainer.fit(
        setup.model,
        train_loader,
        val_loader,
        setup.optimizer,
        extra_state={
            "run_config": asdict(cfg),
            "transfer_config": asdict(transfer_template),
            "model_cfg": asdict(setup.model.cfg),
            "label_to_index": dict(cfg.data.label_to_index),
            "architecture_summary": asdict(setup.architecture_summary),
            "transfer_load_summary": asdict(setup.transfer_load_summary),
            "transfer_freeze_summary": asdict(setup.freeze_summary),
            "transfer_optimizer_summary": asdict(setup.optimizer_summary),
            "loss_weight_summary": asdict(resolved_loss_weights),
        },
    )


if __name__ == "__main__":
    main()
