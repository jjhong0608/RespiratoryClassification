from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from src.data.fsd50k_dataset import build_fsd50k_dataset, build_fsd50k_loader
from src.training.ast_setup import build_ast_model, summarize_model_architecture
from src.training.multilabel_trainer import (
    MultiLabelTrainer,
    MultiLabelTrainerConfig,
    compute_multilabel_target_stats,
    compute_pos_weight_stats,
    compute_sqrt_neg_pos_multilabel_pos_weight,
)
from src.training.transfer import (
    build_fsd50k_multilabel_optimizer,
    load_compatible_model_state,
)
from src.utils.config import JsonConfigLoader
from src.utils.fs import Fs
from src.utils.logging import configure_terminal_width, enable_file_logging, logger
from src.utils.reproducibility import Reproducibility


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = JsonConfigLoader.load_fsd50k_supervised(args.config)
    configure_terminal_width(cfg.terminal.width)
    generators = Reproducibility.seed_everything(cfg.experiment.seed)
    run_dir = Fs.ensure_dir(Path(cfg.experiment.output_dir) / cfg.experiment.name)
    Fs.copy_file(args.config, run_dir)
    enable_file_logging(run_dir / "run.log", mode="w")

    train_dataset = build_fsd50k_dataset(cfg.data, split="train")
    val_dataset = build_fsd50k_dataset(cfg.data, split="val")
    train_targets = train_dataset.targets
    val_targets = val_dataset.targets
    target_stats = compute_multilabel_target_stats(
        train_targets,
        val_targets,
        class_names=tuple(train_dataset.vocabulary.index_to_label),
    )
    logger.info(
        "FSD50K target stats | train_samples=%d | val_samples=%d | "
        "num_classes=%d | train_mean_labels_per_sample=%.6f | "
        "train_min_labels_per_sample=%.1f | train_max_labels_per_sample=%.1f | "
        "val_mean_labels_per_sample=%.6f | val_min_labels_per_sample=%.1f | "
        "val_max_labels_per_sample=%.1f | train_classes_with_positive_count=%d | "
        "val_classes_with_positive_count=%d",
        target_stats["train_samples"],
        target_stats["val_samples"],
        target_stats["num_classes"],
        target_stats["train_mean_labels_per_sample"],
        target_stats["train_min_labels_per_sample"],
        target_stats["train_max_labels_per_sample"],
        target_stats["val_mean_labels_per_sample"],
        target_stats["val_min_labels_per_sample"],
        target_stats["val_max_labels_per_sample"],
        target_stats["train_num_classes_with_positive_count"],
        target_stats["val_num_classes_with_positive_count"],
    )
    logger.info(
        "FSD50K target positive class counts | top10=%s | bottom10_nonzero=%s",
        target_stats["top10_positive_class_counts"],
        target_stats["bottom10_nonzero_positive_class_counts"],
    )
    train_loader = build_fsd50k_loader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
        generator=generators.train_loader,
    )
    val_loader = build_fsd50k_loader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        generator=generators.val_loader,
    )
    model = build_ast_model(
        cfg.model,
        num_mel_bins=train_dataset.num_mel_bins,
        max_length=train_dataset.max_length,
        num_classes=train_dataset.num_classes,
    )
    initialization_summary = None
    if cfg.model.encoder.init_from is not None:
        initialization_summary = load_compatible_model_state(
            model=model,
            checkpoint_path=cfg.model.encoder.init_from,
            reset_classifier=True,
            strict=False,
            map_location="cpu",
        )
        logger.info(
            "Loaded SSL initialization | checkpoint=%s | loaded=%d | skipped=%d | "
            "missing=%d | unexpected=%d",
            initialization_summary.checkpoint_path,
            len(initialization_summary.loaded_keys),
            len(initialization_summary.skipped_keys),
            len(initialization_summary.missing_keys),
            len(initialization_summary.unexpected_keys),
        )
    pos_weight = compute_sqrt_neg_pos_multilabel_pos_weight(
        train_targets,
        cap=cfg.train.loss.pos_weight.cap,
    )
    pos_weight_stats = compute_pos_weight_stats(
        pos_weight,
        cap=cfg.train.loss.pos_weight.cap,
    )
    logger.info(
        "FSD50K pos_weight stats | min=%.6f | mean=%.6f | median=%.6f | "
        "max=%.6f | num_capped_classes=%d",
        pos_weight_stats["pos_weight_min"],
        pos_weight_stats["pos_weight_mean"],
        pos_weight_stats["pos_weight_median"],
        pos_weight_stats["pos_weight_max"],
        pos_weight_stats["num_capped_classes"],
    )
    optimizer, optimizer_summary = build_fsd50k_multilabel_optimizer(
        model,
        encoder_lr=cfg.train.optimizer.encoder_lr,
        body_lr=cfg.train.optimizer.body_lr,
        head_lr=cfg.train.optimizer.head_lr,
        weight_decay=cfg.train.optimizer.weight_decay,
    )
    architecture_summary = summarize_model_architecture(model)
    logger.info(
        "FSD50K supervised | train=%d | val=%d | classes=%d | "
        "macro_AP monitor | optimizer_groups=%d",
        len(train_dataset),
        len(val_dataset),
        train_dataset.num_classes,
        optimizer_summary.param_group_count,
    )
    trainer = MultiLabelTrainer(
        MultiLabelTrainerConfig(
            device=cfg.experiment.device,
            epochs=cfg.train.epochs,
            warmup_ratio=cfg.train.scheduler.warmup_ratio,
            max_grad_norm=(
                cfg.train.gradient_clipping.max_norm
                if cfg.train.gradient_clipping.enabled
                else None
            ),
            run_dir=run_dir,
            threshold=cfg.metrics.f1_threshold,
            analysis=cfg.analysis,
            class_names=tuple(train_dataset.vocabulary.index_to_label),
            checkpointing=cfg.checkpointing,
            terminal_width=cfg.terminal.width,
        ),
        pos_weight=pos_weight,
    )
    trainer.fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        extra_state={
            "run_config": asdict(cfg),
            "model_cfg": asdict(model.cfg),
            "architecture_summary": asdict(architecture_summary),
            "optimizer_summary": asdict(optimizer_summary),
            "initialization_summary": (
                asdict(initialization_summary)
                if initialization_summary is not None
                else None
            ),
            "fsd50k_index_to_label": list(train_dataset.vocabulary.index_to_label),
            "fsd50k_index_to_mid": list(train_dataset.vocabulary.index_to_mid),
            "pos_weight": pos_weight.cpu().tolist(),
            "pos_weight_stats": pos_weight_stats,
            "target_stats": target_stats,
            "class_positive_counts": train_targets.sum(dim=0).cpu().tolist(),
        },
    )


if __name__ == "__main__":
    main()
