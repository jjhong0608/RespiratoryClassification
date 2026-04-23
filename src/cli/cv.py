from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import torch

from src.data.loaders import build_clip_loader, build_dataset
from src.training.ast_setup import (
    apply_encoder_adaptation,
    build_ast_model,
    build_grouped_optimizer,
    summarize_model_architecture,
)
from src.training.imbalance import (
    build_weighted_sampler,
    collect_targets,
    resolve_imbalance,
)
from src.training.trainer import Trainer, TrainerConfig
from src.utils.config import JsonConfigLoader
from src.utils.fs import Fs
from src.utils.logging import enable_file_logging, logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = JsonConfigLoader.load_cv(args.config)
    torch.manual_seed(cfg.experiment.seed)

    base_dir = Fs.ensure_dir(Path(cfg.experiment.output_dir) / cfg.experiment.name)
    Fs.copy_file(args.config, base_dir)
    num_classes = len(cfg.data.label_to_index)

    for fold in cfg.folds:
        fold_dir = Fs.ensure_dir(base_dir / fold.name)
        enable_file_logging(fold_dir / "run.log", mode="w")
        fold_data = replace(
            cfg.data,
            train_dirs=list(fold.train_dirs),
            val_dirs=list(fold.val_dirs),
        )
        train_dataset = build_dataset(fold_data, split="train")
        val_dataset = build_dataset(fold_data, split="val")
        model = build_ast_model(
            cfg.model,
            num_mel_bins=train_dataset.num_mel_bins,
            max_length=train_dataset.max_length,
            num_classes=num_classes,
        )
        architecture_summary = summarize_model_architecture(model)
        adaptation_summary = apply_encoder_adaptation(
            model, model.cfg.encoder.adaptation
        )
        logger.info(
            "[%s] Model architecture | encoder_type=%s | hidden_size=%d | num_heads=%d | "
            "branch_token_counts=%s | branch_time_lengths=%s | total_patch_tokens=%d | "
            "total_temporal_length=%d | rdt_enabled=%s | rdt_steps=%d | "
            "top_tokens_per_branch=%d | branch_auxiliary=%s",
            fold.name,
            architecture_summary.encoder_type,
            architecture_summary.hidden_size,
            architecture_summary.num_attention_heads,
            list(architecture_summary.branch_token_counts),
            list(architecture_summary.branch_time_lengths),
            architecture_summary.total_patch_token_count,
            architecture_summary.total_temporal_length,
            architecture_summary.rdt_enabled,
            architecture_summary.rdt_steps,
            architecture_summary.rdt_top_tokens_per_branch,
            cfg.train.loss.branch_auxiliary.enabled,
        )
        logger.info(
            "[%s] Encoder adaptation | mode=%s | num_layers=%d | trainable_params=%d | frozen_params=%d",
            fold.name,
            adaptation_summary.mode,
            adaptation_summary.num_layers,
            adaptation_summary.trainable_parameters,
            adaptation_summary.frozen_parameters,
        )

        train_targets = collect_targets(train_dataset)
        imbalance = resolve_imbalance(
            targets=train_targets,
            num_classes=num_classes,
            pos_weight=cfg.train.loss.pos_weight,
            auto_pos_weight=cfg.train.loss.auto_pos_weight,
            weighted_random=cfg.train.sampler.weighted_random,
        )
        logger.info(
            "[%s] Train class counts: %s", fold.name, dict(imbalance.class_counts)
        )
        if imbalance.pos_weight is not None:
            if cfg.train.loss.auto_pos_weight:
                logger.info(
                    "[%s] Using auto-computed positive-class weight=%.6f from training clips (loss=%s)",
                    fold.name,
                    imbalance.pos_weight,
                    cfg.train.loss.type,
                )
            else:
                logger.info(
                    "[%s] Using positive-class weight=%.6f (loss=%s)",
                    fold.name,
                    imbalance.pos_weight,
                    cfg.train.loss.type,
                )
        sampler = (
            build_weighted_sampler(train_targets) if imbalance.weighted_random else None
        )
        if sampler is not None:
            logger.info("[%s] Using weighted random sampler", fold.name)

        train_loader = build_clip_loader(
            train_dataset,
            batch_size=fold_data.batch_size,
            num_workers=fold_data.num_workers,
            shuffle=sampler is None,
            sampler=sampler,
        )
        val_loader = build_clip_loader(
            val_dataset,
            batch_size=fold_data.batch_size,
            num_workers=fold_data.num_workers,
            shuffle=False,
        )
        logger.info(
            "[%s] Train clips: %d | Val clips: %d",
            fold.name,
            len(train_dataset),
            len(val_dataset),
        )
        optimizer, optimizer_summary = build_grouped_optimizer(
            model,
            encoder_lr=cfg.train.optimizer.encoder_lr,
            head_lr=cfg.train.optimizer.head_lr,
            weight_decay=cfg.train.optimizer.weight_decay,
        )
        logger.info(
            "[%s] Optimizer groups | encoder_lr=%.8f | head_lr=%.8f | encoder_params=%d | head_params=%d | groups=%d",
            fold.name,
            optimizer_summary.encoder_lr,
            optimizer_summary.head_lr,
            optimizer_summary.encoder_trainable_parameters,
            optimizer_summary.head_trainable_parameters,
            optimizer_summary.param_group_count,
        )

        trainer = Trainer(
            TrainerConfig(
                device=cfg.experiment.device,
                epochs=cfg.train.epochs,
                encoder_lr=cfg.train.optimizer.encoder_lr,
                head_lr=cfg.train.optimizer.head_lr,
                weight_decay=cfg.train.optimizer.weight_decay,
                warmup_ratio=cfg.train.scheduler.warmup_ratio,
                max_grad_norm=cfg.train.max_grad_norm,
                top_k=cfg.train.top_k,
                run_dir=fold_dir,
                num_classes=num_classes,
                loss_type=cfg.train.loss.type,
                gamma=cfg.train.loss.gamma,
                pos_weight=imbalance.pos_weight,
                branch_auxiliary=cfg.train.loss.branch_auxiliary,
                analysis=cfg.analysis,
                early_stopping=cfg.train.early_stopping,
            )
        )
        trainer.fit(
            model,
            train_loader,
            val_loader,
            optimizer,
            extra_state={
                "run_config": asdict(cfg),
                "fold_name": fold.name,
                "model_cfg": asdict(model.cfg),
                "label_to_index": dict(cfg.data.label_to_index),
                "architecture_summary": asdict(architecture_summary),
                "adaptation_summary": asdict(adaptation_summary),
                "optimizer_summary": asdict(optimizer_summary),
            },
        )


if __name__ == "__main__":
    main()
