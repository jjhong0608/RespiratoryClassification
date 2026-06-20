from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from src.data.loaders import build_clip_loader, build_dataset
from src.training.imbalance import (
    build_weighted_sampler,
    collect_targets,
    resolve_imbalance,
)
from src.training.model_setup import build_model_bundle
from src.training.trainer import Trainer, TrainerConfig
from src.utils.config import JsonConfigLoader
from src.utils.fs import Fs
from src.utils.logging import enable_file_logging, logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = JsonConfigLoader.load_training(args.config)
    torch.manual_seed(cfg.experiment.seed)

    run_dir = Fs.ensure_dir(Path(cfg.experiment.output_dir) / cfg.experiment.name)
    Fs.copy_file(args.config, run_dir)
    enable_file_logging(run_dir / "run.log", mode="w")

    train_dataset = build_dataset(cfg.data, split="train")
    val_dataset = build_dataset(cfg.data, split="val")
    num_classes = len(cfg.data.label_to_index)
    bundle = build_model_bundle(
        cfg.model,
        cfg.train,
        train_dataset,
        num_classes=num_classes,
    )
    model = bundle.model
    if bundle.pretrained_info is not None:
        logger.info(
            "Loaded pretrained %s encoder | info=%s",
            bundle.model_family,
            bundle.pretrained_info,
        )
    logger.info(
        "Encoder adaptation | family=%s | mode=%s | num_layers=%d | "
        "trainable_params=%d | frozen_params=%d",
        bundle.model_family,
        bundle.adaptation_summary.mode,
        bundle.adaptation_summary.num_layers,
        bundle.adaptation_summary.trainable_parameters,
        bundle.adaptation_summary.frozen_parameters,
    )

    train_targets = collect_targets(train_dataset)
    imbalance = resolve_imbalance(
        targets=train_targets,
        num_classes=num_classes,
        pos_weight=cfg.train.loss.pos_weight,
        auto_pos_weight=cfg.train.loss.auto_pos_weight,
        weighted_random=cfg.train.sampler.weighted_random,
    )
    logger.info("Train class counts: %s", dict(imbalance.class_counts))
    if imbalance.pos_weight is not None:
        if cfg.train.loss.auto_pos_weight:
            logger.info(
                "Using auto-computed positive-class weight=%.6f from training clips (loss=%s)",
                imbalance.pos_weight,
                cfg.train.loss.type,
            )
        else:
            logger.info(
                "Using positive-class weight=%.6f (loss=%s)",
                imbalance.pos_weight,
                cfg.train.loss.type,
            )

    sampler = (
        build_weighted_sampler(train_targets) if imbalance.weighted_random else None
    )
    if sampler is not None:
        logger.info("Using weighted random sampler for clip training")

    train_loader = build_clip_loader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=sampler is None,
        sampler=sampler,
    )
    val_loader = build_clip_loader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )
    logger.info("Train clips: %d | Val clips: %d", len(train_dataset), len(val_dataset))
    logger.info(
        "Optimizer groups | encoder_lr=%.8f | head_lr=%.8f | encoder_params=%d | head_params=%d | groups=%d",
        bundle.optimizer_summary.encoder_lr,
        bundle.optimizer_summary.head_lr,
        bundle.optimizer_summary.encoder_trainable_parameters,
        bundle.optimizer_summary.head_trainable_parameters,
        bundle.optimizer_summary.param_group_count,
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
            run_dir=run_dir,
            num_classes=num_classes,
            loss_type=cfg.train.loss.type,
            gamma=cfg.train.loss.gamma,
            pos_weight=imbalance.pos_weight,
            analysis=cfg.analysis,
            early_stopping=cfg.train.early_stopping,
        )
    )
    trainer.fit(
        model,
        train_loader,
        val_loader,
        bundle.optimizer,
        extra_state={
            "run_config": asdict(cfg),
            "model_family": bundle.model_family,
            "model_cfg": asdict(bundle.model_cfg),
            "feature_cfg": bundle.feature_cfg,
            "label_to_index": dict(cfg.data.label_to_index),
            "num_classes": num_classes,
            "pretrained_info": (
                asdict(bundle.pretrained_info)
                if bundle.pretrained_info is not None
                else None
            ),
            "adaptation_summary": asdict(bundle.adaptation_summary),
            "optimizer_summary": asdict(bundle.optimizer_summary),
        },
    )


if __name__ == "__main__":
    main()
