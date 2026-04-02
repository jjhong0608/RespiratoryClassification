from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import torch

from src.data.loaders import build_bag_loader, build_dataset
from src.training.imbalance import (
    build_weighted_sampler,
    collect_targets,
    resolve_imbalance,
)
from src.training.mil_setup import build_mil_model, maybe_initialize_encoder
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
        model = build_mil_model(
            cfg.model,
            segment_n_mels=train_dataset.segment_n_mels,
            segment_audio_ctx=train_dataset.segment_audio_ctx,
        )
        pretrained_info = maybe_initialize_encoder(model, cfg.model)
        if pretrained_info is not None:
            logger.info(
                "[%s] Loaded pretrained encoder | source=%s | path=%s | loaded_keys=%d | missing=%d | unexpected=%d | freeze_encoder=%s",
                fold.name,
                pretrained_info.source,
                pretrained_info.resolved_path,
                pretrained_info.loaded_keys,
                len(pretrained_info.missing_keys),
                len(pretrained_info.unexpected_keys),
                cfg.model.encoder.freeze,
            )

        train_targets = collect_targets(train_dataset)
        imbalance = resolve_imbalance(
            targets=train_targets,
            pos_weight=cfg.train.loss.pos_weight,
            weighted_random=cfg.train.sampler.weighted_random,
        )
        logger.info(
            "[%s] Train class counts: %s", fold.name, dict(imbalance.class_counts)
        )
        sampler = (
            build_weighted_sampler(train_targets) if imbalance.weighted_random else None
        )
        if sampler is not None:
            logger.info("[%s] Using weighted random sampler", fold.name)

        train_loader = build_bag_loader(
            train_dataset,
            batch_size=fold_data.batch_size,
            num_workers=fold_data.num_workers,
            shuffle=sampler is None,
            sampler=sampler,
        )
        val_loader = build_bag_loader(
            val_dataset,
            batch_size=fold_data.batch_size,
            num_workers=fold_data.num_workers,
            shuffle=False,
        )
        logger.info(
            "[%s] Train bags: %d | Val bags: %d",
            fold.name,
            len(train_dataset),
            len(val_dataset),
        )

        trainer = Trainer(
            TrainerConfig(
                device=cfg.experiment.device,
                epochs=cfg.train.epochs,
                learning_rate=cfg.train.optimizer.lr,
                weight_decay=cfg.train.optimizer.weight_decay,
                warmup_ratio=cfg.train.warmup_ratio,
                max_grad_norm=cfg.train.max_grad_norm,
                top_k=cfg.train.top_k,
                run_dir=fold_dir,
                pos_weight=imbalance.pos_weight,
                analysis=cfg.analysis,
            )
        )
        trainer.fit(
            model,
            train_loader,
            val_loader,
            extra_state={
                "run_config": asdict(cfg),
                "fold_name": fold.name,
                "model_cfg": asdict(model.cfg),
                "label_to_index": dict(cfg.data.label_to_index),
                "pretrained_info": (
                    asdict(pretrained_info) if pretrained_info is not None else None
                ),
            },
        )


if __name__ == "__main__":
    main()
