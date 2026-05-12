from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

from src.data.loaders import build_clip_loader, build_dataset
from src.training.ast_setup import (
    apply_encoder_adaptation,
    build_ast_model,
    build_grouped_optimizer,
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
from src.utils.config import JsonConfigLoader
from src.utils.fs import Fs
from src.utils.logging import configure_terminal_width, enable_file_logging, logger
from src.utils.reproducibility import Reproducibility


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = JsonConfigLoader.load_cv(args.config)
    configure_terminal_width(cfg.terminal.width)
    Reproducibility.seed_everything(cfg.experiment.seed)

    base_dir = Fs.ensure_dir(Path(cfg.experiment.output_dir) / cfg.experiment.name)
    Fs.copy_file(args.config, base_dir)
    num_classes = len(cfg.data.label_to_index)

    for fold_index, fold in enumerate(cfg.folds):
        fold_seed = cfg.experiment.seed + fold_index
        generators = Reproducibility.create_generators(fold_seed)
        fold_dir = Fs.ensure_dir(base_dir / fold.name)
        enable_file_logging(fold_dir / "run.log", mode="w")
        fold_data = replace(
            cfg.data,
            train_dirs=list(fold.train_dirs),
            val_dirs=list(fold.val_dirs),
        )
        train_dataset = build_dataset(fold_data, split="train")
        val_dataset = build_dataset(fold_data, split="val")
        augmentation = fold_data.augmentation
        logger.info(
            "[%s] Data augmentation | enabled=%s | train_apply=%s | waveform=%s | "
            "waveform_probability=%.4f | gain=%s | noise=%s | time_shift=%s | "
            "fbank=%s | fbank_probability=%.4f | time_mask=%s | freq_mask=%s",
            fold.name,
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
            "top_tokens_per_branch=%d | evidence_score_source=%s | "
            "exclude_branches_from_evidence=%s | attention_temperature=%.4f | "
            "evidence_pooling=%s | branch_auxiliary=%s",
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
            architecture_summary.rdt_evidence_score_source,
            list(architecture_summary.rdt_exclude_branches_from_evidence),
            architecture_summary.mil_attention_temperature,
            architecture_summary.evidence_pooling_type,
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
        logger.info(
            "[%s] Train class counts: %s", fold.name, dict(imbalance.class_counts)
        )
        logger.info(
            "[%s] Train loss weights | class_counts=%s | class_weights=%s | "
            "main_to_binary=%s | binary_counts=%s | branch_binary_pos_weight=%s",
            fold.name,
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
        sampler = None
        if cfg.train.sampler.enabled:
            sampler, sampler_summary = build_sqrt_inverse_class_sampler(
                train_targets,
                num_classes=num_classes,
                cfg=cfg.train.sampler,
                generator=generators.sampler,
            )
            index_to_label = {
                int(index): label_name
                for label_name, index in cfg.data.label_to_index.items()
            }
            class_counts_by_label = {
                index_to_label.get(index, str(index)): count
                for index, count in enumerate(sampler_summary.class_counts)
            }
            class_weights_by_label = {
                index_to_label.get(index, str(index)): weight
                for index, weight in enumerate(sampler_summary.class_weights)
            }
            expected_probabilities_by_label = {
                index_to_label.get(index, str(index)): probability
                for index, probability in enumerate(
                    sampler_summary.expected_class_probabilities
                )
            }
            logger.info(
                "[%s] Sqrt-inverse class sampler enabled | type=%s | "
                "replacement=%s | num_samples=%d | source=%s",
                fold.name,
                cfg.train.sampler.type,
                cfg.train.sampler.replacement,
                sampler_summary.num_samples,
                cfg.train.sampler.source,
            )
            logger.info(
                "[%s] Sampler class counts: %s", fold.name, class_counts_by_label
            )
            logger.info(
                "[%s] Sampler class weights: %s",
                fold.name,
                class_weights_by_label,
            )
            logger.info(
                "[%s] Sampler expected class probabilities: %s",
                fold.name,
                expected_probabilities_by_label,
            )
        elif imbalance.weighted_random:
            sampler = build_weighted_sampler(
                train_targets, generator=generators.sampler
            )
            logger.info("[%s] Using legacy weighted random sampler", fold.name)

        train_loader = build_clip_loader(
            train_dataset,
            batch_size=fold_data.batch_size,
            num_workers=fold_data.num_workers,
            shuffle=sampler is None,
            sampler=sampler,
            generator=generators.train_loader,
        )
        val_loader = build_clip_loader(
            val_dataset,
            batch_size=fold_data.batch_size,
            num_workers=fold_data.num_workers,
            shuffle=False,
            generator=generators.val_loader,
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
                class_weights=resolved_loss_weights.class_weights,
                label_smoothing=cfg.train.loss.label_smoothing,
                branch_auxiliary=cfg.train.loss.branch_auxiliary,
                branch_binary_auxiliary=cfg.train.loss.branch_binary_auxiliary,
                branch_binary_pos_weight=(
                    resolved_loss_weights.branch_binary_pos_weight
                ),
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
                "loss_weight_summary": asdict(resolved_loss_weights),
            },
        )


if __name__ == "__main__":
    main()
