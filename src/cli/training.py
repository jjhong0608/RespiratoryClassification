from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from src.data.audio import AudioPreprocessConfig
from src.data.loaders import build_dataset, build_loader
from src.models.model import WhisperClassifierConfig, WhisperEncoderClassifier
from src.models.whisper_encoder import WhisperEncoderDims
from src.plots.reporter import Curve, PlotlyReporter
from src.pretrained.whisper import OpenAIWhisperCheckpointLoader
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

    cfg = JsonConfigLoader.load_training(args.config)
    torch.manual_seed(cfg.seed)

    run_dir = Fs.ensure_dir(Path(cfg.output_dir) / cfg.run_name)
    Fs.copy_file(args.config, run_dir)
    enable_file_logging(run_dir / "run.log", mode="w")

    preprocess = AudioPreprocessConfig(
        sample_rate=cfg.data.sample_rate,
        n_mels=cfg.model.n_mels,
        clip_seconds=cfg.data.clip_seconds,
        source_type=cfg.data.source_type,
        bandpass_enabled=cfg.data.bandpass.enabled,
        bandpass_low_freq=cfg.data.bandpass.low_freq,
        bandpass_high_freq=cfg.data.bandpass.high_freq,
        bandpass_q=cfg.data.bandpass.q,
    )
    if cfg.model.n_audio_ctx != preprocess.n_audio_ctx:
        raise ValueError(
            "Config mismatch: model.n_audio_ctx must equal "
            f"{preprocess.n_audio_ctx} for clip_seconds={cfg.data.clip_seconds} "
            f"and hop_length={preprocess.hop_length}; got {cfg.model.n_audio_ctx}"
        )
    num_classes = len(cfg.data.label_to_index)
    encoder_dims = WhisperEncoderDims(
        n_mels=cfg.model.n_mels,
        n_audio_ctx=cfg.model.n_audio_ctx,
        n_audio_state=cfg.model.n_audio_state,
        n_audio_head=cfg.model.n_audio_head,
        n_audio_layer=cfg.model.n_audio_layer,
    )
    model = WhisperEncoderClassifier(
        WhisperClassifierConfig(
            encoder=encoder_dims,
            num_classes=num_classes,
            head_type="hf",
            pooling=cfg.model.pooling,
            use_weighted_layer_sum=cfg.model.use_weighted_layer_sum,
            classifier_proj_size=cfg.model.classifier_proj_size,
        )
    )

    pretrained_info = None
    if cfg.pretrained is not None:
        loader = OpenAIWhisperCheckpointLoader()
        pretrained_info = loader.load_encoder_into(model, cfg.pretrained)
        logger.info(
            "Loaded pretrained encoder | "
            f"source={pretrained_info.source} | "
            f"path={pretrained_info.resolved_path} | "
            f"loaded_keys={pretrained_info.loaded_keys} | "
            f"missing={len(pretrained_info.missing_keys)} | "
            f"unexpected={len(pretrained_info.unexpected_keys)} | "
            f"freeze_encoder={cfg.pretrained.freeze_encoder}"
        )

    train_dataset = build_dataset(
        list(cfg.data.train_dirs),
        preprocess,
        dict(cfg.data.label_to_index),
    )
    val_dataset = build_dataset(
        list(cfg.data.val_dirs),
        preprocess,
        dict(cfg.data.label_to_index),
    )
    train_targets = collect_targets(train_dataset)
    imbalance = resolve_imbalance(
        cfg.imbalance,
        train_targets,
        num_classes=num_classes,
    )
    logger.info(f"Train class counts: {dict(imbalance.class_counts)}")
    if imbalance.pos_weight is not None:
        logger.info(f"Resolved pos_weight={imbalance.pos_weight:.6f}")
    sampler = None
    if imbalance.sampler == "weighted_random":
        sampler = build_weighted_sampler(train_targets)
        logger.info("Using weighted random sampler for training loader")

    train_loader = build_loader(
        train_dataset,
        cfg.data.batch_size,
        cfg.data.num_workers,
        shuffle=sampler is None,
        sampler=sampler,
        pin_memory=True,
    )
    val_loader = build_loader(
        val_dataset,
        cfg.data.batch_size,
        cfg.data.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    trainer = Trainer(
        TrainerConfig(
            device=cfg.device,
            epochs=cfg.training.epochs,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            warmup_ratio=cfg.training.warmup_ratio,
            max_grad_norm=cfg.training.max_grad_norm,
            top_k=cfg.top_k,
            num_classes=num_classes,
            run_dir=run_dir,
            pos_weight=imbalance.pos_weight,
            threshold_optimization=cfg.threshold_optimization,
        )
    )

    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    trainer.fit(
        model,
        train_loader,
        val_loader,
        extra_state={
            "model_cfg": asdict(model.cfg),
            "label_to_index": dict(cfg.data.label_to_index),
            "preprocess_cfg": asdict(preprocess),
            "pretrained_cfg": asdict(cfg.pretrained) if cfg.pretrained else None,
            "pretrained_info": asdict(pretrained_info) if pretrained_info else None,
            "imbalance_cfg": asdict(cfg.imbalance),
            "imbalance_info": asdict(imbalance),
        },
    )

    last = torch.load(run_dir / "last.pt", map_location="cpu")
    train_losses = [float(x) for x in last.get("train_losses", [])]
    val_losses = [float(x) for x in last.get("val_losses", [])]
    epochs = list(range(1, len(train_losses) + 1))
    reporter = PlotlyReporter(run_dir / "plots")
    reporter.save_curves(
        "loss_curves.html",
        "Loss Curves",
        [
            Curve(name="train_loss", x=epochs, y=train_losses),
            Curve(name="val_loss", x=epochs, y=val_losses),
        ],
    )


if __name__ == "__main__":
    main()
