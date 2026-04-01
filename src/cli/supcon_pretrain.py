from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.audio import AudioPreprocessConfig
from src.data.contrastive import ContrastiveViewDataset, SpectrogramAugmenter
from src.data.loaders import build_dataset
from src.models.supcon import (
    SupervisedContrastiveEncoder,
    SupervisedContrastiveEncoderConfig,
)
from src.models.whisper_encoder import WhisperEncoderDims
from src.plots.reporter import Curve, PlotlyReporter
from src.pretrained.whisper import OpenAIWhisperCheckpointLoader
from src.training.imbalance import collect_targets
from src.training.supcon_trainer import (
    SupConTrainerConfig,
    SupervisedContrastiveTrainer,
)
from src.utils.config import JsonConfigLoader
from src.utils.fs import Fs
from src.utils.logging import enable_file_logging, logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = JsonConfigLoader.load_supcon_pretrain(args.config)
    torch.manual_seed(cfg.seed)

    run_dir = Fs.ensure_dir(Path(cfg.output_dir) / cfg.run_name)
    Fs.copy_file(args.config, run_dir)
    enable_file_logging(run_dir / "run.log", mode="w")

    preprocess = AudioPreprocessConfig(
        sample_rate=cfg.data.sample_rate,
        n_mels=cfg.encoder.n_mels,
        clip_seconds=cfg.data.clip_seconds,
        source_type=cfg.data.source_type,
        bandpass_enabled=cfg.data.bandpass.enabled,
        bandpass_low_freq=cfg.data.bandpass.low_freq,
        bandpass_high_freq=cfg.data.bandpass.high_freq,
        bandpass_q=cfg.data.bandpass.q,
    )
    if cfg.encoder.n_audio_ctx != preprocess.n_audio_ctx:
        raise ValueError(
            "Config mismatch: encoder.n_audio_ctx must equal "
            f"{preprocess.n_audio_ctx} for clip_seconds={cfg.data.clip_seconds} "
            f"and hop_length={preprocess.hop_length}; got {cfg.encoder.n_audio_ctx}"
        )

    encoder_dims = WhisperEncoderDims(
        n_mels=cfg.encoder.n_mels,
        n_audio_ctx=cfg.encoder.n_audio_ctx,
        n_audio_state=cfg.encoder.n_audio_state,
        n_audio_head=cfg.encoder.n_audio_head,
        n_audio_layer=cfg.encoder.n_audio_layer,
    )
    model = SupervisedContrastiveEncoder(
        SupervisedContrastiveEncoderConfig(
            encoder=encoder_dims,
            projection_hidden_dim=cfg.supervised_contrastive.projection_head.hidden_dim,
            projection_output_dim=cfg.supervised_contrastive.projection_head.output_dim,
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
            f"unexpected={len(pretrained_info.unexpected_keys)}"
        )

    base_train_dataset = build_dataset(
        list(cfg.data.train_dirs),
        preprocess,
        dict(cfg.data.label_to_index),
    )
    base_val_dataset = build_dataset(
        list(cfg.data.val_dirs),
        preprocess,
        dict(cfg.data.label_to_index),
    )
    train_targets = collect_targets(base_train_dataset)
    logger.info(f"Train class counts: {dict(sorted(Counter(train_targets).items()))}")

    augmentation_cfg = cfg.supervised_contrastive.augmentation
    train_dataset = ContrastiveViewDataset(
        base_train_dataset,
        SpectrogramAugmenter(augmentation_cfg),
    )
    val_dataset = ContrastiveViewDataset(
        base_val_dataset,
        SpectrogramAugmenter(augmentation_cfg),
        deterministic_seed_base=cfg.seed + 10_000,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    trainer = SupervisedContrastiveTrainer(
        SupConTrainerConfig(
            device=cfg.device,
            epochs=cfg.training.epochs,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            warmup_ratio=cfg.training.warmup_ratio,
            max_grad_norm=cfg.training.max_grad_norm,
            top_k=cfg.top_k,
            run_dir=run_dir,
            temperature=cfg.supervised_contrastive.temperature,
            normalize=cfg.supervised_contrastive.normalize,
        )
    )
    trainer.fit(
        model,
        train_loader,
        val_loader,
        extra_state={
            "encoder_cfg": asdict(cfg.encoder),
            "preprocess_cfg": asdict(preprocess),
            "pretrained_cfg": asdict(cfg.pretrained) if cfg.pretrained else None,
            "pretrained_info": asdict(pretrained_info) if pretrained_info else None,
            "supervised_contrastive_cfg": asdict(cfg.supervised_contrastive),
            "label_to_index": dict(cfg.data.label_to_index),
        },
    )

    last = torch.load(run_dir / "last.pt", map_location="cpu")
    train_losses = [float(x) for x in last.get("train_losses", [])]
    val_losses = [float(x) for x in last.get("val_losses", [])]
    epochs = list(range(1, len(train_losses) + 1))
    reporter = PlotlyReporter(run_dir / "plots")
    reporter.save_curves(
        "supcon_loss_curves.html",
        "Supervised Contrastive Loss Curves",
        [
            Curve(name="train_supcon_loss", x=epochs, y=train_losses),
            Curve(name="val_supcon_loss", x=epochs, y=val_losses),
        ],
    )


if __name__ == "__main__":
    main()
