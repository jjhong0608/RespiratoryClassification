from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import AdamW

from src.data.audio import AudioPreprocessConfig
from src.data.loaders import build_dataset, build_loader
from src.models.model import WhisperEncoderClassifier
from src.training.imbalance import (
    build_weighted_sampler,
    collect_targets,
    resolve_imbalance,
)
from src.training.trainer import Trainer, TrainerConfig
from src.utils.checkpoint import load_checkpoint, parse_model_cfg
from src.utils.config import JsonConfigLoader
from src.utils.fs import Fs
from src.utils.logging import LoggingMixin, enable_file_logging, logger


class FineTuneRunner(LoggingMixin):
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.cfg = JsonConfigLoader.load_finetune(self.config_path)

    @staticmethod
    def build_optimizer(
        model: WhisperEncoderClassifier,
        *,
        encoder_lr: float,
        classifier_lr: float,
        weight_decay: float,
    ) -> AdamW:
        return AdamW(
            [
                {
                    "params": model.encoder.parameters(),
                    "lr": encoder_lr,
                    "weight_decay": weight_decay,
                },
                {
                    "params": model.classifier.parameters(),
                    "lr": classifier_lr,
                    "weight_decay": weight_decay,
                },
            ]
        )

    def _resolve_preprocess(self, model_cfg, checkpoint: dict) -> AudioPreprocessConfig:
        preprocess_raw = checkpoint.get("preprocess_cfg")
        if isinstance(preprocess_raw, dict):
            return AudioPreprocessConfig(**preprocess_raw)
        if isinstance(preprocess_raw, AudioPreprocessConfig):
            return preprocess_raw
        return AudioPreprocessConfig(
            sample_rate=self.cfg.data.sample_rate,
            n_mels=model_cfg.encoder.n_mels,
            clip_seconds=self.cfg.data.clip_seconds,
            source_type=self.cfg.data.source_type,
            bandpass_enabled=self.cfg.data.bandpass.enabled,
            bandpass_low_freq=self.cfg.data.bandpass.low_freq,
            bandpass_high_freq=self.cfg.data.bandpass.high_freq,
            bandpass_q=self.cfg.data.bandpass.q,
        )

    def run(self) -> None:
        torch.manual_seed(self.cfg.seed)

        run_dir = Fs.ensure_dir(Path(self.cfg.output_dir) / self.cfg.run_name)
        Fs.copy_file(self.config_path, run_dir)
        enable_file_logging(run_dir / "run.log", mode="w")

        ckpt = load_checkpoint(
            self.cfg.finetune.checkpoint_path,
            device=torch.device(self.cfg.device),
            unsafe=getattr(self.cfg.finetune, "unsafe_pickle_load", False),
        )
        model_cfg = parse_model_cfg(ckpt.get("model_cfg"))
        model = WhisperEncoderClassifier(model_cfg)
        model.load_state_dict(ckpt["model_state_dict"])

        for p in model.encoder.parameters():
            p.requires_grad = True

        preprocess = self._resolve_preprocess(model_cfg, ckpt)
        if model_cfg.encoder.n_audio_ctx != preprocess.n_audio_ctx:
            raise ValueError(
                "Config mismatch: model.n_audio_ctx must equal "
                f"{preprocess.n_audio_ctx} for clip_seconds={preprocess.clip_seconds} "
                f"and hop_length={preprocess.hop_length}; got {model_cfg.encoder.n_audio_ctx}"
            )

        num_classes = len(self.cfg.data.label_to_index)
        if num_classes != model_cfg.num_classes:
            raise ValueError(
                f"num_classes mismatch: checkpoint has {model_cfg.num_classes}, "
                f"but label_to_index has {num_classes}"
            )

        train_dataset = build_dataset(
            list(self.cfg.data.train_dirs),
            preprocess,
            dict(self.cfg.data.label_to_index),
        )
        val_dataset = build_dataset(
            list(self.cfg.data.val_dirs),
            preprocess,
            dict(self.cfg.data.label_to_index),
        )
        train_targets = collect_targets(train_dataset)
        imbalance = resolve_imbalance(
            self.cfg.imbalance,
            train_targets,
            num_classes=num_classes,
        )
        logger.info(f"Train class counts: {dict(imbalance.class_counts)}")
        if imbalance.pos_weight is not None:
            logger.info(f"Resolved pos_weight={imbalance.pos_weight:.6f}")
        sampler = None
        if imbalance.sampler == "weighted_random":
            sampler = build_weighted_sampler(train_targets)
            logger.info("Using weighted random sampler for fine-tuning")

        train_loader = build_loader(
            train_dataset,
            self.cfg.data.batch_size,
            self.cfg.data.num_workers,
            shuffle=sampler is None,
            sampler=sampler,
        )
        val_loader = build_loader(
            val_dataset,
            self.cfg.data.batch_size,
            self.cfg.data.num_workers,
            shuffle=False,
        )

        classifier_lr = (
            self.cfg.finetune.classifier_lr or self.cfg.training.learning_rate
        )
        encoder_lr = self.cfg.finetune.encoder_lr or (classifier_lr * 0.1)
        logger.info(
            f"Fine-tune LRs | encoder_lr={encoder_lr:.8f} | classifier_lr={classifier_lr:.8f}"
        )

        optimizer = self.build_optimizer(
            model,
            encoder_lr=encoder_lr,
            classifier_lr=classifier_lr,
            weight_decay=self.cfg.training.weight_decay,
        )

        trainer = Trainer(
            TrainerConfig(
                device=self.cfg.device,
                epochs=self.cfg.training.epochs,
                learning_rate=self.cfg.training.learning_rate,
                weight_decay=self.cfg.training.weight_decay,
                warmup_ratio=self.cfg.training.warmup_ratio,
                max_grad_norm=self.cfg.training.max_grad_norm,
                top_k=self.cfg.top_k,
                num_classes=num_classes,
                run_dir=run_dir,
                pos_weight=imbalance.pos_weight,
                threshold_optimization=self.cfg.threshold_optimization,
            )
        )

        trainer.fit(
            model,
            train_loader,
            val_loader,
            optimizer,
            extra_state={
                "base_checkpoint_path": self.cfg.finetune.checkpoint_path,
                "model_cfg": asdict(model_cfg),
                "preprocess_cfg": asdict(preprocess),
                "finetune_cfg": asdict(self.cfg.finetune),
                "label_to_index": dict(self.cfg.data.label_to_index),
                "imbalance_cfg": asdict(self.cfg.imbalance),
                "imbalance_info": asdict(imbalance),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    FineTuneRunner(args.config).run()


if __name__ == "__main__":
    main()
