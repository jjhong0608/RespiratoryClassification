from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from src.data.fsd50k_dataset import build_fsd50k_dataset, build_fsd50k_loader
from src.models.ssl import MaskedFbankSSLWrapper
from src.training.ast_setup import build_ast_model, summarize_model_architecture
from src.training.ssl_trainer import SslTrainer, SslTrainerConfig
from src.utils.config import JsonConfigLoader
from src.utils.fs import Fs
from src.utils.logging import enable_file_logging, logger
from src.utils.reproducibility import Reproducibility


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = JsonConfigLoader.load_fsd50k_ssl(args.config)
    generators = Reproducibility.seed_everything(cfg.experiment.seed)
    run_dir = Fs.ensure_dir(Path(cfg.experiment.output_dir) / cfg.experiment.name)
    Fs.copy_file(args.config, run_dir)
    enable_file_logging(run_dir / "run.log", mode="w")

    train_dataset = build_fsd50k_dataset(cfg.data, split="train")
    val_dataset = build_fsd50k_dataset(cfg.data, split="val")
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
    base_model = build_ast_model(
        cfg.model,
        num_mel_bins=train_dataset.num_mel_bins,
        max_length=train_dataset.max_length,
        num_classes=2,
    )
    wrapper = MaskedFbankSSLWrapper(base_model, cfg.ssl)
    architecture_summary = summarize_model_architecture(base_model)
    logger.info(
        "FSD50K SSL | train=%d | val=%d | classes=%d | branch_grids=%s | "
        "token_mask_ratio=%.4f",
        len(train_dataset),
        len(val_dataset),
        train_dataset.num_classes,
        list(base_model.encoder.branch_token_grids),
        cfg.ssl.masking.token_mask_ratio,
    )
    trainer = SslTrainer(
        SslTrainerConfig(
            device=cfg.experiment.device,
            epochs=cfg.train.epochs,
            lr=cfg.train.optimizer.lr,
            weight_decay=cfg.train.optimizer.weight_decay,
            warmup_ratio=cfg.train.scheduler.warmup_ratio,
            max_grad_norm=cfg.train.max_grad_norm,
            run_dir=run_dir,
            checkpointing=cfg.checkpointing,
        )
    )
    trainer.fit(
        wrapper,
        train_loader,
        val_loader,
        extra_state={
            "run_config": asdict(cfg),
            "model_cfg": asdict(base_model.cfg),
            "ssl_cfg": asdict(cfg.ssl),
            "architecture_summary": asdict(architecture_summary),
            "fsd50k_index_to_label": list(train_dataset.vocabulary.index_to_label),
            "fsd50k_index_to_mid": list(train_dataset.vocabulary.index_to_mid),
        },
    )


if __name__ == "__main__":
    main()
