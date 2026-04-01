from __future__ import annotations

import json
from pathlib import Path

import torch
from src.cli.evaluate_all import main as eval_all_main


def test_evaluate_all_writes_unique_metrics(tmp_path: Path, monkeypatch) -> None:
    # Create fake checkpoints in two directories
    root = tmp_path / "root"
    fold_a = root / "fold_a"
    fold_b = root / "fold_b"
    fold_a.mkdir(parents=True)
    fold_b.mkdir(parents=True)

    ckpt = {
        "model_state_dict": {},
        "model_cfg": {
            "encoder": {
                "n_mels": 80,
                "n_audio_ctx": 2,
                "n_audio_state": 32,
                "n_audio_head": 4,
                "n_audio_layer": 1,
            },
            "num_classes": 2,
            "pooling": "mean",
            "classifier_type": "linear",
            "hidden_dim": 256,
            "dropout": 0.0,
        },
        "label_to_index": {"a": 0, "b": 1},
        "preprocess_cfg": {"sample_rate": 16000, "n_mels": 80, "clip_seconds": 0.1},
    }

    for p in [fold_a / "last.pt", fold_a / "best_loss_0.1.pt", fold_b / "last.pt"]:
        torch.save(ckpt, p)

    eval_cfg = {
        "device": "cpu",
        "checkpoint_path": str(fold_a / "last.pt"),
        "unsafe_pickle_load": False,
        "data": {
            "eval_dirs": [str(tmp_path)],  # empty dataset; won't be used in this test
            "label_to_index": {"a": 0, "b": 1},
            "sample_rate": 16000,
            "clip_seconds": 0.1,
            "batch_size": 1,
            "num_workers": 0,
        },
    }
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps(eval_cfg), encoding="utf-8")

    # Stub evaluate_checkpoint to avoid heavy model execution
    import src.cli.evaluate_all as eval_all

    def _stub_eval(cfg, checkpoint_path):
        return {"accuracy": 1.0, "checkpoint": str(checkpoint_path)}

    monkeypatch.setattr(eval_all, "evaluate_checkpoint", _stub_eval)
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_all.py",
            "--config",
            str(eval_path),
            "--root",
            str(root),
        ],
    )
    eval_all_main()

    assert (fold_a / "eval_metrics__last.json").exists()
    assert (fold_a / "eval_metrics__best_loss_0.1.json").exists()
    assert (fold_b / "eval_metrics__last.json").exists()
