from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from src.cli.evaluate import evaluate_checkpoint
from src.models.model import WhisperClassifierConfig, WhisperEncoderClassifier
from src.models.whisper_encoder import WhisperEncoderDims
from src.utils.config import JsonConfigLoader
from torch.utils.data import DataLoader, TensorDataset


def _write_binary_checkpoint(tmp_path: Path, *, saved_threshold: float | None) -> Path:
    dims = WhisperEncoderDims(
        n_mels=80,
        n_audio_ctx=2,
        n_audio_state=32,
        n_audio_head=4,
        n_audio_layer=1,
    )
    model = WhisperEncoderClassifier(
        WhisperClassifierConfig(
            encoder=dims,
            num_classes=2,
            head_type="linear",
            pooling="mean",
        )
    )
    ckpt_path = tmp_path / "ckpt.pt"
    threshold_result = None
    if saved_threshold is not None:
        threshold_result = {
            "enabled": True,
            "applied": True,
            "selected_metric": "f1",
            "selected_threshold": saved_threshold,
            "selected_score": 1.0,
            "reason": None,
            "f1": {"metric": "f1", "threshold": saved_threshold, "score": 1.0},
            "balanced_accuracy": None,
            "youden_j": None,
        }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
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
            "label_to_index": {"negative": 0, "positive": 1},
            "preprocess_cfg": {"sample_rate": 16000, "n_mels": 80, "clip_seconds": 0.1},
            "threshold_optimization_result": threshold_result,
        },
        ckpt_path,
    )
    return ckpt_path


def _write_eval_config(
    tmp_path: Path,
    ckpt_path: Path,
    *,
    manual_threshold: float | None = None,
):
    eval_cfg = {
        "device": "cpu",
        "checkpoint_path": str(ckpt_path),
        "unsafe_pickle_load": False,
        "data": {
            "eval_dirs": [str(tmp_path)],
            "label_to_index": {"negative": 0, "positive": 1},
            "sample_rate": 16000,
            "clip_seconds": 0.1,
            "batch_size": 2,
            "num_workers": 0,
        },
    }
    if manual_threshold is not None:
        eval_cfg["threshold"] = {"manual": manual_threshold}
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps(eval_cfg), encoding="utf-8")
    return JsonConfigLoader.load_eval(eval_path)


def _stub_eval_loader(tmp_path: Path, monkeypatch) -> None:
    x = torch.zeros((2, 80, 4), dtype=torch.float32)
    x[0, 0, 0] = 0.4054651081  # sigmoid -> 0.6
    x[1, 0, 0] = 2.1972245773  # sigmoid -> 0.9
    y = torch.tensor([0, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    file_paths = [tmp_path / "a.wav", tmp_path / "b.wav"]

    monkeypatch.setattr(
        "src.cli.evaluate.build_loader",
        lambda roots, preprocess, label_to_index, batch_size, num_workers: (
            loader,
            file_paths,
        ),
    )

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        del self
        diffs = batch[:, 0, 0]
        zeros = torch.zeros_like(diffs)
        return torch.stack([zeros, diffs], dim=-1)

    monkeypatch.setattr(WhisperEncoderClassifier, "forward", _forward)


def test_evaluate_uses_checkpoint_threshold(tmp_path: Path, monkeypatch) -> None:
    ckpt_path = _write_binary_checkpoint(tmp_path, saved_threshold=0.8)
    cfg = _write_eval_config(tmp_path, ckpt_path)
    _stub_eval_loader(tmp_path, monkeypatch)

    metrics, rows = evaluate_checkpoint(cfg, ckpt_path, return_predictions=True)

    assert metrics["decision_threshold"] == pytest.approx(0.8)
    assert metrics["decision_threshold_source"] == "checkpoint"
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert rows[0].predicted_index == 0
    assert rows[0].predicted_probability == pytest.approx(0.4, rel=1e-4)
    assert rows[1].predicted_index == 1
    assert rows[1].predicted_probability == pytest.approx(0.9, rel=1e-4)


def test_evaluate_manual_threshold_overrides_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    ckpt_path = _write_binary_checkpoint(tmp_path, saved_threshold=0.8)
    cfg = _write_eval_config(tmp_path, ckpt_path, manual_threshold=0.95)
    _stub_eval_loader(tmp_path, monkeypatch)

    metrics, rows = evaluate_checkpoint(cfg, ckpt_path, return_predictions=True)

    assert metrics["decision_threshold"] == pytest.approx(0.95)
    assert metrics["decision_threshold_source"] == "manual"
    assert rows[0].predicted_index == 0
    assert rows[1].predicted_index == 0


def test_evaluate_falls_back_to_default_threshold_when_missing(
    tmp_path: Path, monkeypatch
) -> None:
    ckpt_path = _write_binary_checkpoint(tmp_path, saved_threshold=None)
    cfg = _write_eval_config(tmp_path, ckpt_path)
    _stub_eval_loader(tmp_path, monkeypatch)

    metrics, rows = evaluate_checkpoint(cfg, ckpt_path, return_predictions=True)

    assert metrics["decision_threshold"] == pytest.approx(0.5)
    assert metrics["decision_threshold_source"] == "default"
    assert rows[0].predicted_index == 1
    assert rows[1].predicted_index == 1
