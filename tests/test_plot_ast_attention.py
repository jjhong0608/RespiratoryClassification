from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
import soundfile as sf
import torch
from src.cli.plot_ast_attention import (
    AstAttentionInputLoader,
    AstAttentionModelRunner,
    AstAttentionVisualizer,
    AttentionCliConfig,
    AttentionScoreExtractor,
    PatchGridMapper,
    load_attention_checkpoint,
    parse_formats,
)
from src.models.model import (
    AstArchitectureConfig,
    AstEncoderConfig,
    AstFeatureDims,
    AstModelConfig,
    ClassifierConfig,
    RespiratoryAstModel,
)
from src.utils.config import (
    AstFbankConfig,
    AudioConfig,
    BandPassConfig,
    DataConfig,
    ExperimentConfig,
    PreprocessingConfig,
)


def _write_wav(path: Path, *, sample_rate: int = 16000) -> None:
    t = np.linspace(0.0, 0.8, int(sample_rate * 0.8), endpoint=False)
    audio = 0.2 * np.sin(2.0 * np.pi * 320.0 * t)
    sf.write(path, audio.astype(np.float32), sample_rate)


def _data_cfg(tmp_path: Path) -> DataConfig:
    return DataConfig(
        train_dirs=[str(tmp_path / "train")],
        val_dirs=[str(tmp_path / "val")],
        eval_dirs=[str(tmp_path / "eval")],
        label_to_index={"normal": 0, "wheeze": 1, "crackle": 2},
        batch_size=1,
        num_workers=0,
        audio=AudioConfig(sample_rate=16000, clip_duration_sec=1.0),
        preprocessing=PreprocessingConfig(
            source_type="original",
            bandpass=BandPassConfig(enabled=False),
            ast_fbank=AstFbankConfig(
                num_mel_bins=32,
                max_length=32,
                do_normalize=True,
            ),
        ),
    )


def _model_cfg(num_classes: int = 3) -> AstModelConfig:
    return AstModelConfig(
        encoder=AstEncoderConfig(
            feature_dims=AstFeatureDims(num_mel_bins=32, max_length=32),
            pretrained_name_or_path=None,
            architecture=AstArchitectureConfig(
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=64,
                patch_size=16,
                frequency_stride=10,
                time_stride=10,
            ),
        ),
        classifier=ClassifierConfig(
            type="linear", hidden_dim=32, dropout=0.0, pooling="cls"
        ),
        num_classes=num_classes,
    )


def _checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(7)
    data_cfg = _data_cfg(tmp_path)
    model = RespiratoryAstModel(_model_cfg())
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_cfg": asdict(model.cfg),
            "model_state_dict": model.state_dict(),
            "label_to_index": dict(data_cfg.label_to_index),
            "run_config": {
                "experiment": asdict(
                    ExperimentConfig(
                        name="test",
                        task="classification",
                        mode="clip",
                        seed=0,
                        device="cpu",
                        output_dir=str(tmp_path / "run"),
                    )
                ),
                "data": asdict(data_cfg),
            },
        },
        path,
    )
    return path


def _loaded_inputs_runner(
    tmp_path: Path,
) -> tuple[Path, Path, AstAttentionModelRunner, torch.Tensor]:
    checkpoint = _checkpoint(tmp_path)
    wav = tmp_path / "sample.wav"
    _write_wav(wav)
    loaded = load_attention_checkpoint(checkpoint, "cpu")
    inputs = AstAttentionInputLoader(loaded.data_cfg).load(wav)
    runner = AstAttentionModelRunner(loaded)
    return checkpoint, wav, runner, inputs.input_values


def test_checkpoint_loading_and_fbank_shape(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    wav = tmp_path / "sample.wav"
    _write_wav(wav)

    loaded = load_attention_checkpoint(checkpoint, "cpu")
    inputs = AstAttentionInputLoader(loaded.data_cfg).load(wav)

    assert loaded.label_to_index == {"normal": 0, "wheeze": 1, "crackle": 2}
    assert inputs.fbank.shape == (32, 32)
    assert inputs.input_values.shape == (1, 32, 32)


def test_runner_forces_eager_attention_and_returns_attention_shape(
    tmp_path: Path,
) -> None:
    _, _, runner, input_values = _loaded_inputs_runner(tmp_path)

    result = runner.run(
        input_values,
        attention_method="last_cls_patch_head_mean",
        target_class="predicted",
    )

    assert runner.model.encoder.config._attn_implementation == "eager"
    assert len(result.attentions) == 2
    assert result.attentions[-1].shape == (1, 4, 6, 6)


def test_patch_grid_mapping_matches_ast_flatten_order(tmp_path: Path) -> None:
    _, _, runner, _ = _loaded_inputs_runner(tmp_path)
    mapper = PatchGridMapper.from_model(runner)

    record = mapper.patch_record(
        rank=1,
        patch_index=3,
        attention_score=0.5,
        normalized_attention_score=1.0,
    )

    assert mapper.geometry.frequency_patches == 2
    assert mapper.geometry.time_patches == 2
    assert record.freq_index == 1
    assert record.time_index == 1
    assert record.freq_start_bin == 10
    assert record.time_start_frame == 10
    assert record.time_start_sec == pytest.approx(0.1)


def test_attention_score_methods_return_finite_patch_scores(tmp_path: Path) -> None:
    _, _, runner, input_values = _loaded_inputs_runner(tmp_path)
    mapper = PatchGridMapper.from_model(runner)
    extractor = AttentionScoreExtractor(mapper)

    run_result = runner.run(
        input_values,
        attention_method="last_cls_patch_head_mean",
        target_class="predicted",
    )
    for method, head in [
        ("last_cls_patch", "0"),
        ("last_cls_patch_head_mean", "0"),
        ("attention_rollout", "0"),
    ]:
        scores = extractor.extract(run_result, method=method, head=head, top_k=2)
        assert scores.raw_scores.shape == (4,)
        assert np.isfinite(scores.raw_scores).all()
        assert scores.normalized_scores.min() >= 0.0
        assert scores.normalized_scores.max() <= 1.0
        assert len(scores.top_patches) == 2

    gradient_result = runner.run(
        input_values,
        attention_method="class_gradient_attention",
        target_class="wheeze",
    )
    gradient_scores = extractor.extract(
        gradient_result,
        method="class_gradient_attention",
        head="0",
        top_k=3,
    )
    assert gradient_scores.raw_scores.shape == (4,)
    assert np.isfinite(gradient_scores.raw_scores).all()


def test_invalid_head_and_format_raise_errors(tmp_path: Path) -> None:
    _, _, runner, input_values = _loaded_inputs_runner(tmp_path)
    mapper = PatchGridMapper.from_model(runner)
    extractor = AttentionScoreExtractor(mapper)
    run_result = runner.run(
        input_values,
        attention_method="last_cls_patch",
        target_class="predicted",
    )

    with pytest.raises(ValueError, match="out of range"):
        extractor.extract(run_result, method="last_cls_patch", head="99", top_k=2)

    with pytest.raises(ValueError, match="Unsupported format"):
        parse_formats("html,jpg")


def test_visualizer_exports_expected_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    wav = tmp_path / "sample.wav"
    _write_wav(wav)

    def fake_write_html(self: go.Figure, path: str | Path) -> None:
        del self
        Path(path).write_text("<html></html>", encoding="utf-8")

    def fake_write_image(self: go.Figure, path: str | Path) -> None:
        del self
        Path(path).write_bytes(b"image")

    monkeypatch.setattr(go.Figure, "write_html", fake_write_html)
    monkeypatch.setattr(go.Figure, "write_image", fake_write_image)

    config = AttentionCliConfig(
        checkpoint=checkpoint,
        wav=wav,
        out_dir=tmp_path / "attention",
        attention_method="last_cls_patch_head_mean",
        visualization="both",
        target_class="predicted",
        head="0",
        top_k=2,
        formats={"html", "png"},
        device_override="cpu",
        install_chrome=False,
    )

    metadata = AstAttentionVisualizer(config).run()

    assert (tmp_path / "attention" / "sample_fbank.html").exists()
    assert (tmp_path / "attention" / "sample_attention_heatmap_overlay.png").exists()
    assert (tmp_path / "attention" / "sample_attention_rectangle_overlay.html").exists()
    assert (tmp_path / "attention" / "sample_top_patches.csv").exists()
    assert metadata["attention_method"] == "last_cls_patch_head_mean"
    assert set(metadata["outputs"]) == {
        "fbank",
        "heatmap_overlay",
        "rectangle_overlay",
    }


def test_missing_checkpoint_metadata_raises_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"model_cfg": asdict(_model_cfg()), "model_state_dict": {}}, path)

    with pytest.raises(RuntimeError, match="run_config"):
        load_attention_checkpoint(path, "cpu")
