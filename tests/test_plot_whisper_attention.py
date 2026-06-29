from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
import soundfile as sf
import torch
from src.cli.plot_whisper_attention import (
    AttentionCliConfig,
    AttentionScoreExtractor,
    WhisperAttentionInputLoader,
    WhisperAttentionModelRunner,
    WhisperAttentionVisualizer,
    WhisperTimeTokenMapper,
    load_attention_checkpoint,
    parse_args,
    parse_formats,
)
from src.models.model import EncoderAdaptationConfig
from src.models.whisper_encoder import WhisperEncoderDims
from src.models.whisper_model import RespiratoryWhisperModel, WhisperModelConfig
from src.utils.config import (
    AudioConfig,
    BandPassConfig,
    DataConfig,
    ExperimentConfig,
    LogMelConfig,
    PreprocessingConfig,
)


def _write_wav(path: Path, *, sample_rate: int = 16000) -> None:
    t = np.linspace(0.0, 0.2, int(sample_rate * 0.2), endpoint=False)
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
        audio=AudioConfig(sample_rate=16000, clip_duration_sec=0.2),
        preprocessing=PreprocessingConfig(
            feature_type="log_mel",
            source_type="original",
            bandpass=BandPassConfig(enabled=False),
            log_mel=LogMelConfig(
                n_fft=400,
                hop_length=160,
                win_length=400,
                n_mels=16,
            ),
        ),
    )


def _model_cfg(num_classes: int = 3) -> WhisperModelConfig:
    return WhisperModelConfig(
        encoder=WhisperEncoderDims(
            n_mels=16,
            n_audio_ctx=10,
            n_audio_state=32,
            n_audio_head=4,
            n_audio_layer=2,
        ),
        num_classes=num_classes,
        adaptation=EncoderAdaptationConfig(mode="frozen", num_layers=1),
        head_type="hf",
        pooling="mean",
        use_weighted_layer_sum=True,
        classifier_proj_size=16,
    )


def _checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(11)
    data_cfg = _data_cfg(tmp_path)
    model = RespiratoryWhisperModel(_model_cfg())
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
) -> tuple[Path, Path, WhisperAttentionModelRunner, torch.Tensor]:
    checkpoint = _checkpoint(tmp_path)
    wav = tmp_path / "sample.wav"
    _write_wav(wav)
    loaded = load_attention_checkpoint(checkpoint, "cpu")
    inputs = WhisperAttentionInputLoader(loaded.data_cfg).load(wav)
    runner = WhisperAttentionModelRunner(loaded)
    return checkpoint, wav, runner, inputs.input_values


def test_checkpoint_loading_and_log_mel_shape(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    wav = tmp_path / "sample.wav"
    _write_wav(wav)

    loaded = load_attention_checkpoint(checkpoint, "cpu")
    inputs = WhisperAttentionInputLoader(loaded.data_cfg).load(wav)

    assert loaded.label_to_index == {"normal": 0, "wheeze": 1, "crackle": 2}
    assert inputs.log_mel.shape == (16, 20)
    assert inputs.input_values.shape == (1, 16, 20)


def test_runner_returns_whisper_attention_shape(tmp_path: Path) -> None:
    _, _, runner, input_values = _loaded_inputs_runner(tmp_path)

    result = runner.run(
        input_values,
        attention_method="last_time_attention_head_mean",
        target_class="predicted",
    )

    assert len(result.attentions) == 2
    assert result.attentions[-1].shape == (1, 4, 10, 10)


def test_time_token_mapping_uses_log_mel_frame_contract(tmp_path: Path) -> None:
    _, _, runner, _ = _loaded_inputs_runner(tmp_path)
    mapper = WhisperTimeTokenMapper.from_runner(runner)

    record = mapper.token_record(
        rank=1,
        token_index=3,
        attention_score=0.5,
        normalized_attention_score=1.0,
    )

    assert mapper.geometry.num_frames == 20
    assert mapper.geometry.n_audio_ctx == 10
    assert record.frame_start == 6
    assert record.frame_end == 8
    assert record.time_start_sec == pytest.approx(0.06)
    assert mapper.upsample_scores(np.ones(10)).shape == (16, 20)


def test_attention_score_methods_return_finite_time_scores(tmp_path: Path) -> None:
    _, _, runner, input_values = _loaded_inputs_runner(tmp_path)
    mapper = WhisperTimeTokenMapper.from_runner(runner)
    extractor = AttentionScoreExtractor(mapper)

    run_result = runner.run(
        input_values,
        attention_method="last_time_attention_head_mean",
        target_class="predicted",
    )
    for method, head in [
        ("last_time_attention", "0"),
        ("last_time_attention_head_mean", "0"),
        ("time_attention_rollout", "0"),
    ]:
        scores = extractor.extract(run_result, method=method, head=head, top_k=2)
        assert scores.raw_scores.shape == (10,)
        assert np.isfinite(scores.raw_scores).all()
        assert scores.normalized_scores.min() >= 0.0
        assert scores.normalized_scores.max() <= 1.0
        assert len(scores.top_time_spans) == 2

    gradient_result = runner.run(
        input_values,
        attention_method="class_gradient_time_attention",
        target_class="wheeze",
    )
    gradient_scores = extractor.extract(
        gradient_result,
        method="class_gradient_time_attention",
        head="0",
        top_k=3,
    )
    assert gradient_scores.raw_scores.shape == (10,)
    assert np.isfinite(gradient_scores.raw_scores).all()


def test_invalid_head_format_and_default_method_raise_or_parse(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    wav = tmp_path / "sample.wav"
    _write_wav(wav)
    _, _, runner, input_values = _loaded_inputs_runner(tmp_path)
    mapper = WhisperTimeTokenMapper.from_runner(runner)
    extractor = AttentionScoreExtractor(mapper)
    run_result = runner.run(
        input_values,
        attention_method="last_time_attention",
        target_class="predicted",
    )

    with pytest.raises(ValueError, match="out of range"):
        extractor.extract(run_result, method="last_time_attention", head="99", top_k=2)

    with pytest.raises(ValueError, match="Unsupported format"):
        parse_formats("html,jpg")

    config = parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--audio",
            str(wav),
            "--output-dir",
            str(tmp_path / "attention"),
        ]
    )
    assert config.attention_method == "last_time_attention_head_mean"
    assert config.formats == {"html", "json"}


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

    def fake_write_json(self: go.Figure, path: str | Path) -> None:
        del self
        Path(path).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(go.Figure, "write_html", fake_write_html)
    monkeypatch.setattr(go.Figure, "write_json", fake_write_json)

    config = AttentionCliConfig(
        checkpoint=checkpoint,
        wav=wav,
        out_dir=tmp_path / "attention",
        attention_method="last_time_attention_head_mean",
        target_class="predicted",
        head="0",
        top_k=2,
        formats={"html", "json"},
        device_override="cpu",
        install_chrome=False,
    )

    metadata = WhisperAttentionVisualizer(config).run()

    assert (tmp_path / "attention" / "sample_log_mel.html").exists()
    assert (tmp_path / "attention" / "sample_attention_time_curve.json").exists()
    assert (tmp_path / "attention" / "sample_attention_temporal_overlay.html").exists()
    assert (tmp_path / "attention" / "sample_top_time_spans_overlay.json").exists()
    assert (tmp_path / "attention" / "sample_top_time_spans.csv").exists()
    assert metadata["attention_method"] == "last_time_attention_head_mean"
    assert metadata["time_token_geometry"]["n_audio_ctx"] == 10
    assert set(metadata["outputs"]) == {
        "log_mel",
        "attention_time_curve",
        "temporal_overlay",
        "top_time_spans_overlay",
    }


def test_missing_checkpoint_metadata_raises_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"model_cfg": asdict(_model_cfg()), "model_state_dict": {}}, path)

    with pytest.raises(RuntimeError, match="run_config"):
        load_attention_checkpoint(path, "cpu")
