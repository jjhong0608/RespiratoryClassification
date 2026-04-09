from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.models.whisper_encoder import WhisperEncoderDims
from src.utils.config import SegmentEncoderConfig
from src.utils.logging import LoggingMixin

_OPENAI_WHISPER_MODELS: dict[str, str] = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large-v3-turbo": "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
    "turbo": "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
}


@dataclass(frozen=True)
class LoadedPretrainedInfo:
    resolved_path: str
    source: str
    loaded_keys: int
    missing_keys: list[str]
    unexpected_keys: list[str]


class OpenAIWhisperCheckpointLoader(LoggingMixin):
    def available_models(self) -> list[str]:
        return sorted(_OPENAI_WHISPER_MODELS.keys())

    def _default_download_root(self) -> Path:
        default = Path.home() / ".cache" / "whisper"
        xdg = os.getenv("XDG_CACHE_HOME")
        if xdg:
            return Path(xdg) / "whisper"
        return default

    def _download(self, url: str, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        expected_sha256 = url.split("/")[-2]
        target = root / Path(url).name

        if target.exists():
            data = target.read_bytes()
            if hashlib.sha256(data).hexdigest() == expected_sha256:
                return target
            self.logger.info("Checksum mismatch; re-downloading.")

        with urllib.request.urlopen(url) as source, target.open("wb") as output:
            size = int(source.info().get("Content-Length", "0"))
            with tqdm(total=size, ncols=80, unit="iB", unit_scale=True) as bar:
                while True:
                    buf = source.read(8192)
                    if not buf:
                        break
                    output.write(buf)
                    bar.update(len(buf))

        data = target.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise RuntimeError(
                "Downloaded model checksum mismatch; retry download/load."
            )
        return target

    def resolve_checkpoint_path(
        self, name_or_path: str, download_root: str | None
    ) -> Path:
        local_path = Path(name_or_path)
        if local_path.exists():
            return local_path
        if name_or_path in _OPENAI_WHISPER_MODELS:
            root = (
                Path(download_root) if download_root else self._default_download_root()
            )
            return self._download(_OPENAI_WHISPER_MODELS[name_or_path], root)
        raise ValueError(
            f"Unknown pretrained '{name_or_path}'. Use a local .pt file or one of: {self.available_models()}"
        )

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"map_location": "cpu"}
        if "weights_only" in torch.load.__code__.co_varnames:
            kwargs["weights_only"] = True
        checkpoint = torch.load(path, **kwargs)
        if not isinstance(checkpoint, dict):
            raise TypeError("Expected checkpoint dict")
        if "dims" not in checkpoint or "model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint must contain `dims` and `model_state_dict`")
        return checkpoint

    def inspect_encoder_dims(
        self,
        name_or_path: str,
        *,
        download_root: str | None = None,
    ) -> WhisperEncoderDims:
        checkpoint_path = self.resolve_checkpoint_path(name_or_path, download_root)
        checkpoint = self.load_checkpoint(checkpoint_path)
        raw_dims = checkpoint["dims"]
        if not isinstance(raw_dims, dict):
            raise TypeError("checkpoint['dims'] must be a dict")
        return WhisperEncoderDims(
            n_mels=int(raw_dims["n_mels"]),
            n_audio_ctx=int(raw_dims["n_audio_ctx"]),
            n_audio_state=int(raw_dims["n_audio_state"]),
            n_audio_head=int(raw_dims["n_audio_head"]),
            n_audio_layer=int(raw_dims["n_audio_layer"]),
        )

    def _validate_encoder_dims(
        self,
        checkpoint_dims: dict[str, Any],
        target_dims: WhisperEncoderDims,
    ) -> None:
        got = WhisperEncoderDims(
            n_mels=int(checkpoint_dims["n_mels"]),
            n_audio_ctx=int(checkpoint_dims["n_audio_ctx"]),
            n_audio_state=int(checkpoint_dims["n_audio_state"]),
            n_audio_head=int(checkpoint_dims["n_audio_head"]),
            n_audio_layer=int(checkpoint_dims["n_audio_layer"]),
        )
        # MIL segments can use shorter contexts than the original 30s Whisper checkpoints.
        if (
            got.n_mels != target_dims.n_mels
            or got.n_audio_state != target_dims.n_audio_state
            or got.n_audio_head != target_dims.n_audio_head
            or got.n_audio_layer != target_dims.n_audio_layer
        ):
            raise ValueError(
                "Encoder dims mismatch.\n"
                f"- checkpoint: {got}\n"
                f"- config:      {target_dims}\n\n"
                "For MIL segmentation, `n_audio_ctx` may differ and is derived from the segment length, "
                "but the remaining Whisper encoder dims must match."
            )

    def _extract_encoder_state(self, model_state: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in model_state.items():
            if not isinstance(key, str):
                continue
            stripped: str | None = None
            if key.startswith("encoder."):
                stripped = key.removeprefix("encoder.")
            elif key.startswith("segment_encoder.encoder."):
                stripped = key.removeprefix("segment_encoder.encoder.")
            if stripped is None or stripped == "positional_embedding":
                continue
            out[stripped] = value
        if not out:
            raise KeyError("No encoder.* keys found in checkpoint model_state_dict")
        return out

    def load_encoder_into(
        self,
        model: Any,
        cfg: SegmentEncoderConfig,
        *,
        target_dims: WhisperEncoderDims,
    ) -> LoadedPretrainedInfo:
        if cfg.pretrained_name_or_path is None:
            raise ValueError(
                "model.segment_encoder.pretrained_name_or_path must be set"
            )
        if not hasattr(model, "encoder"):
            raise TypeError("model must expose an encoder attribute")

        checkpoint_path = self.resolve_checkpoint_path(
            cfg.pretrained_name_or_path,
            cfg.download_root,
        )
        checkpoint = self.load_checkpoint(checkpoint_path)
        dims = checkpoint["dims"]
        if not isinstance(dims, dict):
            raise TypeError("checkpoint['dims'] must be a dict")
        self._validate_encoder_dims(dims, target_dims)

        model_state = checkpoint["model_state_dict"]
        if not isinstance(model_state, dict):
            raise TypeError("checkpoint['model_state_dict'] must be a dict")
        encoder_state = self._extract_encoder_state(model_state)
        incompatible = model.encoder.load_state_dict(encoder_state, strict=False)

        allowed_missing = {"positional_embedding"}
        missing = [
            item for item in incompatible.missing_keys if item not in allowed_missing
        ]
        unexpected = list(incompatible.unexpected_keys)
        if cfg.strict and (missing or unexpected):
            raise RuntimeError(
                "Strict encoder load failed.\n"
                f"missing={missing}\n"
                f"unexpected={unexpected}"
            )

        source = (
            "local_path"
            if Path(cfg.pretrained_name_or_path).exists()
            else "openai_registry"
        )
        return LoadedPretrainedInfo(
            resolved_path=str(checkpoint_path),
            source=source,
            loaded_keys=len(encoder_state),
            missing_keys=list(incompatible.missing_keys),
            unexpected_keys=unexpected,
        )
