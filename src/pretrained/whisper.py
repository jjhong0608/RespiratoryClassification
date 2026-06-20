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
class LoadedWhisperPretrainedInfo:
    resolved_path: str
    source: str
    loaded_keys: int
    missing_keys: list[str]
    unexpected_keys: list[str]


@dataclass(frozen=True)
class WhisperPretrainedConfig:
    name_or_path: str = "tiny"
    load_encoder_only: bool = True
    strict: bool = True
    freeze_encoder: bool = True
    download_root: str | None = None


class OpenAIWhisperCheckpointLoader(LoggingMixin):
    def available_models(self) -> list[str]:
        return sorted(_OPENAI_WHISPER_MODELS.keys())

    def _default_download_root(self) -> Path:
        xdg = os.getenv("XDG_CACHE_HOME")
        if xdg:
            return Path(xdg) / "whisper"
        return Path.home() / ".cache" / "whisper"

    def _download(self, url: str, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        expected_sha256 = url.split("/")[-2]
        target = root / Path(url).name
        if (
            target.exists()
            and hashlib.sha256(target.read_bytes()).hexdigest() == expected_sha256
        ):
            return target
        with urllib.request.urlopen(url) as source, target.open("wb") as output:
            size = int(source.info().get("Content-Length", "0"))
            with tqdm(total=size, ncols=80, unit="iB", unit_scale=True) as bar:
                while True:
                    buf = source.read(8192)
                    if not buf:
                        break
                    output.write(buf)
                    bar.update(len(buf))
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected_sha256:
            raise RuntimeError("Downloaded Whisper checkpoint checksum mismatch")
        return target

    def resolve_checkpoint_path(
        self,
        name_or_path: str,
        download_root: str | None,
    ) -> Path:
        path = Path(name_or_path)
        if path.exists():
            return path
        if name_or_path in _OPENAI_WHISPER_MODELS:
            root = (
                Path(download_root) if download_root else self._default_download_root()
            )
            return self._download(_OPENAI_WHISPER_MODELS[name_or_path], root)
        raise ValueError(
            f"Unknown Whisper pretrained '{name_or_path}'. "
            f"Use a local .pt file or one of: {self.available_models()}"
        )

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"map_location": "cpu"}
        if "weights_only" in torch.load.__code__.co_varnames:
            kwargs["weights_only"] = True
        checkpoint = torch.load(path, **kwargs)
        if not isinstance(checkpoint, dict):
            raise TypeError("Expected Whisper checkpoint dict")
        if "dims" not in checkpoint or "model_state_dict" not in checkpoint:
            raise KeyError("Whisper checkpoint must contain dims and model_state_dict")
        return checkpoint

    def inspect_encoder_dims(
        self,
        name_or_path: str,
        *,
        download_root: str | None = None,
    ) -> WhisperEncoderDims:
        checkpoint = self.load_checkpoint(
            self.resolve_checkpoint_path(name_or_path, download_root)
        )
        dims = checkpoint["dims"]
        if not isinstance(dims, dict):
            raise TypeError("checkpoint['dims'] must be a dict")
        return self._dims_from_mapping(dims)

    @staticmethod
    def _dims_from_mapping(dims: dict[str, Any]) -> WhisperEncoderDims:
        required = [
            "n_mels",
            "n_audio_ctx",
            "n_audio_state",
            "n_audio_head",
            "n_audio_layer",
        ]
        missing = [key for key in required if key not in dims]
        if missing:
            raise KeyError(f"Whisper checkpoint dims missing keys: {missing}")
        return WhisperEncoderDims(
            n_mels=int(dims["n_mels"]),
            n_audio_ctx=int(dims["n_audio_ctx"]),
            n_audio_state=int(dims["n_audio_state"]),
            n_audio_head=int(dims["n_audio_head"]),
            n_audio_layer=int(dims["n_audio_layer"]),
        )

    @staticmethod
    def _extract_encoder_state(model_state: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in model_state.items():
            if isinstance(key, str) and key.startswith("encoder."):
                out[key.removeprefix("encoder.")] = value
        if not out:
            raise KeyError("No encoder.* keys found in Whisper checkpoint")
        return out

    def load_encoder_into(
        self,
        model: Any,
        cfg: WhisperPretrainedConfig,
    ) -> LoadedWhisperPretrainedInfo:
        if not cfg.load_encoder_only:
            raise ValueError("Only load_encoder_only=true is supported")
        if not hasattr(model, "encoder"):
            raise TypeError("model must expose an encoder attribute")
        target_dims = getattr(getattr(model, "cfg", None), "encoder", None)
        if not isinstance(target_dims, WhisperEncoderDims):
            raise TypeError("model.cfg.encoder must be a WhisperEncoderDims instance")
        checkpoint_path = self.resolve_checkpoint_path(
            cfg.name_or_path, cfg.download_root
        )
        checkpoint = self.load_checkpoint(checkpoint_path)
        dims = checkpoint["dims"]
        if not isinstance(dims, dict):
            raise TypeError("checkpoint['dims'] must be a dict")
        got = self._dims_from_mapping(dims)
        if got != target_dims:
            raise ValueError(
                "Whisper encoder dims mismatch.\n"
                f"- checkpoint: {got}\n"
                f"- config:      {target_dims}"
            )
        model_state = checkpoint["model_state_dict"]
        if not isinstance(model_state, dict):
            raise TypeError("checkpoint['model_state_dict'] must be a dict")
        encoder_state = self._extract_encoder_state(model_state)
        incompatible = model.encoder.load_state_dict(encoder_state, strict=cfg.strict)
        if cfg.freeze_encoder:
            for parameter in model.encoder.parameters():
                parameter.requires_grad = False
        source = "local_path" if Path(cfg.name_or_path).exists() else "openai_registry"
        return LoadedWhisperPretrainedInfo(
            resolved_path=str(checkpoint_path),
            source=source,
            loaded_keys=len(encoder_state),
            missing_keys=list(incompatible.missing_keys),
            unexpected_keys=list(incompatible.unexpected_keys),
        )
