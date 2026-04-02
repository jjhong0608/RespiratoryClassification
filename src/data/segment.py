from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from src.utils.config import AudioConfig, SegmentationConfig


@dataclass(frozen=True)
class SegmentMetadata:
    index: int
    start_sample: int
    end_sample: int
    valid_end_sample: int
    start_sec: float
    end_sec: float
    padded: bool


class BagSegmenter:
    def __init__(self, *, audio: AudioConfig, segment: SegmentationConfig):
        self.audio = audio
        self.segment = segment

    @property
    def segment_length_samples(self) -> int:
        return int(
            round(
                self.segment.effective_length_sec(self.audio.clip_duration_sec)
                * self.audio.sample_rate
            )
        )

    @property
    def stride_samples(self) -> int:
        return int(
            round(
                self.segment.effective_stride_sec(self.audio.clip_duration_sec)
                * self.audio.sample_rate
            )
        )

    def __call__(self, audio: Tensor) -> tuple[list[Tensor], list[SegmentMetadata]]:
        if audio.ndim != 1:
            raise ValueError(f"Expected mono waveform (T,), got {tuple(audio.shape)}")
        if audio.numel() == 0:
            raise ValueError("Cannot segment an empty waveform")
        if self.segment.mode == "full_clip":
            return self._full_clip(audio)
        return self._windowed(audio)

    def _full_clip(self, audio: Tensor) -> tuple[list[Tensor], list[SegmentMetadata]]:
        clip_samples = int(round(self.audio.clip_duration_sec * self.audio.sample_rate))
        valid_end = min(audio.numel(), clip_samples)
        metadata = SegmentMetadata(
            index=0,
            start_sample=0,
            end_sample=clip_samples,
            valid_end_sample=valid_end,
            start_sec=0.0,
            end_sec=clip_samples / self.audio.sample_rate,
            padded=valid_end < clip_samples,
        )
        return [audio[:valid_end]], [metadata]

    def _windowed(self, audio: Tensor) -> tuple[list[Tensor], list[SegmentMetadata]]:
        length = self.segment_length_samples
        stride = self.stride_samples
        total = audio.numel()
        segments: list[Tensor] = []
        metadata: list[SegmentMetadata] = []

        start = 0
        index = 0
        while start + length <= total:
            end = start + length
            segments.append(audio[start:end])
            metadata.append(
                SegmentMetadata(
                    index=index,
                    start_sample=start,
                    end_sample=end,
                    valid_end_sample=end,
                    start_sec=start / self.audio.sample_rate,
                    end_sec=end / self.audio.sample_rate,
                    padded=False,
                )
            )
            start += stride
            index += 1

        if start < total and self.segment.pad_last:
            segments.append(audio[start:total])
            metadata.append(
                SegmentMetadata(
                    index=index,
                    start_sample=start,
                    end_sample=start + length,
                    valid_end_sample=total,
                    start_sec=start / self.audio.sample_rate,
                    end_sec=(start + length) / self.audio.sample_rate,
                    padded=True,
                )
            )

        # Keep one short segment when the bag is shorter than the configured window.
        if not segments:
            segments.append(audio)
            metadata.append(
                SegmentMetadata(
                    index=0,
                    start_sample=0,
                    end_sample=length,
                    valid_end_sample=total,
                    start_sec=0.0,
                    end_sec=length / self.audio.sample_rate,
                    padded=total < length,
                )
            )

        return segments, metadata
