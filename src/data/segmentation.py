from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentBounds:
    index: int
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float


def compute_sliding_window_segments(
    *,
    num_samples: int,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
    tail_policy: str = "cover_end",
) -> list[SegmentBounds]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")
    if window_sec <= 0:
        raise ValueError("window_sec must be greater than zero")
    if hop_sec <= 0:
        raise ValueError("hop_sec must be greater than zero")
    if tail_policy != "cover_end":
        raise ValueError("Only tail_policy='cover_end' is supported")

    window_samples = max(1, int(round(window_sec * sample_rate)))
    hop_samples = max(1, int(round(hop_sec * sample_rate)))
    effective_num_samples = max(0, int(num_samples))

    if effective_num_samples <= window_samples:
        starts = [0]
    else:
        max_regular_start = effective_num_samples - window_samples
        starts = list(range(0, max_regular_start + 1, hop_samples))
        final_start = max(0, effective_num_samples - window_samples)
        if not starts or starts[-1] != final_start:
            starts.append(final_start)

    starts = sorted(dict.fromkeys(starts))
    segments: list[SegmentBounds] = []
    for index, start in enumerate(starts):
        end = min(start + window_samples, effective_num_samples)
        segments.append(
            SegmentBounds(
                index=index,
                start_sample=int(start),
                end_sample=int(end),
                start_sec=float(start) / float(sample_rate),
                end_sec=float(end) / float(sample_rate),
            )
        )
    return segments
