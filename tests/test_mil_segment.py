from __future__ import annotations

import torch
from src.data.segment import BagSegmenter
from src.utils.config import AudioConfig, SegmentationConfig


def test_sliding_window_segmenter_keeps_padded_tail() -> None:
    segmenter = BagSegmenter(
        audio=AudioConfig(sample_rate=4, clip_duration_sec=5.25),
        segment=SegmentationConfig(
            mode="sliding_window",
            length_sec=2.0,
            stride_sec=1.0,
            pad_last=True,
            drop_last=False,
        ),
    )
    audio = torch.arange(21, dtype=torch.float32)

    segments, metadata = segmenter(audio)

    assert len(segments) == 5
    assert [item.start_sample for item in metadata] == [0, 4, 8, 12, 16]
    assert metadata[-1].padded is True
    assert metadata[-1].valid_end_sample == 21


def test_non_overlap_segmenter_drops_short_tail() -> None:
    segmenter = BagSegmenter(
        audio=AudioConfig(sample_rate=4, clip_duration_sec=5.25),
        segment=SegmentationConfig(
            mode="non_overlap",
            length_sec=2.0,
            stride_sec=None,
            pad_last=False,
            drop_last=True,
        ),
    )
    audio = torch.arange(21, dtype=torch.float32)

    segments, metadata = segmenter(audio)

    assert len(segments) == 2
    assert [item.start_sample for item in metadata] == [0, 8]
    assert all(item.padded is False for item in metadata)


def test_full_clip_segmenter_returns_single_instance() -> None:
    segmenter = BagSegmenter(
        audio=AudioConfig(sample_rate=4, clip_duration_sec=5.0),
        segment=SegmentationConfig(
            mode="full_clip",
            length_sec=None,
            stride_sec=None,
            pad_last=True,
            drop_last=False,
        ),
    )
    audio = torch.arange(6, dtype=torch.float32)

    segments, metadata = segmenter(audio)

    assert len(segments) == 1
    assert metadata[0].start_sample == 0
    assert metadata[0].valid_end_sample == 6
