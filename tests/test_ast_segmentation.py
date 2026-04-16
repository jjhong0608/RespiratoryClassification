from __future__ import annotations

from src.data.segmentation import compute_sliding_window_segments


def test_short_recording_yields_single_segment() -> None:
    segments = compute_sliding_window_segments(
        num_samples=8000,
        sample_rate=16000,
        window_sec=2.0,
        hop_sec=1.0,
    )

    assert len(segments) == 1
    assert segments[0].start_sec == 0.0
    assert segments[0].end_sec == 0.5


def test_exact_window_recording_yields_single_segment() -> None:
    segments = compute_sliding_window_segments(
        num_samples=32000,
        sample_rate=16000,
        window_sec=2.0,
        hop_sec=1.0,
    )

    assert len(segments) == 1
    assert segments[0].start_sec == 0.0
    assert segments[0].end_sec == 2.0


def test_overlap_progression_covers_regular_grid() -> None:
    segments = compute_sliding_window_segments(
        num_samples=80000,
        sample_rate=16000,
        window_sec=2.0,
        hop_sec=1.0,
    )

    assert [segment.start_sec for segment in segments] == [0.0, 1.0, 2.0, 3.0]
    assert [segment.end_sec for segment in segments] == [2.0, 3.0, 4.0, 5.0]


def test_tail_alignment_adds_final_end_aligned_window() -> None:
    segments = compute_sliding_window_segments(
        num_samples=int(3.4 * 16000),
        sample_rate=16000,
        window_sec=2.0,
        hop_sec=1.0,
    )

    assert [segment.start_sec for segment in segments] == [0.0, 1.0, 1.4]
    assert [segment.end_sec for segment in segments] == [2.0, 3.0, 3.4]
