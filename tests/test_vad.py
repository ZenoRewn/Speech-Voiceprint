"""Tests for the VAD pre-filter.

We exercise the RMS backend (offline-safe) only. Silero requires torch.hub
network access; CI environments without internet would hang or fail. The
auto/silero codepath is covered by the explicit fall-back-on-error path in
the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from stt.vad_filter import extract_speech_windows, speech_coverage


def _synth(seconds: float, sr: int = 16000, signal_spans: list[tuple[float, float]] | None = None) -> np.ndarray:
    """Create a 16k mono float32 waveform: silence, with optional speech-like
    1kHz tone segments inserted at the given (start, end) ranges."""
    n = int(seconds * sr)
    out = np.zeros(n, dtype=np.float32)
    for s, e in signal_spans or []:
        i0 = int(s * sr)
        i1 = int(e * sr)
        t = np.arange(i1 - i0) / sr
        out[i0:i1] = 0.5 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    return out


def test_rms_picks_up_tone_burst():
    """A clean tone in the middle of silence should produce a single span
    that covers the burst (with padding)."""
    samples = _synth(5.0, signal_spans=[(2.0, 3.0)])
    spans = extract_speech_windows(samples, backend="rms")
    assert len(spans) == 1
    s, e = spans[0]
    # Padding of 120ms on each side, but clipped to [0, total_dur].
    assert 1.7 <= s <= 2.0
    assert 3.0 <= e <= 3.3
    cov = speech_coverage(spans, 5.0)
    assert 0.2 <= cov <= 0.35  # ~1.2s out of 5.0


def test_rms_returns_no_spans_on_pure_silence():
    samples = np.zeros(int(2.0 * 16000), dtype=np.float32)
    spans = extract_speech_windows(samples, backend="rms")
    assert spans == []


def test_rms_merges_adjacent_bursts_via_padding():
    """Two bursts 100ms apart should merge after the default 120ms padding
    extends them into each other."""
    samples = _synth(4.0, signal_spans=[(1.0, 1.5), (1.6, 2.1)])
    spans = extract_speech_windows(samples, backend="rms", padding_ms=120)
    assert len(spans) == 1
    s, e = spans[0]
    assert 0.8 <= s <= 1.0
    assert 2.1 <= e <= 2.3


def test_rms_drops_short_bursts_under_min_speech_ms():
    """A 50ms tick should be filtered out when min_speech_ms=200."""
    samples = _synth(2.0, signal_spans=[(0.5, 0.55)])
    spans = extract_speech_windows(samples, backend="rms", min_speech_ms=200)
    assert spans == []


def test_silero_falls_back_to_rms_on_failure(monkeypatch):
    """If silero throws (e.g. no network for torch.hub), we transparently fall
    back to RMS instead of failing the whole call."""

    def _boom(*_a, **_kw):
        raise RuntimeError("torch.hub down")

    monkeypatch.setattr("stt.vad_filter._silero_spans", _boom)
    samples = _synth(3.0, signal_spans=[(1.0, 2.0)])
    spans = extract_speech_windows(samples, backend="silero")
    assert len(spans) == 1


def test_speech_coverage_returns_zero_on_empty():
    assert speech_coverage([], 10.0) == 0.0
    assert speech_coverage([(0, 1)], 0) == 0.0
