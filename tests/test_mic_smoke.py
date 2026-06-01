"""Smoke tests for `MicrophoneSource` without touching real audio hardware.

PortAudio doesn't exist on most CI runners and physical mics are even less
practical, so we monkey-patch the `sounddevice` module with a fake that
behaves enough like the real one to drive `MicrophoneSource.chunks()` end-to-end.

Two scenarios are pinned:
  1. Native 16 kHz → bytes-out should match the chunk_ms exactly (no resample).
  2. Native 48 kHz → bytes-out should still be 16 kHz mono int16 (resampled).
"""

from __future__ import annotations

import sys
import threading
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
class _FakeStream:
    """Stand-in for `sounddevice.InputStream`.

    `__enter__` returns self; `read(n)` hands back `n` zero-int16 samples.
    The fake serves forever — the test stops it from outside via
    `MicrophoneSource.stop()` so the iterator can exit cleanly.
    """

    def __init__(self, samplerate: int, channels: int, dtype: str, blocksize: int, device):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize
        self.device = device
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._closed = True

    def read(self, n: int):
        # int16 zero-array shaped like the real API would return: (n, channels).
        data = np.zeros((n, self.channels), dtype=np.int16)
        return data, False  # (data, overflowed)


def _install_fake_sounddevice(monkeypatch, *, default_samplerate: int):
    """Replace `sounddevice` in sys.modules with a fake that advertises
    the given default rate and accepts `check_input_settings` unconditionally."""
    fake = types.SimpleNamespace(
        query_devices=lambda *a, **k: {"default_samplerate": float(default_samplerate)},
        check_input_settings=lambda **kw: None,
        InputStream=_FakeStream,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    return fake


def _collect_bytes(source, target_chunks: int = 5) -> list[bytes]:
    """Pull `target_chunks` PCM chunks from a MicrophoneSource then stop it.

    We run the iterator on a background thread because `chunks()` blocks in
    `_FakeStream.read` until cancelled.
    """
    out: list[bytes] = []
    err: list[BaseException] = []

    def runner():
        try:
            it = source.chunks()
            for _ in range(target_chunks):
                out.append(next(it))
            source.stop()
            for chunk in it:  # drain so the generator returns cleanly
                out.append(chunk)
        except StopIteration:
            pass
        except Exception as e:  # noqa: BLE001
            err.append(e)

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout=2.0)
    if err:
        raise err[0]
    return out


# ---------------------------------------------------------------------------
def test_microphone_source_16k_native_no_resample(monkeypatch):
    """A 16 kHz device produces bytes that match `chunk_ms` exactly.

    `chunk_ms=100` at 16 kHz, mono int16 → 100ms × 16 samples/ms × 2 bytes = 3200 bytes/chunk.
    """
    _install_fake_sounddevice(monkeypatch, default_samplerate=16000)
    from pipeline.streaming import MicrophoneSource, PCM_SR

    src = MicrophoneSource(chunk_ms=100)
    assert src.device_samplerate == PCM_SR  # picked the native rate

    chunks = _collect_bytes(src, target_chunks=5)
    assert len(chunks) >= 5
    expected = int(PCM_SR * 0.1) * 2  # int16
    for c in chunks[:5]:
        assert len(c) == expected, f"native 16k chunk wrong size: {len(c)} vs {expected}"


def test_microphone_source_48k_resamples_to_16k(monkeypatch):
    """A 48 kHz device must resample down to 16 kHz before yielding bytes.

    chunk_ms=100 at 48 kHz native → 4800 native samples → 1600 output samples → 3200 bytes.
    Tolerance ±2 samples for rounding in the linear interpolator.
    """
    _install_fake_sounddevice(monkeypatch, default_samplerate=48000)
    from pipeline.streaming import MicrophoneSource, PCM_SR

    src = MicrophoneSource(chunk_ms=100)
    assert src.device_samplerate == 48000

    chunks = _collect_bytes(src, target_chunks=3)
    assert len(chunks) >= 3
    # Linear interpolation can be ±1 sample off due to ratio rounding.
    for c in chunks[:3]:
        n_samples = len(c) // 2
        assert abs(n_samples - PCM_SR // 10) <= 2, (
            f"resampled chunk wrong size: {n_samples} samples vs {PCM_SR // 10} expected"
        )


def test_microphone_source_falls_back_when_query_fails(monkeypatch, capsys):
    """If `query_devices` raises, the source must fall back to the first FALLBACK_RATES entry."""
    fake = types.SimpleNamespace(
        query_devices=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("device unavailable")),
        check_input_settings=lambda **kw: None,
        InputStream=_FakeStream,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    from pipeline.streaming import MicrophoneSource

    src = MicrophoneSource(chunk_ms=100)
    assert src.device_samplerate == MicrophoneSource.FALLBACK_RATES[0]
    captured = capsys.readouterr()
    assert "query_devices failed" in captured.err


def test_microphone_source_raises_when_no_rate_works(monkeypatch):
    """`check_input_settings` rejects every candidate → RuntimeError.

    This is the real-world Bluetooth/AirPods bug where the OS lists a device
    but it refuses every sample rate at open-time.
    """
    fake = types.SimpleNamespace(
        query_devices=lambda *a, **k: {"default_samplerate": 16000.0},
        check_input_settings=lambda **kw: (_ for _ in ()).throw(RuntimeError("rate refused")),
        InputStream=_FakeStream,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    from pipeline.streaming import MicrophoneSource

    with pytest.raises(RuntimeError, match="no usable samplerate"):
        MicrophoneSource(chunk_ms=100)
