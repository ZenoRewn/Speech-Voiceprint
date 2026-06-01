"""VAD (voice activity detection) pre-filter.

Why this exists: Azure Fast Transcription bills per audio second, and a 30
minute meeting with a 5-minute pre-roll of silence wastes that 5 minutes.
This module returns the speech-only spans so callers can decide whether to
transcribe / embed / skip a frame.

Two backends:
  - **silero** (preferred): `torch.hub.load('snakers4/silero-vad', 'silero_vad')`.
    ~1.5MB model, MIT-licensed, far better recall on whispered / accented
    speech than energy gates. Requires a successful first-time `torch.hub`
    fetch (cached in `~/.cache/torch/hub` thereafter).
  - **rms** (fallback): a simple energy gate. No deps beyond numpy. Less
    precise — it'll flag music as speech and miss whispers — but it's
    fully offline and doesn't need a working torch install.

`extract_speech_windows()` is what the rest of the pipeline calls. It
returns merged `(start, end)` time spans in seconds, with `padding_ms` of
context tacked on either side so word boundaries don't get clipped.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

# Backend cache so we load the silero model exactly once per process.
_BACKEND_CACHE: dict[str, object] = {}
_BACKEND_LOCK = threading.Lock()


def extract_speech_windows(
    samples: np.ndarray,
    sr: int = 16000,
    *,
    backend: Literal["silero", "rms", "auto"] = "auto",
    min_speech_ms: int = 200,
    padding_ms: int = 120,
    rms_threshold: float = 0.01,
) -> list[tuple[float, float]]:
    """Return merged (start, end) seconds spans containing speech.

    Args:
        samples: float32 audio in `[-1, 1]`, 1-D.
        sr: sample rate, default 16000.
        backend: 'silero' (gated by env / availability), 'rms', or 'auto'.
        min_speech_ms: drop spans shorter than this.
        padding_ms: extend each span on both sides; spans that overlap after
            padding are merged.
        rms_threshold: only used by the rms backend. 30ms frame mean RMS
            below this is considered silence.
    """
    if samples.ndim != 1:
        samples = samples.reshape(-1)
    samples = samples.astype(np.float32, copy=False)

    chosen = _select_backend(backend)
    if chosen == "silero":
        try:
            spans = _silero_spans(samples, sr)
        except Exception as e:  # noqa: BLE001 — broad: covers torch.hub network, missing torchaudio, etc.
            log.warning("silero VAD unavailable (%s); falling back to RMS", e)
            spans = _rms_spans(samples, sr, rms_threshold)
    else:
        spans = _rms_spans(samples, sr, rms_threshold)

    return _post_process(spans, sr, len(samples), min_speech_ms, padding_ms)


def _select_backend(requested: str) -> Literal["silero", "rms"]:
    if requested == "rms":
        return "rms"
    if requested == "silero":
        return "silero"
    # auto: prefer silero if torch is importable; env override for offline.
    if os.environ.get("SV_VAD_FORCE_RMS", "").lower() in ("1", "true", "yes"):
        return "rms"
    try:
        import torch  # noqa: F401
        return "silero"
    except Exception:
        return "rms"


# ---------------------------------------------------------------------------
def _silero_spans(samples: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Use snakers4/silero-vad. Returns spans in seconds."""
    import torch

    with _BACKEND_LOCK:
        cached = _BACKEND_CACHE.get("silero")
    if cached is None:
        # `trust_repo=True` skips the interactive prompt that torch.hub shows
        # the first time a repo is loaded — important for non-TTY runs.
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        with _BACKEND_LOCK:
            _BACKEND_CACHE["silero"] = (model, utils)
        cached = (model, utils)
    model, utils = cached  # type: ignore[misc]
    get_speech_timestamps = utils[0]

    if sr != 16000:
        # Silero's checkpoint is 16k or 8k. Resample for anything else with a
        # cheap linear interp; real callers should already be 16k mono.
        from numpy import interp
        new_len = int(len(samples) * 16000 / sr)
        samples = interp(
            np.linspace(0, len(samples), new_len, endpoint=False),
            np.arange(len(samples)),
            samples,
        ).astype(np.float32)
        sr = 16000

    tensor = torch.from_numpy(samples)
    timestamps = get_speech_timestamps(tensor, model, sampling_rate=sr)
    return [(ts["start"] / sr, ts["end"] / sr) for ts in timestamps]


# ---------------------------------------------------------------------------
def _rms_spans(samples: np.ndarray, sr: int, threshold: float) -> list[tuple[float, float]]:
    """Frame-based RMS gate.

    Splits into 30ms frames; consecutive above-threshold frames become a span.
    No smoothing — `_post_process` merges nearby spans via padding.
    """
    if not len(samples):
        return []
    frame_len = max(1, int(sr * 0.03))
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return []
    framed = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(framed ** 2, axis=1))
    voiced = rms > threshold

    spans: list[tuple[float, float]] = []
    in_span = False
    span_start = 0
    for i, v in enumerate(voiced):
        if v and not in_span:
            in_span = True
            span_start = i
        elif not v and in_span:
            in_span = False
            spans.append((span_start * frame_len / sr, i * frame_len / sr))
    if in_span:
        spans.append((span_start * frame_len / sr, n_frames * frame_len / sr))
    return spans


# ---------------------------------------------------------------------------
def _post_process(
    spans: list[tuple[float, float]],
    sr: int,
    total_samples: int,
    min_speech_ms: int,
    padding_ms: int,
) -> list[tuple[float, float]]:
    if not spans:
        return []
    pad = padding_ms / 1000.0
    min_dur = min_speech_ms / 1000.0
    total_dur = total_samples / sr

    padded: list[tuple[float, float]] = []
    for s, e in spans:
        if e - s < min_dur:
            continue
        padded.append((max(0.0, s - pad), min(total_dur, e + pad)))
    if not padded:
        return []

    # Merge overlapping/adjacent spans after padding.
    padded.sort()
    merged: list[tuple[float, float]] = [padded[0]]
    for s, e in padded[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def speech_coverage(spans: list[tuple[float, float]], total_seconds: float) -> float:
    """How much of the audio (0..1) is speech under these spans."""
    if total_seconds <= 0:
        return 0.0
    voiced = sum(e - s for s, e in spans)
    return min(1.0, voiced / total_seconds)
