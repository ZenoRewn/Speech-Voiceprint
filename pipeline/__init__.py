"""Public Python SDK surface for speech-voiceprint.

Why this exists: an external user shouldn't have to read `orchestrator.py`,
import three modules, instantiate providers, and remember which keyword
arguments are required. `transcribe_file()` is the one-shot entry point —
it returns a typed `PipelineResult` that round-trips with the CLI's JSON
output.

Realtime / streaming has different ergonomics (you push frames, you receive
events) so it stays in `pipeline.streaming`. The SDK intentionally only
exposes the offline modes (fast, batch).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Literal, Optional, Union

from schemas import PipelineResult


def transcribe_file(
    audio: Union[str, Path],
    *,
    mode: Literal["fast", "batch"] = "fast",
    backend: Literal["pyannote", "speechbrain"] = "speechbrain",
    languages: Optional[Iterable[str]] = None,
    audio_url: Optional[str] = None,
    num_speakers: Optional[int] = None,
    majority_threshold: float = 0.7,
    device: str = "auto",
    hf_token: Optional[str] = None,
    registry_path: Optional[Union[str, Path]] = None,
    match_threshold: float = 0.40,
    unknown_threshold: float = 0.60,
    auto_enroll_unknown: bool = True,
    max_per_speaker: int = 5,
    dual_enroll: bool = False,
) -> PipelineResult:
    """Run the full pipeline (STT + voiceprint + alignment + optional registry).

    Parameters mirror the orchestrator CLI flags:
      - `mode='fast'`: Azure Fast Transcription on the local file. `audio` required.
      - `mode='batch'`: Azure Batch v3.2 against `audio_url`. `audio` is still
        required because voiceprint diarization runs on local PCM, not the URL.

    Returns:
      `PipelineResult` (Pydantic). Use `.model_dump_json(exclude_none=True)`
      to get the same JSON shape the CLI writes.
    """
    # Lazy imports keep `pipeline` importable in environments without numpy
    # (e.g. inspecting the schema only).
    from pipeline.orchestrator import build_voiceprint_provider, run_batch, run_fast

    audio_str = str(audio) if audio is not None else None
    languages_list = list(languages) if languages is not None else None
    registry_str = str(registry_path) if registry_path is not None else None
    hf = hf_token or os.environ.get("HF_TOKEN")

    voiceprint = build_voiceprint_provider(backend, hf, device)

    # Optional dual-enroll: build the *other* backend so post-pipeline can write
    # its embeddings under the same speaker_ids the primary resolved. Failures
    # here (e.g. missing HF_TOKEN for pyannote) degrade gracefully.
    secondary = None
    if dual_enroll and registry_str:
        other = "pyannote" if backend == "speechbrain" else "speechbrain"
        try:
            secondary = build_voiceprint_provider(other, hf, device)
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "dual_enroll: secondary backend %s unavailable, skipping", other
            )
            secondary = None

    if mode == "fast":
        if not audio_str:
            raise ValueError("`audio` is required in fast mode")
        payload = run_fast(
            audio_str,
            languages_list,
            voiceprint,
            num_speakers,
            majority_threshold,
            registry_path=registry_str,
            match_threshold=match_threshold,
            unknown_threshold=unknown_threshold,
            auto_enroll_unknown=auto_enroll_unknown,
            max_per_speaker=max_per_speaker,
            secondary_voiceprint=secondary,
        )
    elif mode == "batch":
        if not audio_url:
            raise ValueError("`audio_url` (SAS URL) is required in batch mode")
        if not audio_str:
            raise ValueError("`audio` (local PCM) is required for voiceprint extraction in batch mode")
        payload = run_batch(
            audio_url,
            languages_list,
            voiceprint,
            num_speakers,
            majority_threshold,
            local_audio_for_voiceprint=audio_str,
            registry_path=registry_str,
            match_threshold=match_threshold,
            unknown_threshold=unknown_threshold,
            auto_enroll_unknown=auto_enroll_unknown,
            max_per_speaker=max_per_speaker,
            secondary_voiceprint=secondary,
        )
    else:
        raise ValueError(f"unsupported mode: {mode!r} (use 'fast' or 'batch')")

    return PipelineResult.from_payload(payload)


def get_registry(path: Union[str, Path]):
    """Open or create a registry store.

    Accepts a bare filesystem path (defaults to SQLite) or a URI like
    `sqlite:///abs/path.db`. See `registry.open_store` for the URI scheme.
    """
    from registry import open_store

    return open_store(path)


__all__ = ["transcribe_file", "get_registry", "PipelineResult"]
