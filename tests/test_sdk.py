"""Tests for the public SDK entry point `pipeline.transcribe_file`.

These tests avoid touching real Azure / pyannote / SpeechBrain — they
monkey-patch `pipeline.orchestrator.AzureFastTranscription` and
`build_voiceprint_provider` to return stubs. The point is to lock the
SDK shape and the round-trip contract with `PipelineResult`, not to
exercise the providers.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from stt.base import STTResult, Utterance, Word
from voiceprint.base import SpeakerSegment


class _StubFast:
    def __init__(self, *_a, **_kw) -> None:
        pass

    def transcribe_fast(self, audio_path: str, languages=None) -> STTResult:
        utt = Utterance(
            text="hello world",
            start=0.0,
            end=1.0,
            words=[Word("hello", 0.0, 0.5), Word("world", 0.5, 1.0)],
            azure_speaker="Guest-1",
        )
        return STTResult(utterances=[utt], language="en-US", duration=1.0)


class _StubVoiceprint:
    name = "stub"
    embedding_dim = 4

    def diarize(self, audio_path: str, num_speakers=None) -> list[SpeakerSegment]:
        return [
            SpeakerSegment(0.0, 0.6, "Speaker_A"),
            SpeakerSegment(0.6, 1.0, "Speaker_B"),
        ]

    def embed(self, audio_path: str, start: float, end: float) -> np.ndarray:
        return np.zeros(self.embedding_dim, dtype=np.float32)


@pytest.fixture
def stub_pipeline(monkeypatch, tmp_path):
    """Wire the orchestrator to use stubs, plus give us a real-but-empty audio path."""
    from pipeline import orchestrator

    monkeypatch.setattr(orchestrator, "AzureFastTranscription", _StubFast)
    monkeypatch.setattr(orchestrator, "build_voiceprint_provider", lambda *a, **kw: _StubVoiceprint())

    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"")
    return audio


def test_transcribe_file_returns_pydantic_result(stub_pipeline):
    from pipeline import PipelineResult, transcribe_file

    result = transcribe_file(stub_pipeline, mode="fast", backend="speechbrain")
    assert isinstance(result, PipelineResult)
    assert result.language == "en-US"
    assert result.voiceprint_backend == "stub"
    assert len(result.utterances) == 1
    assert result.utterances[0].words[0].text == "hello"


def test_round_trip_ingests_cli_payload(stub_pipeline):
    """The CLI's `_run_pipeline` dict must validate cleanly into PipelineResult.

    We don't require byte-equality back — the CLI emits `confidence: None`
    in word dicts which `model_dump_json(exclude_none=True)` legitimately
    drops. What matters is that a CLI consumer can swap to the SDK without
    re-parsing.
    """
    from pipeline import PipelineResult, orchestrator, transcribe_file

    direct = orchestrator.run_fast(
        str(stub_pipeline),
        None,
        _StubVoiceprint(),
        None,
        0.7,
    )
    parsed = PipelineResult.from_payload(direct)
    via_sdk = transcribe_file(stub_pipeline, mode="fast", backend="speechbrain")

    assert parsed.model_dump(exclude_none=True) == via_sdk.model_dump(exclude_none=True)
    # And every utterance's text + speaker matches the dict source exactly.
    for d_utt, p_utt in zip(direct["utterances"], parsed.utterances):
        assert d_utt["text"] == p_utt.text
        assert d_utt["speaker"] == p_utt.speaker


def test_audio_label_does_not_expose_paths_or_sas():
    from pipeline.orchestrator import _public_audio_label

    assert _public_audio_label("/Users/me/private/customer.wav") == "customer.wav"
    assert (
        _public_audio_label(
            "https://acct.blob.core.windows.net/in/audio.wav?sig=secret",
            is_url=True,
        )
        == "https://acct.blob.core.windows.net/in/audio.wav"
    )


def test_invalid_mode_raises():
    from pipeline import transcribe_file

    with pytest.raises(ValueError, match="unsupported mode"):
        transcribe_file("anywhere.wav", mode="realtime")  # type: ignore[arg-type]


def test_batch_mode_requires_audio_url(stub_pipeline):
    from pipeline import transcribe_file

    with pytest.raises(ValueError, match="audio_url"):
        transcribe_file(stub_pipeline, mode="batch")
