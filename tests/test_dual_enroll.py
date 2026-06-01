"""Dual-enroll: secondary backend writes voiceprints under primary speaker_ids."""

from __future__ import annotations

import numpy as np

from pipeline.orchestrator import _dual_enroll, _majority_owner
from registry import RegistryStore
from voiceprint.base import SpeakerSegment


class FakeSecondaryProvider:
    """Minimal VoiceprintProvider stand-in. Returns pre-canned diarized segments
    so the test doesn't pull torch/pyannote/speechbrain weights."""

    name = "fake"
    embedding_dim = 8

    def __init__(self, segments: list[SpeakerSegment]) -> None:
        self._segments = segments

    def diarize(self, audio_path: str, num_speakers=None) -> list[SpeakerSegment]:
        return list(self._segments)

    def embed(self, audio_path: str, start: float, end: float) -> np.ndarray:
        raise NotImplementedError("not used in dual_enroll path")


def test_majority_owner_picks_max_overlap():
    spans = [
        (0.0, 4.0, "sp_A"),
        (4.0, 10.0, "sp_B"),
    ]
    # Window [3, 5]: 1s overlap with sp_A, 1s with sp_B → first key wins on tie.
    # We don't depend on ties here; spread the window so sp_B clearly wins.
    assert _majority_owner(3.5, 9.0, spans) == "sp_B"
    assert _majority_owner(0.5, 3.5, spans) == "sp_A"
    assert _majority_owner(20.0, 25.0, spans) is None  # outside any span


def test_dual_enroll_writes_secondary_under_primary_speaker_id(tmp_path):
    db_path = tmp_path / "r.db"

    # Pre-seed registry with one primary speaker who already has a primary-model voiceprint.
    with RegistryStore(path=str(db_path)) as store:
        sp = store.create_speaker(display_name="Katie")
        store.add_voiceprint(sp.id, np.ones(192, dtype=np.float32), "speechbrain-192")
    primary_id = sp.id

    # Build a fake secondary provider whose only diarized segment overlaps the
    # primary speaker's time range.
    sec_emb = np.full(8, 0.5, dtype=np.float32)
    fake = FakeSecondaryProvider([
        SpeakerSegment(start=0.0, end=5.0, local_label="Speaker_A", embedding=sec_emb),
    ])

    primary_spans = [(0.0, 5.0, primary_id)]
    result = _dual_enroll(
        audio_path="ignored.wav",
        secondary=fake,
        primary_spans=primary_spans,
        registry_path=str(db_path),
        max_per_speaker=5,
    )

    assert result["model"] == "fake-8"
    assert result["enrolled_speaker_ids"] == [primary_id]

    with RegistryStore(path=str(db_path)) as store:
        # Both backends now match the same speaker.
        primary_vps = store.list_voiceprints(model="speechbrain-192")
        secondary_vps = store.list_voiceprints(model="fake-8")
        assert len(primary_vps) == 1
        assert len(secondary_vps) == 1
        assert secondary_vps[0].speaker_id == primary_id
        np.testing.assert_array_almost_equal(secondary_vps[0].embedding, sec_emb)


def test_dual_enroll_skips_when_no_overlap(tmp_path):
    db_path = tmp_path / "r.db"
    with RegistryStore(path=str(db_path)) as store:
        sp = store.create_speaker(display_name="Katie")

    # Secondary segment is entirely outside the primary span → no enrollment.
    fake = FakeSecondaryProvider([
        SpeakerSegment(start=20.0, end=25.0, local_label="Speaker_A",
                       embedding=np.ones(8, dtype=np.float32)),
    ])
    result = _dual_enroll(
        audio_path="ignored.wav",
        secondary=fake,
        primary_spans=[(0.0, 5.0, sp.id)],
        registry_path=str(db_path),
        max_per_speaker=5,
    )
    assert result["enrolled_speaker_ids"] == []
    assert "skipped_reason" in result

    with RegistryStore(path=str(db_path)) as store:
        assert store.list_voiceprints(model="fake-8") == []


def test_dual_enroll_skips_segments_without_embedding(tmp_path):
    db_path = tmp_path / "r.db"
    with RegistryStore(path=str(db_path)) as store:
        sp = store.create_speaker(display_name="K")

    fake = FakeSecondaryProvider([
        SpeakerSegment(start=0.0, end=5.0, local_label="Speaker_A", embedding=None),
    ])
    result = _dual_enroll(
        audio_path="x.wav",
        secondary=fake,
        primary_spans=[(0.0, 5.0, sp.id)],
        registry_path=str(db_path),
        max_per_speaker=5,
    )
    assert result["enrolled_speaker_ids"] == []
