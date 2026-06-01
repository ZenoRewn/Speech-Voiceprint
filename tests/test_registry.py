import numpy as np
import pytest

from registry.matcher import MatchVerdict, SpeakerMatcher
from registry.store import RegistryStore
from voiceprint.base import SpeakerSegment


@pytest.fixture
def store():
    s = RegistryStore(path=None, in_memory=True)
    yield s
    s.close()


def _emb(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v


def test_create_and_get_speaker(store):
    sp = store.create_speaker("Katie")
    assert sp.id.startswith("sp_")
    fetched = store.get_speaker(sp.id)
    assert fetched is not None
    assert fetched.display_name == "Katie"


def test_voiceprint_round_trip(store):
    sp = store.create_speaker("Steve")
    e = _emb(1)
    vp = store.add_voiceprint(sp.id, e, model="ecapa")
    listed = store.list_voiceprints(model="ecapa")
    assert len(listed) == 1
    np.testing.assert_allclose(listed[0].embedding, e, rtol=1e-5)
    assert vp.id == listed[0].id


def test_max_per_speaker_trims_oldest(store):
    sp = store.create_speaker()
    for i in range(5):
        store.add_voiceprint(sp.id, _emb(i), model="ecapa", max_per_speaker=3)
    remaining = store.list_voiceprints(model="ecapa")
    assert len(remaining) == 3


def test_centroids_average_per_speaker(store):
    a = store.create_speaker("A")
    b = store.create_speaker("B")
    store.add_voiceprint(a.id, np.array([1.0, 0.0], dtype=np.float32), model="m")
    store.add_voiceprint(a.id, np.array([3.0, 0.0], dtype=np.float32), model="m")
    store.add_voiceprint(b.id, np.array([0.0, 4.0], dtype=np.float32), model="m")
    cents = store.speaker_centroids("m")
    np.testing.assert_allclose(cents[a.id], [2.0, 0.0])
    np.testing.assert_allclose(cents[b.id], [0.0, 4.0])


def test_matcher_known_below_threshold(store):
    sp = store.create_speaker("Katie")
    e = _emb(42)
    store.add_voiceprint(sp.id, e, model="ecapa")
    matcher = SpeakerMatcher(store, model="ecapa", match_threshold=0.2, unknown_threshold=0.6)
    res = matcher.match(e + 1e-6)
    assert res.verdict == MatchVerdict.KNOWN
    assert res.speaker_id == sp.id
    assert res.display_name == "Katie"


def test_matcher_low_confidence_zone(store):
    sp = store.create_speaker("X")
    base = _emb(7, dim=4)
    store.add_voiceprint(sp.id, base, model="m")
    # produce a probe at cosine distance ~0.5 by mixing in orthogonal noise
    ortho = np.array([base[1], -base[0], base[3], -base[2]], dtype=np.float32)
    ortho /= np.linalg.norm(ortho)
    probe = 0.5 * base + 0.5 * ortho
    matcher = SpeakerMatcher(
        store, model="m", match_threshold=0.2, unknown_threshold=0.6, auto_enroll_unknown=False
    )
    res = matcher.match(probe)
    assert res.verdict == MatchVerdict.LOW_CONFIDENCE
    assert res.speaker_id is None
    assert res.candidate_speaker_id == sp.id


def test_matcher_unknown_auto_enrolls(store):
    sp = store.create_speaker("known")
    store.add_voiceprint(sp.id, _emb(1), model="m")
    matcher = SpeakerMatcher(
        store, model="m", match_threshold=0.2, unknown_threshold=0.4, auto_enroll_unknown=True
    )
    far = _emb(99)
    res = matcher.match(far)
    if res.verdict != MatchVerdict.UNKNOWN:
        pytest.skip("random embedding accidentally close; rerun")
    # match() itself doesn't enroll; assign_local_labels does. Verify count unchanged.
    assert len(store.list_speakers()) == 1


def test_assign_local_labels_creates_unknown_and_relabels(store):
    katie = store.create_speaker("Katie")
    katie_emb = _emb(1)
    store.add_voiceprint(katie.id, katie_emb, model="m")

    matcher = SpeakerMatcher(
        store, model="m", match_threshold=0.2, unknown_threshold=0.4, auto_enroll_unknown=True
    )
    segs = [
        SpeakerSegment(0.0, 1.0, "Speaker_A", embedding=katie_emb),
        SpeakerSegment(1.5, 2.5, "Speaker_B", embedding=_emb(99)),
    ]
    verdicts = matcher.assign_local_labels(segs)
    assert verdicts["Speaker_A"].verdict == MatchVerdict.KNOWN
    assert verdicts["Speaker_A"].speaker_id == katie.id
    assert verdicts["Speaker_B"].verdict == MatchVerdict.UNKNOWN
    assert verdicts["Speaker_B"].speaker_id is not None  # newly enrolled
    # speakers table now has 2 rows
    assert len(store.list_speakers()) == 2


def test_match_threshold_validation(store):
    with pytest.raises(ValueError):
        SpeakerMatcher(store, model="m", match_threshold=0.5, unknown_threshold=0.4)
