"""Three-state speaker matcher.

Given a probe embedding and a registry, produce one of:
  * `KNOWN`         — distance < match_threshold
  * `LOW_CONFIDENCE`— between match_threshold and unknown_threshold
  * `UNKNOWN`       — distance >= unknown_threshold (worth enrolling as new)

`SpeakerMatcher.assign_local_labels` consumes the diarized SpeakerSegments and
either reuses the registry's display name (or `sp_xxxxxxxx` id) or assigns a
fresh `Speaker_<uuid>` for unknowns. `low_confidence` matches keep the local
clustering label so downstream code doesn't accidentally claim an identity it's
not sure about, but the candidate match is exposed for human review.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from voiceprint.base import SpeakerSegment

from .store import RegistryStore


class MatchVerdict(str, Enum):
    KNOWN = "known"
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN = "unknown"


@dataclass
class MatchResult:
    verdict: MatchVerdict
    speaker_id: str | None
    display_name: str | None
    distance: float
    candidate_speaker_id: str | None = None     # filled when low_confidence
    candidate_distance: float | None = None


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    sim = float(np.dot(a, b) / (na * nb))
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


class SpeakerMatcher:
    def __init__(
        self,
        store: RegistryStore,
        model: str,
        *,
        match_threshold: float = 0.40,
        unknown_threshold: float = 0.60,
        max_voiceprints_per_speaker: int = 5,
        auto_enroll_unknown: bool = True,
    ) -> None:
        if match_threshold > unknown_threshold:
            raise ValueError("match_threshold must be <= unknown_threshold")
        self.store = store
        self.model = model
        self.match_threshold = match_threshold
        self.unknown_threshold = unknown_threshold
        self.max_voiceprints_per_speaker = max_voiceprints_per_speaker
        self.auto_enroll_unknown = auto_enroll_unknown
        self._centroids = self.store.speaker_centroids(model)

    def _refresh(self) -> None:
        self._centroids = self.store.speaker_centroids(self.model)

    def match(self, embedding: np.ndarray) -> MatchResult:
        if embedding is None:
            return MatchResult(MatchVerdict.UNKNOWN, None, None, distance=1.0)
        if not self._centroids:
            return MatchResult(MatchVerdict.UNKNOWN, None, None, distance=1.0)
        dists = {sid: _cosine_distance(embedding, c) for sid, c in self._centroids.items()}
        best_sid, best_d = min(dists.items(), key=lambda kv: kv[1])

        if best_d < self.match_threshold:
            sp = self.store.get_speaker(best_sid)
            return MatchResult(
                verdict=MatchVerdict.KNOWN,
                speaker_id=best_sid,
                display_name=sp.display_name if sp else None,
                distance=best_d,
            )
        if best_d < self.unknown_threshold:
            sp = self.store.get_speaker(best_sid)
            return MatchResult(
                verdict=MatchVerdict.LOW_CONFIDENCE,
                speaker_id=None,
                display_name=None,
                distance=best_d,
                candidate_speaker_id=best_sid,
                candidate_distance=best_d,
            )
        return MatchResult(MatchVerdict.UNKNOWN, None, None, distance=best_d)

    def assign_local_labels(
        self, segments: list[SpeakerSegment]
    ) -> dict[str, MatchResult]:
        """Resolve each *clustering* label to a registry identity.

        Multiple segments share a `local_label`, so we average their embeddings
        before matching to be more stable than any single short segment. The
        return is a `local_label -> MatchResult` map; callers can rewrite the
        segment labels in place.
        """
        clusters: dict[str, list[np.ndarray]] = {}
        for seg in segments:
            if seg.embedding is None:
                continue
            clusters.setdefault(seg.local_label, []).append(seg.embedding)

        verdicts: dict[str, MatchResult] = {}
        for label, embs in clusters.items():
            mean_emb = np.mean(np.stack(embs), axis=0)
            verdicts[label] = self.match(mean_emb)

            res = verdicts[label]
            if res.verdict == MatchVerdict.KNOWN and res.speaker_id is not None:
                self.store.add_voiceprint(
                    res.speaker_id,
                    mean_emb,
                    self.model,
                    quality=1.0 - res.distance,
                    max_per_speaker=self.max_voiceprints_per_speaker,
                )
                self._refresh()
            elif res.verdict == MatchVerdict.UNKNOWN and self.auto_enroll_unknown:
                sp = self.store.create_speaker(display_name=None)
                self.store.add_voiceprint(
                    sp.id,
                    mean_emb,
                    self.model,
                    quality=1.0,
                    max_per_speaker=self.max_voiceprints_per_speaker,
                )
                verdicts[label] = MatchResult(
                    verdict=MatchVerdict.UNKNOWN,
                    speaker_id=sp.id,
                    display_name=None,
                    distance=res.distance,
                )
                self._refresh()
        return verdicts

    @staticmethod
    def label_for(result: MatchResult, fallback: str) -> str:
        if result.verdict == MatchVerdict.KNOWN:
            return result.display_name or result.speaker_id or fallback
        if result.verdict == MatchVerdict.UNKNOWN and result.speaker_id is not None:
            return result.display_name or result.speaker_id
        return fallback  # low_confidence: keep local clustering label
