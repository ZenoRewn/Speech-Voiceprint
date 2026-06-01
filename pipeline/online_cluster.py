"""Tiny online clustering for streaming voiceprints.

State: a list of `(id, centroid, count)` tuples. For each incoming embedding we
do a cosine NN against all centroids, attach to the closest if its distance is
below `threshold`, else start a new cluster. Centroids update via running mean
(weighted by count) so they stabilize as more samples arrive.

Designed for tens of speakers and a few thousand windows per session — fine on
CPU, no need for FAISS.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

import numpy as np


@dataclass
class _Cluster:
    label: str
    centroid: np.ndarray
    count: int


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    sim = float(np.dot(a, b) / (na * nb))
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


def _label_for(idx: int) -> str:
    if idx < 26:
        return f"Speaker_{string.ascii_uppercase[idx]}"
    return f"Speaker_{idx}"


class OnlineSpeakerCluster:
    def __init__(self, threshold: float = 0.6, smoothing: float = 0.2) -> None:
        self.threshold = threshold
        self.smoothing = smoothing
        self._clusters: list[_Cluster] = []

    def update(self, embedding: np.ndarray) -> str:
        if embedding is None:
            return "Speaker_Unknown"
        if not self._clusters:
            label = _label_for(0)
            self._clusters.append(_Cluster(label=label, centroid=embedding.astype(np.float32), count=1))
            return label

        dists = [_cosine_distance(embedding, c.centroid) for c in self._clusters]
        i = int(np.argmin(dists))
        if dists[i] <= self.threshold:
            c = self._clusters[i]
            # Running mean with mild smoothing — older samples keep most weight
            new_centroid = c.centroid + self.smoothing * (embedding - c.centroid)
            c.centroid = new_centroid.astype(np.float32)
            c.count += 1
            return c.label

        label = _label_for(len(self._clusters))
        self._clusters.append(_Cluster(label=label, centroid=embedding.astype(np.float32), count=1))
        return label

    def label_for_centroid(self, embedding: np.ndarray) -> str | None:
        """Read-only: closest cluster within threshold, or None."""
        if not self._clusters or embedding is None:
            return None
        dists = [_cosine_distance(embedding, c.centroid) for c in self._clusters]
        i = int(np.argmin(dists))
        if dists[i] <= self.threshold:
            return self._clusters[i].label
        return None

    def centroids(self) -> dict[str, np.ndarray]:
        return {c.label: c.centroid.copy() for c in self._clusters}
