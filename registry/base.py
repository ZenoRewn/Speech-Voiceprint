"""SpeakerStore — backend-agnostic interface for the voiceprint registry.

The shipped implementation is SQLite (`registry.store.RegistryStore`). External
DBs (MySQL, Postgres) plug in by implementing this Protocol and registering a
URI scheme handler in `registry.open_store`.

Embeddings are model-tagged (e.g. `"speechbrain-192"`, `"pyannote-512"`) so
backends with different dimensions coexist; the matcher filters by `model`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Speaker:
    id: str
    display_name: str | None
    created_at: float
    updated_at: float


@dataclass
class Voiceprint:
    id: str
    speaker_id: str
    embedding: np.ndarray
    model: str
    quality: float
    created_at: float


@runtime_checkable
class SpeakerStore(Protocol):
    """Persistence contract every registry backend must satisfy.

    Implementations are expected to be thread-safe enough for a single-writer /
    multi-reader workload — that's what the streaming + jobs paths assume. The
    SQLite implementation gets this from WAL mode; a SQL-server backend gets it
    from the server itself.
    """

    # ----- speakers -----
    def create_speaker(self, display_name: str | None = None, speaker_id: str | None = None) -> Speaker: ...

    def rename_speaker(self, speaker_id: str, display_name: str) -> None: ...

    def get_speaker(self, speaker_id: str) -> Speaker | None: ...

    def list_speakers(self) -> list[Speaker]: ...

    def delete_speaker(self, speaker_id: str) -> None: ...

    # ----- voiceprints -----
    def add_voiceprint(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        model: str,
        *,
        quality: float = 0.0,
        max_per_speaker: int | None = None,
    ) -> Voiceprint: ...

    def list_voiceprints(self, model: str | None = None) -> list[Voiceprint]: ...

    def speaker_centroids(self, model: str) -> dict[str, np.ndarray]: ...

    # ----- lifecycle -----
    def close(self) -> None: ...

    def __enter__(self) -> "SpeakerStore": ...

    def __exit__(self, *exc) -> None: ...
