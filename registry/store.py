"""SQLite-backed voiceprint registry.

Schema is intentionally minimal: a `speakers` row per persistent identity, plus
a `voiceprints` row per enrolled embedding. Embeddings are stored as raw float32
bytes plus their dimension and source model — backends with different dims (192
for ECAPA, 512 for pyannote) coexist by filtering on `model` at match time.
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np

# Re-export Speaker/Voiceprint from registry.base so legacy imports from
# `registry.store` keep working while the canonical types live with the
# Protocol that defines the storage contract.
from .base import Speaker, SpeakerStore, Voiceprint  # noqa: F401


SCHEMA = """
CREATE TABLE IF NOT EXISTS speakers (
    id TEXT PRIMARY KEY,
    display_name TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS voiceprints (
    id TEXT PRIMARY KEY,
    speaker_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL,
    model TEXT NOT NULL,
    quality REAL NOT NULL DEFAULT 0.0,
    created_at REAL NOT NULL,
    FOREIGN KEY (speaker_id) REFERENCES speakers(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_voiceprints_speaker ON voiceprints(speaker_id);
CREATE INDEX IF NOT EXISTS idx_voiceprints_model ON voiceprints(model);
"""


def _emb_to_blob(emb: np.ndarray) -> bytes:
    return np.ascontiguousarray(emb.astype(np.float32)).tobytes()


def _blob_to_emb(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim).copy()


class RegistryStore:
    def __init__(self, path: str | os.PathLike, *, in_memory: bool = False) -> None:
        if in_memory:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            target = Path(path).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            # `isolation_level=None` plus explicit transactions would also work,
            # but the default deferred mode + WAL gives us readers concurrent
            # with one writer — exactly what multiple streaming sessions need.
            self._conn = sqlite3.connect(str(target), check_same_thread=False, timeout=15.0)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=15000")
            self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ----- speakers -----
    def create_speaker(self, display_name: str | None = None, speaker_id: str | None = None) -> Speaker:
        now = time.time()
        sid = speaker_id or f"sp_{uuid.uuid4().hex[:8]}"
        self._conn.execute(
            "INSERT INTO speakers(id, display_name, created_at, updated_at) VALUES (?,?,?,?)",
            (sid, display_name, now, now),
        )
        self._conn.commit()
        return Speaker(id=sid, display_name=display_name, created_at=now, updated_at=now)

    def rename_speaker(self, speaker_id: str, display_name: str) -> None:
        self._conn.execute(
            "UPDATE speakers SET display_name = ?, updated_at = ? WHERE id = ?",
            (display_name, time.time(), speaker_id),
        )
        self._conn.commit()

    def get_speaker(self, speaker_id: str) -> Speaker | None:
        row = self._conn.execute(
            "SELECT id, display_name, created_at, updated_at FROM speakers WHERE id = ?",
            (speaker_id,),
        ).fetchone()
        if not row:
            return None
        return Speaker(*row)

    def list_speakers(self) -> list[Speaker]:
        rows = self._conn.execute(
            "SELECT id, display_name, created_at, updated_at FROM speakers ORDER BY created_at"
        ).fetchall()
        return [Speaker(*r) for r in rows]

    def delete_speaker(self, speaker_id: str) -> None:
        self._conn.execute("DELETE FROM voiceprints WHERE speaker_id = ?", (speaker_id,))
        self._conn.execute("DELETE FROM speakers WHERE id = ?", (speaker_id,))
        self._conn.commit()

    # ----- voiceprints -----
    def add_voiceprint(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        model: str,
        *,
        quality: float = 0.0,
        max_per_speaker: int | None = None,
    ) -> Voiceprint:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        vp = Voiceprint(
            id=f"vp_{uuid.uuid4().hex[:10]}",
            speaker_id=speaker_id,
            embedding=emb,
            model=model,
            quality=quality,
            created_at=time.time(),
        )
        self._conn.execute(
            "INSERT INTO voiceprints(id, speaker_id, embedding, dim, model, quality, created_at) VALUES (?,?,?,?,?,?,?)",
            (vp.id, vp.speaker_id, _emb_to_blob(emb), int(emb.size), model, quality, vp.created_at),
        )
        self._conn.execute(
            "UPDATE speakers SET updated_at = ? WHERE id = ?",
            (vp.created_at, speaker_id),
        )
        self._conn.commit()

        if max_per_speaker is not None and max_per_speaker > 0:
            self._trim_voiceprints(speaker_id, model, max_per_speaker)
        return vp

    def _trim_voiceprints(self, speaker_id: str, model: str, keep: int) -> None:
        rows = self._conn.execute(
            "SELECT id FROM voiceprints WHERE speaker_id = ? AND model = ? ORDER BY created_at DESC",
            (speaker_id, model),
        ).fetchall()
        stale = [r[0] for r in rows[keep:]]
        if stale:
            self._conn.executemany("DELETE FROM voiceprints WHERE id = ?", [(s,) for s in stale])
            self._conn.commit()

    def list_voiceprints(self, model: str | None = None) -> list[Voiceprint]:
        sql = "SELECT id, speaker_id, embedding, dim, model, quality, created_at FROM voiceprints"
        args: tuple = ()
        if model is not None:
            sql += " WHERE model = ?"
            args = (model,)
        rows = self._conn.execute(sql, args).fetchall()
        out: list[Voiceprint] = []
        for r in rows:
            out.append(
                Voiceprint(
                    id=r[0],
                    speaker_id=r[1],
                    embedding=_blob_to_emb(r[2], r[3]),
                    model=r[4],
                    quality=r[5],
                    created_at=r[6],
                )
            )
        return out

    def speaker_centroids(self, model: str) -> dict[str, np.ndarray]:
        """Return mean-normalized embedding per speaker for fast matching."""
        out: dict[str, list[np.ndarray]] = {}
        for vp in self.list_voiceprints(model=model):
            out.setdefault(vp.speaker_id, []).append(vp.embedding)
        return {sid: np.mean(np.stack(embs), axis=0) for sid, embs in out.items()}

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RegistryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
