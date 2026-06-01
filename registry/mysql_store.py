"""MySQL-backed voiceprint registry.

This mirrors `registry.store.RegistryStore` but stores speakers and model-tagged
embeddings in MySQL. Audio uploads and result JSONs intentionally stay on the
filesystem/PVC; MySQL owns identity metadata and vector blobs only.
"""

from __future__ import annotations

import threading
import time
import uuid
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from .base import Speaker, SpeakerStore, Voiceprint  # noqa: F401


SCHEMA = """
CREATE TABLE IF NOT EXISTS speakers (
    id VARCHAR(64) PRIMARY KEY,
    display_name VARCHAR(255) NULL,
    created_at DOUBLE NOT NULL,
    updated_at DOUBLE NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS voiceprints (
    id VARCHAR(64) PRIMARY KEY,
    speaker_id VARCHAR(64) NOT NULL,
    embedding LONGBLOB NOT NULL,
    dim INT NOT NULL,
    model VARCHAR(128) NOT NULL,
    quality DOUBLE NOT NULL DEFAULT 0.0,
    created_at DOUBLE NOT NULL,
    INDEX idx_voiceprints_speaker (speaker_id),
    INDEX idx_voiceprints_model (model),
    CONSTRAINT fk_voiceprints_speaker
      FOREIGN KEY (speaker_id) REFERENCES speakers(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""


def _emb_to_blob(emb: np.ndarray) -> bytes:
    return np.ascontiguousarray(emb.astype(np.float32)).tobytes()


def _blob_to_emb(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim).copy()


def _connect_kwargs(uri: str) -> dict:
    parsed = urlparse(uri)
    if not parsed.hostname:
        raise ValueError("MySQL registry URI must include a host")
    database = parsed.path.lstrip("/")
    if not database:
        raise ValueError("MySQL registry URI must include a database name")
    query = parse_qs(parsed.query)
    kwargs = {
        "host": parsed.hostname,
        "port": parsed.port or 3306,
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "database": database,
        "charset": query.get("charset", ["utf8mb4"])[0],
        "autocommit": False,
    }
    ssl_ca = query.get("ssl_ca", [None])[0]
    if ssl_ca:
        kwargs["ssl"] = {"ca": ssl_ca}
    connect_timeout = query.get("connect_timeout", [None])[0]
    if connect_timeout:
        kwargs["connect_timeout"] = int(connect_timeout)
    return kwargs


class MySQLRegistryStore:
    def __init__(self, uri: str) -> None:
        try:
            import pymysql
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "MySQL registry requires optional dependency: pip install -e '.[mysql]'"
            ) from e

        self._conn = pymysql.connect(**_connect_kwargs(uri))
        self._lock = threading.Lock()
        with self._conn.cursor() as cur:
            for stmt in [s.strip() for s in SCHEMA.split(";") if s.strip()]:
                cur.execute(stmt)
        self._conn.commit()

    # ----- speakers -----
    def create_speaker(self, display_name: str | None = None, speaker_id: str | None = None) -> Speaker:
        now = time.time()
        sid = speaker_id or f"sp_{uuid.uuid4().hex[:8]}"
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO speakers(id, display_name, created_at, updated_at) VALUES (%s,%s,%s,%s)",
                (sid, display_name, now, now),
            )
            self._conn.commit()
        return Speaker(id=sid, display_name=display_name, created_at=now, updated_at=now)

    def rename_speaker(self, speaker_id: str, display_name: str) -> None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "UPDATE speakers SET display_name = %s, updated_at = %s WHERE id = %s",
                (display_name, time.time(), speaker_id),
            )
            self._conn.commit()

    def get_speaker(self, speaker_id: str) -> Speaker | None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, display_name, created_at, updated_at FROM speakers WHERE id = %s",
                (speaker_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return Speaker(*row)

    def list_speakers(self) -> list[Speaker]:
        with self._lock, self._conn.cursor() as cur:
            cur.execute("SELECT id, display_name, created_at, updated_at FROM speakers ORDER BY created_at")
            rows = cur.fetchall()
        return [Speaker(*r) for r in rows]

    def delete_speaker(self, speaker_id: str) -> None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute("DELETE FROM speakers WHERE id = %s", (speaker_id,))
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
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO voiceprints(id, speaker_id, embedding, dim, model, quality, created_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                """,
                (vp.id, vp.speaker_id, _emb_to_blob(emb), int(emb.size), model, quality, vp.created_at),
            )
            cur.execute("UPDATE speakers SET updated_at = %s WHERE id = %s", (vp.created_at, speaker_id))
            self._conn.commit()

        if max_per_speaker is not None and max_per_speaker > 0:
            self._trim_voiceprints(speaker_id, model, max_per_speaker)
        return vp

    def _trim_voiceprints(self, speaker_id: str, model: str, keep: int) -> None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM voiceprints
                WHERE speaker_id = %s AND model = %s
                ORDER BY created_at DESC
                """,
                (speaker_id, model),
            )
            stale = [r[0] for r in cur.fetchall()[keep:]]
            if stale:
                cur.executemany("DELETE FROM voiceprints WHERE id = %s", [(s,) for s in stale])
                self._conn.commit()

    def list_voiceprints(self, model: str | None = None) -> list[Voiceprint]:
        sql = "SELECT id, speaker_id, embedding, dim, model, quality, created_at FROM voiceprints"
        args: tuple = ()
        if model is not None:
            sql += " WHERE model = %s"
            args = (model,)
        with self._lock, self._conn.cursor() as cur:
            cur.execute(sql, args)
            rows = cur.fetchall()
        return [
            Voiceprint(
                id=r[0],
                speaker_id=r[1],
                embedding=_blob_to_emb(r[2], r[3]),
                model=r[4],
                quality=r[5],
                created_at=r[6],
            )
            for r in rows
        ]

    def speaker_centroids(self, model: str) -> dict[str, np.ndarray]:
        out: dict[str, list[np.ndarray]] = {}
        for vp in self.list_voiceprints(model=model):
            out.setdefault(vp.speaker_id, []).append(vp.embedding)
        return {sid: np.mean(np.stack(embs), axis=0) for sid, embs in out.items()}

    # ----- lifecycle -----
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MySQLRegistryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
