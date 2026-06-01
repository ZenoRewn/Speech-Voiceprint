"""Voiceprint registry public surface.

`open_store(uri)` is the factory callers should use. It picks an implementation
from a URI scheme:

- `sqlite:///abs/path.db` or bare path (e.g. `~/.sv/registry.db`) → SQLite
- `mysql+pymysql://user:pw@host/db` → MySQL backend
- `postgresql://user:pw@host/db`    (planned) → SQL-server backend

The legacy `RegistryStore(path=...)` constructor still works — `open_store` is
the recommended entry point because it isolates callers from backend choice.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from .base import Speaker, SpeakerStore, Voiceprint
from .matcher import MatchResult, MatchVerdict, SpeakerMatcher
from .store import RegistryStore


def open_store(uri: str | os.PathLike) -> SpeakerStore:
    """Open the registry described by `uri`.

    Accepts:
      - `sqlite:///<path>` — explicit SQLite URI (path may be absolute or relative)
      - bare filesystem path — treated as SQLite (back-compat with old configs)
      - `mysql+...://...` — MySQL backend (requires `pip install -e '.[mysql]'`)
      - `postgresql://...` — raises NotImplementedError; the Protocol is in place
        so adding it is a single new file.
    """
    raw = os.fspath(uri).strip()
    if not raw:
        raise ValueError("empty registry URI")

    scheme = urlparse(raw).scheme.lower() if "://" in raw else ""

    if scheme in ("", "sqlite", "sqlite3"):
        # SQLAlchemy-style: `sqlite:///abs.db` (3 slashes = absolute), `sqlite://./rel.db`
        # (2 slashes + relative). Strip the scheme prefix and pass whatever's left to
        # SQLite. Bare paths fall through unchanged.
        if raw.startswith("sqlite:///") or raw.startswith("sqlite3:///"):
            path = raw.split("///", 1)[1]
            path = "/" + path  # keep absolute marker
        elif raw.startswith("sqlite://") or raw.startswith("sqlite3://"):
            path = raw.split("://", 1)[1]
        elif raw.startswith("sqlite:") or raw.startswith("sqlite3:"):
            path = raw.split(":", 1)[1]
        else:
            path = raw
        return RegistryStore(path=path)

    if scheme.startswith("mysql"):
        from .mysql_store import MySQLRegistryStore

        return MySQLRegistryStore(raw)

    if scheme.startswith(("postgres", "postgresql")):
        raise NotImplementedError(
            f"registry backend '{scheme}' isn't implemented yet — see registry/base.py "
            "SpeakerStore Protocol; implement it in a new module and route here."
        )

    raise ValueError(f"unsupported registry URI scheme: {scheme!r}")


__all__ = [
    "RegistryStore",
    "Speaker",
    "SpeakerMatcher",
    "SpeakerStore",
    "MatchResult",
    "MatchVerdict",
    "Voiceprint",
    "open_store",
]
