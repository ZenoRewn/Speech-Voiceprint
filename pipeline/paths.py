"""Single source of truth for filesystem layout.

The api needs five directories:
  resources/ — read-only inputs (audio the user dropped in)
  uploads/   — multipart uploads from /api/jobs/transcribe-upload
  stream/    — multipart uploads from /api/sessions/{id}/stream-file
  outputs/   — finished Job result JSONs (downloadable from /api/jobs/{id}/download)
  registry/  — SQLite WAL files for the speaker registry (when no explicit --registry)

All five default under `SV_DATA_DIR` (env, falls back to `<repo>/data/`). Each
field is also overridable individually so AKS can mount uploads/outputs on a
different PVC than the registry. Directories are created lazily on first use,
not at import time, so unit tests don't pollute the filesystem.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DataPaths:
    root: Path
    resources: Path
    uploads: Path
    stream: Path
    outputs: Path
    registry: Path

    def ensure(self, *subset: str) -> None:
        """Create the requested subdirectories. With no args, create them all."""
        names = subset or ("resources", "uploads", "stream", "outputs", "registry")
        for n in names:
            getattr(self, n).mkdir(parents=True, exist_ok=True)

    def default_registry_db(self) -> Path:
        return self.registry / "speakers.db"


def _env_path(name: str, default: Path) -> Path:
    val = os.environ.get(name)
    return Path(val).expanduser() if val else default


def get_paths() -> DataPaths:
    """Resolve paths from env at call time.

    Re-reading env on each call lets tests monkey-patch SV_DATA_DIR without a
    full module reload. Cheap — just a few env lookups and Path() ctors.
    """
    root = _env_path("SV_DATA_DIR", _REPO_ROOT / "data")
    return DataPaths(
        root=root,
        resources=_env_path("SV_RESOURCES_DIR", root / "resources"),
        uploads=_env_path("SV_UPLOADS_DIR", root / "uploads"),
        stream=_env_path("SV_STREAM_DIR", root / "stream"),
        outputs=_env_path("SV_OUTPUTS_DIR", root / "outputs"),
        registry=_env_path("SV_REGISTRY_DIR", root / "registry"),
    )


__all__ = ["DataPaths", "get_paths"]
