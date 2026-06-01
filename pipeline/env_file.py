"""Minimal .env loader — no external dependency.

Parses simple `KEY=value` lines. Supports:
- `# comment` lines and trailing comments after unquoted values
- `export KEY=value` (the `export` prefix is stripped)
- single- or double-quoted values (quotes are removed; no escape processing)
- blank lines

Does NOT override variables already present in `os.environ`, so explicit
`export FOO=bar` in the shell still wins over the file. This keeps the
file useful for dev defaults without surprising production deploys that
inject credentials via the orchestrator.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def load_env_file(path: str | os.PathLike | None = None) -> dict[str, str]:
    """Load `.env` from `path` (or cwd's `.env` if omitted).

    Returns the dict of keys actually applied (skips already-set keys).
    Silently no-ops if the file does not exist.
    """
    p = Path(path) if path else Path.cwd() / ".env"
    if not p.is_file():
        return {}

    applied: dict[str, str] = {}
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("env file %s unreadable: %s", p, e)
        return {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not _is_valid_key(key):
            continue
        value = value.strip()
        if (len(value) >= 2) and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            # strip trailing inline `# ...` comment for unquoted values
            hash_idx = value.find(" #")
            if hash_idx >= 0:
                value = value[:hash_idx].rstrip()
        if key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value

    if applied:
        log.info("env file %s applied %d var(s): %s", p, len(applied), ", ".join(sorted(applied)))
    return applied


def _is_valid_key(key: str) -> bool:
    if not key:
        return False
    if not (key[0].isalpha() or key[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in key)
