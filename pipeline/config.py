"""YAML config loader with env-var expansion and CLI override semantics.

Three layers, latest wins:
  1. Defaults baked into click options
  2. YAML file (if `--config` is given)
  3. Explicit CLI flags

`${ENV_VAR}` and `${ENV_VAR:default}` are expanded at load time so the same
config file works on Mac dev and Azure VM without shell-side templating.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::([^}]*))?\}")


def _expand_envs(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            default = match.group(2) or ""
            return os.environ.get(name, default)

        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_envs(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_envs(v) for v in value]
    return value


def load_config(path: str | os.PathLike | None) -> dict:
    """Load a YAML config file, expanding `${VAR}` against the environment.

    Returns an empty dict when path is None or the file is empty/missing.
    """
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {p}")
    return _expand_envs(raw)


def pick(
    cfg: dict,
    dotted: str,
    default: Any = None,
) -> Any:
    """Navigate `cfg` by dotted key path, returning `default` on miss."""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def merge_cli(
    cfg: dict,
    cli_value: Any,
    dotted: str,
    *,
    default: Any = None,
) -> Any:
    """CLI > YAML > default. `None` and empty tuples count as "user did not set"."""
    if cli_value is not None and cli_value != ():
        return cli_value
    yaml_value = pick(cfg, dotted)
    if yaml_value is not None:
        return yaml_value
    return default
