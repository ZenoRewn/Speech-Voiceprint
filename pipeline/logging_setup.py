"""Single place to configure logging for the project.

Why this exists: the API server, job worker, and Azure adapters need a
consistent format so an operator on Azure VM can grep one stream rather than
piecing together several. CLI tools that already write user-facing warnings
to stderr (e.g. `--list-devices`, mic fallback notices) keep doing that —
they aren't logs.

Usage:
    from pipeline.logging_setup import configure_logging
    configure_logging()  # respects SV_LOG_LEVEL / SV_LOG_JSON env

Then anywhere:
    log = logging.getLogger(__name__)
    log.info("loaded model", extra={"model": "speechbrain-192"})

JSON mode (`SV_LOG_JSON=1`) emits one object per line so log shippers
(fluent-bit, Vector) can ingest directly. Human mode is the default and
matches how the existing CLI scripts surface errors.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any


_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    """Minimal JSON line formatter — no third-party deps."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Anything passed via `extra=` lands as record attributes; pluck the
        # ones that aren't built-in. This lets call sites attach structured
        # context without having to touch the formatter.
        for key, value in record.__dict__.items():
            if key in _STD_ATTRS or key.startswith("_"):
                continue
            try:
                json.dumps(value)
            except TypeError:
                value = repr(value)
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


_STD_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName",
}


def configure_logging(
    *,
    level: str | int | None = None,
    json_format: bool | None = None,
    stream=None,
) -> None:
    """Idempotent root-logger setup. Subsequent calls are no-ops unless force=True is added later.

    Priority: explicit args > env (`SV_LOG_LEVEL`, `SV_LOG_JSON`) > defaults (INFO, human).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    if level is None:
        level = os.environ.get("SV_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if json_format is None:
        json_format = os.environ.get("SV_LOG_JSON", "").lower() in ("1", "true", "yes")

    handler = logging.StreamHandler(stream or sys.stderr)
    if json_format:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    root = logging.getLogger()
    # Drop any handlers that uvicorn / pytest may have left, to avoid duplicate
    # lines. We're the application owner of stderr.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet noisy third-party loggers; they're rarely useful at INFO.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("azure").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper so callers can `from pipeline.logging_setup import get_logger`."""
    return logging.getLogger(name)
