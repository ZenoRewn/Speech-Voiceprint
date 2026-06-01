"""Loop-agnostic multi-session pub/sub state for the streaming viewer.

Why a separate module: `pipeline/ws_server.py` runs its own asyncio loop on a
daemon thread (used by `pipeline.streaming`). `pipeline/api.py` runs uvicorn
with a primary loop; we don't want a second loop. Both want the same data
layout (sessions → clients, sessions → history) so they extract it here.

The hub deliberately does *not* know about a specific WebSocket library — it
exposes `broadcast(rec, session)` to publishers and tracks a list of sender
callables (`async def send(str)`) per session for subscribers. FastAPI's
`WebSocket.send_text` and websockets' `ServerConnection.send` both fit.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Awaitable, Callable

_MAX_SESSION_LEN = 64
_MAX_HISTORY = 1000
_HISTORY_TRIM_TO = 500

Sender = Callable[[str], Awaitable[None]]


def normalize_session(value: str | None) -> str:
    """Bound a session label to 64 chars; empty falls back to `default`."""
    if not value:
        return "default"
    s = str(value).strip()
    if not s:
        return "default"
    return s[:_MAX_SESSION_LEN]


class SessionHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: dict[str, set[Sender]] = {}
        self._histories: dict[str, list[dict]] = {}

    # ---- subscriptions ------------------------------------------------
    def subscribe(self, session: str, sender: Sender) -> str:
        sess = normalize_session(session)
        with self._lock:
            self._subs.setdefault(sess, set()).add(sender)
        return sess

    def unsubscribe(self, session: str, sender: Sender) -> None:
        sess = normalize_session(session)
        with self._lock:
            subs = self._subs.get(sess)
            if subs:
                subs.discard(sender)
                if not subs and not self._histories.get(sess):
                    self._subs.pop(sess, None)

    def history(self, session: str) -> list[dict]:
        sess = normalize_session(session)
        with self._lock:
            return list(self._histories.get(sess, []))

    # ---- publishing ---------------------------------------------------
    def append(self, record: dict, session: str) -> tuple[str, dict]:
        """Tag the record with its session and append to history.

        Returns `(session, record)` so callers can fan out the same dict that
        was stored — useful so subscribers see the canonical session label.
        """
        sess = normalize_session(session)
        record = {**record, "session": sess}
        with self._lock:
            hist = self._histories.setdefault(sess, [])
            hist.append(record)
            if len(hist) > _MAX_HISTORY:
                self._histories[sess] = hist[-_HISTORY_TRIM_TO:]
        return sess, record

    async def broadcast(self, record: dict, session: str) -> None:
        sess, record = self.append(record, session)
        msg = json.dumps(record, ensure_ascii=False)
        with self._lock:
            senders = list(self._subs.get(sess, set()))
        if not senders:
            return
        await asyncio.gather(
            *(s(msg) for s in senders),
            return_exceptions=True,
        )

    def rewrite_history_speaker(
        self, old_label: str, new_label: str, *, session: str
    ) -> list[dict]:
        sess = normalize_session(session)
        updated: list[dict] = []
        with self._lock:
            for rec in self._histories.get(sess, []):
                if rec.get("speaker") == old_label:
                    rec["speaker"] = new_label
                    rec["event"] = "revised"
                    updated.append(rec)
        return updated

    # ---- introspection ------------------------------------------------
    def list_sessions(self) -> list[dict]:
        with self._lock:
            keys = set(self._histories) | set(self._subs)
            return [
                {
                    "session": s,
                    "clients": len(self._subs.get(s, set())),
                    "history": len(self._histories.get(s, [])),
                }
                for s in sorted(keys)
            ]

    def drop_session(self, session: str) -> bool:
        """Forget the session's history. Active subscribers stay connected
        (they'll keep receiving new events) but the replay buffer is wiped."""
        sess = normalize_session(session)
        with self._lock:
            had = sess in self._histories
            self._histories.pop(sess, None)
            return had
