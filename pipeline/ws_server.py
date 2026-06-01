"""Tiny multi-session WebSocket fan-out for streaming events.

`WebSocketSink.broadcast(rec, session=...)` ships a JSON message to every
client subscribed to that session. The server runs in its own asyncio loop on
a background thread so the rest of the streaming pipeline can keep using plain
blocking calls.

Clients pick which session they want by appending `?session=<id>` to the WS
URL (`/events?session=katiesteve`). Without the parameter they fall back to
the sink's default `session_id`. Late joiners get the session's full history
replayed before live events start streaming.

Inbound messages from clients are forwarded to a `command_handler(msg, session)`
callback — the streaming pipeline registers one to handle `rename` / `forget`
operations on the SpeakerRegistry.

Also serves a static `web/` directory on the same port over plain HTTP plus a
JSON `/sessions` endpoint that lists active sessions, so a viewer can show a
session picker without opening a separate control plane.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlsplit

import websockets
from websockets.asyncio.server import serve as ws_serve
from websockets.datastructures import Headers
from websockets.http11 import Response

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"

CommandHandler = Callable[[dict, str], dict | None]
_MAX_SESSION_LEN = 64
_MAX_HISTORY = 1000
_HISTORY_TRIM_TO = 500


class WebSocketSink:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        path: str = "/events",
        command_handler: CommandHandler | None = None,
        session_id: str = "default",
    ) -> None:
        self.host = host
        self.port = port
        self.path = path
        self.command_handler = command_handler
        # Default session label used when a publisher / client doesn't specify
        # one. Normalized to bound storage and avoid empty keys.
        self.session_id = self._normalize_session(session_id)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        # Per-session client sets and history. A client subscribes to a session
        # via `?session=<id>` on the WS URL, falling back to `self.session_id`.
        self._sessions: dict[str, set[websockets.ServerConnection]] = {}
        self._histories: dict[str, list[dict]] = {}
        self._sessions_lock = threading.Lock()
        self._ready = threading.Event()
        # Created lazily on the loop thread; `stop()` flips it via
        # call_soon_threadsafe so the loop wakes and shuts down gracefully.
        self._stop_event: asyncio.Event | None = None  # type: ignore[assignment]

    # public API --------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError(f"WebSocket server failed to start on {self.host}:{self.port}")

    def stop(self) -> None:
        """Ask the server to close cleanly. Safe to call more than once."""
        if self._loop is None:
            return
        # Setting `_stop_event` from inside the loop lets `ws_serve`'s
        # async-context-manager teardown finish before the loop closes,
        # avoiding "Event loop is closed" warnings during pytest GC.
        try:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        except RuntimeError:
            return  # loop already closed
        if self._thread:
            self._thread.join(timeout=3.0)

    def broadcast(self, record: dict, session: str | None = None) -> None:
        if self._loop is None:
            return
        sess = self._normalize_session(session)
        record = {**record, "session": sess}  # tag every event so multi-tenant viewers can filter
        with self._sessions_lock:
            hist = self._histories.setdefault(sess, [])
            hist.append(record)
            if len(hist) > _MAX_HISTORY:
                self._histories[sess] = hist[-_HISTORY_TRIM_TO:]
        msg = json.dumps(record, ensure_ascii=False)
        asyncio.run_coroutine_threadsafe(self._fanout(msg, sess), self._loop)

    @staticmethod
    def _normalize_session(value: str | None) -> str:
        if not value:
            return "default"
        s = str(value).strip()
        if not s:
            return "default"
        # Be permissive but bounded — these end up as dict keys and history
        # entries, so an unbounded id from a malicious client could waste RAM.
        return s[:_MAX_SESSION_LEN]

    def rewrite_history_speaker(
        self, old_label: str, new_label: str, session: str | None = None
    ) -> list[dict]:
        """Update prior records when a speaker has been renamed.

        Returns the updated records so the caller can fan them out as
        `revised` events to currently connected clients.
        """
        sess = self._normalize_session(session)
        updated: list[dict] = []
        with self._sessions_lock:
            for rec in self._histories.get(sess, []):
                if rec.get("speaker") == old_label:
                    rec["speaker"] = new_label
                    rec["event"] = "revised"
                    updated.append(rec)
        return updated

    def list_sessions(self) -> list[dict]:
        """Snapshot of active sessions for the `/sessions` endpoint and tests."""
        with self._sessions_lock:
            keys = set(self._histories) | set(self._sessions)
            return [
                {
                    "session": s,
                    "clients": len(self._sessions.get(s, set())),
                    "history": len(self._histories.get(s, [])),
                }
                for s in sorted(keys)
            ]

    # internals ---------------------------------------------------------
    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._stop_event = asyncio.Event()

        async def main() -> None:
            async with ws_serve(self._handler, self.host, self.port, process_request=self._http_static):
                self._ready.set()
                # Wait for stop() instead of an unawaited Future — this lets
                # the `async with` exit graceful before the loop is torn down.
                await self._stop_event.wait()

        try:
            self._loop.run_until_complete(main())
        except (asyncio.CancelledError, RuntimeError):
            pass
        finally:
            self._loop.close()

    def _session_from_request(self, request_path: str) -> str:
        """Extract `?session=` from a WS URL path. Falls back to the sink default."""
        try:
            qs = parse_qs(urlsplit(request_path).query, keep_blank_values=False)
        except ValueError:
            return self.session_id
        values = qs.get("session") or []
        return self._normalize_session(values[0] if values else None) if values else self.session_id

    async def _handler(self, ws) -> None:
        sess = self._session_from_request(ws.request.path or "/")
        with self._sessions_lock:
            self._sessions.setdefault(sess, set()).add(ws)
            history = list(self._histories.get(sess, []))
        try:
            # Replay this session's history so a late-joining client sees the
            # full transcript. Each record was already tagged with its session
            # name when broadcast(), so the client can verify-on-receive.
            for rec in history:
                await ws.send(json.dumps(rec, ensure_ascii=False))
            async for raw in ws:
                # Inbound messages are commands (rename, forget…). Run them in
                # a thread so any SQLite call never blocks the event loop, and
                # echo the response (or the error) back to the sender. The
                # session label is passed through so command_handler can scope
                # its mutations to the right registry / history.
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send(json.dumps({"type": "error", "error": "invalid json"}))
                    continue
                if not isinstance(msg, dict):
                    await ws.send(json.dumps({"type": "error", "error": "expected json object"}))
                    continue
                if self.command_handler is None:
                    await ws.send(json.dumps({"type": "error", "error": "no command handler"}))
                    continue
                try:
                    response = await asyncio.to_thread(self.command_handler, msg, sess)
                except Exception as e:  # noqa: BLE001 — surface to client
                    response = {"type": "error", "error": f"{type(e).__name__}: {e}"}
                if response is not None:
                    await ws.send(json.dumps(response, ensure_ascii=False))
        except websockets.ConnectionClosed:
            pass
        finally:
            with self._sessions_lock:
                clients = self._sessions.get(sess)
                if clients is not None:
                    clients.discard(ws)
                    if not clients:
                        # Drop the empty set so list_sessions doesn't list a
                        # ghost session that has no live clients and no history.
                        if not self._histories.get(sess):
                            self._sessions.pop(sess, None)

    async def _fanout(self, msg: str, sess: str) -> None:
        with self._sessions_lock:
            clients = list(self._sessions.get(sess, set()))
        if not clients:
            return
        await asyncio.gather(
            *(c.send(msg) for c in clients),
            return_exceptions=True,
        )

    def _http_static(self, connection, request):
        """Serve a tiny static site at `/` so the WebSocket and viewer share a port.

        Returning `None` lets websockets perform the standard WS upgrade. Any
        non-`/events*` path is served from `WEB_DIR` as a regular HTTP response.
        Also exposes `GET /sessions` as JSON for the viewer's session picker.
        """
        split = urlsplit(request.path or "/")
        path = split.path or "/"
        if path.startswith("/events"):
            return None
        if path == "/sessions":
            body = json.dumps(self.list_sessions(), ensure_ascii=False).encode("utf-8")
            headers = Headers([
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
            ])
            return Response(200, "OK", headers, body)
        rel = path.lstrip("/") or "index.html"
        full = (WEB_DIR / rel).resolve()
        try:
            full.relative_to(WEB_DIR.resolve())
        except ValueError:
            return Response(403, "Forbidden", Headers([("Content-Type", "text/plain")]), b"forbidden\n")
        if not full.is_file():
            return Response(404, "Not Found", Headers([("Content-Type", "text/plain")]), b"not found\n")
        body = full.read_bytes()
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json; charset=utf-8",
        }.get(full.suffix, "application/octet-stream")
        headers = Headers([("Content-Type", ctype), ("Content-Length", str(len(body)))])
        return Response(200, "OK", headers, body)
