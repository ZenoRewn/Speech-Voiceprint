"""StreamingCoordinator — drive `pipeline.streaming.stream()` from arbitrary
PCM producers (HTTP file upload, browser WebSocket, ...) instead of a CLI
generator.

The existing `stream()` function is event-driven and already handles all the
threading (Azure recognizer, voiceprint windowing, online cluster, registry
resolver). It only requires an `AudioSource` that yields 16k mono int16 PCM
chunks. Here we plug in a thread-safe queue-backed source so callers in the
api process can:

  coord = StreamingCoordinator(session="demo", hub=hub, language="en-US")
  coord.start()                 # spawns background pipeline thread
  coord.push_audio(pcm_bytes)   # called many times by the producer
  coord.end_audio()             # producer is done
  coord.stop()                  # wait for the pipeline to drain & cleanup

Events are forwarded into `SessionHub.broadcast()` so any browser client
subscribed to `session` sees them live, and `hub.history(session)` replays
on a late-joining tab.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Any

log = logging.getLogger(__name__)


class _QueueAudioSource:
    """AudioSource implementation backed by a producer/consumer queue.

    The pipeline thread calls `chunks()` (a generator) and blocks on
    `queue.get()`; the api request handler calls `push()` per inbound chunk
    and `close()` when the producer is done. `close()` is idempotent so
    callers don't need to track lifecycle perfectly — duplicate close from
    cleanup paths is fine.
    """

    _SENTINEL = object()

    def __init__(self) -> None:
        self._q: queue.Queue[Any] = queue.Queue()
        self._closed = threading.Event()
        self._lock = threading.Lock()

    def push(self, chunk: bytes) -> None:
        if self._closed.is_set() or not chunk:
            return
        self._q.put(chunk)

    def close(self) -> None:
        with self._lock:
            if self._closed.is_set():
                return
            self._closed.set()
            self._q.put(self._SENTINEL)

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def chunks(self):
        while True:
            item = self._q.get()
            if item is self._SENTINEL:
                return
            yield item


class StreamingCoordinator:
    """Lifecycle wrapper that runs the realtime pipeline inside a background
    thread and exposes push/end/stop entry points safe to call from FastAPI
    handlers (sync or async).

    `loop` is the asyncio event loop owned by the api process — required so
    we can schedule `hub.broadcast(...)` (an async coroutine) from the
    pipeline worker thread. If `loop` is None, events are still appended to
    the hub's history (sync) but live WS subscribers won't be notified.
    """

    def __init__(
        self,
        *,
        session: str,
        hub: Any,                  # pipeline.session_hub.SessionHub
        language: str = "en-US",
        languages: list[str] | None = None,
        registry_path: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        chunk_ms: int = 100,
        cluster_threshold: float = 0.7,
        window_seconds: float = 2.0,
        hop_seconds: float = 0.75,
        match_threshold: float = 0.40,
        unknown_threshold: float = 0.60,
        auto_enroll_unknown: bool = True,
        max_per_speaker: int = 5,
        # Test seam — production callers leave this as None to use the real
        # `pipeline.streaming.stream`. Tests inject a stub so we can exercise
        # push/end/stop without real Azure SDK.
        stream_fn: Any = None,
    ) -> None:
        self.session = session
        self.hub = hub
        self.languages = languages or [language]
        self.registry_path = registry_path
        self._loop = loop
        self._chunk_ms = chunk_ms
        self._common = dict(
            cluster_threshold=cluster_threshold,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
            registry_path=registry_path,
            match_threshold=match_threshold,
            unknown_threshold=unknown_threshold,
            auto_enroll_unknown=auto_enroll_unknown,
            max_per_speaker=max_per_speaker,
        )

        if stream_fn is None:
            from pipeline.streaming import stream as _stream
            stream_fn = _stream
        self._stream_fn = stream_fn

        self._source = _QueueAudioSource()
        self._resolver_holder: dict = {}
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._finished = threading.Event()
        self._error: BaseException | None = None
        # bytes pushed (for `/api/sessions/{id}/stream-status`)
        self._bytes_pushed = 0

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        if self._started.is_set():
            return
        self._started.set()
        self._thread = threading.Thread(
            target=self._run, name=f"sv-stream-{self.session}", daemon=True,
        )
        self._thread.start()

    def push_audio(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        self._bytes_pushed += len(pcm_bytes)
        self._source.push(pcm_bytes)

    def end_audio(self) -> None:
        """Signal the producer side is done. Pipeline drains then cleans up."""
        self._source.close()

    def stop(self, *, timeout: float = 30.0) -> None:
        """Block until the pipeline thread exits (or `timeout` elapses)."""
        self._source.close()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # -- introspection --------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._started.is_set() and not self._finished.is_set()

    @property
    def bytes_pushed(self) -> int:
        return self._bytes_pushed

    @property
    def error(self) -> BaseException | None:
        return self._error

    def status(self) -> dict:
        return {
            "session": self.session,
            "running": self.is_running,
            "finished": self._finished.is_set(),
            "bytes_pushed": self._bytes_pushed,
            "source_closed": self._source.closed,
            "languages": self.languages,
            "registry_path": self.registry_path,
            "error": (f"{type(self._error).__name__}: {self._error}"
                      if self._error else None),
        }

    # -- commands -------------------------------------------------------
    def handle_command(self, msg: dict) -> dict:
        """Inbound rename/forget routed from the WS subscriber.

        Mirrors `pipeline.streaming.main._command_handler` but scoped to a
        single session — the api's WS already does session routing.
        """
        kind = msg.get("type")
        resolver = self._resolver_holder.get("resolver")
        if kind == "rename":
            label = msg.get("speaker") or msg.get("old") or msg.get("label")
            new_name = (msg.get("display_name") or msg.get("new") or "").strip()
            if not label or not new_name:
                return {"type": "error", "error": "rename needs speaker + display_name"}
            if resolver is None:
                # registry not enabled — surface as a hub-level rewrite anyway
                updated = self.hub.rewrite_history_speaker(label, new_name, session=self.session)
                self._broadcast_updates(updated)
                return {
                    "type": "rename_ok", "speaker_id": None,
                    "old_label": label, "new_label": new_name,
                    "updated_records": len(updated),
                }
            speaker_id, resolved = resolver.rename(label, new_name)
            updated = self.hub.rewrite_history_speaker(label, resolved, session=self.session)
            self._broadcast_updates(updated)
            return {
                "type": "rename_ok", "speaker_id": speaker_id,
                "old_label": label, "new_label": resolved,
                "updated_records": len(updated),
            }
        if kind == "forget":
            speaker_id = msg.get("speaker_id")
            if resolver is None or not speaker_id:
                return {"type": "error", "error": "forget needs registry + speaker_id"}
            resolver.store.delete_speaker(speaker_id)
            resolver.matcher._refresh()
            return {"type": "forget_ok", "speaker_id": speaker_id}
        return {"type": "error", "error": f"unknown command {kind!r}"}

    # -- internals ------------------------------------------------------
    def _run(self) -> None:
        try:
            self._stream_fn(
                self._source,
                self.languages,
                chunk_ms=self._chunk_ms,
                on_event=self._publish,
                resolver_holder=self._resolver_holder,
                **self._common,
            )
        except BaseException as e:  # noqa: BLE001
            self._error = e
            log.exception("StreamingCoordinator session=%s crashed", self.session)
        finally:
            self._finished.set()
            # Tell subscribers the stream is over — the dashboard hides the
            # 'live' badge once it sees this.
            self._publish({"event": "stream_end", "ok": self._error is None,
                           "error": (str(self._error) if self._error else None)})

    def _publish(self, rec: dict) -> None:
        """Forward an event into the SessionHub.

        Pipeline thread → asyncio loop hop. We schedule `hub.broadcast()`
        on the api's loop with `run_coroutine_threadsafe`. If no loop is
        set (e.g. tests), fall back to `hub.append()` for sync history.
        """
        if self._loop is not None and self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(
                self.hub.broadcast(rec, self.session), self._loop,
            )
            try:
                fut.result(timeout=2.0)
            except Exception:  # noqa: BLE001
                # Don't let a broadcast hiccup take down the pipeline.
                log.warning("hub.broadcast failed for session=%s", self.session)
        else:
            self.hub.append(rec, self.session)

    def _broadcast_updates(self, updated: list[dict]) -> None:
        for rec in updated:
            self._publish(rec)


# ---------------------------------------------------------------------------
# Process-wide registry of live coordinators, keyed by session label.
# `pipeline.api` looks up coordinators here when `/ws/ingest` or
# `POST /api/sessions/{id}/stream-file` arrives, so the same StreamingHub
# instance is reused even if multiple WS clients connect mid-stream.

class CoordinatorRegistry:
    """Thread-safe map of session -> StreamingCoordinator.

    `acquire` returns the existing coordinator if one is running for the
    session, otherwise creates+starts a fresh one. `release` removes the
    entry once the coordinator finishes (call from the api after `stop()`).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._table: dict[str, StreamingCoordinator] = {}

    def get(self, session: str) -> StreamingCoordinator | None:
        with self._lock:
            return self._table.get(session)

    def register(self, coord: StreamingCoordinator) -> None:
        with self._lock:
            existing = self._table.get(coord.session)
            if existing is not None and existing.is_running:
                raise RuntimeError(f"session {coord.session!r} already streaming")
            self._table[coord.session] = coord

    def release(self, session: str) -> None:
        with self._lock:
            self._table.pop(session, None)

    def list_active(self) -> list[dict]:
        with self._lock:
            return [c.status() for c in self._table.values()]
