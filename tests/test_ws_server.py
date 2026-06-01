"""Regression tests for the multi-session WebSocketSink.

The ws_server underpins the live viewer; before this test the per-session
refactor was half-finished — `_handler` referenced fields that no longer
existed and `_fanout`'s signature drifted from `broadcast`'s call site —
but the prod path crashed only when an actual client connected, so the
test suite let the regression sail through. These cases pin the contract.

We use websockets' synchronous client so no asyncio test plumbing is
needed; each test picks an OS-assigned free port to avoid CI flake.
"""

from __future__ import annotations

import json
import socket
import time

import pytest
from websockets.sync.client import connect as ws_connect

from pipeline.ws_server import WebSocketSink


def _free_port() -> int:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _wait_for(predicate, timeout: float = 2.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def sink():
    """A started, host-local sink. Stop is best-effort — tests that close
    sockets fast can race the loop shutdown, but the daemon thread won't
    keep the interpreter alive."""
    port = _free_port()
    s = WebSocketSink(host="127.0.0.1", port=port)
    s.start()
    try:
        yield s
    finally:
        s.stop()


def _drain(client, count: int, timeout: float = 1.0) -> list[dict]:
    out: list[dict] = []
    deadline = time.monotonic() + timeout
    while len(out) < count and time.monotonic() < deadline:
        try:
            raw = client.recv(timeout=max(0.05, deadline - time.monotonic()))
        except TimeoutError:
            break
        out.append(json.loads(raw))
    return out


def test_session_isolation_broadcast(sink):
    """Records broadcast to session 'A' must not reach 'B' subscribers.

    Connect both viewers, push a record into A, then push a record into B.
    Each side should see only its own session's record.
    """
    url = f"ws://127.0.0.1:{sink.port}/events?session={{}}"
    with ws_connect(url.format("A")) as a, ws_connect(url.format("B")) as b:
        # Wait until the sink has both clients registered before broadcasting,
        # otherwise we race the handshake and lose messages to the void.
        assert _wait_for(lambda: len(sink._sessions.get("A", set())) == 1)
        assert _wait_for(lambda: len(sink._sessions.get("B", set())) == 1)

        sink.broadcast({"text": "for-A", "speaker": "Alice"}, session="A")
        sink.broadcast({"text": "for-B", "speaker": "Bob"}, session="B")

        msgs_a = _drain(a, 1)
        msgs_b = _drain(b, 1)

    assert len(msgs_a) == 1 and msgs_a[0]["text"] == "for-A" and msgs_a[0]["session"] == "A"
    assert len(msgs_b) == 1 and msgs_b[0]["text"] == "for-B" and msgs_b[0]["session"] == "B"


def test_history_replay_is_per_session(sink):
    """A late joiner gets exactly the prior records for the session it picks."""
    sink.broadcast({"text": "early-A", "speaker": "S1"}, session="A")
    sink.broadcast({"text": "early-B", "speaker": "S2"}, session="B")
    sink.broadcast({"text": "early-A2", "speaker": "S1"}, session="A")

    with ws_connect(f"ws://127.0.0.1:{sink.port}/events?session=A") as a:
        history = _drain(a, 2, timeout=1.5)
    assert [m["text"] for m in history] == ["early-A", "early-A2"]
    assert all(m["session"] == "A" for m in history)


def test_rewrite_history_speaker_scoped_to_session(sink):
    """Renaming a speaker in session A must not touch session B's transcript."""
    sink.broadcast({"text": "hi", "speaker": "Speaker_A"}, session="A")
    sink.broadcast({"text": "hello", "speaker": "Speaker_A"}, session="B")

    updated = sink.rewrite_history_speaker("Speaker_A", "Katie", session="A")
    assert len(updated) == 1
    assert updated[0]["session"] == "A"
    assert updated[0]["speaker"] == "Katie"
    assert updated[0]["event"] == "revised"

    # B's history must be untouched even though its speaker label was the same.
    histories_b = sink._histories.get("B", [])
    assert len(histories_b) == 1
    assert histories_b[0]["speaker"] == "Speaker_A"


def test_list_sessions_reports_clients_and_history(sink):
    sink.broadcast({"text": "x", "speaker": "S"}, session="alpha")
    sink.broadcast({"text": "y", "speaker": "S"}, session="alpha")
    sink.broadcast({"text": "z", "speaker": "S"}, session="beta")

    with ws_connect(f"ws://127.0.0.1:{sink.port}/events?session=alpha") as _:
        assert _wait_for(lambda: any(
            s["session"] == "alpha" and s["clients"] == 1 for s in sink.list_sessions()
        ))
        items = {row["session"]: row for row in sink.list_sessions()}

    assert items["alpha"]["history"] == 2
    assert items["beta"]["history"] == 1
    assert items["beta"]["clients"] == 0  # nobody subscribed to beta


def test_command_handler_receives_session(sink):
    """The session label of the *issuing* client must reach command_handler."""
    received: list[tuple[dict, str]] = []

    def handler(msg: dict, session: str) -> dict:
        received.append((msg, session))
        return {"type": "ok", "echo": msg, "session": session}

    sink.command_handler = handler
    with ws_connect(f"ws://127.0.0.1:{sink.port}/events?session=katiesteve") as a:
        a.send(json.dumps({"type": "rename", "speaker": "Speaker_A", "display_name": "Katie"}))
        reply = json.loads(a.recv(timeout=1.5))

    assert reply == {"type": "ok", "echo": {"type": "rename", "speaker": "Speaker_A", "display_name": "Katie"}, "session": "katiesteve"}
    assert len(received) == 1
    assert received[0][1] == "katiesteve"


def test_session_query_falls_back_to_default(sink):
    """A client that connects to /events with no ?session= lands on the sink default."""
    # Pre-load some history under the sink's default session so the new client gets a replay.
    sink.broadcast({"text": "default-1", "speaker": "S"}, session=None)

    with ws_connect(f"ws://127.0.0.1:{sink.port}/events") as a:
        history = _drain(a, 1, timeout=1.0)

    assert history and history[0]["session"] == sink.session_id == "default"
    assert history[0]["text"] == "default-1"


def test_sessions_endpoint_returns_json(sink):
    """`GET /sessions` should return a JSON array used by the viewer's picker."""
    import urllib.request

    sink.broadcast({"text": "a", "speaker": "S"}, session="alpha")
    sink.broadcast({"text": "b", "speaker": "S"}, session="beta")

    with urllib.request.urlopen(f"http://127.0.0.1:{sink.port}/sessions") as r:
        payload = json.loads(r.read().decode("utf-8"))

    by_name = {row["session"]: row for row in payload}
    assert "alpha" in by_name and "beta" in by_name
    assert by_name["alpha"]["history"] == 1
    assert by_name["beta"]["history"] == 1


def test_normalize_session_bounds_and_defaults():
    assert WebSocketSink._normalize_session(None) == "default"
    assert WebSocketSink._normalize_session("") == "default"
    assert WebSocketSink._normalize_session("   ") == "default"
    assert WebSocketSink._normalize_session("katie") == "katie"
    long = "x" * 200
    assert WebSocketSink._normalize_session(long) == "x" * 64
