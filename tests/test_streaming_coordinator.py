"""Tests for `pipeline.streaming_coordinator.StreamingCoordinator`.

We don't exercise the real Azure recognizer — the coordinator's job is
plumbing (queue source, lifecycle, hub forwarding, command routing). A
stub `stream()` consumes chunks from the source and emits a synthetic
event per chunk so we can verify push_audio/end_audio/stop wiring.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from pipeline.session_hub import SessionHub
from pipeline.streaming_coordinator import (
    CoordinatorRegistry,
    StreamingCoordinator,
    _QueueAudioSource,
)


# ---------------------------------------------------------------------------
def _stub_stream(source, languages, *, on_event=None, resolver_holder=None, **_kw):
    """Drain the source and emit one event per chunk seen."""
    idx = 0
    for chunk in source.chunks():
        if on_event:
            on_event({
                "start": float(idx), "end": float(idx + 1),
                "speaker": "Speaker_A", "speaker_confidence": 0.9,
                "azure_speaker": "Guest-1", "cluster_label": "Speaker_A",
                "text": f"chunk-{idx} ({len(chunk)}B)",
                "event": "final",
            })
        idx += 1


# ---------------------------------------------------------------------------
def test_queue_source_yields_pushed_chunks_then_stops_on_close():
    src = _QueueAudioSource()
    out: list[bytes] = []

    def consumer():
        for c in src.chunks():
            out.append(c)

    t = threading.Thread(target=consumer)
    t.start()

    src.push(b"\x00\x01")
    src.push(b"\x02\x03")
    src.close()
    t.join(timeout=2.0)

    assert out == [b"\x00\x01", b"\x02\x03"]
    assert src.closed


def test_queue_source_close_is_idempotent():
    src = _QueueAudioSource()
    src.close()
    src.close()  # must not raise / deadlock


def test_queue_source_drops_pushes_after_close():
    src = _QueueAudioSource()
    src.close()
    src.push(b"late")  # dropped — no chunks generator is running anyway
    out = list(src.chunks())
    assert out == []


# ---------------------------------------------------------------------------
def test_coordinator_forwards_chunks_through_stub_to_hub():
    hub = SessionHub()
    coord = StreamingCoordinator(
        session="t1", hub=hub, language="en-US",
        stream_fn=_stub_stream, loop=None,
    )
    coord.start()
    coord.push_audio(b"\x00" * 320)
    coord.push_audio(b"\x01" * 320)
    coord.end_audio()
    coord.stop(timeout=5.0)

    history = hub.history("t1")
    # 2 chunk events + 1 stream_end marker
    assert len(history) == 3
    assert history[0]["text"].startswith("chunk-0")
    assert history[1]["text"].startswith("chunk-1")
    assert history[2]["event"] == "stream_end"
    assert history[2]["ok"] is True
    assert coord.bytes_pushed == 640


def test_coordinator_status_reflects_lifecycle():
    hub = SessionHub()
    coord = StreamingCoordinator(
        session="t2", hub=hub, stream_fn=_stub_stream, loop=None,
    )
    assert coord.status()["running"] is False

    coord.start()
    assert coord.is_running
    assert coord.status()["running"] is True

    coord.end_audio()
    coord.stop(timeout=5.0)
    s = coord.status()
    assert s["running"] is False
    assert s["finished"] is True
    assert s["error"] is None


def test_coordinator_captures_stream_errors():
    def crashing_stream(source, languages, **kw):
        # Drain at least one chunk so push_audio reaches the queue, then die.
        for _ in source.chunks():
            raise RuntimeError("simulated azure failure")

    hub = SessionHub()
    coord = StreamingCoordinator(
        session="t3", hub=hub, stream_fn=crashing_stream, loop=None,
    )
    coord.start()
    coord.push_audio(b"\x00" * 320)
    coord.end_audio()
    coord.stop(timeout=5.0)

    assert coord.error is not None
    assert "simulated azure failure" in str(coord.error)
    history = hub.history("t3")
    # Final event is a stream_end marker with ok=False
    assert history[-1]["event"] == "stream_end"
    assert history[-1]["ok"] is False
    assert "simulated" in (history[-1]["error"] or "")


def test_coordinator_publishes_via_asyncio_loop_when_provided():
    """When the api's event loop is available, broadcasts should reach
    subscribers (not just the history)."""
    hub = SessionHub()
    received: list[str] = []
    loop_holder: dict = {}
    loop_ready = threading.Event()
    stop_loop = threading.Event()

    def run_loop():
        loop = asyncio.new_event_loop()
        loop_holder["loop"] = loop
        asyncio.set_event_loop(loop)
        loop_ready.set()
        try:
            loop.run_until_complete(_idle_until(stop_loop))
        finally:
            loop.close()

    async def _idle_until(stop_event):
        while not stop_event.is_set():
            await asyncio.sleep(0.02)

    async def sub(msg):
        received.append(msg)

    t = threading.Thread(target=run_loop, daemon=True)
    t.start()
    loop_ready.wait(timeout=2.0)

    hub.subscribe("t4", sub)

    coord = StreamingCoordinator(
        session="t4", hub=hub, stream_fn=_stub_stream,
        loop=loop_holder["loop"],
    )
    coord.start()
    coord.push_audio(b"\x00" * 320)
    coord.end_audio()
    coord.stop(timeout=5.0)

    # Give the loop a moment to flush the last few coroutines.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and len(received) < 2:
        time.sleep(0.05)
    stop_loop.set()
    t.join(timeout=2.0)

    # 1 chunk + stream_end
    assert len(received) >= 2


# ---------------------------------------------------------------------------
def test_handle_command_rename_rewrites_history_without_registry():
    hub = SessionHub()
    coord = StreamingCoordinator(
        session="t5", hub=hub, stream_fn=_stub_stream, loop=None,
    )
    coord.start()
    coord.push_audio(b"\x00" * 320)
    coord.end_audio()
    coord.stop(timeout=5.0)

    reply = coord.handle_command({
        "type": "rename", "speaker": "Speaker_A", "display_name": "Alice",
    })
    assert reply["type"] == "rename_ok"
    assert reply["new_label"] == "Alice"
    assert reply["updated_records"] >= 1
    history = hub.history("t5")
    assert any(r.get("speaker") == "Alice" for r in history)


def test_handle_command_unknown_kind_returns_error():
    hub = SessionHub()
    coord = StreamingCoordinator(
        session="t6", hub=hub, stream_fn=_stub_stream, loop=None,
    )
    reply = coord.handle_command({"type": "delete_universe"})
    assert reply["type"] == "error"


def test_handle_command_rename_validates_inputs():
    hub = SessionHub()
    coord = StreamingCoordinator(
        session="t7", hub=hub, stream_fn=_stub_stream, loop=None,
    )
    reply = coord.handle_command({"type": "rename"})
    assert reply["type"] == "error"


# ---------------------------------------------------------------------------
def test_coordinator_registry_prevents_concurrent_streams_on_same_session():
    reg = CoordinatorRegistry()
    hub = SessionHub()
    c1 = StreamingCoordinator(session="dup", hub=hub, stream_fn=_stub_stream)
    c1.start()
    reg.register(c1)
    try:
        c2 = StreamingCoordinator(session="dup", hub=hub, stream_fn=_stub_stream)
        with pytest.raises(RuntimeError, match="already streaming"):
            reg.register(c2)
    finally:
        c1.end_audio()
        c1.stop(timeout=5.0)
        reg.release("dup")
    assert reg.get("dup") is None


def test_coordinator_registry_lists_active_sessions():
    reg = CoordinatorRegistry()
    hub = SessionHub()
    c = StreamingCoordinator(session="s_active", hub=hub, stream_fn=_stub_stream)
    c.start()
    reg.register(c)
    try:
        active = reg.list_active()
        assert any(s["session"] == "s_active" and s["running"] for s in active)
    finally:
        c.end_audio()
        c.stop(timeout=5.0)
        reg.release("s_active")
