"""Tests for `pipeline.api`.

Coverage:
  - health route happy path
  - bearer auth: enabled vs. disabled
  - registry CRUD round-trip
  - session history endpoint reflects hub broadcasts
  - WS endpoint replays history then receives live broadcasts
  - jobs queue: submit → poll → done (with monkey-patched orchestrator)
  - schema endpoint returns a JSON Schema with `utterances`

We deliberately don't spin uvicorn here — `TestClient` from starlette runs
the ASGI app directly, which covers the WS path. For real network tests,
`tests/test_ws_server.py` already exercises the daemon-thread variant.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pipeline.api import AppConfig, create_app
from registry.store import RegistryStore
from stt.base import STTResult, Utterance, Word
from voiceprint.base import SpeakerSegment


# ---------------------------------------------------------------------------
# Stub providers shared with test_sdk.py — we stub at the orchestrator level
# rather than mocking the entire SDK so the api → SDK → orchestrator wiring
# is exercised, only the heavy provider deps are swapped out.

class _StubFast:
    def __init__(self, *_a, **_kw) -> None:
        pass

    def transcribe_fast(self, audio_path: str, languages=None) -> STTResult:
        return STTResult(
            utterances=[
                Utterance(
                    text="hello world",
                    start=0.0, end=1.0,
                    words=[Word("hello", 0.0, 0.5), Word("world", 0.5, 1.0)],
                )
            ],
            language="en-US", duration=1.0,
        )


class _StubVoiceprint:
    name = "stub"
    embedding_dim = 4

    def diarize(self, audio_path: str, num_speakers=None):
        return [
            SpeakerSegment(0.0, 0.6, "Speaker_A"),
            SpeakerSegment(0.6, 1.0, "Speaker_B"),
        ]

    def embed(self, audio_path, start, end):
        return np.zeros(self.embedding_dim, dtype=np.float32)


@pytest.fixture
def app_no_auth(tmp_path, monkeypatch):
    # Force-clear any inherited token so this fixture is unambiguous.
    monkeypatch.delenv("SV_API_TOKEN", raising=False)
    cfg = AppConfig(registry_path=str(tmp_path / "registry.db"))
    return create_app(cfg)


@pytest.fixture
def app_with_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("SV_API_TOKEN", "secret-test-token")
    cfg = AppConfig(registry_path=str(tmp_path / "registry.db"))
    return create_app(cfg)


@pytest.fixture
def stubbed_orchestrator(monkeypatch):
    """Patch the orchestrator so submitted jobs run on stubs (no Azure calls)."""
    from pipeline import orchestrator

    monkeypatch.setattr(orchestrator, "AzureFastTranscription", _StubFast)
    monkeypatch.setattr(orchestrator, "build_voiceprint_provider", lambda *a, **kw: _StubVoiceprint())


# ---------------------------------------------------------------------------
def test_health_returns_ok(app_no_auth):
    with TestClient(app_no_auth) as client:
        r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["auth_enabled"] is False
    assert isinstance(data["sessions"], int)


def test_auth_required_when_token_set(app_with_auth):
    with TestClient(app_with_auth) as client:
        r = client.get("/api/health")
        assert r.status_code == 401
        r = client.get("/api/health", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401
        r = client.get("/api/health", headers={"Authorization": "Bearer secret-test-token"})
        assert r.status_code == 200


def test_container_probes_skip_auth(app_with_auth):
    with TestClient(app_with_auth) as client:
        assert client.get("/livez").status_code == 200
        assert client.get("/readyz").status_code == 200


def test_schema_endpoint_describes_pipeline_result(app_no_auth):
    with TestClient(app_no_auth) as client:
        r = client.get("/api/schema")
    assert r.status_code == 200
    schema = r.json()
    assert schema["title"] == "PipelineResult"
    assert "utterances" in schema["properties"]


def test_registry_crud_round_trip(app_no_auth):
    cfg: AppConfig = app_no_auth.state.cfg
    # Pre-seed a speaker so list isn't empty.
    with RegistryStore(path=cfg.registry_path) as store:
        sp = store.create_speaker(display_name="Alice")

    with TestClient(app_no_auth) as client:
        r = client.get("/api/registry/speakers")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["display_name"] == "Alice"

        r = client.patch(
            f"/api/registry/speakers/{sp.id}",
            json={"display_name": "Alicia"},
        )
        assert r.status_code == 200
        assert r.json()["display_name"] == "Alicia"

        r = client.delete(f"/api/registry/speakers/{sp.id}")
        assert r.status_code == 204

        r = client.get(f"/api/registry/speakers/{sp.id}")
        assert r.status_code == 404


def test_session_event_post_appears_in_history(app_no_auth):
    with TestClient(app_no_auth) as client:
        rec = {"start": 0.0, "end": 1.0, "speaker": "Speaker_A", "text": "hi"}
        r = client.post("/api/sessions/test/events", json=rec)
        assert r.status_code == 200

        r = client.get("/api/sessions/test/history")
        assert r.status_code == 200
        history = r.json()
        assert len(history) == 1
        assert history[0]["text"] == "hi"
        assert history[0]["session"] == "test"


def test_ws_replays_history_and_receives_live(app_no_auth):
    """A late-joining WS client should replay the session's prior events
    and then keep receiving new broadcasts."""
    with TestClient(app_no_auth) as client:
        # Pre-seed.
        client.post("/api/sessions/live/events", json={"speaker": "A", "text": "hi"})
        with client.websocket_connect("/ws/events?session=live") as ws:
            replay = ws.receive_text()
            replay_msg = json.loads(replay)
            assert replay_msg["text"] == "hi"
            # Now publish a new event after the WS is connected.
            client.post("/api/sessions/live/events", json={"speaker": "B", "text": "yo"})
            live = ws.receive_text()
            live_msg = json.loads(live)
            assert live_msg["text"] == "yo"
            assert live_msg["session"] == "live"


def test_ws_rename_rewrites_history(app_no_auth):
    with TestClient(app_no_auth) as client:
        client.post("/api/sessions/r/events", json={"speaker": "Speaker_A", "text": "one"})
        client.post("/api/sessions/r/events", json={"speaker": "Speaker_A", "text": "two"})

        with client.websocket_connect("/ws/events?session=r") as ws:
            ws.receive_text(); ws.receive_text()  # drain initial replay
            ws.send_text(json.dumps({"type": "rename", "old": "Speaker_A", "new": "Alice"}))
            # Server fans out two `revised` records, then acks.
            seen = []
            for _ in range(3):
                seen.append(json.loads(ws.receive_text()))
            ack = next(m for m in seen if m.get("type") == "ack")
            assert ack["op"] == "rename"
            assert ack["updated"] == 2

        r = client.get("/api/sessions/r/history")
        assert all(rec["speaker"] == "Alice" for rec in r.json())


def test_submit_job_runs_through_stubs(app_no_auth, stubbed_orchestrator, tmp_path):
    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"")

    with TestClient(app_no_auth) as client:
        r = client.post(
            "/api/jobs/transcribe",
            json={"mode": "fast", "audio_path": str(audio), "backend": "speechbrain"},
        )
        assert r.status_code == 200
        job_id = r.json()["job_id"]

        # Job runs in the threadpool; poll briefly.
        for _ in range(50):
            r = client.get(f"/api/jobs/{job_id}")
            assert r.status_code == 200
            payload = r.json()
            if payload["status"] in ("done", "failed"):
                break
            import time as _t
            _t.sleep(0.02)

        assert payload["status"] == "done", payload
        assert payload["result"]["voiceprint_backend"] == "stub"
        assert len(payload["result"]["utterances"]) == 1


def test_submit_job_validates_required_fields(app_no_auth):
    with TestClient(app_no_auth) as client:
        r = client.post("/api/jobs/transcribe", json={"mode": "fast"})  # no audio_path
        assert r.status_code == 400
        r = client.post("/api/jobs/transcribe", json={"mode": "batch"})  # no audio_url
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Browser-driven Live ingest (Phase B)

def _stub_streaming_pipeline(source, languages, *, on_event=None, resolver_holder=None, **_):
    """Stand-in for `pipeline.streaming.stream`. Drains the source, emits
    one synthetic event per chunk."""
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


def _make_wav_bytes(seconds: float = 0.4, sr: int = 16000) -> bytes:
    """Build a minimal 16k mono int16 WAV in memory."""
    import io
    import wave

    n = int(seconds * sr)
    pcm = (np.sin(2 * np.pi * 440 * np.arange(n) / sr) * 16000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


@pytest.fixture
def app_with_stub_stream(tmp_path, monkeypatch):
    monkeypatch.delenv("SV_API_TOKEN", raising=False)
    cfg = AppConfig(registry_path=str(tmp_path / "registry.db"))
    app = create_app(cfg)
    app.state.stream_fn = _stub_streaming_pipeline
    return app


def test_stream_file_drives_pipeline_and_emits_history(app_with_stub_stream):
    wav = _make_wav_bytes(seconds=0.3)
    with TestClient(app_with_stub_stream) as client:
        r = client.post(
            "/api/sessions/livefile/stream-file",
            files={"upload": ("clip.wav", wav, "audio/wav")},
            data={"language": "en-US", "realtime_pacing": "false"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["session"] == "livefile"
        assert r.json()["status"] == "streaming"

        # The feed thread runs as fast as it can with realtime_pacing=False;
        # poll until status drops out of running. After completion the
        # coordinator is released from the registry, so status returns the
        # "no coord" shape.
        import time as _t
        for _ in range(200):
            s = client.get("/api/sessions/livefile/stream-status").json()
            if not s.get("running", True):
                break
            _t.sleep(0.05)
        assert s.get("running") is False, s

        history = client.get("/api/sessions/livefile/history").json()
        # at least: 1 chunk event + the stream_end marker
        assert len(history) >= 2
        assert history[-1]["event"] == "stream_end"
        assert history[-1]["ok"] is True


def test_stream_file_rejects_concurrent_session(app_with_stub_stream):
    """If a session is already streaming, a second `stream-file` POST 409s."""
    # Inject a coordinator already running for `dup`.
    from pipeline.streaming_coordinator import StreamingCoordinator

    coords = app_with_stub_stream.state.coords
    blocking = StreamingCoordinator(
        session="dup", hub=app_with_stub_stream.state.hub,
        stream_fn=lambda *a, **kw: __import__("time").sleep(2),
    )
    blocking.start()
    coords.register(blocking)
    try:
        with TestClient(app_with_stub_stream) as client:
            r = client.post(
                "/api/sessions/dup/stream-file",
                files={"upload": ("clip.wav", _make_wav_bytes(), "audio/wav")},
            )
            assert r.status_code == 409
    finally:
        blocking.end_audio()
        blocking.stop(timeout=5.0)
        coords.release("dup")


def test_stream_status_returns_idle_for_unknown_session(app_with_stub_stream):
    with TestClient(app_with_stub_stream) as client:
        r = client.get("/api/sessions/never_started/stream-status")
        assert r.status_code == 200
        assert r.json() == {"session": "never_started", "running": False}


def test_ws_ingest_accepts_binary_pcm_and_emits_events(app_with_stub_stream):
    with TestClient(app_with_stub_stream) as client:
        with client.websocket_connect("/ws/ingest?session=mic1") as ws:
            ready = json.loads(ws.receive_text())
            assert ready["type"] == "ingest_ready"
            assert ready["session"] == "mic1"

            ws.send_bytes(b"\x00\x01" * 1600)  # 200 ms of fake PCM
            ws.send_bytes(b"\x02\x03" * 1600)
            ws.send_text("stop")

        # After the WS closes the coordinator drains in a worker thread.
        import time as _t
        for _ in range(50):
            s = client.get("/api/sessions/mic1/stream-status").json()
            if not s.get("running", False):
                break
            _t.sleep(0.05)

        history = client.get("/api/sessions/mic1/history").json()
        # 2 chunks + stream_end
        assert len(history) >= 3
        assert history[-1]["event"] == "stream_end"


def test_ws_ingest_routes_rename_command_to_coordinator(app_with_stub_stream):
    with TestClient(app_with_stub_stream) as client:
        with client.websocket_connect("/ws/ingest?session=cmd1") as ws:
            json.loads(ws.receive_text())  # ready
            ws.send_bytes(b"\x00\x01" * 1600)
            # Wait until the chunk event lands in history.
            import time as _t
            for _ in range(40):
                if client.get("/api/sessions/cmd1/history").json():
                    break
                _t.sleep(0.05)

            ws.send_text(json.dumps({
                "type": "rename", "speaker": "Speaker_A", "display_name": "Bob",
            }))
            reply = json.loads(ws.receive_text())
            assert reply["type"] == "rename_ok"
            assert reply["new_label"] == "Bob"

            ws.send_text("stop")


def test_ready_flips_to_503_after_lifespan_shutdown(app_no_auth):
    """Readiness probe must report 503 once SIGTERM-driven drain begins."""
    with TestClient(app_no_auth) as client:
        r = client.get("/api/ready")
        assert r.status_code == 200
        assert r.json()["ready"] is True
    # After context exits, lifespan shutdown has flipped ready=False.
    assert app_no_auth.state.ready is False


def test_ready_endpoint_skips_auth(app_with_auth):
    """Probes shouldn't need a bearer token. /api/ready stays auth-free."""
    with TestClient(app_with_auth) as client:
        r = client.get("/api/ready")
        assert r.status_code == 200


def test_maintenance_usage_lists_all_scopes(app_no_auth, tmp_path, monkeypatch):
    monkeypatch.setenv("SV_DATA_DIR", str(tmp_path))
    # Re-create the app so it picks up the new env-driven paths.
    from pipeline.api import AppConfig, create_app
    monkeypatch.delenv("SV_API_TOKEN", raising=False)
    app = create_app(AppConfig(registry_path=str(tmp_path / "reg.db")))
    # Seed a file so uploads count > 0.
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "a.wav").write_bytes(b"hello")

    with TestClient(app) as client:
        r = client.get("/api/maintenance/usage")
        assert r.status_code == 200
        data = r.json()
        for scope in ("uploads", "stream", "outputs", "jobs", "sessions"):
            assert scope in data
        assert data["uploads"]["count"] == 1
        assert data["uploads"]["bytes"] == 5


def test_maintenance_cleanup_dry_run_doesnt_delete(app_no_auth, tmp_path, monkeypatch):
    monkeypatch.setenv("SV_DATA_DIR", str(tmp_path))
    from pipeline.api import AppConfig, create_app
    monkeypatch.delenv("SV_API_TOKEN", raising=False)
    app = create_app(AppConfig(registry_path=str(tmp_path / "reg.db")))
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "a.wav").write_bytes(b"hello")
    (uploads / "b.wav").write_bytes(b"world!")

    with TestClient(app) as client:
        r = client.post("/api/maintenance/cleanup",
                        json={"scope": "uploads", "dry_run": True})
        assert r.status_code == 200
        body = r.json()
        assert body["deleted"] == 2
        assert body["bytes_freed"] == len(b"hello") + len(b"world!")
        # Files still on disk.
        assert (uploads / "a.wav").exists()

        r = client.post("/api/maintenance/cleanup",
                        json={"scope": "uploads", "dry_run": False})
        assert r.status_code == 200
        assert r.json()["deleted"] == 2
        assert not (uploads / "a.wav").exists()
        assert not (uploads / "b.wav").exists()


def test_job_download_returns_persisted_json(app_no_auth, stubbed_orchestrator, tmp_path):
    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"")

    with TestClient(app_no_auth) as client:
        r = client.post("/api/jobs/transcribe",
                        json={"mode": "fast", "audio_path": str(audio), "backend": "speechbrain"})
        job_id = r.json()["job_id"]

        for _ in range(50):
            r = client.get(f"/api/jobs/{job_id}")
            if r.json()["status"] in ("done", "failed"):
                break
            import time as _t; _t.sleep(0.02)

        r = client.get(f"/api/jobs/{job_id}/download")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")
        assert b"voiceprint_backend" in r.content


def test_delete_job_drops_record_and_file(app_no_auth, stubbed_orchestrator, tmp_path):
    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"")

    with TestClient(app_no_auth) as client:
        r = client.post("/api/jobs/transcribe",
                        json={"mode": "fast", "audio_path": str(audio), "backend": "speechbrain"})
        job_id = r.json()["job_id"]
        for _ in range(50):
            r = client.get(f"/api/jobs/{job_id}")
            if r.json()["status"] in ("done", "failed"):
                break
            import time as _t; _t.sleep(0.02)

        r = client.delete(f"/api/jobs/{job_id}")
        assert r.status_code == 204

        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 404


def test_submit_job_respects_auto_enroll_unknown_flag(app_no_auth, monkeypatch, tmp_path):
    """auto_enroll_unknown=False must reach transcribe_file unchanged."""
    captured = {}

    def fake_transcribe_file(audio_path, **kwargs):
        captured.update(kwargs)
        captured["audio_path"] = audio_path
        # Return a minimal valid result.
        from schemas.output import PipelineResult
        return PipelineResult.from_payload({
            "audio": str(audio_path),
            "voiceprint_backend": "stub",
            "language": "en-US",
            "utterances": [],
            "duration": 0.0,
        })

    import pipeline as pkg
    monkeypatch.setattr(pkg, "transcribe_file", fake_transcribe_file)

    audio = tmp_path / "x.wav"
    audio.write_bytes(b"")

    with TestClient(app_no_auth) as client:
        r = client.post("/api/jobs/transcribe", json={
            "mode": "fast",
            "audio_path": str(audio),
            "backend": "speechbrain",
            "auto_enroll_unknown": False,
        })
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        for _ in range(50):
            r = client.get(f"/api/jobs/{job_id}")
            if r.json()["status"] in ("done", "failed"):
                break
            import time as _t; _t.sleep(0.02)
        assert r.json()["status"] == "done"

    assert captured.get("auto_enroll_unknown") is False


def test_comparison_id_round_trips_through_listing(app_no_auth, stubbed_orchestrator, tmp_path):
    """When the dashboard groups two backends under one comparison_id, both
    JobRecords must echo it back so the listing can render them as a pair."""
    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"")
    cmp_id = "cmp_demo_42"

    with TestClient(app_no_auth) as client:
        r1 = client.post("/api/jobs/transcribe", json={
            "mode": "fast", "audio_path": str(audio),
            "backend": "speechbrain", "comparison_id": cmp_id,
        })
        r2 = client.post("/api/jobs/transcribe", json={
            "mode": "fast", "audio_path": str(audio),
            "backend": "pyannote", "comparison_id": cmp_id,
        })
        assert r1.status_code == r2.status_code == 200
        assert r1.json()["comparison_id"] == cmp_id
        assert r2.json()["comparison_id"] == cmp_id

        listing = client.get("/api/jobs").json()
        ours = [j for j in listing if j["comparison_id"] == cmp_id]
        assert len(ours) == 2
        backends = {j["request"]["backend"] for j in ours}
        assert backends == {"speechbrain", "pyannote"}
