"""FastAPI surface for speech-voiceprint.

Brings together:
  - REST: health, schema, sessions/history, registry CRUD, jobs
  - WebSocket: `/ws/events?session=<id>` — same protocol as the legacy
    `pipeline.ws_server` viewer endpoint (replay → live → command echo)
  - Static: `/` and `/static/*` serve the bundled web management UI

Auth: if `SV_API_TOKEN` is set in env, every route (REST + WS) requires
`Authorization: Bearer <token>`. Unset → no auth (local dev).

Run as: `python -m pipeline.api --port 8080 --registry /path/to/speakers.db`
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline.logging_setup import configure_logging
from pipeline.paths import DataPaths, get_paths
from pipeline.session_hub import SessionHub, normalize_session
from pipeline.streaming_coordinator import (
    CoordinatorRegistry,
    StreamingCoordinator,
)
from schemas import PipelineResult

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
STATIC_DIR = WEB_DIR / "static"
UPLOAD_CHUNK_BYTES = 1024 * 1024

# ---------------------------------------------------------------------------
# Config the app picks up on startup. Routes use `request.app.state.cfg`.

class AppConfig:
    def __init__(
        self,
        *,
        registry_path: str | None = None,
        api_token: str | None = None,
        worker_count: int = 2,
        cors_origins: tuple[str, ...] = (),
    ) -> None:
        # Resolution order: explicit arg > SV_REGISTRY_PATH env > data/registry/speakers.db default.
        # Falling back to the default keeps a single-binary launch (./start.sh) working
        # without forcing every user to discover the env var first.
        resolved = registry_path or os.environ.get("SV_REGISTRY_PATH") or None
        if not resolved:
            from pipeline.paths import get_paths
            paths = get_paths()
            paths.ensure("registry")
            resolved = str(paths.default_registry_db())
        self.registry_path = resolved
        self.api_token = api_token if api_token is not None else os.environ.get("SV_API_TOKEN") or None
        self.worker_count = worker_count
        self.cors_origins = cors_origins


# ---------------------------------------------------------------------------
# Job state (in-memory). One process owns it; that's the deployment we ship.

class JobRecord(BaseModel):
    job_id: str
    mode: str
    status: str = "queued"  # queued | running | done | failed
    submitted_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    log_lines: list[str] = Field(default_factory=list)
    result: dict | None = None  # PipelineResult.model_dump(exclude_none=True)
    request: dict = Field(default_factory=dict)
    # Optional grouping key. When the dashboard submits the same audio against
    # both backends, both jobs share this id so the UI can render a side-by-side
    # comparison. Server only persists/echoes the id; it doesn't auto-spawn.
    comparison_id: str | None = None


class JobStore:
    """Bounded in-memory job table.

    Single-process job state. Writes are serialized on the executor thread
    plus the api thread; we use a lock for clarity even though the GIL would
    technically suffice for dict mutations.
    """

    def __init__(self, *, max_jobs: int = 200) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    def submit(self, mode: str, request: dict, *, comparison_id: str | None = None) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        rec = JobRecord(
            job_id=job_id, mode=mode, status="queued",
            submitted_at=time.time(), request=request,
            comparison_id=comparison_id,
        )
        with self._lock:
            self._jobs[job_id] = rec
            self._order.append(job_id)
            if len(self._order) > self._max_jobs:
                evict = self._order.pop(0)
                self._jobs.pop(evict, None)
        return rec

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return
            for k, v in fields.items():
                setattr(rec, k, v)

    def append_log(self, job_id: str, line: str) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec:
                rec.log_lines.append(line)
                # Bound to the most recent 500 lines per job.
                if len(rec.log_lines) > 500:
                    rec.log_lines = rec.log_lines[-500:]

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_(self) -> list[JobRecord]:
        with self._lock:
            return [self._jobs[j] for j in reversed(self._order) if j in self._jobs]

    def delete(self, job_id: str) -> bool:
        with self._lock:
            if job_id not in self._jobs:
                return False
            self._jobs.pop(job_id, None)
            try:
                self._order.remove(job_id)
            except ValueError:
                pass
            return True


# ---------------------------------------------------------------------------
# Auth dependency

def _auth_dep(authorization: str | None = Header(default=None)):
    """Bearer token gate. Returns the validated token (or None when disabled)."""
    expected = os.environ.get("SV_API_TOKEN")
    if not expected:
        return None  # auth disabled (local dev)
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.split(None, 1)[1].strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="invalid bearer token")
    return token


def _safe_upload_name(name: str | None, fallback: str) -> str:
    base = Path(name or fallback).name or fallback
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)[:160] or fallback


async def _save_upload(upload: UploadFile, target: Path, *, max_bytes: int | None = None) -> int:
    total = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                try:
                    target.unlink()
                except OSError:
                    pass
                raise HTTPException(status_code=413, detail=f"upload exceeds {max_bytes} bytes")
            f.write(chunk)
    return total


def _max_upload_bytes() -> int:
    raw = os.environ.get("SV_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024))
    try:
        return max(1, int(raw))
    except ValueError:
        return 200 * 1024 * 1024


# ---------------------------------------------------------------------------
# Pydantic request/response models for routes

class SpeakerSummary(BaseModel):
    id: str
    display_name: str | None
    voiceprint_count: int
    models: list[str] = Field(default_factory=list)
    created_at: float | None = None
    updated_at: float | None = None


class RenameSpeakerBody(BaseModel):
    display_name: str = Field(min_length=1, max_length=128)


class SubmitJobBody(BaseModel):
    mode: str = Field(pattern="^(fast|batch)$")
    audio_path: Optional[str] = None
    audio_url: Optional[str] = None
    backend: str = Field(default="speechbrain", pattern="^(speechbrain|pyannote)$")
    languages: list[str] = Field(default_factory=list)
    num_speakers: Optional[int] = None
    majority_threshold: float = 0.7
    device: str = "auto"
    registry_path: Optional[str] = None
    # Match-only vs match-and-enroll. When False, voiceprint matching still
    # tags utterances with known speakers (KNOWN/LOW_CONFIDENCE) but unknown
    # voices stay as session-local labels and are NOT inserted into the
    # registry. Use this for read-only experiments / one-off transcriptions
    # that shouldn't pollute the persistent identity store.
    auto_enroll_unknown: bool = True
    # Cross-backend enrollment: after the primary backend resolves speakers,
    # also run the *other* backend on the same audio and write its embeddings
    # under the same speaker_ids (matched by time overlap). One identity ends
    # up addressable by both speechbrain and pyannote queries.
    dual_enroll: bool = False
    # Optional grouping key shared between two jobs that compare backends on
    # the same audio. The frontend mints a UUID and sends both submissions
    # with the same value; server echoes it on the JobRecord.
    comparison_id: Optional[str] = None


class HealthResponse(BaseModel):
    ok: bool = True
    version: str = "0.1.0"
    registry_path: Optional[str]
    auth_enabled: bool
    workers: int
    sessions: int
    jobs: int


class CleanupBody(BaseModel):
    scope: str = Field(pattern="^(uploads|stream|outputs|jobs|sessions)$")
    older_than_days: Optional[int] = None
    dry_run: bool = False


# ---------------------------------------------------------------------------
# App factory

def create_app(cfg: AppConfig | None = None) -> FastAPI:
    configure_logging()
    cfg = cfg or AppConfig()
    hub = SessionHub()
    jobs = JobStore()
    coords = CoordinatorRegistry()
    job_futures: set[Future] = set()
    job_futures_lock = threading.Lock()
    executor = ThreadPoolExecutor(
        max_workers=cfg.worker_count, thread_name_prefix="sv-job",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info(
            "api starting",
            extra={"registry_path": cfg.registry_path, "auth_enabled": bool(cfg.api_token)},
        )
        # Mark ready as soon as routes are mounted. Heavy model loading is
        # lazy (per-job); /api/ready stays green from t=0 because the app can
        # accept requests — model warm-up happens inside individual jobs and
        # is reported via job status, not via readiness.
        app.state.ready = True
        yield
        log.info("api draining")
        app.state.ready = False  # Stop reporting healthy once SIGTERM lands.
        # Give in-flight jobs a window to finish before we cut the executor.
        # Configurable via SV_SHUTDOWN_TIMEOUT (seconds), default 60.
        timeout = float(os.environ.get("SV_SHUTDOWN_TIMEOUT", "60"))
        log.info("api waiting up to %.0fs for in-flight jobs", timeout)
        deadline = time.time() + max(0.0, timeout)
        while True:
            with job_futures_lock:
                pending = [f for f in job_futures if not f.done()]
            if not pending or time.time() >= deadline:
                break
            await asyncio.sleep(0.2)
        with job_futures_lock:
            pending = [f for f in job_futures if not f.done()]
        if pending:
            for fut in pending:
                fut.cancel()
            log.warning("api shutdown timeout reached; %d job(s) still running", len(pending))
        executor.shutdown(wait=False, cancel_futures=True)
        log.info("api shutdown complete")

    app = FastAPI(title="Speech_Voiceprint API", version="0.1.0", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.hub = hub
    app.state.jobs = jobs
    app.state.coords = coords
    app.state.executor = executor
    app.state.job_futures = job_futures
    app.state.job_futures_lock = job_futures_lock
    app.state.paths = get_paths()
    app.state.ready = False  # Flipped to True in lifespan startup.
    # Test seam — tests inject a stub `stream` so we don't hit the real
    # Azure SDK. Production leaves this None and the coordinator uses
    # `pipeline.streaming.stream`.
    app.state.stream_fn = None

    if cfg.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cfg.cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )

    _register_routes(app)
    _register_static(app)
    return app


# ---------------------------------------------------------------------------
# Registry helpers — each request opens & closes the store. Cheap (SQLite WAL)
# and avoids long-lived handles competing with job-side writers.

def _open_store(request: Request):
    cfg: AppConfig = request.app.state.cfg
    if not cfg.registry_path:
        raise HTTPException(status_code=503, detail="registry not configured (set SV_REGISTRY_PATH)")
    from registry import open_store

    return open_store(cfg.registry_path)


def _speaker_summary(store, speaker) -> SpeakerSummary:
    vps = store.list_voiceprints()
    mine = [v for v in vps if v.speaker_id == speaker.id]
    return SpeakerSummary(
        id=speaker.id,
        display_name=speaker.display_name,
        voiceprint_count=len(mine),
        models=sorted({v.model for v in mine}),
        created_at=getattr(speaker, "created_at", None),
        updated_at=getattr(speaker, "updated_at", None),
    )


# ---------------------------------------------------------------------------
# Routes

def _register_routes(app: FastAPI) -> None:

    @app.get("/livez", include_in_schema=False)
    def livez():
        """Unauthenticated liveness probe for container orchestrators."""
        return {"ok": True}

    @app.get("/readyz", include_in_schema=False)
    def readyz(request: Request):
        """Unauthenticated readiness probe for container orchestrators."""
        if not getattr(request.app.state, "ready", False):
            return JSONResponse({"ready": False}, status_code=503)
        return {"ready": True}

    @app.get("/api/health", response_model=HealthResponse)
    def health(request: Request, _=Depends(_auth_dep)):
        """Detailed health for the dashboard and authenticated operators."""
        cfg: AppConfig = request.app.state.cfg
        hub: SessionHub = request.app.state.hub
        jobs: JobStore = request.app.state.jobs
        return HealthResponse(
            registry_path=cfg.registry_path,
            auth_enabled=bool(cfg.api_token or os.environ.get("SV_API_TOKEN")),
            workers=cfg.worker_count,
            sessions=len(hub.list_sessions()),
            jobs=len(jobs.list_()),
        )

    @app.get("/api/ready")
    def ready(request: Request):
        """Readiness — process is willing to accept new traffic.

        Returns 503 once SIGTERM has flipped `app.state.ready=False` so the
        K8s ingress stops routing new requests while in-flight jobs drain.
        Auth-free on purpose: probes shouldn't need a token.
        """
        if not getattr(request.app.state, "ready", False):
            return JSONResponse({"ready": False}, status_code=503)
        return {"ready": True}

    @app.get("/api/schema")
    def schema(_=Depends(_auth_dep)):
        return PipelineResult.model_json_schema()

    # ---- sessions / history (live viewer state) -----------------------
    @app.get("/api/sessions")
    def list_sessions(request: Request, _=Depends(_auth_dep)):
        hub: SessionHub = request.app.state.hub
        return hub.list_sessions()

    @app.get("/api/sessions/{session_id}/history")
    def session_history(session_id: str, request: Request, _=Depends(_auth_dep)):
        hub: SessionHub = request.app.state.hub
        return hub.history(session_id)

    @app.delete("/api/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str, request: Request, _=Depends(_auth_dep)):
        """Drop a session's history buffer. Live subscribers stay connected."""
        hub: SessionHub = request.app.state.hub
        if not hub.drop_session(session_id):
            raise HTTPException(status_code=404, detail="session not found")
        return Response(status_code=204)

    @app.post("/api/sessions/{session_id}/events")
    async def post_session_event(
        session_id: str, record: dict, request: Request, _=Depends(_auth_dep)
    ):
        """Cross-process publish endpoint.

        Streaming runs as its own process; this is how it pushes events into
        the api's hub so the browser viewer sees them. Same payload shape as
        the WS broadcasts (see `pipeline.streaming.on_event`).
        """
        hub: SessionHub = request.app.state.hub
        await hub.broadcast(record, session_id)
        return {"ok": True}

    # ---- in-process streaming ingest ---------------------------------
    # Two surfaces: (a) `POST /api/sessions/{id}/stream-file` accepts a WAV
    # upload and drives the realtime pipeline against it (used by the
    # dashboard's Live → "Use audio file" tab); (b) `WS /ws/ingest?session=`
    # accepts raw 16k mono int16 PCM frames pushed live from the browser
    # microphone (resampled by an AudioWorklet). Both share one
    # StreamingCoordinator per session so concurrent producers reuse state.
    @app.post("/api/sessions/{session_id}/stream-file")
    async def stream_file(
        session_id: str,
        request: Request,
        upload: UploadFile = File(...),
        language: str = Form("en-US"),
        registry_path: str | None = Form(default=None),
        realtime_pacing: bool = Form(default=True),
        _=Depends(_auth_dep),
    ):
        """Upload a 16k mono int16 WAV and stream it through the realtime
        pipeline. Returns immediately with a session handle; the browser
        subscribes to `/ws/events?session=<id>` to watch events roll in.
        """
        sess = normalize_session(session_id)
        coords: CoordinatorRegistry = request.app.state.coords
        if coords.get(sess) is not None and coords.get(sess).is_running:  # type: ignore[union-attr]
            raise HTTPException(409, detail=f"session {sess!r} already streaming")

        paths: DataPaths = request.app.state.paths
        paths.ensure("stream")
        saved = paths.stream / f"{uuid.uuid4().hex}-{_safe_upload_name(upload.filename, 'audio.wav')}"
        await _save_upload(upload, saved, max_bytes=_max_upload_bytes())

        cfg: AppConfig = request.app.state.cfg
        loop = asyncio.get_running_loop()
        coord = StreamingCoordinator(
            session=sess,
            hub=request.app.state.hub,
            language=language,
            registry_path=registry_path or cfg.registry_path,
            loop=loop,
            # Browser-driven streams are exploratory — never silently
            # auto-enroll new speakers into the persistent registry, and
            # use a tighter clustering threshold so 2-second-window jitter
            # doesn't split one speaker into many `Speaker_*` cluster ids.
            # Users can still rename/enroll explicitly via the chip menu.
            auto_enroll_unknown=False,
            cluster_threshold=0.55,
            stream_fn=request.app.state.stream_fn,
        )
        coords.register(coord)
        coord.start()

        # Drive a WavSource → coord.push_audio in a background thread so this
        # request returns immediately. The pipeline thread inside the
        # coordinator does the heavy lifting.
        def _feed_wav():
            try:
                from pipeline.streaming import WavSource
                src = WavSource(str(saved), realtime_pacing=realtime_pacing)
                for chunk in src.chunks():
                    coord.push_audio(chunk)
            except Exception as e:  # noqa: BLE001
                log.exception("stream-file feed failed for %s", sess)
                coord._error = e  # surfaced via /stream-status
            finally:
                coord.end_audio()
                # Block-on-finish so we can release the registry slot.
                coord.stop(timeout=120.0)
                coords.release(sess)

        threading.Thread(
            target=_feed_wav, name=f"sv-stream-feed-{sess}", daemon=True,
        ).start()

        return {"session": sess, "status": "streaming", "audio": saved.name}

    @app.post("/api/sessions/{session_id}/stream-stop")
    def stream_stop(session_id: str, request: Request, _=Depends(_auth_dep)):
        sess = normalize_session(session_id)
        coords: CoordinatorRegistry = request.app.state.coords
        coord = coords.get(sess)
        if coord is None:
            raise HTTPException(404, detail="no active stream for session")
        coord.end_audio()
        # Don't block the request thread waiting for full drain; the feed
        # thread will release the slot when stream() returns.
        return {"session": sess, "status": "stopping"}

    @app.get("/api/sessions/{session_id}/stream-status")
    def stream_status(session_id: str, request: Request, _=Depends(_auth_dep)):
        sess = normalize_session(session_id)
        coords: CoordinatorRegistry = request.app.state.coords
        coord = coords.get(sess)
        if coord is None:
            return {"session": sess, "running": False}
        return coord.status()

    # ---- registry ----------------------------------------------------
    @app.get("/api/registry/speakers", response_model=list[SpeakerSummary])
    def list_speakers(request: Request, _=Depends(_auth_dep)):
        with _open_store(request) as store:
            return [_speaker_summary(store, s) for s in store.list_speakers()]

    @app.delete("/api/registry/speakers")
    def bulk_delete_speakers(
        request: Request,
        scope: str = "all",
        _=Depends(_auth_dep),
    ):
        """Bulk delete speakers from the registry.

        scope=all       — wipe everything (use with caution)
        scope=unnamed   — only those without a display_name (i.e. auto-enrolled
                          rows the dashboard exposes as `sp_xxxxxxxx`)
        """
        if scope not in ("all", "unnamed"):
            raise HTTPException(status_code=400, detail="scope must be 'all' or 'unnamed'")
        with _open_store(request) as store:
            speakers = store.list_speakers()
            targets = [s for s in speakers if (scope == "all" or not s.display_name)]
            for sp in targets:
                store.delete_speaker(sp.id)
        return {"deleted": len(targets), "scope": scope}

    @app.get("/api/registry/speakers/{speaker_id}", response_model=SpeakerSummary)
    def get_speaker(speaker_id: str, request: Request, _=Depends(_auth_dep)):
        with _open_store(request) as store:
            sp = store.get_speaker(speaker_id)
            if not sp:
                raise HTTPException(status_code=404, detail="speaker not found")
            return _speaker_summary(store, sp)

    @app.patch("/api/registry/speakers/{speaker_id}", response_model=SpeakerSummary)
    def rename_speaker(
        speaker_id: str,
        body: RenameSpeakerBody,
        request: Request,
        _=Depends(_auth_dep),
    ):
        with _open_store(request) as store:
            sp = store.get_speaker(speaker_id)
            if not sp:
                raise HTTPException(status_code=404, detail="speaker not found")
            store.rename_speaker(speaker_id, body.display_name)
            sp = store.get_speaker(speaker_id)
            return _speaker_summary(store, sp)

    @app.delete("/api/registry/speakers/{speaker_id}", status_code=204)
    def delete_speaker(speaker_id: str, request: Request, _=Depends(_auth_dep)):
        with _open_store(request) as store:
            sp = store.get_speaker(speaker_id)
            if not sp:
                raise HTTPException(status_code=404, detail="speaker not found")
            store.delete_speaker(speaker_id)
        return Response(status_code=204)

    # ---- jobs --------------------------------------------------------
    def _enqueue(request: Request, req: SubmitJobBody) -> dict:
        if req.mode == "fast" and not req.audio_path:
            raise HTTPException(status_code=400, detail="audio_path required for fast mode")
        if req.mode == "batch" and not req.audio_url:
            raise HTTPException(status_code=400, detail="audio_url required for batch mode")
        rec = request.app.state.jobs.submit(
            req.mode, req.model_dump(), comparison_id=req.comparison_id,
        )
        fut = request.app.state.executor.submit(_run_job, request.app.state, rec.job_id, req)
        with request.app.state.job_futures_lock:
            request.app.state.job_futures.add(fut)

        def _drop_future(done: Future) -> None:
            with request.app.state.job_futures_lock:
                request.app.state.job_futures.discard(done)

        fut.add_done_callback(_drop_future)
        return {"job_id": rec.job_id, "status": rec.status, "comparison_id": rec.comparison_id}

    @app.post("/api/jobs/transcribe")
    async def submit_job_json(request: Request, body: SubmitJobBody, _=Depends(_auth_dep)):
        """JSON body — for callers that already have the audio on a shared path."""
        return _enqueue(request, body)

    @app.post("/api/jobs/transcribe-upload")
    async def submit_job_upload(
        request: Request,
        upload: UploadFile = File(...),
        mode: str = Form("fast"),
        backend: str = Form("speechbrain"),
        languages: str = Form(""),
        num_speakers: int | None = Form(default=None),
        registry_path: str | None = Form(default=None),
        auto_enroll_unknown: bool = Form(default=True),
        dual_enroll: bool = Form(default=False),
        comparison_id: str | None = Form(default=None),
        _=Depends(_auth_dep),
    ):
        """Multipart upload — convenience for the dashboard job page."""
        paths: DataPaths = request.app.state.paths
        paths.ensure("uploads")
        saved = paths.uploads / f"{uuid.uuid4().hex}-{_safe_upload_name(upload.filename, 'audio.bin')}"
        await _save_upload(upload, saved, max_bytes=_max_upload_bytes())
        req = SubmitJobBody(
            mode=mode,  # type: ignore[arg-type]
            audio_path=str(saved),
            backend=backend,  # type: ignore[arg-type]
            languages=[s.strip() for s in (languages or "").split(",") if s.strip()],
            num_speakers=num_speakers,
            registry_path=registry_path,
            auto_enroll_unknown=auto_enroll_unknown,
            dual_enroll=dual_enroll,
            comparison_id=comparison_id,
        )
        return _enqueue(request, req)

    @app.get("/api/jobs", response_model=list[JobRecord])
    def list_jobs(request: Request, _=Depends(_auth_dep)):
        return request.app.state.jobs.list_()

    @app.get("/api/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str, request: Request, _=Depends(_auth_dep)):
        rec = request.app.state.jobs.get(job_id)
        if not rec:
            raise HTTPException(status_code=404, detail="job not found")
        return rec

    @app.get("/api/jobs/{job_id}/download")
    def download_job(job_id: str, request: Request, _=Depends(_auth_dep)):
        """Stream the persisted result JSON. Available once job is `done`."""
        rec = request.app.state.jobs.get(job_id)
        if not rec:
            raise HTTPException(status_code=404, detail="job not found")
        if rec.status != "done" or not rec.result:
            raise HTTPException(status_code=409, detail=f"job not ready (status={rec.status})")
        paths: DataPaths = request.app.state.paths
        f = paths.outputs / f"{job_id}.json"
        if not f.is_file():
            # Lazy backfill — older jobs from before persistence shipped won't
            # have a file on disk; write it now so subsequent calls hit fast.
            paths.ensure("outputs")
            f.write_text(json.dumps(rec.result, ensure_ascii=False, indent=2))
        return FileResponse(
            str(f),
            media_type="application/json",
            filename=f"job_{job_id}.json",
        )

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str, request: Request, _=Depends(_auth_dep)):
        rec = request.app.state.jobs.get(job_id)
        if not rec:
            raise HTTPException(status_code=404, detail="job not found")
        request.app.state.jobs.delete(job_id)
        paths: DataPaths = request.app.state.paths
        f = paths.outputs / f"{job_id}.json"
        if f.is_file():
            try:
                f.unlink()
            except OSError:
                pass
        return Response(status_code=204)

    # ---- maintenance --------------------------------------------------
    @app.get("/api/maintenance/usage")
    def maintenance_usage(request: Request, _=Depends(_auth_dep)):
        """Snapshot of disk + in-memory state the maintenance page renders.

        Each scope reports `count` and `bytes` (sum of file sizes; missing
        directories are silently 0). `jobs.kept` and `sessions.kept` are
        in-memory; clearing them is cheap.
        """
        paths: DataPaths = request.app.state.paths
        hub: SessionHub = request.app.state.hub
        jobs: JobStore = request.app.state.jobs

        def _scan(p: Path) -> dict:
            if not p.is_dir():
                return {"count": 0, "bytes": 0, "path": str(p)}
            files = [f for f in p.iterdir() if f.is_file()]
            return {
                "count": len(files),
                "bytes": sum(f.stat().st_size for f in files),
                "path": str(p),
            }

        return {
            "uploads":  _scan(paths.uploads),
            "stream":   _scan(paths.stream),
            "outputs":  _scan(paths.outputs),
            "sessions": {"count": len(hub.list_sessions())},
            "jobs":     {"count": len(jobs.list_())},
            "data_root": str(paths.root),
        }

    @app.post("/api/maintenance/cleanup")
    def maintenance_cleanup(
        request: Request,
        body: CleanupBody,
        _=Depends(_auth_dep),
    ):
        """Remove ephemeral state. With `dry_run=true` returns counts only.

        scope:
          uploads  — multipart job uploads (data/uploads/)
          stream   — browser stream-file uploads (data/stream/)
          outputs  — persisted job result JSONs (data/outputs/)
          jobs     — in-memory job table (and their on-disk outputs)
          sessions — in-memory session histories (subscribers stay connected)

        `older_than_days` filters by mtime for filesystem scopes; ignored for
        in-memory scopes (jobs/sessions are always wiped fully when targeted).
        """
        paths: DataPaths = request.app.state.paths
        hub: SessionHub = request.app.state.hub
        jobs: JobStore = request.app.state.jobs
        cutoff = (
            time.time() - body.older_than_days * 86400
            if body.older_than_days is not None and body.older_than_days >= 0
            else None
        )

        deleted = 0
        bytes_freed = 0

        if body.scope in ("uploads", "stream", "outputs"):
            target = {
                "uploads": paths.uploads,
                "stream":  paths.stream,
                "outputs": paths.outputs,
            }[body.scope]
            if target.is_dir():
                for f in target.iterdir():
                    if not f.is_file():
                        continue
                    if cutoff is not None and f.stat().st_mtime > cutoff:
                        continue
                    sz = f.stat().st_size
                    if not body.dry_run:
                        try:
                            f.unlink()
                        except OSError:
                            continue
                    deleted += 1
                    bytes_freed += sz

        elif body.scope == "jobs":
            for rec in list(jobs.list_()):
                if not body.dry_run:
                    jobs.delete(rec.job_id)
                    out = paths.outputs / f"{rec.job_id}.json"
                    if out.is_file():
                        try:
                            bytes_freed += out.stat().st_size
                            out.unlink()
                        except OSError:
                            pass
                deleted += 1

        elif body.scope == "sessions":
            for s in list(hub.list_sessions()):
                if not body.dry_run:
                    hub.drop_session(s["session"])
                deleted += 1

        return {
            "scope": body.scope,
            "dry_run": body.dry_run,
            "deleted": deleted,
            "bytes_freed": bytes_freed,
        }

    # ---- websocket ---------------------------------------------------
    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket, session: str = Query(default="default")):
        # FastAPI doesn't run Depends() for WS; do auth manually.
        expected = os.environ.get("SV_API_TOKEN")
        if expected:
            auth = ws.headers.get("authorization") or ""
            token = auth.split(None, 1)[1].strip() if auth.lower().startswith("bearer ") else ""
            # Allow `?token=` fallback for browsers that can't set headers.
            if not token:
                token = ws.query_params.get("token", "")
            if token != expected:
                await ws.close(code=4401)
                return
        await ws.accept()
        sess = normalize_session(session)
        hub: SessionHub = ws.app.state.hub

        async def sender(msg: str) -> None:
            try:
                await ws.send_text(msg)
            except Exception:
                # Drop on broken pipe; cleanup happens in the finally below.
                pass

        hub.subscribe(sess, sender)
        try:
            for rec in hub.history(sess):
                await ws.send_text(json.dumps(rec, ensure_ascii=False))
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_text(json.dumps({"type": "error", "error": "invalid json"}))
                    continue
                if not isinstance(msg, dict):
                    await ws.send_text(json.dumps({"type": "error", "error": "expected json object"}))
                    continue
                # Inbound rename/forget: scope to this session's history.
                op = msg.get("type")
                if op == "rename":
                    old = msg.get("old"); new = msg.get("new")
                    if not old or not new:
                        await ws.send_text(json.dumps({"type": "error", "error": "rename: old/new required"}))
                        continue
                    updated = hub.rewrite_history_speaker(old, new, session=sess)
                    for rec in updated:
                        await hub.broadcast(rec, sess)
                    await ws.send_text(json.dumps({"type": "ack", "op": "rename", "updated": len(updated)}))
                else:
                    await ws.send_text(json.dumps({"type": "error", "error": f"unknown op: {op!r}"}))
        except WebSocketDisconnect:
            pass
        finally:
            hub.unsubscribe(sess, sender)

    # ---- websocket: browser microphone ingest ------------------------
    @app.websocket("/ws/ingest")
    async def ws_ingest(
        ws: WebSocket,
        session: str = Query(default="default"),
        language: str = Query(default="en-US"),
    ):
        """Accept raw 16k mono int16 PCM from a browser AudioWorklet.

        Wire format:
          - binary frames: PCM bytes (any chunk length; coordinator buffers).
          - text "stop": producer is done; pipeline drains then closes.

        Auth identical to `/ws/events` — Bearer header or `?token=`.
        """
        expected = os.environ.get("SV_API_TOKEN")
        if expected:
            auth = ws.headers.get("authorization") or ""
            token = auth.split(None, 1)[1].strip() if auth.lower().startswith("bearer ") else ""
            if not token:
                token = ws.query_params.get("token", "")
            if token != expected:
                await ws.close(code=4401)
                return
        await ws.accept()
        sess = normalize_session(session)
        coords: CoordinatorRegistry = ws.app.state.coords
        cfg: AppConfig = ws.app.state.cfg

        existing = coords.get(sess)
        if existing is not None and existing.is_running:
            await ws.send_text(json.dumps({
                "type": "error", "error": f"session {sess!r} already streaming",
            }))
            await ws.close(code=4409)
            return

        loop = asyncio.get_running_loop()
        coord = StreamingCoordinator(
            session=sess,
            hub=ws.app.state.hub,
            language=language,
            registry_path=cfg.registry_path,
            loop=loop,
            # Same safer defaults as POST /stream-file — see comment there.
            auto_enroll_unknown=False,
            cluster_threshold=0.55,
            stream_fn=ws.app.state.stream_fn,
        )
        coords.register(coord)
        coord.start()
        await ws.send_text(json.dumps({
            "type": "ingest_ready", "session": sess, "language": language,
        }))

        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if "bytes" in msg and msg["bytes"] is not None:
                    coord.push_audio(msg["bytes"])
                    continue
                text = msg.get("text")
                if text is None:
                    continue
                if text.strip() == "stop":
                    break
                # accept JSON commands too (rename / forget)
                try:
                    payload = json.loads(text)
                    if isinstance(payload, dict):
                        reply = coord.handle_command(payload)
                        await ws.send_text(json.dumps(reply))
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            pass
        finally:
            coord.end_audio()
            # Hand off final cleanup to a worker thread so the WS can close
            # promptly without blocking on Azure SDK drain.
            def _drain():
                try:
                    coord.stop(timeout=60.0)
                finally:
                    coords.release(sess)
            threading.Thread(target=_drain, daemon=True).start()
            try:
                await ws.close()
            except Exception:
                pass


def _run_job(app_state, job_id: str, body: SubmitJobBody) -> None:
    """Worker entry. Runs in the api's executor thread."""
    jobs: JobStore = app_state.jobs
    log.info("job %s starting (%s)", job_id, body.mode)
    jobs.update(job_id, status="running", started_at=time.time())
    jobs.append_log(job_id, f"started mode={body.mode} backend={body.backend}")

    try:
        from pipeline import transcribe_file

        # Fall back to the api-wide registry when the request didn't specify one,
        # so dashboard-submitted jobs persist speakers without needing to retype
        # the path on every form.
        cfg: AppConfig = app_state.cfg
        effective_registry = body.registry_path or cfg.registry_path

        result = transcribe_file(
            body.audio_path or "",
            mode=body.mode,  # type: ignore[arg-type]
            backend=body.backend,  # type: ignore[arg-type]
            languages=body.languages or None,
            audio_url=body.audio_url,
            num_speakers=body.num_speakers,
            majority_threshold=body.majority_threshold,
            device=body.device,
            registry_path=effective_registry,
            auto_enroll_unknown=body.auto_enroll_unknown,
            dual_enroll=body.dual_enroll,
        )
        result_dict = result.model_dump(exclude_none=True)
        jobs.update(
            job_id,
            status="done",
            finished_at=time.time(),
            result=result_dict,
        )
        # Persist a copy on disk so it survives api restarts and the dashboard
        # can offer a download link. Best-effort — failure here doesn't fail
        # the job (the in-memory record still has it).
        try:
            paths: DataPaths = app_state.paths
            paths.ensure("outputs")
            (paths.outputs / f"{job_id}.json").write_text(
                json.dumps(result_dict, ensure_ascii=False, indent=2)
            )
        except OSError as oe:
            log.warning("job %s outputs write failed: %s", job_id, oe)
        jobs.append_log(job_id, f"done utterances={len(result.utterances)}")
        log.info("job %s done", job_id)
    except Exception as e:  # noqa: BLE001 — surface to the dashboard, not the test runner
        tb = traceback.format_exc(limit=4)
        jobs.update(job_id, status="failed", finished_at=time.time(), error=f"{type(e).__name__}: {e}")
        jobs.append_log(job_id, tb.rstrip())
        log.exception("job %s failed", job_id)


# ---------------------------------------------------------------------------
# Static UI mount (Phase 5 fills out web/static; this just routes everything)

def _register_static(app: FastAPI) -> None:
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        idx = WEB_DIR / "index.html"
        if not idx.is_file():
            return JSONResponse({"detail": "web UI not bundled"}, status_code=404)
        return FileResponse(str(idx), media_type="text/html; charset=utf-8")


# ---------------------------------------------------------------------------
# CLI entry — `python -m pipeline.api`

def main() -> None:
    from pipeline.env_file import load_env_file
    load_env_file()
    parser = argparse.ArgumentParser(description="Speech_Voiceprint API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--registry", default=None, help="SQLite registry path")
    parser.add_argument("--workers", type=int, default=2, help="job worker pool size")
    parser.add_argument(
        "--cors",
        default=None,
        help="comma-separated origins to allow CORS for (e.g. http://localhost:5173)",
    )
    args = parser.parse_args()

    cfg = AppConfig(
        registry_path=args.registry,
        worker_count=max(1, args.workers),
        cors_origins=tuple(s.strip() for s in (args.cors or "").split(",") if s.strip()),
    )
    app = create_app(cfg)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
