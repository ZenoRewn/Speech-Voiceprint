"""Realtime streaming pipeline.

Audio is pushed in small chunks to two consumers in parallel:

  1. Azure realtime transcriber (`AzureRealtimeTranscriber`) returns transcribed
     utterance events with word-level timestamps.
  2. A rolling raw-PCM buffer driven by a background thread that, every
     `hop_seconds`, embeds the most recent `window_seconds` of audio with the
     SpeechBrain ECAPA encoder and feeds the embedding to an
     `OnlineSpeakerCluster`. SpeechBrain's RTF (~0.02 on Mac CPU in M1) means
     this can keep up with realtime even while STT is also running.

When an utterance event arrives, we look up which voiceprint windows overlap
its time range and assign the cluster label that holds the majority. Optional
`SpeakerRegistry` matching resolves cluster centroids to persistent identities.

The `--audio FILE` mode plays the file at 1× wall-clock so the realtime path
behaves the same as a microphone — handy for repeatable demos.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from stt.azure_realtime import AzureRealtimeTranscriber, TranscriptionEvent
from stt.base import Word
from voiceprint.speechbrain_provider import SpeechBrainProvider

from .online_cluster import OnlineSpeakerCluster

PCM_SR = 16000
PCM_BYTES_PER_SAMPLE = 2  # int16


# ---------------------------------------------------------------------------
@dataclass
class WindowEmbedding:
    start: float
    end: float
    label: str


class RollingPCMBuffer:
    """Thread-safe append-only int16 PCM buffer with elapsed-time bookkeeping."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._samples_seen = 0
        self._lock = threading.Lock()

    def append(self, pcm: bytes) -> None:
        with self._lock:
            self._buf.extend(pcm)
            self._samples_seen = len(self._buf) // PCM_BYTES_PER_SAMPLE

    def slice_seconds(self, start_s: float, end_s: float) -> np.ndarray | None:
        with self._lock:
            i0 = max(0, int(start_s * PCM_SR) * PCM_BYTES_PER_SAMPLE)
            i1 = int(end_s * PCM_SR) * PCM_BYTES_PER_SAMPLE
            if i1 > len(self._buf) or i1 - i0 < PCM_BYTES_PER_SAMPLE * 100:
                return None
            chunk = bytes(self._buf[i0:i1])
        arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        return arr

    @property
    def elapsed_seconds(self) -> float:
        with self._lock:
            return self._samples_seen / PCM_SR


# ---------------------------------------------------------------------------
class StreamingVoiceprint:
    """Background thread: consume buffer windows, emit (start, end, label)."""

    def __init__(
        self,
        encoder: SpeechBrainProvider,
        buffer: RollingPCMBuffer,
        cluster: OnlineSpeakerCluster,
        window_seconds: float = 2.0,
        hop_seconds: float = 0.75,
    ) -> None:
        self.encoder = encoder
        self.buffer = buffer
        self.cluster = cluster
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self._windows: list[WindowEmbedding] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        next_start = 0.0
        while not self._stop.is_set():
            elapsed = self.buffer.elapsed_seconds
            if elapsed < next_start + self.window_seconds:
                time.sleep(0.05)
                continue
            window_end = next_start + self.window_seconds
            clip = self.buffer.slice_seconds(next_start, window_end)
            if clip is None:
                time.sleep(0.05)
                continue
            try:
                emb = self.encoder._embed_clip(clip, PCM_SR)
                label = self.cluster.update(emb)
                with self._lock:
                    self._windows.append(WindowEmbedding(start=next_start, end=window_end, label=label))
            except Exception as e:  # noqa: BLE001 — surface but keep streaming
                sys.stderr.write(f"[voiceprint] window {next_start:.2f}-{window_end:.2f} failed: {e}\n")
            next_start += self.hop_seconds

    def label_for_range(self, start: float, end: float) -> tuple[str, float]:
        with self._lock:
            windows = list(self._windows)
        votes: Counter[str] = Counter()
        for w in windows:
            ov = min(end, w.end) - max(start, w.start)
            if ov > 0:
                votes[w.label] += int(ov * 100)
        if not votes:
            return "Speaker_pending", 0.0
        top, count = votes.most_common(1)[0]
        total = sum(votes.values())
        return top, count / total if total else 0.0


# ---------------------------------------------------------------------------
class _RegistryResolver:
    """Adapter: take an OnlineSpeakerCluster, look its centroids up against a
    persistent SpeakerRegistry, and produce a stable cluster_label -> identity
    map. Re-resolve as cluster centroids drift; cache verdicts so we don't hit
    SQLite on every utterance.
    """

    def __init__(
        self,
        cluster: OnlineSpeakerCluster,
        registry_path: str,
        backend_name: str,
        embedding_dim: int,
        *,
        match_threshold: float,
        unknown_threshold: float,
        auto_enroll_unknown: bool,
        max_per_speaker: int,
    ) -> None:
        from registry import open_store
        from registry.matcher import SpeakerMatcher

        self.cluster = cluster
        self.store = open_store(registry_path)
        self.matcher = SpeakerMatcher(
            self.store,
            model=f"{backend_name}-{embedding_dim}",
            match_threshold=match_threshold,
            unknown_threshold=unknown_threshold,
            max_voiceprints_per_speaker=max_per_speaker,
            auto_enroll_unknown=auto_enroll_unknown,
        )
        self._resolved: dict[str, str] = {}      # cluster_label -> display label
        self._verdicts: dict[str, dict] = {}     # debug info per label

    def resolve(self, cluster_label: str) -> str:
        if cluster_label in self._resolved:
            return self._resolved[cluster_label]
        centroid = self.cluster.centroids().get(cluster_label)
        if centroid is None:
            return cluster_label
        from registry.matcher import MatchVerdict, SpeakerMatcher

        result = self.matcher.match(centroid)
        new_label = SpeakerMatcher.label_for(result, fallback=cluster_label)
        if result.verdict == MatchVerdict.KNOWN and result.speaker_id:
            self.store.add_voiceprint(
                result.speaker_id,
                centroid,
                self.matcher.model,
                quality=1.0 - result.distance,
                max_per_speaker=self.matcher.max_voiceprints_per_speaker,
            )
            self.matcher._refresh()
        elif result.verdict == MatchVerdict.UNKNOWN and self.matcher.auto_enroll_unknown:
            sp = self.store.create_speaker()
            self.store.add_voiceprint(
                sp.id,
                centroid,
                self.matcher.model,
                quality=1.0,
                max_per_speaker=self.matcher.max_voiceprints_per_speaker,
            )
            new_label = sp.id
            self.matcher._refresh()

        self._resolved[cluster_label] = new_label
        self._verdicts[cluster_label] = {
            "verdict": result.verdict.value,
            "speaker_id": result.speaker_id,
            "display_name": result.display_name,
            "distance": round(result.distance, 4),
            "resolved_label": new_label,
        }
        return new_label

    def verdicts(self) -> dict[str, dict]:
        return dict(self._verdicts)

    def close(self) -> None:
        self.store.close()

    def rename(self, current_label: str, display_name: str) -> tuple[str, str]:
        """Rename whichever speaker is currently labeled `current_label`.

        Resolves the cluster → speaker_id (creating one if needed), updates
        the registry's `display_name`, refreshes the cached label cache so
        future utterances pick up the new name. Returns `(speaker_id, new_label)`.
        """
        speaker_id: str | None = None
        # If the live label is a registry id (sp_xxx) or already a display
        # name, walk the resolved-label map back to find the underlying id.
        for cluster_label, resolved in list(self._resolved.items()):
            if resolved == current_label:
                # Look up this cluster's verdict for the canonical speaker_id.
                verdict = self._verdicts.get(cluster_label, {})
                speaker_id = verdict.get("speaker_id")
                # Bind this cluster to the new display_name
                self._resolved[cluster_label] = display_name
                break
        # Fallback: treat current_label itself as a registry id
        if speaker_id is None and current_label.startswith("sp_"):
            speaker_id = current_label
        # Cluster-only label (Speaker_A, etc.): need to enroll its centroid
        if speaker_id is None:
            centroid = self.cluster.centroids().get(current_label)
            if centroid is None:
                raise ValueError(f"cannot rename unknown label {current_label!r}")
            sp = self.store.create_speaker(display_name=display_name)
            self.store.add_voiceprint(
                sp.id,
                centroid,
                self.matcher.model,
                quality=1.0,
                max_per_speaker=self.matcher.max_voiceprints_per_speaker,
            )
            speaker_id = sp.id
            self._resolved[current_label] = display_name
        else:
            self.store.rename_speaker(speaker_id, display_name)
        self.matcher._refresh()
        return speaker_id, display_name


def stream(
    source: "AudioSource",
    languages: list[str],
    *,
    chunk_ms: int = 100,
    cluster_threshold: float = 0.7,
    window_seconds: float = 2.0,
    hop_seconds: float = 0.75,
    registry_path: str | None = None,
    match_threshold: float = 0.40,
    unknown_threshold: float = 0.60,
    auto_enroll_unknown: bool = True,
    max_per_speaker: int = 5,
    on_event=None,
    resolver_holder: dict | None = None,
):
    """Drive the realtime pipeline from any AudioSource (file, mic, ...).

    The source yields raw 16k mono int16 PCM bytes; this function plumbs them
    into Azure realtime + the voiceprint windowing thread, then merges the
    transcribed events with cluster labels (and optional registry identities).

    `resolver_holder` is an optional dict the caller passes in; when registry
    is enabled, we set `resolver_holder["resolver"]` so the WS command handler
    can issue rename/forget against the live registry.
    """
    transcriber = AzureRealtimeTranscriber(languages=languages, diarization=True)
    transcriber.start()

    buffer = RollingPCMBuffer()
    cluster = OnlineSpeakerCluster(threshold=cluster_threshold)
    encoder = SpeechBrainProvider(device="cpu")
    encoder._encoder()  # warm
    voiceprint = StreamingVoiceprint(
        encoder=encoder,
        buffer=buffer,
        cluster=cluster,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
    )
    voiceprint.start()

    resolver: _RegistryResolver | None = None
    if registry_path:
        resolver = _RegistryResolver(
            cluster,
            registry_path=registry_path,
            backend_name=encoder.name,
            embedding_dim=encoder.embedding_dim,
            match_threshold=match_threshold,
            unknown_threshold=unknown_threshold,
            auto_enroll_unknown=auto_enroll_unknown,
            max_per_speaker=max_per_speaker,
        )
        if resolver_holder is not None:
            resolver_holder["resolver"] = resolver

    emitted: list[dict] = []
    pending_min_confidence = 0.5
    pending: list[dict] = []

    def _emit(rec: dict, kind: str) -> None:
        if on_event:
            on_event({**rec, "event": kind})
        else:
            tag = "TENTATIVE" if kind == "tentative" else ("REVISED  " if kind == "revised" else "FINAL    ")
            sys.stdout.write(
                f"[{rec['start']:6.2f}-{rec['end']:6.2f}] {tag} "
                f"vp={rec['speaker']:14s} (conf={rec['speaker_confidence']:.2f}, "
                f"az={rec['azure_speaker']})  {rec['text']}\n"
            )
            sys.stdout.flush()

    def producer() -> None:
        try:
            for chunk in source.chunks():
                buffer.append(chunk)
                transcriber.push(chunk)
        except KeyboardInterrupt:
            pass
        finally:
            time.sleep(window_seconds)
            transcriber.end_input(drain_timeout=60.0)
            transcriber.close()

    prod = threading.Thread(target=producer, daemon=True)
    prod.start()

    try:
        for evt in transcriber.events():
            cluster_label, conf = voiceprint.label_for_range(evt.start, evt.end)
            resolved = resolver.resolve(cluster_label) if resolver else cluster_label
            rec = {
                "start": round(evt.start, 3),
                "end": round(evt.end, 3),
                "azure_speaker": evt.azure_speaker,
                "cluster_label": cluster_label,
                "speaker": resolved,
                "speaker_confidence": round(conf, 3),
                "text": evt.text,
            }
            emitted.append(rec)
            if conf < pending_min_confidence:
                pending.append(rec)
                _emit(rec, "tentative")
            else:
                _emit(rec, "final")

        time.sleep(window_seconds)
        for rec in pending:
            cluster_label, conf = voiceprint.label_for_range(rec["start"], rec["end"])
            resolved = resolver.resolve(cluster_label) if resolver else cluster_label
            if resolved != rec["speaker"] or conf > rec["speaker_confidence"] + 0.1:
                rec["cluster_label"] = cluster_label
                rec["speaker"] = resolved
                rec["speaker_confidence"] = round(conf, 3)
                _emit(rec, "revised")
    finally:
        voiceprint.stop()
        prod.join(timeout=5.0)
        if resolver is not None:
            resolver.close()

    return emitted


class AudioSource:
    """Yield raw 16k mono int16 PCM bytes in chunks."""

    def chunks(self):  # pragma: no cover - protocol
        raise NotImplementedError


class WavSource(AudioSource):
    """Read a 16k mono int16 WAV; if `realtime_pacing` is True, sleep between
    chunks so playback simulates a real microphone."""

    def __init__(self, wav_path: str, *, chunk_ms: int = 100, realtime_pacing: bool = True) -> None:
        self.wav_path = wav_path
        self.chunk_ms = chunk_ms
        self.realtime_pacing = realtime_pacing

    def chunks(self):
        with wave.open(self.wav_path, "rb") as wf:
            if wf.getframerate() != PCM_SR or wf.getsampwidth() != 2 or wf.getnchannels() != 1:
                raise ValueError("expected 16k mono int16 WAV; transcode first")
            chunk_frames = int(PCM_SR * self.chunk_ms / 1000)
            t0 = time.perf_counter()
            sample_idx = 0
            while True:
                frames = wf.readframes(chunk_frames)
                if not frames:
                    return
                yield frames
                sample_idx += chunk_frames
                if self.realtime_pacing:
                    target = sample_idx / PCM_SR
                    sleep_for = target - (time.perf_counter() - t0)
                    if sleep_for > 0:
                        time.sleep(sleep_for)


class MicrophoneSource(AudioSource):
    """Pull from the system mic via sounddevice. Yields chunks of `chunk_ms`.

    Stop with Ctrl-C or by calling `stop()` from another thread. Resamples
    automatically if the device's native rate isn't 16k.
    """

    # Most consumer audio devices land on one of these. We try the device's
    # advertised default first, then fall through this list.
    FALLBACK_RATES: tuple[int, ...] = (16000, 48000, 44100, 32000, 22050)

    def __init__(
        self,
        *,
        chunk_ms: int = 100,
        device: int | str | None = None,
        device_samplerate: int | None = None,
    ) -> None:
        import sounddevice  # imported here to keep file usable without portaudio

        self._sd = sounddevice
        self.chunk_ms = chunk_ms
        self.device = device
        self._stop = threading.Event()

        # Pick a samplerate the device actually accepts. `query_devices` can
        # silently return a default that the device then refuses to open
        # (some virtual mics / Bluetooth headsets advertise 44.1k but only
        # accept 16k, etc.); explicit `check_input_settings` catches that.
        if device_samplerate is not None:
            self.device_samplerate = device_samplerate
        else:
            try:
                info = sounddevice.query_devices(device, "input") if device is not None else sounddevice.query_devices(kind="input")
                preferred = int(info["default_samplerate"])
            except Exception as e:
                sys.stderr.write(f"[mic] query_devices failed ({e!r}); falling back to {self.FALLBACK_RATES[0]} Hz\n")
                preferred = self.FALLBACK_RATES[0]

            candidates: list[int] = [preferred] + [r for r in self.FALLBACK_RATES if r != preferred]
            chosen: int | None = None
            errors: list[str] = []
            for rate in candidates:
                try:
                    sounddevice.check_input_settings(device=device, samplerate=rate, channels=1, dtype="int16")
                    chosen = rate
                    break
                except Exception as e:
                    errors.append(f"{rate}: {type(e).__name__}")
            if chosen is None:
                raise RuntimeError(
                    f"no usable samplerate found for input device {device!r}; tried {candidates}, errors {errors}"
                )
            if chosen != preferred:
                sys.stderr.write(f"[mic] device rejected {preferred} Hz, falling back to {chosen} Hz\n")
            self.device_samplerate = chosen

    def stop(self) -> None:
        self._stop.set()

    def chunks(self):
        import numpy as np

        chunk_frames_native = int(self.device_samplerate * self.chunk_ms / 1000)
        ratio = self.device_samplerate / PCM_SR

        with self._sd.InputStream(
            samplerate=self.device_samplerate,
            channels=1,
            dtype="int16",
            blocksize=chunk_frames_native,
            device=self.device,
        ) as stream:
            sys.stderr.write(
                f"[mic] capturing from device={self.device or 'default'} "
                f"native_rate={self.device_samplerate} -> {PCM_SR}; press Ctrl-C to stop\n"
            )
            while not self._stop.is_set():
                data, _overflowed = stream.read(chunk_frames_native)
                pcm_native = data.reshape(-1).astype(np.int16)
                if self.device_samplerate != PCM_SR:
                    # naive linear-interp resample is fine for speech
                    out_n = int(round(len(pcm_native) / ratio))
                    if out_n <= 0:
                        continue
                    xs = np.arange(out_n) * ratio
                    i0 = np.floor(xs).astype(int)
                    i1 = np.minimum(i0 + 1, len(pcm_native) - 1)
                    frac = (xs - i0).astype(np.float32)
                    out = (1 - frac) * pcm_native[i0] + frac * pcm_native[i1]
                    pcm = out.astype(np.int16)
                else:
                    pcm = pcm_native
                yield pcm.tobytes()


def stream_from_wav(
    wav_path: str,
    languages: list[str],
    *,
    chunk_ms: int = 100,
    realtime_pacing: bool = True,
    **kwargs,
):
    """Backwards-compatible WAV entrypoint."""
    src = WavSource(wav_path, chunk_ms=chunk_ms, realtime_pacing=realtime_pacing)
    return stream(src, languages, chunk_ms=chunk_ms, **kwargs)


def stream_from_microphone(
    languages: list[str],
    *,
    chunk_ms: int = 100,
    device: int | str | None = None,
    **kwargs,
):
    """Capture from the system microphone."""
    src = MicrophoneSource(chunk_ms=chunk_ms, device=device)
    return stream(src, languages, chunk_ms=chunk_ms, **kwargs)


# ---------------------------------------------------------------------------
def _forward_to_api(url: str, record: dict, token: str | None) -> None:
    """Best-effort POST to the central api server's `/api/sessions/{id}/events`.

    Failures are non-fatal — we don't want a temporary api outage to take
    down the streaming process. The caller's stderr surfaces the error so
    operators can spot a misconfigured target.
    """
    import requests as _r
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        _r.post(url, json=record, headers=headers, timeout=2.0)
    except _r.RequestException as e:
        sys.stderr.write(f"[api] forward failed ({e!r}); continuing\n")


def main() -> None:
    from pipeline.env_file import load_env_file
    load_env_file()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML config file; CLI flags override these values")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--audio", help="16k mono WAV (transcode MP3 first)")
    src.add_argument("--mic", action="store_true", help="capture from system microphone")
    ap.add_argument("--mic-device", default=None, help="sounddevice index or substring; default = system default")
    ap.add_argument("--list-devices", action="store_true", help="print input devices and exit")
    ap.add_argument("--language", action="append", default=None)
    ap.add_argument("--chunk-ms", type=int, default=None)
    ap.add_argument("--no-pacing", action="store_true", help="WAV mode: run as fast as possible")
    ap.add_argument("--cluster-threshold", type=float, default=None)
    ap.add_argument("--window-seconds", type=float, default=None)
    ap.add_argument("--hop-seconds", type=float, default=None)
    ap.add_argument("--registry", default=None, help="enable cross-session speaker registry SQLite path")
    ap.add_argument("--match-threshold", type=float, default=None)
    ap.add_argument("--unknown-threshold", type=float, default=None)
    ap.add_argument("--no-auto-enroll", action="store_true")
    ap.add_argument("--max-per-speaker", type=int, default=None)
    ap.add_argument("--ws-port", type=int, default=None, help="also broadcast events on ws://0.0.0.0:PORT")
    ap.add_argument(
        "--session",
        default=None,
        help="session label clients select with ?session= (default: 'default')",
    )
    ap.add_argument(
        "--api-target",
        default=None,
        help="forward each event to a remote api server (e.g. http://api:8080); uses SV_API_TOKEN if set",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd

        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                print(f"{idx:3d}  {dev['name']}  ({int(dev['default_samplerate'])} Hz, {dev['max_input_channels']} ch)")
        return

    from pipeline.config import load_config, merge_cli, pick

    cfg = load_config(args.config)

    audio_path = merge_cli(cfg, args.audio, "audio.path")
    use_mic = args.mic or bool(pick(cfg, "audio.mic", False))
    if not audio_path and not use_mic:
        ap.error("either --audio or --mic (or audio.path / audio.mic in YAML) must be provided")

    languages = merge_cli(cfg, args.language, "azure.languages", default=["en-US"])
    if isinstance(languages, str):
        languages = [languages]

    chunk_ms = merge_cli(cfg, args.chunk_ms, "stream.chunk_ms", default=100)
    cluster_threshold = merge_cli(cfg, args.cluster_threshold, "voiceprint.online_cluster_threshold", default=0.7)
    window_seconds = merge_cli(cfg, args.window_seconds, "voiceprint.window_seconds", default=2.0)
    hop_seconds = merge_cli(cfg, args.hop_seconds, "voiceprint.hop_seconds", default=0.75)
    registry_path = merge_cli(cfg, args.registry, "registry.store_path")
    if pick(cfg, "registry.enabled") is False:
        registry_path = None
    match_threshold = merge_cli(cfg, args.match_threshold, "registry.match_threshold", default=0.40)
    unknown_threshold = merge_cli(cfg, args.unknown_threshold, "registry.unknown_threshold", default=0.60)
    max_per_speaker = merge_cli(cfg, args.max_per_speaker, "registry.max_voiceprints_per_speaker", default=5)
    auto_enroll_unknown = not args.no_auto_enroll
    if pick(cfg, "registry.auto_enroll_unknown") is False:
        auto_enroll_unknown = False
    ws_port = merge_cli(cfg, args.ws_port, "output.ws_port")
    session_id = merge_cli(cfg, args.session, "output.ws_session", default="default")
    api_target = merge_cli(cfg, args.api_target, "output.api_target")
    out_path = merge_cli(cfg, args.out, "output.path")
    mic_device = merge_cli(cfg, args.mic_device, "audio.mic_device")
    realtime_pacing = not (args.no_pacing or bool(pick(cfg, "stream.no_pacing", False)))

    common = dict(
        cluster_threshold=cluster_threshold,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        registry_path=registry_path,
        match_threshold=match_threshold,
        unknown_threshold=unknown_threshold,
        auto_enroll_unknown=auto_enroll_unknown,
        max_per_speaker=max_per_speaker,
    )

    # Hold a reference to the resolver here so the WS command handler can
    # reach it; `stream_from_*` will create one if registry is enabled.
    resolver_holder: dict[str, object] = {}

    def _command_handler(msg: dict, sess: str) -> dict:
        """Inbound from browser. Currently supports `rename` and `forget`.

        `sess` is the session label of the client that issued the command
        (extracted from `?session=` on the WS URL). Mutations are scoped to
        that session's history so renaming Katie in `katiesteve` doesn't
        rewrite a different stream's transcript.
        """
        kind = msg.get("type")
        if kind == "rename":
            label = msg.get("speaker") or msg.get("label")
            new_name = (msg.get("display_name") or "").strip()
            if not label or not new_name:
                return {"type": "error", "error": "rename needs speaker + display_name"}
            resolver = resolver_holder.get("resolver")
            if resolver is None:
                return {"type": "error", "error": "registry not enabled in this stream"}
            speaker_id, resolved = resolver.rename(label, new_name)  # type: ignore[attr-defined]
            updates = ws_sink.rewrite_history_speaker(label, resolved, session=sess) if ws_sink else []
            for upd in updates:
                ws_sink.broadcast(upd, session=sess)  # type: ignore[union-attr]
            return {
                "type": "rename_ok",
                "speaker_id": speaker_id,
                "old_label": label,
                "new_label": resolved,
                "updated_records": len(updates),
            }
        if kind == "forget":
            speaker_id = msg.get("speaker_id")
            resolver = resolver_holder.get("resolver")
            if resolver is None or not speaker_id:
                return {"type": "error", "error": "forget needs registry + speaker_id"}
            resolver.store.delete_speaker(speaker_id)  # type: ignore[attr-defined]
            resolver.matcher._refresh()  # type: ignore[attr-defined]
            return {"type": "forget_ok", "speaker_id": speaker_id}
        return {"type": "error", "error": f"unknown command {kind!r}"}

    ws_sink = None
    if ws_port:
        from pipeline.ws_server import WebSocketSink

        ws_sink = WebSocketSink(
            port=int(ws_port),
            command_handler=_command_handler,
            session_id=str(session_id),
        )
        ws_sink.start()
        sys.stderr.write(
            f"[ws] listening on ws://0.0.0.0:{int(ws_port)}/events?session={ws_sink.session_id}\n"
        )

    api_session = str(session_id)
    api_post_url: str | None = None
    api_token: str | None = None
    if api_target:
        api_post_url = f"{api_target.rstrip('/')}/api/sessions/{api_session}/events"
        api_token = os.environ.get("SV_API_TOKEN")
        sys.stderr.write(f"[api] forwarding events to {api_post_url}\n")

    def on_event(rec: dict) -> None:
        # Mirror the pretty stdout format and feed the sinks in one place.
        tag = {"tentative": "TENTATIVE", "revised": "REVISED  ", "final": "FINAL    "}.get(rec.get("event"), "         ")
        sys.stdout.write(
            f"[{rec['start']:6.2f}-{rec['end']:6.2f}] {tag} "
            f"vp={rec['speaker']:14s} (conf={rec['speaker_confidence']:.2f}, "
            f"az={rec['azure_speaker']})  {rec['text']}\n"
        )
        sys.stdout.flush()
        if ws_sink:
            ws_sink.broadcast(rec, session=ws_sink.session_id)
        if api_post_url:
            _forward_to_api(api_post_url, rec, api_token)

    if use_mic:
        device: int | str | None = mic_device
        if isinstance(device, str) and device.lstrip("-").isdigit():
            device = int(device)
        events = stream_from_microphone(
            languages=languages, chunk_ms=chunk_ms, device=device, on_event=on_event,
            resolver_holder=resolver_holder, **common,
        )
    else:
        events = stream_from_wav(
            audio_path,
            languages=languages,
            chunk_ms=chunk_ms,
            realtime_pacing=realtime_pacing,
            on_event=on_event,
            resolver_holder=resolver_holder,
            **common,
        )

    if ws_sink:
        ws_sink.stop()
    if out_path:
        Path(out_path).write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
        sys.stderr.write(f"wrote {out_path}\n")


if __name__ == "__main__":
    main()
