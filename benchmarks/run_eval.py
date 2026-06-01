"""Compare voiceprint backends on a fixed audio set.

For each (audio, backend) pair we measure:
  * STT cache hit (avoid double-billing Azure)
  * Voiceprint diarization wall time and RTF
  * Detected speaker count, segment count
  * Aligned utterance count, mixed-ratio (uncertainty), unknown-ratio

STT response is cached per-audio under .cache/stt/<sha1>.json so we only call
Azure Fast Transcription once per file across multiple backends.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from merger.aligner import MIXED, UNKNOWN, align
from stt.azure_fast import AzureFastTranscription
from stt.base import STTResult, Utterance, Word


CACHE_DIR = ROOT / ".cache" / "stt"


def _audio_fingerprint(audio_path: str) -> str:
    h = hashlib.sha1()
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _stt_cache_path(audio_path: str, languages: list[str] | None) -> Path:
    fp = _audio_fingerprint(audio_path)
    lang_tag = "+".join(languages) if languages else "auto"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{fp}__{lang_tag}.json"


def stt_cached(audio_path: str, languages: list[str] | None, refresh: bool = False) -> STTResult:
    cache = _stt_cache_path(audio_path, languages)
    if cache.exists() and not refresh:
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return AzureFastTranscription._parse_response(raw)
    client = AzureFastTranscription()
    t0 = time.perf_counter()
    result = client.transcribe_fast(audio_path, languages=languages)
    elapsed = time.perf_counter() - t0
    cache.write_text(json.dumps(result.raw, ensure_ascii=False), encoding="utf-8")
    sys.stderr.write(f"  [STT] {Path(audio_path).name}: {elapsed:.1f}s -> cached\n")
    return result


def build_provider(backend: str, device: str):
    if backend == "pyannote":
        from voiceprint.pyannote_provider import PyannoteProvider
        import os

        return PyannoteProvider(hf_token=os.environ.get("HF_TOKEN"), device=device)
    if backend == "speechbrain":
        from voiceprint.speechbrain_provider import SpeechBrainProvider

        return SpeechBrainProvider(device=device)
    raise ValueError(f"unknown backend {backend}")


def evaluate(
    audio_paths: list[str],
    backends: list[str],
    languages: list[str] | None,
    device: str,
    out_dir: Path,
    num_speakers: int | None = None,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for audio in audio_paths:
        audio_name = Path(audio).stem
        stt_result = stt_cached(audio, languages)
        duration = stt_result.duration or 0.0
        sys.stderr.write(
            f"== {audio_name}  duration={duration:.1f}s  utterances={len(stt_result.utterances)}\n"
        )

        for backend in backends:
            sys.stderr.write(f"  -- backend={backend}\n")
            provider = build_provider(backend, device)

            t0 = time.perf_counter()
            segments = provider.diarize(audio, num_speakers=num_speakers)
            wall = time.perf_counter() - t0
            rtf = wall / duration if duration > 0 else float("nan")

            aligned = align(stt_result, segments)
            utt_count = len(aligned)
            mixed = sum(1 for u in aligned if u.speaker == MIXED)
            unknown = sum(1 for u in aligned if u.speaker == UNKNOWN)
            speaker_set = {u.speaker for u in aligned if u.speaker not in (MIXED, UNKNOWN)}

            payload = {
                "audio": str(Path(audio).resolve()),
                "audio_name": audio_name,
                "duration": duration,
                "language": stt_result.language,
                "backend": backend,
                "voiceprint_segments": [
                    {"start": round(s.start, 3), "end": round(s.end, 3), "local_label": s.local_label}
                    for s in segments
                ],
                "utterances": [u.to_dict() for u in aligned],
                "metrics": {
                    "diarize_wall_seconds": round(wall, 3),
                    "rtf": round(rtf, 4),
                    "segment_count": len(segments),
                    "speaker_count": len(speaker_set),
                    "utterance_count": utt_count,
                    "mixed_utterances": mixed,
                    "unknown_utterances": unknown,
                    "mixed_ratio": round(mixed / utt_count, 3) if utt_count else 0.0,
                },
            }

            json_path = out_dir / f"{audio_name}__{backend}.json"
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            row = {"audio": audio_name, **{"duration": duration}, **payload["metrics"], "backend": backend}
            sys.stderr.write(
                f"     {wall:.1f}s wall  RTF={rtf:.3f}  "
                f"speakers={row['speaker_count']}  segs={row['segment_count']}  "
                f"mixed={row['mixed_utterances']}/{row['utterance_count']}\n"
            )
            rows.append(row)

    return rows


def render_report(rows: list[dict], out_path: Path) -> None:
    lines: list[str] = ["# Voiceprint Backend Comparison", ""]
    lines.append("| audio | duration(s) | backend | RTF | speakers | segments | utterances | mixed | mixed_ratio |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['audio']} | {r['duration']:.1f} | {r['backend']} | {r['rtf']:.3f} | "
            f"{r['speaker_count']} | {r['segment_count']} | {r['utterance_count']} | "
            f"{r['mixed_utterances']} | {r['mixed_ratio']:.3f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stderr.write(f"report -> {out_path}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", action="append", required=True, help="repeatable")
    ap.add_argument("--backend", action="append", default=None)
    ap.add_argument("--language", action="append", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "benchmarks" / "out"))
    ap.add_argument("--report", default=str(ROOT / "benchmarks" / "report.md"))
    args = ap.parse_args()

    backends = args.backend or ["pyannote", "speechbrain"]
    rows = evaluate(args.audio, backends, args.language, args.device, Path(args.out_dir), args.num_speakers)
    render_report(rows, Path(args.report))


if __name__ == "__main__":
    main()
