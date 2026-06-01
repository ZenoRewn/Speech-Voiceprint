from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import click

import numpy as np

from merger.aligner import MIXED, UNKNOWN, AlignedUtterance, align
from pipeline.config import load_config, merge_cli, pick
from stt.azure_batch import AzureBatchTranscription
from stt.azure_fast import AzureFastTranscription
from stt.base import STTResult
from voiceprint.base import SpeakerSegment, VoiceprintProvider


def build_voiceprint_provider(backend: str, hf_token: str | None, device: str) -> VoiceprintProvider:
    if backend == "pyannote":
        from voiceprint.pyannote_provider import PyannoteProvider

        return PyannoteProvider(hf_token=hf_token, device=device)
    if backend == "speechbrain":
        from voiceprint.speechbrain_provider import SpeechBrainProvider

        return SpeechBrainProvider(device=device)
    raise click.BadParameter(f"unknown voiceprint backend: {backend}")


def _public_audio_label(value: str, *, is_url: bool = False) -> str:
    """Return a non-sensitive input label for result JSON.

    Local runs used to serialize absolute host paths, and batch mode used to
    serialize the full SAS URL. Both are useful for debugging but unsafe for
    shared artifacts, so results keep only the basename or URL path.
    """
    if is_url:
        parts = urlsplit(value)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    return Path(value).name


def _apply_registry(
    segments: list[SpeakerSegment],
    aligned: list[AlignedUtterance],
    backend_name: str,
    embedding_dim: int,
    *,
    registry_path: str,
    match_threshold: float,
    unknown_threshold: float,
    auto_enroll_unknown: bool,
    max_per_speaker: int,
) -> dict:
    """Resolve cluster-local labels to persistent registry identities.

    Mutates `segments` and `aligned` in place. The returned dict feeds the
    output payload so downstream consumers can see which utterances were
    matched to a known speaker, enrolled as unknown, or kept as low_confidence.
    """
    from registry import open_store
    from registry.matcher import MatchVerdict, SpeakerMatcher

    store = open_store(registry_path)
    model_id = f"{backend_name}-{embedding_dim}"
    matcher = SpeakerMatcher(
        store,
        model=model_id,
        match_threshold=match_threshold,
        unknown_threshold=unknown_threshold,
        max_voiceprints_per_speaker=max_per_speaker,
        auto_enroll_unknown=auto_enroll_unknown,
    )

    verdicts = matcher.assign_local_labels(segments)
    label_map: dict[str, str] = {}
    label_meta: dict[str, dict] = {}
    for local_label, res in verdicts.items():
        new_label = SpeakerMatcher.label_for(res, fallback=local_label)
        label_map[local_label] = new_label
        label_meta[local_label] = {
            "verdict": res.verdict.value,
            "speaker_id": res.speaker_id,
            "display_name": res.display_name,
            "distance": round(res.distance, 4),
            "candidate_speaker_id": res.candidate_speaker_id,
            "candidate_distance": round(res.candidate_distance, 4) if res.candidate_distance else None,
            "resolved_label": new_label,
        }

    for seg in segments:
        if seg.local_label in label_map:
            seg.local_label = label_map[seg.local_label]
    for utt in aligned:
        if utt.speaker in label_map:
            utt.speaker = label_map[utt.speaker]
        for w in utt.words:
            if w.speaker in label_map:
                w.speaker = label_map[w.speaker]

    store.close()
    return {
        "registry_path": registry_path,
        "model": model_id,
        "match_threshold": match_threshold,
        "unknown_threshold": unknown_threshold,
        "label_resolutions": label_meta,
    }


def _majority_owner(
    start: float, end: float, primary_spans: list[tuple[float, float, str | None]]
) -> str | None:
    """Return the primary speaker_id whose segments cover the most of [start,end]."""
    if end <= start:
        return None
    overlap: dict[str, float] = {}
    for ps, pe, sid in primary_spans:
        if not sid:
            continue
        ovl = max(0.0, min(end, pe) - max(start, ps))
        if ovl > 0:
            overlap[sid] = overlap.get(sid, 0.0) + ovl
    if not overlap:
        return None
    return max(overlap.items(), key=lambda kv: kv[1])[0]


def _dual_enroll(
    *,
    audio_path: str,
    secondary: VoiceprintProvider,
    primary_spans: list[tuple[float, float, str | None]],
    registry_path: str,
    max_per_speaker: int,
) -> dict:
    """Run the secondary backend on the same audio and write its embeddings
    under the same speaker_ids the primary already resolved.

    The secondary never *creates* speakers — it only adds voiceprints to ones
    the primary identified by time overlap. That keeps the registry's source of
    truth (segment boundaries + speaker_ids) anchored to one backend per run.
    """
    from registry import open_store

    sec_segments = secondary.diarize(audio_path)
    sec_dim = getattr(secondary, "embedding_dim", 0)
    sec_model = f"{secondary.name}-{sec_dim}"

    # Group secondary embeddings by which primary speaker they overlap most.
    bucket: dict[str, list[np.ndarray]] = {}
    for seg in sec_segments:
        if seg.embedding is None:
            continue
        owner = _majority_owner(seg.start, seg.end, primary_spans)
        if owner:
            bucket.setdefault(owner, []).append(seg.embedding)

    if not bucket:
        return {
            "model": sec_model,
            "enrolled_speaker_ids": [],
            "skipped_reason": "no overlap with primary speakers",
        }

    enrolled: list[str] = []
    with open_store(registry_path) as store:
        for sid, embs in bucket.items():
            mean_emb = np.mean(np.stack(embs), axis=0)
            store.add_voiceprint(
                sid, mean_emb, sec_model, quality=1.0, max_per_speaker=max_per_speaker
            )
            enrolled.append(sid)
    return {"model": sec_model, "enrolled_speaker_ids": enrolled}


def _run_pipeline(
    *,
    stt_result: STTResult,
    audio_path_for_voiceprint: str,
    voiceprint: VoiceprintProvider,
    num_speakers: int | None,
    majority_threshold: float,
    registry_path: str | None,
    match_threshold: float,
    unknown_threshold: float,
    auto_enroll_unknown: bool,
    max_per_speaker: int,
    audio_label: str,
    secondary_voiceprint: VoiceprintProvider | None = None,
) -> dict:
    """Shared post-STT plumbing:
    1. Run voiceprint diarization on the local audio
    2. Align word/utterance timestamps with speaker segments
    3. Optionally resolve cluster labels to persistent registry identities
    """
    segments: list[SpeakerSegment] = voiceprint.diarize(
        audio_path_for_voiceprint, num_speakers=num_speakers
    )
    aligned = align(stt_result, segments, utterance_majority_threshold=majority_threshold)

    payload = {
        "audio": audio_label,
        "language": stt_result.language,
        "duration": stt_result.duration,
        "voiceprint_backend": voiceprint.name,
        "voiceprint_segments": [
            {"start": round(s.start, 3), "end": round(s.end, 3), "local_label": s.local_label}
            for s in segments
        ],
    }
    if registry_path:
        # Snapshot pre-mutation (start, end, original_local_label) so dual-enroll
        # can map segments_B's time spans back to primary speaker_ids via label_meta.
        original_spans = [(s.start, s.end, s.local_label) for s in segments]
        payload["registry"] = _apply_registry(
            segments,
            aligned,
            backend_name=voiceprint.name,
            embedding_dim=getattr(voiceprint, "embedding_dim", 0),
            registry_path=registry_path,
            match_threshold=match_threshold,
            unknown_threshold=unknown_threshold,
            auto_enroll_unknown=auto_enroll_unknown,
            max_per_speaker=max_per_speaker,
        )
        payload["voiceprint_segments"] = [
            {"start": round(s.start, 3), "end": round(s.end, 3), "local_label": s.local_label}
            for s in segments
        ]

        if secondary_voiceprint is not None:
            label_meta = payload["registry"].get("label_resolutions", {})
            primary_spans = [
                (start, end, (label_meta.get(orig_label) or {}).get("speaker_id"))
                for (start, end, orig_label) in original_spans
            ]
            try:
                payload["dual_enroll"] = _dual_enroll(
                    audio_path=audio_path_for_voiceprint,
                    secondary=secondary_voiceprint,
                    primary_spans=primary_spans,
                    registry_path=registry_path,
                    max_per_speaker=max_per_speaker,
                )
            except Exception as e:  # noqa: BLE001 — best-effort side enrollment
                import logging
                logging.getLogger(__name__).warning("dual_enroll failed: %s", e)
                payload["dual_enroll"] = {"error": f"{type(e).__name__}: {e}"}

    payload["utterances"] = [u.to_dict() for u in aligned]
    return payload


def run_fast(
    audio_path: str,
    languages: list[str] | None,
    voiceprint: VoiceprintProvider,
    num_speakers: int | None,
    majority_threshold: float,
    *,
    registry_path: str | None = None,
    match_threshold: float = 0.40,
    unknown_threshold: float = 0.60,
    auto_enroll_unknown: bool = True,
    max_per_speaker: int = 5,
    secondary_voiceprint: VoiceprintProvider | None = None,
) -> dict:
    stt_client = AzureFastTranscription()
    stt_result: STTResult = stt_client.transcribe_fast(audio_path, languages=languages)
    return _run_pipeline(
        stt_result=stt_result,
        audio_path_for_voiceprint=audio_path,
        voiceprint=voiceprint,
        num_speakers=num_speakers,
        majority_threshold=majority_threshold,
        registry_path=registry_path,
        match_threshold=match_threshold,
        unknown_threshold=unknown_threshold,
        auto_enroll_unknown=auto_enroll_unknown,
        max_per_speaker=max_per_speaker,
        audio_label=_public_audio_label(audio_path),
        secondary_voiceprint=secondary_voiceprint,
    )


def run_batch(
    audio_url: str,
    languages: list[str] | None,
    voiceprint: VoiceprintProvider,
    num_speakers: int | None,
    majority_threshold: float,
    *,
    local_audio_for_voiceprint: str,
    registry_path: str | None = None,
    match_threshold: float = 0.40,
    unknown_threshold: float = 0.60,
    auto_enroll_unknown: bool = True,
    max_per_speaker: int = 5,
    on_status=None,
    secondary_voiceprint: VoiceprintProvider | None = None,
) -> dict:
    """Submit a SAS URL to Azure Batch v3.2 and run voiceprint locally.

    Voiceprint diarization always runs on the *local* audio file because we
    need raw PCM to embed; the SAS URL is only for STT. Caller is responsible
    for ensuring `local_audio_for_voiceprint` matches the audio at the URL.
    """
    stt_client = AzureBatchTranscription()
    stt_result: STTResult = stt_client.transcribe_batch(
        audio_url,
        languages=languages,
        diarization=True,
        max_speakers=max(num_speakers or 6, 2),
        on_status=on_status,
    )
    return _run_pipeline(
        stt_result=stt_result,
        audio_path_for_voiceprint=local_audio_for_voiceprint,
        voiceprint=voiceprint,
        num_speakers=num_speakers,
        majority_threshold=majority_threshold,
        registry_path=registry_path,
        match_threshold=match_threshold,
        unknown_threshold=unknown_threshold,
        auto_enroll_unknown=auto_enroll_unknown,
        max_per_speaker=max_per_speaker,
        audio_label=_public_audio_label(audio_url, is_url=True),
        secondary_voiceprint=secondary_voiceprint,
    )


@click.command()
@click.option("--config", "config_path", type=click.Path(dir_okay=False), default=None,
              help="YAML config file; CLI flags override these values")
@click.option("--audio", "audio_path", type=click.Path(exists=True, dir_okay=False), default=None,
              help="local audio file (required for fast; required-as-voiceprint-source for batch)")
@click.option("--audio-url", default=None,
              help="SAS URL for batch mode")
@click.option("--mode", type=click.Choice(["fast", "batch", "realtime"]), default=None)
@click.option("--backend", type=click.Choice(["pyannote", "speechbrain"]), default=None)
@click.option("--language", "languages", multiple=True, help="locale, repeatable")
@click.option("--num-speakers", type=int, default=None)
@click.option("--majority-threshold", type=float, default=None)
@click.option("--device", default=None)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=None)
@click.option("--registry", "registry_path", type=click.Path(dir_okay=False), default=None,
              help="enable cross-session speaker registry at this SQLite path")
@click.option("--match-threshold", type=float, default=None)
@click.option("--unknown-threshold", type=float, default=None)
@click.option("--no-auto-enroll", is_flag=True, default=False, help="don't auto-enroll unknowns")
@click.option("--max-per-speaker", type=int, default=None)
def main(
    config_path: str | None,
    audio_path: str | None,
    audio_url: str | None,
    mode: str | None,
    backend: str | None,
    languages: tuple[str, ...],
    num_speakers: int | None,
    majority_threshold: float | None,
    device: str | None,
    out_path: str | None,
    registry_path: str | None,
    match_threshold: float | None,
    unknown_threshold: float | None,
    no_auto_enroll: bool,
    max_per_speaker: int | None,
) -> None:
    from pipeline.env_file import load_env_file
    load_env_file()
    cfg = load_config(config_path)

    # Three-layer merge: CLI flag > YAML > hard-coded default.
    mode = merge_cli(cfg, mode, "stt.mode", default="fast")
    backend = merge_cli(cfg, backend, "voiceprint.backend", default="pyannote")
    audio_path = merge_cli(cfg, audio_path, "audio.path")
    audio_url = merge_cli(cfg, audio_url, "audio.url")
    out_path = merge_cli(cfg, out_path, "output.path")
    registry_path = merge_cli(cfg, registry_path, "registry.store_path")
    languages = merge_cli(cfg, list(languages) or None, "azure.languages")
    num_speakers = merge_cli(cfg, num_speakers, "voiceprint.num_speakers")
    majority_threshold = merge_cli(cfg, majority_threshold, "aligner.utterance_majority_threshold", default=0.7)
    device = merge_cli(cfg, device, "voiceprint.device", default="auto")
    match_threshold = merge_cli(cfg, match_threshold, "registry.match_threshold", default=0.40)
    unknown_threshold = merge_cli(cfg, unknown_threshold, "registry.unknown_threshold", default=0.60)
    max_per_speaker = merge_cli(cfg, max_per_speaker, "registry.max_voiceprints_per_speaker", default=5)
    auto_enroll_unknown = not no_auto_enroll
    if "registry" in cfg and pick(cfg, "registry.auto_enroll_unknown") is False:
        auto_enroll_unknown = False
    if "registry" in cfg and pick(cfg, "registry.enabled") is False:
        registry_path = None  # explicit YAML disable wins over a stale CLI flag

    if mode == "realtime":
        raise click.BadParameter("realtime: use `python -m pipeline.streaming` instead")
    if mode == "fast":
        if not audio_path:
            raise click.BadParameter("--audio is required in fast mode")
        voiceprint = build_voiceprint_provider(backend, os.environ.get("HF_TOKEN"), device)
        payload = run_fast(
            audio_path,
            languages or None,
            voiceprint,
            num_speakers,
            majority_threshold,
            registry_path=registry_path,
            match_threshold=match_threshold,
            unknown_threshold=unknown_threshold,
            auto_enroll_unknown=auto_enroll_unknown,
            max_per_speaker=max_per_speaker,
        )
    else:  # batch
        if not audio_url:
            raise click.BadParameter("--audio-url (SAS URL) is required in batch mode")
        if not audio_path:
            raise click.BadParameter("--audio (local copy) is required for voiceprint extraction in batch mode")
        voiceprint = build_voiceprint_provider(backend, os.environ.get("HF_TOKEN"), device)
        click.echo(f"submitting batch transcription for {audio_url}", err=True)
        payload = run_batch(
            audio_url,
            languages or None,
            voiceprint,
            num_speakers,
            majority_threshold,
            local_audio_for_voiceprint=audio_path,
            registry_path=registry_path,
            match_threshold=match_threshold,
            unknown_threshold=unknown_threshold,
            auto_enroll_unknown=auto_enroll_unknown,
            max_per_speaker=max_per_speaker,
            on_status=lambda status, _payload: click.echo(f"  status: {status}", err=True),
        )

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
        click.echo(f"wrote {out_path}", err=True)
    else:
        sys.stdout.write(text)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
