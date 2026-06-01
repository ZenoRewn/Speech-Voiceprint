"""Azure Speech Batch Transcription v3.2 adapter.

Workflow:
  1. POST /speechtotext/transcriptions:submit?api-version=2024-11-15 with
     `contentUrls` (list of SAS URLs) + properties (diarization, word
     timestamps, optional language identification).
  2. Poll the returned `self` URI until `status == "Succeeded"`.
  3. GET `<self>/files`, find the file whose `kind == "Transcription"`, follow
     its `links.contentUrl`, download the JSON payload.
  4. Reshape to STTResult — the v3.2 transcript schema is similar to Fast
     Transcription but uses `recognizedPhrases` instead of `phrases`, with
     `offset`/`duration` as ISO-8601 strings rather than millisecond ints.

For diarization the SDK accepts:
  * 2 speakers     → properties.diarizationEnabled = true
  * 3-35 speakers  → properties.diarization = {speakers: {minCount, maxCount}}
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Iterator

import requests

from ._retry import retrying_request
from .azure_host import host_for
from .base import STTResult, Utterance, Word

API_VERSION = "2024-11-15"
DEFAULT_POLL_INTERVAL = 10.0
DEFAULT_POLL_INTERVAL_MAX = 60.0
DEFAULT_POLL_TIMEOUT = 6 * 3600  # 6 hours

log = logging.getLogger(__name__)


def _parse_iso_duration(value: str) -> float:
    """Convert an ISO-8601 duration like `PT2.04S` or `PT1M3.4S` to seconds.

    Batch v3.2 returns offsets/durations in this format. Whisper-style models
    sometimes emit `PT0M0.84S`; the regex covers both.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?", value)
    if not m:
        return 0.0
    h = float(m.group(1) or 0)
    mi = float(m.group(2) or 0)
    s = float(m.group(3) or 0)
    return h * 3600 + mi * 60 + s


class AzureBatchTranscription:
    def __init__(
        self,
        speech_key: str | None = None,
        speech_region: str | None = None,
        endpoint_host: str | None = None,
    ) -> None:
        self.speech_key = speech_key or os.environ.get("AZURE_SPEECH_KEY")
        self.speech_region = speech_region or os.environ.get("AZURE_SPEECH_REGION")
        if not self.speech_key:
            raise ValueError("AZURE_SPEECH_KEY missing")
        self.host = host_for(self.speech_region, "batch", override=endpoint_host)

    @property
    def _headers(self) -> dict[str, str]:
        return {"Ocp-Apim-Subscription-Key": self.speech_key, "Content-Type": "application/json"}

    # ------------------------------------------------------------------ submit
    def submit(
        self,
        content_urls: list[str],
        *,
        locale: str = "en-US",
        candidate_locales: list[str] | None = None,
        diarization: bool = True,
        max_speakers: int = 6,
        word_timestamps: bool = True,
        display_name: str = "speech-voiceprint batch",
        time_to_live_hours: int = 48,
    ) -> str:
        url = f"{self.host}/speechtotext/transcriptions:submit?api-version={API_VERSION}"
        properties: dict = {
            "wordLevelTimestampsEnabled": word_timestamps,
            "timeToLiveHours": time_to_live_hours,
        }
        if diarization:
            if max_speakers <= 2:
                properties["diarizationEnabled"] = True
            else:
                properties["diarization"] = {
                    "speakers": {"minCount": 2, "maxCount": int(max_speakers)},
                }
        if candidate_locales:
            properties["languageIdentification"] = {
                "candidateLocales": list(candidate_locales)
            }
        body = {
            "contentUrls": content_urls,
            "locale": locale,
            "displayName": display_name,
            "properties": properties,
        }
        resp = retrying_request(
            lambda: requests.post(url, headers=self._headers, json=body, timeout=60),
            op_name="azure-batch-submit",
        )
        resp.raise_for_status()
        log.info("batch submitted", extra={"uri": resp.json().get("self")})
        return resp.json()["self"]  # transcription URI

    # ------------------------------------------------------------------ poll
    def wait(
        self,
        transcription_uri: str,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        poll_interval_max: float = DEFAULT_POLL_INTERVAL_MAX,
        timeout: float = DEFAULT_POLL_TIMEOUT,
        on_status=None,
    ) -> dict:
        deadline = time.time() + timeout
        current_interval = poll_interval
        while True:
            resp = retrying_request(
                lambda: requests.get(transcription_uri, headers=self._headers, timeout=30),
                op_name="azure-batch-poll",
            )
            resp.raise_for_status()
            payload = resp.json()
            status = payload.get("status", "Unknown")
            if on_status:
                on_status(status, payload)
            if status == "Succeeded":
                return payload
            if status == "Failed":
                err = payload.get("properties", {}).get("error", {})
                raise RuntimeError(f"batch failed: {err}")
            if time.time() > deadline:
                raise TimeoutError(f"batch transcription timed out after {timeout}s")
            log.debug("batch status %s, sleeping %.1fs", status, current_interval)
            time.sleep(current_interval)
            current_interval = min(current_interval * 1.5, poll_interval_max)

    # ------------------------------------------------------------------ fetch
    def fetch_results(self, transcription_uri: str) -> list[dict]:
        files_url = f"{transcription_uri.rstrip('/')}/files"
        sep = "&" if "?" in files_url else "?"
        if "api-version=" not in files_url:
            files_url = f"{files_url}{sep}api-version={API_VERSION}"
        resp = retrying_request(
            lambda: requests.get(files_url, headers=self._headers, timeout=60),
            op_name="azure-batch-files",
        )
        resp.raise_for_status()
        files = resp.json().get("values", [])
        results: list[dict] = []
        for f in files:
            if f.get("kind") != "Transcription":
                continue
            content_url = f["links"]["contentUrl"]
            r2 = retrying_request(
                lambda url=content_url: requests.get(url, timeout=300),
                op_name="azure-batch-download",
            )
            r2.raise_for_status()
            results.append(r2.json())
        return results

    # ------------------------------------------------------------------ run
    def transcribe_batch(
        self,
        audio_url: str | list[str],
        *,
        languages: list[str] | None = None,
        diarization: bool = True,
        max_speakers: int = 6,
        on_status=None,
    ) -> STTResult:
        urls = [audio_url] if isinstance(audio_url, str) else list(audio_url)
        if not urls:
            raise ValueError("at least one content URL required")
        primary_locale = (languages or ["en-US"])[0]
        candidates = languages if languages and len(languages) > 1 else None
        uri = self.submit(
            urls,
            locale=primary_locale,
            candidate_locales=candidates,
            diarization=diarization,
            max_speakers=max_speakers,
        )
        if on_status:
            on_status("Submitted", {"self": uri})
        self.wait(uri, on_status=on_status)
        files = self.fetch_results(uri)
        if not files:
            raise RuntimeError("batch succeeded but produced no transcription files")
        return self._merge_files(files)

    # ------------------------------------------------------------------ parse
    @classmethod
    def _merge_files(cls, files: list[dict]) -> STTResult:
        utterances: list[Utterance] = []
        total_duration = 0.0
        language: str | None = None
        for data in files:
            duration_str = data.get("duration")
            total_duration = max(total_duration, _parse_iso_duration(duration_str))
            for phrase in data.get("recognizedPhrases", []):
                phrase_start = _parse_iso_duration(phrase.get("offset"))
                phrase_dur = _parse_iso_duration(phrase.get("duration"))
                if not language:
                    language = phrase.get("locale")
                speaker = phrase.get("speaker")
                azure_speaker = f"Guest-{speaker}" if speaker is not None else None

                nbest = phrase.get("nBest") or []
                if not nbest:
                    continue
                top = nbest[0]
                phrase_text = top.get("display") or top.get("lexical") or ""
                words: list[Word] = []
                for w in top.get("displayWords") or top.get("words") or []:
                    w_start = _parse_iso_duration(w.get("offset"))
                    w_dur = _parse_iso_duration(w.get("duration"))
                    words.append(
                        Word(
                            text=w.get("displayText") or w.get("word") or "",
                            start=w_start,
                            end=w_start + w_dur,
                            confidence=top.get("confidence"),
                        )
                    )
                utterances.append(
                    Utterance(
                        text=phrase_text,
                        start=phrase_start,
                        end=phrase_start + phrase_dur,
                        words=words,
                        azure_speaker=azure_speaker,
                    )
                )
        utterances.sort(key=lambda u: u.start)
        return STTResult(utterances=utterances, language=language, duration=total_duration, raw=files)

    # parity with STTProvider protocol
    def transcribe_fast(self, audio_path: str, languages: list[str] | None = None) -> STTResult:
        raise NotImplementedError("use AzureFastTranscription for inline files")

    def transcribe_stream(self, audio_chunks: Iterator[bytes]) -> Iterator[Utterance]:
        raise NotImplementedError("use AzureRealtimeTranscriber for streaming")
