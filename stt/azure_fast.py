from __future__ import annotations

import json
import logging
import os
from typing import Iterator

import requests

from ._retry import retrying_request
from .azure_host import host_for
from .base import STTResult, Utterance, Word

API_VERSION = "2025-10-15"
DEFAULT_TIMEOUT = 600

log = logging.getLogger(__name__)


class AzureFastTranscription:
    """Azure Speech Fast Transcription REST adapter.

    Endpoint: POST {host}/speechtotext/transcriptions:transcribe?api-version=2025-10-15
    Returns display-form transcription with word-level timestamps in every phrase.
    """

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
        self.host = host_for(self.speech_region, "fast", override=endpoint_host)

    def transcribe_fast(
        self,
        audio_path: str,
        languages: list[str] | None = None,
        max_speakers_hint: int = 6,
        diarization: bool = True,
        phrase_list: list[str] | None = None,
    ) -> STTResult:
        url = f"{self.host}/speechtotext/transcriptions:transcribe?api-version={API_VERSION}"

        definition: dict = {}
        if languages:
            definition["locales"] = languages
        if diarization:
            definition["diarization"] = {"enabled": True, "maxSpeakers": max_speakers_hint}
        if phrase_list:
            definition["phraseList"] = {"phrases": phrase_list}

        headers = {"Ocp-Apim-Subscription-Key": self.speech_key}

        def _post() -> requests.Response:
            with open(audio_path, "rb") as f:
                files = {
                    "audio": (os.path.basename(audio_path), f, "application/octet-stream"),
                    "definition": (None, json.dumps(definition), "application/json"),
                }
                return requests.post(url, headers=headers, files=files, timeout=DEFAULT_TIMEOUT)

        log.info("fast transcribe submit", extra={"audio": audio_path, "languages": languages})
        resp = retrying_request(_post, op_name="azure-fast-transcribe")
        resp.raise_for_status()
        return self._parse_response(resp.json())

    def transcribe_batch(self, audio_url: str, languages: list[str] | None = None) -> STTResult:
        # M4 will switch to v3.2 batch endpoint with SAS URL + polling.
        raise NotImplementedError("Use transcribe_fast for now; batch lands in M4")

    def transcribe_stream(self, audio_chunks: Iterator[bytes]) -> Iterator[Utterance]:
        # M3 will implement realtime via azure-cognitiveservices-speech SDK.
        raise NotImplementedError("Realtime streaming lands in M3")

    @staticmethod
    def _parse_response(data: dict) -> STTResult:
        utterances: list[Utterance] = []
        for phrase in data.get("phrases", []):
            phrase_start = phrase.get("offsetMilliseconds", 0) / 1000.0
            phrase_dur = phrase.get("durationMilliseconds", 0) / 1000.0
            words: list[Word] = []
            for w in phrase.get("words", []):
                w_start = w.get("offsetMilliseconds", 0) / 1000.0
                w_dur = w.get("durationMilliseconds", 0) / 1000.0
                words.append(
                    Word(
                        text=w["text"],
                        start=w_start,
                        end=w_start + w_dur,
                        confidence=phrase.get("confidence"),
                    )
                )
            azure_speaker = phrase.get("speaker")
            azure_speaker_label = f"Guest-{azure_speaker}" if azure_speaker is not None else None
            utterances.append(
                Utterance(
                    text=phrase.get("text", ""),
                    start=phrase_start,
                    end=phrase_start + phrase_dur,
                    words=words,
                    azure_speaker=azure_speaker_label,
                )
            )

        duration = data.get("durationMilliseconds", 0) / 1000.0
        language = None
        phrases = data.get("phrases", [])
        if phrases:
            language = phrases[0].get("locale")
        return STTResult(utterances=utterances, language=language, duration=duration, raw=data)
