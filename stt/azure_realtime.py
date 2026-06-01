"""Azure realtime STT with word timestamps via ConversationTranscriber.

We use ConversationTranscriber for utterance + word timing; Azure's own
speaker_id (`Guest-N`) is recorded as a fallback hint but the authoritative
speaker label is supplied by the streaming voiceprint pipeline.
"""

from __future__ import annotations

import os
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass

import azure.cognitiveservices.speech as speechsdk

from .base import Utterance, Word

PCM_BYTES_PER_SECOND = 16000 * 2  # 16k mono int16


@dataclass
class TranscriptionEvent:
    text: str
    start: float
    end: float
    words: list[Word]
    azure_speaker: str | None
    is_final: bool


class AzureRealtimeTranscriber:
    """Push raw 16k mono PCM chunks; consume `events()` to receive transcribed
    utterances. Exits cleanly when the producer calls `close()`.
    """

    def __init__(
        self,
        speech_key: str | None = None,
        speech_region: str | None = None,
        languages: list[str] | None = None,
        diarization: bool = True,
    ) -> None:
        self.speech_key = speech_key or os.environ.get("AZURE_SPEECH_KEY")
        self.speech_region = speech_region or os.environ.get("AZURE_SPEECH_REGION")
        if not self.speech_key or not self.speech_region:
            raise ValueError("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION are required")
        self.languages = languages or ["en-US"]
        self.diarization = diarization

        self._event_queue: queue.Queue[TranscriptionEvent | None] = queue.Queue()
        self._push_stream: speechsdk.audio.PushAudioInputStream | None = None
        self._transcriber: speechsdk.transcription.ConversationTranscriber | None = None
        self._stopped = threading.Event()

    # ------------------------------------------------------------------
    def start(self) -> None:
        speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
        speech_config.request_word_level_timestamps()
        speech_config.set_property(
            property_id=speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults,
            value="true" if self.diarization else "false",
        )
        speech_config.output_format = speechsdk.OutputFormat.Detailed

        format = speechsdk.audio.AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)
        self._push_stream = speechsdk.audio.PushAudioInputStream(format)
        audio_config = speechsdk.audio.AudioConfig(stream=self._push_stream)

        if len(self.languages) > 1:
            auto = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=self.languages)
            self._transcriber = speechsdk.transcription.ConversationTranscriber(
                speech_config=speech_config,
                audio_config=audio_config,
                auto_detect_source_language_config=auto,
            )
        else:
            speech_config.speech_recognition_language = self.languages[0]
            self._transcriber = speechsdk.transcription.ConversationTranscriber(
                speech_config=speech_config,
                audio_config=audio_config,
            )

        self._transcriber.transcribed.connect(self._on_transcribed)
        self._transcriber.session_stopped.connect(self._on_session_end)
        self._transcriber.canceled.connect(self._on_cancel)

        self._transcriber.start_transcribing_async().get()

    # ------------------------------------------------------------------
    def push(self, pcm: bytes) -> None:
        if self._push_stream is None:
            raise RuntimeError("call start() first")
        self._push_stream.write(pcm)

    def end_input(self, drain_timeout: float = 30.0) -> bool:
        """Signal end of audio and wait for Azure to flush remaining utterances.

        Returns True if `session_stopped` fired within `drain_timeout`. After
        this returns the SDK has emitted every utterance it intends to; safe
        to call `close()`.
        """
        if self._push_stream is not None:
            self._push_stream.close()
        return self._stopped.wait(drain_timeout)

    def close(self) -> None:
        if self._push_stream is not None:
            try:
                self._push_stream.close()
            except Exception:
                pass
        if self._transcriber is not None:
            self._transcriber.stop_transcribing_async().get()
        self._event_queue.put(None)

    def events(self, timeout: float | None = None) -> Iterator[TranscriptionEvent]:
        while True:
            try:
                evt = self._event_queue.get(timeout=timeout)
            except queue.Empty:
                return
            if evt is None:
                return
            yield evt

    # ------------------------------------------------------------------
    def _on_transcribed(self, evt) -> None:
        result = evt.result
        if result.reason != speechsdk.ResultReason.RecognizedSpeech:
            return
        text = result.text or ""
        if not text.strip():
            return
        offset_s = result.offset / 10_000_000
        duration_s = result.duration / 10_000_000

        words = self._extract_words(result, offset_s)
        speaker = getattr(result, "speaker_id", None)
        ev = TranscriptionEvent(
            text=text,
            start=offset_s,
            end=offset_s + duration_s,
            words=words,
            azure_speaker=speaker if speaker and speaker != "Unknown" else None,
            is_final=True,
        )
        self._event_queue.put(ev)

    def _on_session_end(self, _evt) -> None:
        self._stopped.set()
        self._event_queue.put(None)

    def _on_cancel(self, evt) -> None:
        self._stopped.set()
        self._event_queue.put(None)

    @staticmethod
    def _extract_words(result, fallback_start: float) -> list[Word]:
        try:
            import json

            raw_json = result.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult
            )
            if not raw_json:
                return []
            data = json.loads(raw_json)
        except Exception:
            return []
        nbest = data.get("NBest") or []
        if not nbest:
            return []
        words_data = nbest[0].get("Words") or []
        words: list[Word] = []
        for w in words_data:
            offset = w.get("Offset", 0) / 10_000_000
            dur = w.get("Duration", 0) / 10_000_000
            words.append(
                Word(
                    text=w.get("Word", ""),
                    start=offset,
                    end=offset + dur,
                    confidence=w.get("Confidence"),
                )
            )
        if not words:
            return []
        return words

    # convenience for offline tests / type-checking parity with STTProvider
    def transcribe_fast(self, audio_path: str, languages: list[str] | None = None):
        raise NotImplementedError("this class is realtime-only; use AzureFastTranscription for files")

    def transcribe_batch(self, audio_url: str, languages: list[str] | None = None):
        raise NotImplementedError

    def transcribe_stream(self, audio_chunks: Iterator[bytes]) -> Iterator[Utterance]:
        self.start()
        producer_done = threading.Event()

        def producer() -> None:
            for chunk in audio_chunks:
                self.push(chunk)
            producer_done.set()
            self.close()

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        for evt in self.events():
            yield Utterance(
                text=evt.text,
                start=evt.start,
                end=evt.end,
                words=evt.words,
                azure_speaker=evt.azure_speaker,
            )
        t.join()
