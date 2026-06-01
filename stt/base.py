from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    confidence: float | None = None

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0


@dataclass
class Utterance:
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)
    azure_speaker: str | None = None  # 来自 Azure 的 Guest-N,仅作降级参考


@dataclass
class STTResult:
    utterances: list[Utterance]
    language: str | None = None
    duration: float | None = None
    raw: dict | None = None

    @property
    def words(self) -> list[Word]:
        out: list[Word] = []
        for u in self.utterances:
            out.extend(u.words)
        return out


class STTProvider(Protocol):
    def transcribe_fast(self, audio_path: str, languages: list[str] | None = None) -> STTResult: ...

    def transcribe_batch(self, audio_url: str, languages: list[str] | None = None) -> STTResult: ...

    def transcribe_stream(self, audio_chunks: Iterator[bytes]) -> Iterator[Utterance]: ...
