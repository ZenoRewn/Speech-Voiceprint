from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from stt.base import STTResult, Utterance, Word
from voiceprint.base import SpeakerSegment

UNKNOWN = "Speaker_Unknown"
MIXED = "mixed"


@dataclass
class AlignedWord:
    text: str
    start: float
    end: float
    speaker: str
    confidence: float | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "speaker": self.speaker,
            "confidence": self.confidence,
        }


@dataclass
class AlignedUtterance:
    text: str
    start: float
    end: float
    speaker: str
    speaker_confidence: float
    azure_speaker: str | None
    words: list[AlignedWord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "speaker": self.speaker,
            "speaker_confidence": round(self.speaker_confidence, 3),
            "azure_speaker": self.azure_speaker,
            "words": [w.to_dict() for w in self.words],
        }


def _segment_for_word(word: Word, segments: list[SpeakerSegment]) -> SpeakerSegment | None:
    """Pick the speaker segment that maximally overlaps the word.

    Falls back to the segment containing the word midpoint when nothing overlaps.
    """
    best: SpeakerSegment | None = None
    best_overlap = 0.0
    for seg in segments:
        ov = min(word.end, seg.end) - max(word.start, seg.start)
        if ov > best_overlap:
            best_overlap = ov
            best = seg
    if best is not None:
        return best
    mid = word.midpoint
    for seg in segments:
        if seg.start <= mid <= seg.end:
            return seg
    return None


def _utterance_speaker(words: list[AlignedWord], threshold: float) -> tuple[str, float]:
    """Vote majority speaker by word count. Below `threshold`, label as `mixed`."""
    if not words:
        return UNKNOWN, 0.0
    counts: Counter[str] = Counter(w.speaker for w in words if w.speaker != UNKNOWN)
    if not counts:
        return UNKNOWN, 0.0
    top, top_count = counts.most_common(1)[0]
    ratio = top_count / len(words)
    if ratio < threshold:
        return MIXED, ratio
    return top, ratio


def align(
    stt: STTResult,
    segments: list[SpeakerSegment],
    *,
    utterance_majority_threshold: float = 0.7,
    drop_silent_segments: bool = True,
) -> list[AlignedUtterance]:
    """Merge STT word timestamps with voiceprint segments.

    Word-level: assign by maximum overlap; fall back to midpoint containment;
    finally `Speaker_Unknown`. Utterance-level: majority vote weighted by word
    count, label as `mixed` when the dominant speaker holds < threshold share.
    `drop_silent_segments` is currently informational — segments without any
    overlapping STT words contribute nothing to the output, which is the
    desired behavior; the flag is reserved for a future `[silence]` injection.
    """
    _ = drop_silent_segments  # reserved
    sorted_segments = sorted(segments, key=lambda s: s.start)
    out: list[AlignedUtterance] = []

    for utt in stt.utterances:
        aligned_words: list[AlignedWord] = []
        for w in utt.words:
            seg = _segment_for_word(w, sorted_segments)
            speaker = seg.local_label if seg is not None else UNKNOWN
            aligned_words.append(
                AlignedWord(
                    text=w.text,
                    start=w.start,
                    end=w.end,
                    speaker=speaker,
                    confidence=w.confidence,
                )
            )
        speaker, ratio = _utterance_speaker(aligned_words, utterance_majority_threshold)
        out.append(
            AlignedUtterance(
                text=utt.text,
                start=utt.start,
                end=utt.end,
                speaker=speaker,
                speaker_confidence=ratio,
                azure_speaker=utt.azure_speaker,
                words=aligned_words,
            )
        )
    return out
