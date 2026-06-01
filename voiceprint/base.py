from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class SpeakerSegment:
    start: float
    end: float
    local_label: str          # 会话内匿名标签:Speaker_A / Speaker_B
    embedding: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class VoiceprintProvider(Protocol):
    name: str
    embedding_dim: int

    def diarize(self, audio_path: str, num_speakers: int | None = None) -> list[SpeakerSegment]: ...

    def embed(self, audio_path: str, start: float, end: float) -> np.ndarray: ...
