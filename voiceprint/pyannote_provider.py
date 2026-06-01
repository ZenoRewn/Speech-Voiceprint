from __future__ import annotations

import os
import string
import warnings
from functools import lru_cache

import numpy as np

from .base import SpeakerSegment

# pyannote/audio's stat-pooling layer triggers a torch UserWarning
# ("std(): degrees of freedom is <= 0") on extremely short windows. The
# warning is benign — std() falls back to 0 for those windows — and floods the
# log on every diarize() call. Filter it once at import.
warnings.filterwarnings(
    "ignore",
    message=r"std\(\): degrees of freedom is <= 0.*",
    category=UserWarning,
    module=r"pyannote\.audio\..*",
)


def _label_for(idx: int) -> str:
    if idx < 26:
        return f"Speaker_{string.ascii_uppercase[idx]}"
    return f"Speaker_{idx}"


class PyannoteProvider:
    """pyannote.audio backend.

    Uses the speaker-diarization-community-1 pipeline for VAD + segmentation +
    embedding + clustering, then re-embeds each segment with pyannote/embedding
    so we can match against a registry. Imports are lazy to keep `pip install -e .`
    optional unless the user actually opts into this backend.
    """

    name = "pyannote"
    embedding_dim = 512

    def __init__(
        self,
        pipeline_id: str = "pyannote/speaker-diarization-community-1",
        embedding_id: str = "pyannote/embedding",
        hf_token: str | None = None,
        device: str = "auto",
    ) -> None:
        self.pipeline_id = pipeline_id
        self.embedding_id = embedding_id
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.device = self._resolve_device(device)

    @staticmethod
    def _resolve_device(spec: str) -> str:
        if spec != "auto":
            return spec
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @lru_cache(maxsize=1)
    def _pipeline(self):
        from pyannote.audio import Pipeline
        import torch

        pipe = Pipeline.from_pretrained(self.pipeline_id, token=self.hf_token)
        pipe.to(torch.device(self.device))
        return pipe

    @lru_cache(maxsize=1)
    def _inference(self):
        from pyannote.audio import Inference, Model
        import torch

        model = Model.from_pretrained(self.embedding_id, token=self.hf_token)
        inference = Inference(model, window="whole")
        inference.to(torch.device(self.device))
        return inference

    def diarize(self, audio_path: str, num_speakers: int | None = None) -> list[SpeakerSegment]:
        pipe = self._pipeline()
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        # Pre-load to a 16k mono tensor so MP3 decoder rounding doesn't make
        # pyannote's chunk math drift by a sample.
        result = pipe(self._load_for_pipeline(audio_path), **kwargs)

        # community-1 returns DiarizeOutput; legacy 3.1 returns Annotation directly.
        annotation = getattr(result, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(result, "speaker_diarization", result)

        speaker_embeddings = getattr(result, "speaker_embeddings", None)
        speaker_to_emb: dict[str, np.ndarray] = {}
        if speaker_embeddings is not None:
            labels = annotation.labels()
            arr = np.asarray(speaker_embeddings)
            if arr.ndim == 2 and arr.shape[0] == len(labels):
                for label, emb in zip(labels, arr):
                    speaker_to_emb[label] = emb.astype(np.float32)

        raw_label_to_local: dict[str, str] = {}
        out: list[SpeakerSegment] = []
        for segment, _, label in annotation.itertracks(yield_label=True):
            if label not in raw_label_to_local:
                raw_label_to_local[label] = _label_for(len(raw_label_to_local))
            emb = speaker_to_emb.get(label)
            if emb is None:
                try:
                    emb = self.embed(audio_path, segment.start, segment.end)
                except Exception:
                    emb = None
            out.append(
                SpeakerSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    local_label=raw_label_to_local[label],
                    embedding=emb,
                    extra={"raw_label": label},
                )
            )
        out.sort(key=lambda s: s.start)
        return out

    def embed(self, audio_path: str, start: float, end: float) -> np.ndarray:
        from pyannote.core import Segment

        inference = self._inference()
        emb = inference.crop(self._load_for_pipeline(audio_path), Segment(start, end))
        arr = np.asarray(emb).reshape(-1)
        return arr.astype(np.float32)

    @staticmethod
    def _load_for_pipeline(audio_path: str) -> dict:
        import librosa
        import torch

        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
        waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
        return {"waveform": waveform, "sample_rate": sr}
