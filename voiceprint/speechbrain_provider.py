from __future__ import annotations

import os
import string
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from .base import SpeakerSegment


def _label_for(idx: int) -> str:
    if idx < 26:
        return f"Speaker_{string.ascii_uppercase[idx]}"
    return f"Speaker_{idx}"


class SpeechBrainProvider:
    """SpeechBrain backend.

    Pipeline: VAD (vad-crdnn-libriparty) -> sliding window -> ECAPA-TDNN embedding
    (spkrec-ecapa-voxceleb, 192-d) -> agglomerative clustering on cosine distance.
    A small VAD-failure fallback splits the whole clip into uniform 1.5s windows
    so we still get *something* on inputs the VAD silently rejects.
    """

    name = "speechbrain"
    embedding_dim = 192

    def __init__(
        self,
        embedding_id: str = "speechbrain/spkrec-ecapa-voxceleb",
        vad_id: str = "speechbrain/vad-crdnn-libriparty",
        cluster_threshold: float = 0.6,  # legacy AHC threshold, retained for reference
        device: str = "auto",
        window_seconds: float = 2.0,
        hop_seconds: float = 1.0,
        max_speakers: int = 12,
        min_segment_seconds: float = 0.5,
        cache_dir: str | None = None,
    ) -> None:
        self.embedding_id = embedding_id
        self.vad_id = vad_id
        self.cluster_threshold = cluster_threshold
        self.device = self._resolve_device(device)
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.min_segment_seconds = min_segment_seconds
        self.max_speakers = max_speakers
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/speechbrain")

    @staticmethod
    def _resolve_device(spec: str) -> str:
        if spec != "auto":
            # SpeechBrain 1.1.0's Pretrained.__init__ only sets device_type
            # for "cpu" and strings containing "cuda" — passing "mps" raises
            # AttributeError on first inference call. Downgrade to cpu rather
            # than crash; ECAPA is small enough that MPS gives no meaningful
            # speedup on our window sizes.
            return "cpu" if spec == "mps" else spec
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @lru_cache(maxsize=1)
    def _vad(self):
        from speechbrain.inference.VAD import VAD

        return VAD.from_hparams(
            source=self.vad_id,
            savedir=str(Path(self.cache_dir) / "vad"),
            run_opts={"device": self.device},
        )

    @lru_cache(maxsize=1)
    def _encoder(self):
        from speechbrain.inference.speaker import EncoderClassifier

        return EncoderClassifier.from_hparams(
            source=self.embedding_id,
            savedir=str(Path(self.cache_dir) / "ecapa"),
            run_opts={"device": self.device},
        )

    def _vad_segments(self, audio_path: str) -> list[tuple[float, float]]:
        vad = self._vad()
        boundaries = vad.get_speech_segments(audio_path)  # tensor (N, 2) seconds
        out: list[tuple[float, float]] = []
        for row in boundaries.tolist():
            start, end = float(row[0]), float(row[1])
            if end - start >= self.min_segment_seconds:
                out.append((start, end))
        return out

    @staticmethod
    def _ensure_wav_16k(audio_path: str) -> str:
        """Transcode anything that isn't already 16k mono WAV to a temp file.

        SpeechBrain VAD reads the path directly via torchaudio and fails on
        many MP3 codecs, so we normalize once and reuse the temp path.
        """
        if audio_path.lower().endswith(".wav"):
            try:
                import soundfile as sf

                info = sf.info(audio_path)
                if info.samplerate == 16000 and info.channels == 1:
                    return audio_path
            except Exception:
                pass
        import librosa
        import soundfile as sf

        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio, 16000, subtype="PCM_16")
        return tmp.name

    def _windows(self, segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
        windows: list[tuple[float, float]] = []
        for s, e in segments:
            if e - s <= self.window_seconds:
                windows.append((s, e))
                continue
            t = s
            while t + self.window_seconds <= e:
                windows.append((t, t + self.window_seconds))
                t += self.hop_seconds
            tail_start = max(s, e - self.window_seconds)
            if tail_start < t and (e - tail_start) >= self.min_segment_seconds:
                windows.append((tail_start, e))
        return windows

    def _embed_clip(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import torch

        encoder = self._encoder()
        wav = torch.from_numpy(audio).float().unsqueeze(0)
        with torch.no_grad():
            emb = encoder.encode_batch(wav).squeeze().detach().cpu().numpy()
        return emb.astype(np.float32).reshape(-1)

    def diarize(self, audio_path: str, num_speakers: int | None = None) -> list[SpeakerSegment]:
        import soundfile as sf

        wav_path = self._ensure_wav_16k(audio_path)
        try:
            audio, sr = sf.read(wav_path, always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = np.asarray(audio, dtype=np.float32)

            try:
                voiced = self._vad_segments(wav_path)
            except Exception:
                voiced = []
            if not voiced:
                duration = len(audio) / sr
                voiced = [(0.0, duration)]

            windows = self._windows(voiced)
            if not windows:
                return []

            embeddings = []
            for s, e in windows:
                i0 = int(s * sr)
                i1 = int(e * sr)
                clip = audio[i0:i1]
                if clip.size < int(0.3 * sr):
                    continue
                embeddings.append((s, e, self._embed_clip(clip, sr)))

            if not embeddings:
                return []

            X = np.stack([emb for _, _, emb in embeddings])
            labels = self._cluster(X, num_speakers)
            raw_to_local: dict[int, str] = {}
            out: list[SpeakerSegment] = []
            for (s, e, emb), lbl in zip(embeddings, labels):
                if lbl not in raw_to_local:
                    raw_to_local[lbl] = _label_for(len(raw_to_local))
                out.append(
                    SpeakerSegment(
                        start=s,
                        end=e,
                        local_label=raw_to_local[lbl],
                        embedding=emb,
                        extra={"cluster": int(lbl)},
                    )
                )
            return self._merge_adjacent(out)
        finally:
            self._cleanup_temp(wav_path, audio_path)

    def _cluster(self, X: np.ndarray, num_speakers: int | None) -> np.ndarray:
        if X.shape[0] == 1:
            return np.array([0])
        if num_speakers is not None and num_speakers >= 1:
            return self._spectral(X, num_speakers)
        # auto-k: pick k from the largest eigen-gap, then run spectral.
        k = self._auto_k(X, max_k=min(self.max_speakers, X.shape[0] - 1))
        if k <= 1:
            return np.zeros(X.shape[0], dtype=int)
        return self._spectral(X, k)

    @staticmethod
    def _affinity(X: np.ndarray) -> np.ndarray:
        # cosine similarity, clipped to [0, 1] so it's a valid affinity for
        # the symmetric-normalized Laplacian below.
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        sim = Xn @ Xn.T
        sim = np.clip(sim, 0.0, 1.0)
        np.fill_diagonal(sim, 0.0)  # ignore self-similarity for eigen-gap
        return sim

    @staticmethod
    def _filter_outliers(X: np.ndarray, percentile: float = 90.0) -> np.ndarray:
        """Drop the noisiest windows before computing k.

        A single laughter / silence / noise window often forms a degenerate
        size-1 cluster that hides the real speaker count. Use mean kNN cosine
        distance as a density proxy and trim the top `percentile`.
        """
        n = X.shape[0]
        k_nn = min(5, n // 4)
        if k_nn < 2:
            return X
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=k_nn, metric="cosine").fit(X)
        dists, _ = nbrs.kneighbors(X)
        mean_knn = dists[:, 1:].mean(axis=1)
        cutoff = float(np.percentile(mean_knn, percentile))
        inlier = mean_knn <= cutoff
        if inlier.sum() < max(4, n // 2):
            return X
        return X[inlier]

    @classmethod
    def _auto_k(
        cls,
        X: np.ndarray,
        max_k: int,
        monologue_median_distance: float = 0.5,
        min_silhouette: float = 0.10,
    ) -> int:
        """Estimate number of speakers in two stages.

        1. **Monologue gate** — if the median pairwise cosine distance is
           below `monologue_median_distance`, the embeddings are too tightly
           packed to plausibly contain multiple speakers, return 1. This is
           the only thing that reliably distinguishes a single speaker from a
           tight conversation; eigen-gap and silhouette both produce false
           positives on monologues.
        2. **Spectral + silhouette** — run spectral for k = 2..max_k, drop k
           where any cluster has < `min_size` members (degenerate splits from
           outlier windows), keep the best cosine-silhouette. If even the
           best is below `min_silhouette`, give up and return 1.

        For >=3 speakers on noisy real audio this method tends to underestimate
        k (e.g. lump similar voices). pyannote remains the preferred backend
        when speaker-count accuracy matters; SpeechBrain's value here is speed.
        """
        from scipy.spatial.distance import pdist
        from sklearn.metrics import silhouette_score

        n = X.shape[0]
        if n <= 1:
            return 1
        if n == 2:
            return 1 if cls._affinity(X)[0, 1] > 0.7 else 2

        median_pairwise = float(np.median(pdist(X, metric="cosine")))
        if median_pairwise < monologue_median_distance:
            return 1

        X_inlier = cls._filter_outliers(X)
        n_inlier = X_inlier.shape[0]
        max_k = max(2, min(max_k, n_inlier - 1))
        min_size = max(2, int(round(n_inlier * 0.05)))
        best_k = 1
        best_score = -1.0
        for k in range(2, max_k + 1):
            try:
                labels = cls._spectral(X_inlier, k)
            except Exception:
                continue
            sizes = np.bincount(labels)
            if len(sizes) < k or sizes.min() < min_size:
                continue
            score = silhouette_score(X_inlier, labels, metric="cosine")
            if score > best_score:
                best_score = float(score)
                best_k = k
        if best_score < min_silhouette:
            return 1
        return best_k

    @staticmethod
    def _spectral(X: np.ndarray, k: int) -> np.ndarray:
        from sklearn.cluster import SpectralClustering

        k = max(1, min(k, X.shape[0]))
        if k == 1:
            return np.zeros(X.shape[0], dtype=int)
        model = SpectralClustering(
            n_clusters=k,
            affinity="cosine",
            assign_labels="kmeans",
            random_state=0,
        )
        return model.fit_predict(X)

    @staticmethod
    def _merge_adjacent(segments: list[SpeakerSegment]) -> list[SpeakerSegment]:
        if not segments:
            return segments
        segments = sorted(segments, key=lambda s: s.start)
        merged: list[SpeakerSegment] = [segments[0]]
        for cur in segments[1:]:
            prev = merged[-1]
            if cur.local_label == prev.local_label and cur.start <= prev.end + 0.2:
                merged[-1] = SpeakerSegment(
                    start=prev.start,
                    end=max(prev.end, cur.end),
                    local_label=prev.local_label,
                    embedding=prev.embedding,
                    extra=prev.extra,
                )
            else:
                merged.append(cur)
        return merged

    def embed(self, audio_path: str, start: float, end: float) -> np.ndarray:
        import soundfile as sf

        wav_path = self._ensure_wav_16k(audio_path)
        try:
            audio, sr = sf.read(wav_path, always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = np.asarray(audio, dtype=np.float32)
            clip = audio[int(start * sr) : int(end * sr)]
            return self._embed_clip(clip, sr)
        finally:
            self._cleanup_temp(wav_path, audio_path)

    @staticmethod
    def _cleanup_temp(wav_path: str, original_path: str) -> None:
        if wav_path == original_path:
            return
        try:
            Path(wav_path).unlink(missing_ok=True)
        except OSError:
            pass
