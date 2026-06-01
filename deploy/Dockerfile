# syntax=docker/dockerfile:1.7
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/opt/cache/torch \
    HF_HOME=/opt/cache/huggingface \
    SPEECHBRAIN_CACHE=/opt/cache/speechbrain \
    SV_DATA_DIR=/data

# libsndfile/ffmpeg for librosa+soundfile MP3 reading; libgomp1 for sklearn/torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
        ca-certificates \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY stt/ ./stt/
COPY voiceprint/ ./voiceprint/
COPY merger/ ./merger/
COPY registry/ ./registry/
COPY pipeline/ ./pipeline/
COPY schemas/ ./schemas/
COPY benchmarks/ ./benchmarks/
COPY web/ ./web/

# Install deps (CPU torch wheel; pyannote/speechbrain are extras).
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio \
    && pip install -e ".[pyannote,speechbrain,dev]"

# ---------------------------------------------------------------------------
# Optional model warm-up: BAKE_MODELS=1 + HF_TOKEN downloads pyannote weights
# into the image. Prefer BuildKit secret `id=hf_token`; ARG HF_TOKEN remains as
# a local fallback but should not be used in shared CI because build args are not
# secret material.
ARG BAKE_MODELS=0
ARG HF_TOKEN=
RUN --mount=type=secret,id=hf_token,required=false \
    if [ "$BAKE_MODELS" = "1" ]; then \
        HF_TOKEN_SECRET="$(cat /run/secrets/hf_token 2>/dev/null || true)" ; \
        HF_TOKEN="${HF_TOKEN_SECRET:-$HF_TOKEN}" ; \
        if [ -n "$HF_TOKEN" ]; then echo "warming model cache (HF_TOKEN: set)"; else echo "warming model cache (HF_TOKEN: MISSING)"; fi \
        && HF_TOKEN="$HF_TOKEN" python -c "\
	import os; \
	from voiceprint.speechbrain_provider import SpeechBrainProvider; \
	SpeechBrainProvider(device='cpu')._encoder(); \
from voiceprint.pyannote_provider import PyannoteProvider; \
p = PyannoteProvider(hf_token=os.environ.get('HF_TOKEN'), device='cpu'); \
p._pipeline(); p._inference()" ; \
    fi

# Run as non-root. The data + cache dirs are pre-created so PVC mounts don't
# inherit root ownership when the K8s securityContext runs as uid=10001.
RUN useradd --system --uid 10001 --home /home/svuser --create-home svuser \
    && mkdir -p /data /opt/cache \
    && chown -R svuser:svuser /data /opt/cache /app
USER svuser

VOLUME ["/data", "/opt/cache"]

EXPOSE 8080

# Liveness is intentionally auth-free. Readiness probes should target /readyz.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/livez || exit 1

# tini is PID 1 → forwards SIGTERM to uvicorn cleanly so the lifespan
# `drain` block runs before the container exits. Default to running the api;
# override --entrypoint for orchestrator/streaming/bench.
ENTRYPOINT ["/usr/bin/tini", "--", "python", "-m", "pipeline.api"]
CMD ["--port", "8080", "--registry", "/data/registry/speakers.db"]
