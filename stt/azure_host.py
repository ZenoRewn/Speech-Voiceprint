"""Centralize Azure Speech endpoint host selection.

Azure Speech REST APIs live on two slightly different host stems for the same
region:

  * Fast Transcription / Speaker Recognition / older endpoints:
    ``{region}.api.cognitive.microsoft.com``
  * Batch Transcription / SDK token endpoint / newer service plane:
    ``{region}.cognitiveservices.azure.com``

Picking the wrong one yields a confusing 404 — and historically each adapter
wrote its own copy of the rule. This module is the single source of truth.
"""

from __future__ import annotations

import os
from typing import Literal

Service = Literal["fast", "batch", "realtime"]


def host_for(
    region: str | None = None,
    service: Service = "fast",
    *,
    override: str | None = None,
) -> str:
    """Return the canonical https host for a region/service combo.

    `override` (typically passed through from a constructor) bypasses
    inference; useful for sovereign clouds (gov.cognitive.microsoft.com /
    .cognitive.azure.us) where the stem differs.
    """
    if override:
        return override.rstrip("/")
    region = region or os.environ.get("AZURE_SPEECH_REGION")
    if not region:
        raise ValueError("AZURE_SPEECH_REGION not set; pass region or override")
    region = region.strip().lower()
    if service == "batch":
        return f"https://{region}.cognitiveservices.azure.com"
    # Both fast and realtime live on the api.cognitive host.
    return f"https://{region}.api.cognitive.microsoft.com"
