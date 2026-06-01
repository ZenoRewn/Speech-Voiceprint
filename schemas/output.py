"""Pydantic models for `PipelineResult` and friends — the public output schema.

Field names mirror what `pipeline/orchestrator._run_pipeline` produces today
(round-tripping a CLI JSON output through `PipelineResult.from_payload(...)`
and back via `model_dump(mode='json')` should be lossless).

`PipelineResult.model_json_schema()` is what the API exposes at
`GET /api/schema` so SDK users have a typed contract.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class WordModel(BaseModel):
    """Pre-alignment word from the STT layer."""
    model_config = ConfigDict(extra="forbid")

    text: str
    start: float
    end: float
    confidence: Optional[float] = None


class AlignedWordModel(BaseModel):
    """Word with a resolved speaker label."""
    model_config = ConfigDict(extra="forbid")

    text: str
    start: float
    end: float
    speaker: str
    confidence: Optional[float] = None


class AlignedUtteranceModel(BaseModel):
    """Utterance with majority-voted speaker; `mixed` when below threshold."""
    model_config = ConfigDict(extra="forbid")

    text: str
    start: float
    end: float
    speaker: str
    speaker_confidence: float = Field(ge=0.0, le=1.0)
    azure_speaker: Optional[str] = None
    words: List[AlignedWordModel] = Field(default_factory=list)


class VoiceprintSegmentModel(BaseModel):
    """Pre-alignment voiceprint diarization segment."""
    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    local_label: str


class LabelResolutionModel(BaseModel):
    """Per-cluster registry resolution: known/low_confidence/unknown."""
    model_config = ConfigDict(extra="allow")  # registry payload may grow new keys

    verdict: str
    speaker_id: Optional[str] = None
    display_name: Optional[str] = None
    distance: Optional[float] = None
    candidate_speaker_id: Optional[str] = None
    candidate_distance: Optional[float] = None
    resolved_label: Optional[str] = None


class RegistryInfoModel(BaseModel):
    """Registry context attached when the run was registry-aware."""
    model_config = ConfigDict(extra="allow")

    registry_path: str
    model: str
    match_threshold: float
    unknown_threshold: float
    label_resolutions: Dict[str, LabelResolutionModel] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    """Full output of a fast/batch pipeline run.

    Top-level shape matches `_run_pipeline()`'s dict so existing CLI JSON
    files validate cleanly.
    """
    model_config = ConfigDict(extra="forbid")

    audio: str
    language: Optional[str] = None
    duration: Optional[float] = None
    voiceprint_backend: str
    voiceprint_segments: List[VoiceprintSegmentModel] = Field(default_factory=list)
    registry: Optional[RegistryInfoModel] = None
    # Cross-backend dual-enroll summary: { model, enrolled_speaker_ids[, skipped_reason | error ] }.
    # Loose-typed because the structure is informational (UI/log only).
    dual_enroll: Optional[Dict[str, Any]] = None
    utterances: List[AlignedUtteranceModel] = Field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PipelineResult":
        """Adapter from `_run_pipeline()`'s dict.

        Defensive: orchestrator may hand back keys we haven't enumerated yet
        (e.g. future `quality`, `metrics` blocks). `extra="forbid"` on the
        root would reject those, so we route through `model_validate` and
        let nested `extra='allow'` (registry, label resolutions) accept growth.
        For unknown top-level keys we want loud failures during dev — caller
        can drop them before passing if intentional.
        """
        return cls.model_validate(payload)
