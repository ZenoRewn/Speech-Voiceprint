"""Pydantic models for the public output contract.

Why these are separate from the dataclasses in `stt/base.py` and
`voiceprint/base.py`:

- The dataclasses are *internal* — orchestrator passes them around by
  reference, mutates them in `_apply_registry`, and never serializes them
  directly. Adding Pydantic to those types would force every internal call
  site to validate, which is wasted work.
- The Pydantic models are the *boundary* contract — what the SDK returns,
  what the REST API responds with, what `model_json_schema()` exports.
  External consumers should pin to these shapes, not the orchestrator dict.

The orchestrator currently returns a `dict` (see `_run_pipeline`). We keep
that as the internal currency but adapt at the edges via
`PipelineResult.from_payload(dict)`.
"""

from .output import (
    AlignedUtteranceModel,
    AlignedWordModel,
    LabelResolutionModel,
    PipelineResult,
    RegistryInfoModel,
    VoiceprintSegmentModel,
    WordModel,
)

__all__ = [
    "AlignedUtteranceModel",
    "AlignedWordModel",
    "LabelResolutionModel",
    "PipelineResult",
    "RegistryInfoModel",
    "VoiceprintSegmentModel",
    "WordModel",
]
