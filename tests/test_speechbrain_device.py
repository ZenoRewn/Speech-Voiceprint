"""Pin the SpeechBrain device-resolution rules.

SpeechBrain 1.1.0's Pretrained.__init__ only branches on "cpu" and strings
containing "cuda" — passing "mps" leaves device_type unset and the next
torch.autocast call raises AttributeError. We map mps → cpu in the provider
to dodge it; this test fails loudly if anyone restores the mps path before
the upstream bug is fixed.
"""

from __future__ import annotations

from voiceprint.speechbrain_provider import SpeechBrainProvider


def test_explicit_cpu_passthrough():
    assert SpeechBrainProvider._resolve_device("cpu") == "cpu"


def test_explicit_cuda_passthrough():
    assert SpeechBrainProvider._resolve_device("cuda") == "cuda"
    assert SpeechBrainProvider._resolve_device("cuda:0") == "cuda:0"


def test_explicit_mps_downgrades_to_cpu():
    assert SpeechBrainProvider._resolve_device("mps") == "cpu"


def test_auto_never_returns_mps(monkeypatch):
    """Even on Apple Silicon (where torch.backends.mps is available), auto
    must not return 'mps' — it would crash speechbrain at first inference."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Force the mps branch to be available; auto should still pick cpu.
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert SpeechBrainProvider._resolve_device("auto") == "cpu"


def test_auto_prefers_cuda_when_available(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert SpeechBrainProvider._resolve_device("auto") == "cuda"
