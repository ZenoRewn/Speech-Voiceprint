import pytest

from stt.azure_host import host_for


def test_fast_uses_api_cognitive_host():
    assert host_for("eastus", "fast") == "https://eastus.api.cognitive.microsoft.com"


def test_batch_uses_cognitiveservices_host():
    assert host_for("eastus", "batch") == "https://eastus.cognitiveservices.azure.com"


def test_realtime_shares_fast_host():
    assert host_for("southeastasia", "realtime") == "https://southeastasia.api.cognitive.microsoft.com"


def test_override_is_returned_unchanged():
    custom = "https://my-private-edge.example.com"
    assert host_for("eastus", "batch", override=custom) == custom
    # trailing slash trimmed
    assert host_for("eastus", "fast", override=custom + "/") == custom


def test_region_normalization():
    assert host_for(" EastUS ", "fast") == "https://eastus.api.cognitive.microsoft.com"


def test_missing_region_raises(monkeypatch):
    monkeypatch.delenv("AZURE_SPEECH_REGION", raising=False)
    with pytest.raises(ValueError):
        host_for(None, "fast")


def test_env_fallback(monkeypatch):
    monkeypatch.setenv("AZURE_SPEECH_REGION", "westeurope")
    assert host_for(None, "batch") == "https://westeurope.cognitiveservices.azure.com"
