import os
from pathlib import Path

import pytest

from pipeline.config import load_config, merge_cli, pick


def test_pick_dotted_lookup():
    cfg = {"a": {"b": {"c": 42}}}
    assert pick(cfg, "a.b.c") == 42
    assert pick(cfg, "a.b") == {"c": 42}
    assert pick(cfg, "a.x", default="fallback") == "fallback"
    assert pick(cfg, "missing.deep.path", default=None) is None


def test_merge_cli_priority():
    cfg = {"azure": {"region": "eastus"}}
    assert merge_cli(cfg, None, "azure.region", default="x") == "eastus"
    assert merge_cli(cfg, "cn", "azure.region", default="x") == "cn"
    assert merge_cli(cfg, (), "azure.region", default="x") == "eastus"  # () = unset
    assert merge_cli(cfg, None, "missing.key", default="default") == "default"


def test_load_config_expands_env(tmp_path: Path):
    os.environ["SV_TEST_KEY"] = "secret-abc"
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        """
azure:
  speech_key: ${SV_TEST_KEY}
  region: ${SV_TEST_MISSING:southeastasia}
languages:
  - en-US
  - ${SV_TEST_LANG:zh-CN}
""",
        encoding="utf-8",
    )
    try:
        cfg = load_config(str(cfg_file))
    finally:
        del os.environ["SV_TEST_KEY"]
    assert cfg["azure"]["speech_key"] == "secret-abc"
    assert cfg["azure"]["region"] == "southeastasia"
    assert cfg["languages"] == ["en-US", "zh-CN"]


def test_load_config_none_returns_empty():
    assert load_config(None) == {}
    assert load_config("") == {}


def test_load_config_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "not-here.yaml")


def test_load_config_rejects_non_mapping(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(bad))
