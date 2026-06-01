"""Sanity checks for pipeline.paths — env override + lazy mkdir."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.paths import get_paths


def test_default_paths_resolve_under_repo_data(tmp_path, monkeypatch):
    monkeypatch.delenv("SV_DATA_DIR", raising=False)
    p = get_paths()
    # Resolve and check the children are direct subdirs of `root`.
    assert p.resources.parent == p.root
    assert p.uploads.parent == p.root
    assert p.outputs.parent == p.root
    assert p.stream.parent == p.root
    assert p.registry.parent == p.root


def test_sv_data_dir_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("SV_DATA_DIR", str(tmp_path))
    p = get_paths()
    assert p.root == tmp_path
    assert p.uploads == tmp_path / "uploads"


def test_individual_dir_override_wins_over_root(tmp_path, monkeypatch):
    """SV_OUTPUTS_DIR should override even when SV_DATA_DIR is set."""
    monkeypatch.setenv("SV_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SV_OUTPUTS_DIR", str(tmp_path / "elsewhere"))
    p = get_paths()
    assert p.outputs == tmp_path / "elsewhere"
    # Other dirs still under SV_DATA_DIR.
    assert p.uploads == tmp_path / "uploads"


def test_ensure_creates_subset(tmp_path, monkeypatch):
    monkeypatch.setenv("SV_DATA_DIR", str(tmp_path))
    p = get_paths()
    assert not (tmp_path / "uploads").exists()
    p.ensure("uploads", "outputs")
    assert (tmp_path / "uploads").is_dir()
    assert (tmp_path / "outputs").is_dir()
    # Untouched scopes stay missing.
    assert not (tmp_path / "stream").exists()


def test_ensure_no_args_creates_all(tmp_path, monkeypatch):
    monkeypatch.setenv("SV_DATA_DIR", str(tmp_path))
    p = get_paths()
    p.ensure()
    for sub in ("resources", "uploads", "stream", "outputs", "registry"):
        assert (tmp_path / sub).is_dir(), f"missing {sub}"
