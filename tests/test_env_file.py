"""Tests for the minimal .env loader (pipeline/env_file.py)."""

from __future__ import annotations

import os
from pathlib import Path

from pipeline.env_file import load_env_file


def test_load_basic_keys(tmp_path: Path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FOO=bar\nBAZ=qux\n", encoding="utf-8")
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAZ", raising=False)
    applied = load_env_file(env)
    assert applied == {"FOO": "bar", "BAZ": "qux"}
    assert os.environ["FOO"] == "bar"
    assert os.environ["BAZ"] == "qux"


def test_existing_env_wins(tmp_path: Path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("ALREADY=fromfile\n", encoding="utf-8")
    monkeypatch.setenv("ALREADY", "fromshell")
    applied = load_env_file(env)
    assert "ALREADY" not in applied
    assert os.environ["ALREADY"] == "fromshell"


def test_comments_and_quotes(tmp_path: Path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        '# header comment\n'
        'export QUOTED="hello world"\n'
        "SINGLE='a b'\n"
        "WITH_HASH=plain # trailing\n"
        "\n"
        "EMPTY=\n",
        encoding="utf-8",
    )
    for k in ("QUOTED", "SINGLE", "WITH_HASH", "EMPTY"):
        monkeypatch.delenv(k, raising=False)
    applied = load_env_file(env)
    assert applied["QUOTED"] == "hello world"
    assert applied["SINGLE"] == "a b"
    assert applied["WITH_HASH"] == "plain"
    assert applied["EMPTY"] == ""


def test_missing_file_is_noop(tmp_path: Path):
    assert load_env_file(tmp_path / "does-not-exist") == {}


def test_invalid_lines_skipped(tmp_path: Path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "no equals here\n"
        "=missing-key\n"
        "1BAD=startsWithDigit\n"
        "GOOD=ok\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GOOD", raising=False)
    applied = load_env_file(env)
    assert applied == {"GOOD": "ok"}
