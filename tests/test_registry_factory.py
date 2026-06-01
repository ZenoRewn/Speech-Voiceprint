"""URI-scheme dispatch for registry.open_store."""

from __future__ import annotations

import pytest

from registry import RegistryStore, SpeakerStore, open_store
from registry.mysql_store import _connect_kwargs


def test_bare_path_returns_sqlite_store(tmp_path):
    db = tmp_path / "r.db"
    store = open_store(str(db))
    try:
        assert isinstance(store, RegistryStore)
        # Protocol check (structural typing): satisfies SpeakerStore.
        assert isinstance(store, SpeakerStore)
        # Smoke a write to confirm it actually opens.
        sp = store.create_speaker(display_name="t")
        assert sp.id.startswith("sp_")
    finally:
        store.close()
    assert db.exists()


def test_sqlite_triple_slash_uri(tmp_path):
    db = tmp_path / "r2.db"
    uri = f"sqlite://{db}"  # absolute path → resolves to sqlite:///abs/...
    if not uri.startswith("sqlite:///"):
        uri = "sqlite://" + str(db)  # ensure right shape
    store = open_store(f"sqlite://{db}" if str(db).startswith("/") else "sqlite:///" + str(db))
    try:
        store.create_speaker(display_name="x")
        assert len(store.list_speakers()) == 1
    finally:
        store.close()


def test_mysql_uri_dispatches_to_mysql_store(monkeypatch):
    from registry.mysql_store import MySQLRegistryStore

    monkeypatch.setattr(MySQLRegistryStore, "__init__", lambda self, uri: setattr(self, "uri", uri))
    store = open_store("mysql+pymysql://u:p@host/db")
    assert isinstance(store, MySQLRegistryStore)
    assert store.uri == "mysql+pymysql://u:p@host/db"


def test_mysql_uri_parse_connect_kwargs():
    kwargs = _connect_kwargs(
        "mysql+pymysql://user:p%40ss@db.example:3307/speech_voiceprint"
        "?charset=utf8mb4&connect_timeout=7"
    )
    assert kwargs["host"] == "db.example"
    assert kwargs["port"] == 3307
    assert kwargs["user"] == "user"
    assert kwargs["password"] == "p@ss"
    assert kwargs["database"] == "speech_voiceprint"
    assert kwargs["connect_timeout"] == 7


def test_postgres_uri_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="postgres"):
        open_store("postgresql://u:p@host/db")


def test_unknown_scheme_raises_value_error():
    with pytest.raises(ValueError, match="unsupported"):
        open_store("redis://host/0")


def test_empty_uri_raises_value_error():
    with pytest.raises(ValueError, match="empty"):
        open_store("")
