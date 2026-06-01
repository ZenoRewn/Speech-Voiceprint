"""Tests for stt._retry.retrying_request.

Why these specific cases: production took down a job last quarter when Azure
returned a transient 503 mid-batch. retry_on (429/5xx) covers that without
masking real bugs (401 stays bubbled up, 400 stays bubbled up).
"""

from __future__ import annotations

import time
from typing import List

import pytest
import requests

from stt._retry import RetryError, retrying_request


class _FakeResponse:
    def __init__(self, status: int, headers: dict | None = None) -> None:
        self.status_code = status
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Replace time.sleep so the test suite runs in milliseconds, not seconds."""
    monkeypatch.setattr(time, "sleep", lambda *_a, **_kw: None)


def test_retries_on_429_then_succeeds():
    statuses: List[int] = [429, 429, 200]
    calls: list[int] = []

    def request():
        calls.append(1)
        return _FakeResponse(statuses.pop(0))

    resp = retrying_request(request, retries=3, op_name="t")
    assert resp.status_code == 200
    assert len(calls) == 3


def test_retries_until_exhausted_returns_last_response():
    """All attempts fail with retriable code → caller sees the last failed response.

    We intentionally do NOT raise — caller decides via raise_for_status().
    """

    def request():
        return _FakeResponse(503)

    resp = retrying_request(request, retries=2, op_name="t")
    assert resp.status_code == 503


def test_does_not_retry_on_401():
    calls: list[int] = []

    def request():
        calls.append(1)
        return _FakeResponse(401)

    resp = retrying_request(request, retries=3, op_name="t")
    assert resp.status_code == 401
    assert len(calls) == 1


def test_does_not_retry_on_400():
    calls: list[int] = []

    def request():
        calls.append(1)
        return _FakeResponse(400)

    resp = retrying_request(request, retries=3, op_name="t")
    assert resp.status_code == 400
    assert len(calls) == 1


def test_retries_on_connection_error_then_raises():
    calls: list[int] = []

    def request():
        calls.append(1)
        raise requests.ConnectionError("boom")

    with pytest.raises(RetryError) as info:
        retrying_request(request, retries=2, op_name="t")
    assert "connection error" in str(info.value)
    assert len(calls) == 3


def test_honours_retry_after_header(monkeypatch):
    """When Azure returns Retry-After we respect it instead of computing our own delay."""
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda x: sleeps.append(x))

    statuses = [(429, {"Retry-After": "2"}), (200, None)]

    def request():
        status, headers = statuses.pop(0)
        return _FakeResponse(status, headers)

    resp = retrying_request(request, retries=3, op_name="t", backoff_cap=10.0)
    assert resp.status_code == 200
    # First sleep should be ~2s from Retry-After (jitter is added only when computing
    # our own delay). Allow exact match here.
    assert sleeps and sleeps[0] == 2.0
