"""HTTP retry helper for Azure REST adapters.

Problem we're solving: `requests.post(...).raise_for_status()` fails on the
first 429/503 with no retry, so a transient throttle on the Speech endpoint
takes the whole job down. Each adapter (`azure_fast`, `azure_batch`) wraps
its calls with `retrying_request(...)`.

Retries are reserved for retriable status codes and connection errors;
auth (401/403) and bad-request (400) bubble up immediately because retrying
those just wastes time. The backoff is `base * 2 ** attempt` with a jitter
cap so two parallel jobs hitting the same throttle don't perfectly align.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable

import requests

log = logging.getLogger(__name__)

DEFAULT_RETRIABLE = (429, 500, 502, 503, 504)


class RetryError(RuntimeError):
    """Raised when all retry attempts are exhausted."""


def retrying_request(
    request_fn: Callable[[], requests.Response],
    *,
    retries: int = 3,
    backoff_base: float = 0.5,
    backoff_cap: float = 30.0,
    retry_on: tuple[int, ...] = DEFAULT_RETRIABLE,
    op_name: str = "azure",
) -> requests.Response:
    """Run `request_fn` up to `retries + 1` times.

    Connection-level errors (`requests.ConnectionError`, `Timeout`) and
    HTTP statuses listed in `retry_on` trigger retries with exponential
    backoff. The Azure endpoint also returns `Retry-After` for 429/503;
    when present we honour it (capped) instead of computing our own delay.
    """
    last_exc: Exception | None = None
    last_resp: requests.Response | None = None

    for attempt in range(retries + 1):
        try:
            resp = request_fn()
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            if attempt == retries:
                raise RetryError(f"{op_name}: connection error after {attempt + 1} tries: {e}") from e
            sleep_for = _compute_delay(attempt, backoff_base, backoff_cap)
            log.warning(
                "%s connection error, retrying in %.2fs (attempt %d/%d): %s",
                op_name, sleep_for, attempt + 1, retries + 1, e,
            )
            time.sleep(sleep_for)
            continue

        if resp.status_code in retry_on and attempt < retries:
            retry_after = _retry_after_seconds(resp)
            sleep_for = retry_after if retry_after is not None else _compute_delay(
                attempt, backoff_base, backoff_cap
            )
            sleep_for = min(sleep_for, backoff_cap)
            log.warning(
                "%s got HTTP %d, retrying in %.2fs (attempt %d/%d)",
                op_name, resp.status_code, sleep_for, attempt + 1, retries + 1,
            )
            time.sleep(sleep_for)
            last_resp = resp
            continue

        return resp

    # Unreachable normally — kept as defensive fallback.
    if last_resp is not None:
        return last_resp
    raise RetryError(f"{op_name}: exhausted retries ({last_exc})")


def _compute_delay(attempt: int, base: float, cap: float) -> float:
    raw = base * (2 ** attempt)
    jitter = random.uniform(0, base)
    return min(raw + jitter, cap)


def _retry_after_seconds(resp: requests.Response) -> float | None:
    header = resp.headers.get("Retry-After")
    if not header:
        return None
    try:
        return float(header)
    except ValueError:
        return None
