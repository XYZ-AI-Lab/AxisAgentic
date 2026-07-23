# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Configurable timeout and retry policy for web-search outbound calls.

This module centralizes the timeout/backoff logic that ``_scrape_utils`` and
``search`` previously hardcoded. The default factories reproduce the prior
behavior exactly, so a stable endpoint produces an identical trace: the first
request succeeds, no backoff branch is taken, and timeout values are unchanged.

``jitter`` and ``respect_retry_after`` only influence the *retry* (instability)
path; they never affect a request that succeeds on the first attempt.
"""

from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx


@dataclass(frozen=True)
class TimeoutConfig:
    """Per-request timeout budget, mapped onto ``httpx.Timeout``.

    ``connect_seconds`` / ``read_seconds`` fall back to ``total_seconds`` when
    unset, while ``total_seconds`` (the positional ``httpx`` default) also covers
    the write/pool phases. ``None`` means "no limit" for that phase.
    """

    connect_seconds: float | None = None
    read_seconds: float | None = None
    total_seconds: float | None = None

    def to_httpx(self) -> httpx.Timeout:
        connect = self.connect_seconds if self.connect_seconds is not None else self.total_seconds
        read = self.read_seconds if self.read_seconds is not None else self.total_seconds
        return httpx.Timeout(self.total_seconds, connect=connect, read=read)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimeoutConfig:
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> TimeoutConfig:
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class RetryConfig:
    """Retry policy for a transient (endpoint-instability) failure.

    ``max_attempts`` counts the first attempt, so ``max_attempts=5`` allows four
    retries. ``retry_on_server_errors`` keeps the prior "any 5xx is retryable"
    rule while ``retryable_status_codes`` adds explicit non-5xx codes (e.g. 429).
    """

    max_attempts: int = 1
    backoff_base_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    backoff_max_seconds: float = 8.0
    jitter: bool = False
    respect_retry_after: bool = False
    retryable_status_codes: frozenset[int] = frozenset()
    retry_on_server_errors: bool = True
    retry_on_timeout: bool = True
    retry_on_connection_error: bool = True

    def is_retryable_status(self, status_code: int) -> bool:
        if self.retry_on_server_errors and status_code >= 500:
            return True
        return status_code in self.retryable_status_codes

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["retryable_status_codes"] = sorted(self.retryable_status_codes)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetryConfig:
        data = dict(data)
        if "retryable_status_codes" in data:
            data["retryable_status_codes"] = frozenset(data["retryable_status_codes"])
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> RetryConfig:
        return cls.from_dict(json.loads(raw))


def compute_backoff(attempt: int, retry: RetryConfig, *, retry_after: float | None = None) -> float:
    """Seconds to sleep before retry number ``attempt`` (1-based).

    Without ``jitter`` and ``retry_after`` this is the plain capped exponential
    ``base * multiplier**(attempt-1)`` — identical to the previous fixed delay
    sequences. ``jitter`` applies equal jitter (half fixed, half random) and
    ``respect_retry_after`` never sleeps less than the server-requested delay.
    """
    raw = retry.backoff_base_seconds * (retry.backoff_multiplier ** (attempt - 1))
    delay = min(raw, retry.backoff_max_seconds)
    if retry.jitter:
        delay = delay * 0.5 + random.uniform(0.0, delay * 0.5)
    if retry.respect_retry_after and retry_after is not None and retry_after > 0:
        delay = max(delay, retry_after)
    return delay


def coerce_timeout(value: float | TimeoutConfig | None) -> TimeoutConfig | None:
    """Accept a :class:`TimeoutConfig`, a legacy scalar seconds value, or ``None``.

    A bare float/int is treated as the previously documented "request timeout in
    seconds" and mapped to ``TimeoutConfig(total_seconds=...)`` so older callers
    keep working. ``None`` is passed through so the caller can apply its default.
    """
    if value is None or isinstance(value, TimeoutConfig):
        return value
    return TimeoutConfig(total_seconds=float(value))


def should_retry_transport_error(exc: BaseException, retry: RetryConfig) -> bool:
    """Whether a caught transport error is retryable per the timeout/connection flags.

    httpx timeout errors (``ConnectTimeout``, ``ReadTimeout`` …) subclass
    ``TimeoutException`` and are gated by ``retry_on_timeout``; other transport
    errors (``ConnectError`` and the ``RequestError`` base) are treated as
    connection errors gated by ``retry_on_connection_error``. Keeping the two
    flags independent lets a caller disable one without affecting the other.
    """
    if isinstance(exc, httpx.TimeoutException):
        return retry.retry_on_timeout
    return retry.retry_on_connection_error


def parse_retry_after(headers: Any) -> float | None:
    """Parse a ``Retry-After`` header (delta-seconds or HTTP-date) into seconds."""
    if headers is None:
        return None
    value = headers.get("Retry-After") or headers.get("retry-after")
    if not value:
        return None
    value = str(value).strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0.0, (parsed - datetime.now(UTC)).total_seconds())


# ---------------------------------------------------------------------------
# Default factories — these reproduce the previously hardcoded behavior so that
# callers which do not pass an explicit config keep an unchanged stable-path
# trace (jitter off, Retry-After off, same timeouts, same attempt counts).
# ---------------------------------------------------------------------------


def default_summary_llm_timeout() -> TimeoutConfig:
    # Previously httpx.Timeout(None, connect=30, read=300).
    return TimeoutConfig(connect_seconds=30.0, read_seconds=300.0)


def default_summary_llm_retry() -> RetryConfig:
    # Previously retry_delays=[1,2,4,8] (5 total attempts) on status>=500 or {408,429}.
    return RetryConfig(
        max_attempts=5,
        backoff_base_seconds=1.0,
        backoff_multiplier=2.0,
        backoff_max_seconds=8.0,
        retryable_status_codes=frozenset({408, 429}),
        retry_on_server_errors=True,
    )


def default_scrape_timeout() -> TimeoutConfig:
    # Previously SCRAPE_TIMEOUT = httpx.Timeout(None, connect=20, read=60).
    return TimeoutConfig(connect_seconds=20.0, read_seconds=60.0)


def default_scrape_retry() -> RetryConfig:
    # Previously Jina retry_delays=[1,2,4,8]: 4 attempts, sleeps 1,2,4.
    return RetryConfig(
        max_attempts=4,
        backoff_base_seconds=1.0,
        backoff_multiplier=2.0,
        backoff_max_seconds=8.0,
        retryable_status_codes=frozenset({408, 409, 425, 429}),
        retry_on_server_errors=True,
    )


def default_scrape_fallback_retry() -> RetryConfig:
    # Previously HTTP fallback retry_delays=[1,2,4]: 3 attempts, sleeps 1,2.
    return RetryConfig(
        max_attempts=3,
        backoff_base_seconds=1.0,
        backoff_multiplier=2.0,
        backoff_max_seconds=8.0,
        retryable_status_codes=frozenset({408, 409, 425, 429}),
        retry_on_server_errors=True,
    )


def default_search_timeout() -> TimeoutConfig:
    # Previously a 30s scalar timeout passed to create_async_client.
    return TimeoutConfig(total_seconds=30.0)


def default_search_retry() -> RetryConfig:
    # Previously tenacity stop_after_attempt(3) + wait_exponential(min=4,max=10).
    return RetryConfig(
        max_attempts=3,
        backoff_base_seconds=4.0,
        backoff_multiplier=2.0,
        backoff_max_seconds=10.0,
        retryable_status_codes=frozenset({408, 429}),
        retry_on_server_errors=True,
    )
