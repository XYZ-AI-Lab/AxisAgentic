# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Shared YAML schema for outbound-call timeout/retry policy.

These pydantic models are the config-surface for the runtime
``agentic.tools.web_search._retry`` primitives, reused by the web-search and
wide-search recipes so the timeout/retry knobs are identical across recipes.
``to_runtime()`` converts to the frozen runtime dataclasses the
tool layer consumes.

Defaults reproduce the previously hardcoded behavior on the *stable* path;
``jitter`` / ``respect_retry_after`` only affect the retry (instability) path.
The summary-LLM defaults deliberately enable jitter + Retry-After and use a 30s
base backoff, which is the mitigation for summary-endpoint 429 storms.
"""

from __future__ import annotations

from dataclasses import replace

from pydantic import BaseModel, ConfigDict, Field

from agentic.model_clients.openai_client import DEFAULT_TRANSIENT_ENDPOINT_ERROR_STATUS_CODES
from agentic.tools.web_search._retry import RetryConfig as RuntimeRetryConfig
from agentic.tools.web_search._retry import TimeoutConfig as RuntimeTimeoutConfig


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TimeoutConfig(_StrictModel):
    """Per-request timeout budget for an outbound call.

    ``connect_seconds`` / ``read_seconds`` fall back to ``total_seconds`` when
    unset; ``None`` means no limit for that phase.
    """

    connect_seconds: float | None = Field(default=None, gt=0)
    read_seconds: float | None = Field(default=None, gt=0)
    total_seconds: float | None = Field(default=None, gt=0)

    def to_runtime(self) -> RuntimeTimeoutConfig:
        return RuntimeTimeoutConfig(
            connect_seconds=self.connect_seconds,
            read_seconds=self.read_seconds,
            total_seconds=self.total_seconds,
        )


class RetryConfig(_StrictModel):
    """Transient-error retry policy for an outbound call.

    ``max_attempts`` counts the first attempt. ``jitter`` and
    ``respect_retry_after`` only affect the retry path, so a request that
    succeeds on the first attempt is unaffected.
    """

    max_attempts: int = Field(default=1, ge=1)
    backoff_base_seconds: float = Field(default=1.0, ge=0)
    backoff_multiplier: float = Field(default=2.0, ge=1)
    backoff_max_seconds: float = Field(default=8.0, ge=0)
    jitter: bool = False
    respect_retry_after: bool = False
    retryable_status_codes: list[int] = Field(default_factory=list)
    retry_on_server_errors: bool = True
    retry_on_timeout: bool = True
    retry_on_connection_error: bool = True

    def to_runtime(self) -> RuntimeRetryConfig:
        return RuntimeRetryConfig(
            max_attempts=self.max_attempts,
            backoff_base_seconds=self.backoff_base_seconds,
            backoff_multiplier=self.backoff_multiplier,
            backoff_max_seconds=self.backoff_max_seconds,
            jitter=self.jitter,
            respect_retry_after=self.respect_retry_after,
            retryable_status_codes=frozenset(self.retryable_status_codes),
            retry_on_server_errors=self.retry_on_server_errors,
            retry_on_timeout=self.retry_on_timeout,
            retry_on_connection_error=self.retry_on_connection_error,
        )


class ModelTransportRetryConfig(_StrictModel):
    """Transient transport retry policy for OpenAI-compatible model endpoints."""

    retryable_status_codes: list[int] = Field(default_factory=lambda: list(DEFAULT_TRANSIENT_ENDPOINT_ERROR_STATUS_CODES))
    backoff_base_seconds: float = Field(default=60.0, ge=0)
    backoff_multiplier: float = Field(default=2.0, ge=1)
    backoff_max_seconds: float = Field(default=300.0, ge=0)
    jitter: bool = True
    respect_retry_after: bool = True
    retry_on_timeout: bool = True
    retry_on_connection_error: bool = True
    exit_after_seconds: float | None = Field(default=300.0, ge=0)


class ModelTransportConfig(_StrictModel):
    """Timeout/retry for a primary OpenAI-compatible model endpoint."""

    timeout: TimeoutConfig = Field(default_factory=lambda: TimeoutConfig(total_seconds=600.0))
    retry: ModelTransportRetryConfig = Field(default_factory=ModelTransportRetryConfig)

    def timeout_seconds(self) -> float:
        if self.timeout.total_seconds is None:
            msg = "model.transport.timeout.total_seconds is required for model endpoints"
            raise ValueError(msg)
        return self.timeout.total_seconds


class ModelResponseRetryConfig(_StrictModel):
    """Provider-neutral retry policy for malformed or truncated model responses."""

    max_attempts: int = Field(default=10, ge=1)
    wait_seconds: float = Field(default=30.0, ge=0)


def legacy_model_retryable_status_codes(status_codes: list[int]) -> list[int]:
    """Keep only transient-looking status codes from old endpoint-error config."""
    return [code for code in status_codes if code in {408, 409, 425, 429} or 500 <= code <= 599]


# ---------------------------------------------------------------------------
# Per-call-class default factories
# ---------------------------------------------------------------------------


def default_summary_timeout() -> TimeoutConfig:
    return TimeoutConfig(connect_seconds=30.0, read_seconds=300.0)


def default_summary_retry() -> RetryConfig:
    # Coverage matches the prior summary path (status>=500 or {408,429}), but
    # jitter + Retry-After are on and the base backoff is 30s — the mitigation
    # for summary-endpoint 429 storms. Only the retry path is affected.
    return RetryConfig(
        max_attempts=5,
        backoff_base_seconds=30.0,
        backoff_multiplier=2.0,
        backoff_max_seconds=240.0,
        retryable_status_codes=[408, 429],
        jitter=True,
        respect_retry_after=True,
    )


def default_code_exec_retry() -> RetryConfig:
    # E2B sandbox 429 rate-limit mitigation: mirror the summary-LLM retry profile
    # (patient 30s base backoff, jitter + Retry-After on, retry on 408/429 and
    # 5xx) so a rate-limited sandbox call is waited out and retried rather than
    # surfaced to the model as the tool response.
    return RetryConfig(
        max_attempts=5,
        backoff_base_seconds=30.0,
        backoff_multiplier=2.0,
        backoff_max_seconds=240.0,
        retryable_status_codes=[408, 429],
        jitter=True,
        respect_retry_after=True,
    )


def default_scrape_timeout() -> TimeoutConfig:
    return TimeoutConfig(connect_seconds=20.0, read_seconds=60.0)


def default_scrape_retry() -> RetryConfig:
    return RetryConfig(
        max_attempts=4,
        backoff_base_seconds=1.0,
        backoff_multiplier=2.0,
        backoff_max_seconds=8.0,
        retryable_status_codes=[408, 409, 425, 429],
    )


def default_search_timeout() -> TimeoutConfig:
    return TimeoutConfig(total_seconds=30.0)


def default_search_retry() -> RetryConfig:
    return RetryConfig(
        max_attempts=3,
        backoff_base_seconds=4.0,
        backoff_multiplier=2.0,
        backoff_max_seconds=10.0,
        retryable_status_codes=[408, 429],
    )


class ScrapeToolConfig(_StrictModel):
    """Timeout/retry for the Jina fetch and the direct-HTTP fallback."""

    timeout: TimeoutConfig = Field(default_factory=default_scrape_timeout)
    retry: RetryConfig = Field(default_factory=default_scrape_retry)
    fallback_max_attempts: int = Field(default=3, ge=1)

    def fallback_retry_runtime(self) -> RuntimeRetryConfig:
        return replace(self.retry.to_runtime(), max_attempts=self.fallback_max_attempts)


class SearchToolConfig(_StrictModel):
    """Timeout/retry for the Serper web-search request."""

    timeout: TimeoutConfig = Field(default_factory=default_search_timeout)
    retry: RetryConfig = Field(default_factory=default_search_retry)
