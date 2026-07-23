"""Tests for the configurable timeout/retry policy of the web-search tools.

Focus areas:
* defaults reproduce the previously hardcoded behavior (stable-trace preservation),
* the backoff helper and Retry-After parsing,
* summary-LLM and scrape retries on transient endpoint instability,
* the search tool no longer retries non-retryable 4xx.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

import agentic.tools.web_search._scrape_utils as su
from agentic.tools.web_search import search
from agentic.tools.web_search._retry import (
    RetryConfig,
    TimeoutConfig,
    compute_backoff,
    default_scrape_retry,
    default_scrape_timeout,
    default_search_retry,
    default_search_timeout,
    default_summary_llm_retry,
    default_summary_llm_timeout,
    parse_retry_after,
)
from agentic.tools.web_search._scrape_utils import extract_with_llm, scrape_with_jina

# ---------------------------------------------------------------------------
# Backoff / parsing primitives
# ---------------------------------------------------------------------------


def test_compute_backoff_matches_legacy_fixed_delays() -> None:
    retry = default_summary_llm_retry()
    assert [compute_backoff(n, retry) for n in (1, 2, 3, 4)] == [1.0, 2.0, 4.0, 8.0]


def test_compute_backoff_caps_at_max() -> None:
    retry = RetryConfig(backoff_base_seconds=1.0, backoff_multiplier=2.0, backoff_max_seconds=8.0)
    assert compute_backoff(5, retry) == 8.0
    assert compute_backoff(10, retry) == 8.0


def test_compute_backoff_jitter_within_equal_jitter_range() -> None:
    retry = RetryConfig(backoff_base_seconds=4.0, backoff_multiplier=1.0, backoff_max_seconds=4.0, jitter=True)
    for _ in range(200):
        delay = compute_backoff(1, retry)
        assert 2.0 <= delay <= 4.0


def test_compute_backoff_respects_retry_after() -> None:
    retry = RetryConfig(backoff_base_seconds=1.0, backoff_max_seconds=8.0, respect_retry_after=True)
    # Server asks for longer than the computed backoff -> honor the server.
    assert compute_backoff(1, retry, retry_after=5.0) == 5.0
    # Server asks for less -> keep the computed backoff.
    assert compute_backoff(3, retry, retry_after=1.0) == 4.0


def test_parse_retry_after_seconds_and_missing() -> None:
    assert parse_retry_after({"Retry-After": "12"}) == 12.0
    assert parse_retry_after({"retry-after": "0"}) == 0.0
    assert parse_retry_after({}) is None
    assert parse_retry_after(None) is None


def test_default_factories_preserve_legacy_values() -> None:
    st = default_summary_llm_timeout()
    assert (st.connect_seconds, st.read_seconds, st.total_seconds) == (30.0, 300.0, None)
    sr = default_summary_llm_retry()
    assert sr.max_attempts == 5
    assert sr.retryable_status_codes == frozenset({408, 429})
    assert sr.jitter is False and sr.respect_retry_after is False  # tool-layer default = legacy

    ct = default_scrape_timeout()
    assert (ct.connect_seconds, ct.read_seconds) == (20.0, 60.0)
    assert default_scrape_retry().max_attempts == 4
    assert default_scrape_retry().retryable_status_codes == frozenset({408, 409, 425, 429})

    assert default_search_timeout().total_seconds == 30.0
    assert default_search_retry().max_attempts == 3


def test_timeout_to_httpx_phase_fallback() -> None:
    # connect/read fall back to total; total covers write/pool.
    t = TimeoutConfig(total_seconds=30.0).to_httpx()
    assert t.connect == 30.0
    assert t.read == 30.0
    # explicit connect/read with no total leaves write/pool unlimited.
    t2 = TimeoutConfig(connect_seconds=20.0, read_seconds=60.0).to_httpx()
    assert t2.connect == 20.0
    assert t2.read == 60.0


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _SeqClient:
    """Async client returning a scripted sequence of (status, body) responses."""

    def __init__(self, responses: list[tuple[int, dict[str, Any] | None]], *, retry_after: str | None = None) -> None:
        self._responses = responses
        self._retry_after = retry_after
        self.calls = 0

    async def post(self, url: str, **_kwargs: Any) -> httpx.Response:
        status, body = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        headers = {"Retry-After": self._retry_after} if (self._retry_after and status != 200) else {}
        return httpx.Response(status, json=body, headers=headers, request=httpx.Request("POST", url))

    async def get(self, url: str, **_kwargs: Any) -> httpx.Response:
        status, body = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        text = "ok" if body is None else None
        return httpx.Response(status, json=body, text=text, request=httpx.Request("GET", url))


_SUMMARY_OK = {"choices": [{"message": {"content": "the answer"}}], "usage": {"total_tokens": 3}}


@pytest.fixture
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(su.asyncio, "sleep", fake_sleep)
    return recorded


# ---------------------------------------------------------------------------
# Summary LLM extraction
# ---------------------------------------------------------------------------


def test_extract_with_llm_stable_path_single_attempt(_no_sleep: list[float]) -> None:
    client = _SeqClient([(200, _SUMMARY_OK)])
    result = asyncio.run(
        extract_with_llm(
            content="some content",
            info_to_extract="the answer",
            llm_base_url="http://llm.example/v1",
            llm_api_key=None,
            client=client,  # type: ignore[arg-type]
            cache=None,
        )
    )
    assert result["success"] is True
    assert result["extracted_info"] == "the answer"
    assert client.calls == 1
    assert _no_sleep == []  # stable endpoint never sleeps


def test_extract_with_llm_retries_on_429_then_succeeds(_no_sleep: list[float]) -> None:
    client = _SeqClient([(429, None), (429, None), (200, _SUMMARY_OK)], retry_after="2")
    retry = RetryConfig(
        max_attempts=5,
        backoff_base_seconds=1.0,
        backoff_multiplier=2.0,
        backoff_max_seconds=8.0,
        retryable_status_codes=frozenset({429}),
        respect_retry_after=True,
    )
    result = asyncio.run(
        extract_with_llm(
            content="some content",
            info_to_extract="the answer",
            llm_base_url="http://llm.example/v1",
            llm_api_key=None,
            client=client,  # type: ignore[arg-type]
            cache=None,
            retry=retry,
        )
    )
    assert result["success"] is True
    assert client.calls == 3
    # backoff(1)=1 vs Retry-After=2 -> 2; backoff(2)=2 vs 2 -> 2
    assert _no_sleep == [2.0, 2.0]


def test_extract_with_llm_gives_up_after_max_attempts(_no_sleep: list[float]) -> None:
    client = _SeqClient([(429, None)])
    retry = RetryConfig(max_attempts=3, retryable_status_codes=frozenset({429}))
    result = asyncio.run(
        extract_with_llm(
            content="some content",
            info_to_extract="the answer",
            llm_base_url="http://llm.example/v1",
            llm_api_key=None,
            client=client,  # type: ignore[arg-type]
            cache=None,
            retry=retry,
        )
    )
    assert result["success"] is False
    assert "LLM HTTP 429" in result["error"]
    assert client.calls == 3  # 1 initial + 2 retries
    assert len(_no_sleep) == 2


# ---------------------------------------------------------------------------
# Scrape (Jina) retry path
# ---------------------------------------------------------------------------


def test_scrape_with_jina_retries_on_500_then_succeeds(_no_sleep: list[float]) -> None:
    client = _SeqClient([(500, None), (200, None)])
    retry = RetryConfig(max_attempts=4, retryable_status_codes=frozenset({429}))
    result = asyncio.run(
        scrape_with_jina(
            "https://example.com",
            jina_api_key="key",
            jina_base_url="https://r.jina.ai",
            client=client,  # type: ignore[arg-type]
            retry=retry,
        )
    )
    assert result["success"] is True
    assert client.calls == 2
    assert _no_sleep == [1.0]


# ---------------------------------------------------------------------------
# Serper search retry semantics
# ---------------------------------------------------------------------------


def _make_search_tool(monkeypatch: pytest.MonkeyPatch, client: _SeqClient, retry: RetryConfig) -> Any:
    monkeypatch.setattr(search, "create_async_client", lambda *_a, **_k: client)

    async def fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(search.asyncio, "sleep", fake_sleep)
    return search.create_web_search_tool(serper_api_key="key", retry=retry)


def test_web_search_retries_on_503_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _SeqClient([(503, None), (503, None), (200, {"organic": [], "searchParameters": {"q": "x"}})])
    tool = _make_search_tool(monkeypatch, client, default_search_retry())
    result = asyncio.run(tool._fn(query="x"))
    assert result.metadata["success"] is True
    assert client.calls == 3


def test_web_search_does_not_retry_404(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _SeqClient([(404, None)])
    tool = _make_search_tool(monkeypatch, client, default_search_retry())
    result = asyncio.run(tool._fn(query="x"))
    assert result.metadata["success"] is False
    assert client.calls == 1  # 404 is not retryable


# ---------------------------------------------------------------------------
# Review fixes: scalar-timeout coercion + per-flag transport retry
# ---------------------------------------------------------------------------


class _RaisingClient:
    """Async client that raises ``exc`` for the first ``n_fail`` calls then 200s."""

    def __init__(self, exc: Exception, n_fail: int) -> None:
        self._exc = exc
        self._n_fail = n_fail
        self.calls = 0

    async def post(self, url: str, **_kwargs: Any) -> httpx.Response:
        self.calls += 1
        if self.calls <= self._n_fail:
            raise self._exc
        return httpx.Response(200, json={"organic": [], "searchParameters": {"q": "x"}}, request=httpx.Request("POST", url))


def test_coerce_timeout_accepts_scalar_config_and_none() -> None:
    from agentic.tools.web_search._retry import coerce_timeout

    assert coerce_timeout(None) is None
    assert coerce_timeout(10.0) == TimeoutConfig(total_seconds=10.0)
    tc = TimeoutConfig(connect_seconds=5.0)
    assert coerce_timeout(tc) is tc


def test_web_search_tool_accepts_scalar_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    # A legacy float timeout must not crash tool creation (was AttributeError).
    client = _SeqClient([(200, {"organic": [], "searchParameters": {"q": "x"}})])
    monkeypatch.setattr(search, "create_async_client", lambda *_a, **_k: client)
    tool = search.create_web_search_tool(serper_api_key="key", timeout=10.0)
    result = asyncio.run(tool._fn(query="x"))
    assert result.metadata["success"] is True


def test_should_retry_transport_error_respects_each_flag() -> None:
    from agentic.tools.web_search._retry import should_retry_transport_error

    only_conn = RetryConfig(retry_on_timeout=False, retry_on_connection_error=True)
    only_timeout = RetryConfig(retry_on_timeout=True, retry_on_connection_error=False)
    assert should_retry_transport_error(httpx.ConnectError("x"), only_conn) is True
    assert should_retry_transport_error(httpx.ReadTimeout("x"), only_conn) is False
    assert should_retry_transport_error(httpx.ConnectError("x"), only_timeout) is False
    assert should_retry_transport_error(httpx.ReadTimeout("x"), only_timeout) is True
    # ConnectTimeout subclasses TimeoutException -> gated by retry_on_timeout.
    assert should_retry_transport_error(httpx.ConnectTimeout("x"), only_timeout) is True


def test_web_search_disabled_timeout_flag_does_not_retry_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _RaisingClient(httpx.ReadTimeout("slow"), n_fail=10)
    retry = RetryConfig(max_attempts=3, retry_on_timeout=False, retry_on_connection_error=True)
    tool = _make_search_tool(monkeypatch, client, retry)
    result = asyncio.run(tool._fn(query="x"))
    assert result.metadata["success"] is False
    assert client.calls == 1  # timeouts disabled -> no retry despite connection retry on


def test_web_search_connection_retry_still_active_when_timeout_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _RaisingClient(httpx.ConnectError("down"), n_fail=2)
    retry = RetryConfig(max_attempts=3, retry_on_timeout=False, retry_on_connection_error=True)
    tool = _make_search_tool(monkeypatch, client, retry)
    result = asyncio.run(tool._fn(query="x"))
    assert result.metadata["success"] is True
    assert client.calls == 3  # connection errors still retried
