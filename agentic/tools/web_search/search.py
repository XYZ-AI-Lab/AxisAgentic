# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Web search tool powered by the Serper Google Search API.

Provides a general-purpose ``web_search`` tool for agentic pipelines.
The tool queries the Serper API and returns structured JSON with organic results.

Key capabilities beyond a naive wrapper:

* Automatic retry with exponential back-off on transient HTTP errors.
* Quote-retry logic: when the initial query yields no organic results and contains double-quotes, a second attempt strips the quotes automatically.
* Filtering of banned URLs (e.g. HuggingFace dataset/spaces pages that leak benchmark answers).
* Safe URL-decoding that preserves RFC 3986 reserved characters.
* Rich metadata (per-attempt timing, request count, decode latency).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any
from urllib.parse import unquote

import httpx

from agentic.contracts.messages import ToolResultStatus
from agentic.tools.base import CallableTool, ToolResult
from agentic.tools.web_search._retry import (
    RetryConfig,
    TimeoutConfig,
    coerce_timeout,
    compute_backoff,
    default_search_retry,
    default_search_timeout,
    parse_retry_after,
    should_retry_transport_error,
)
from agentic.tools.web_search._utils import DEFAULT_HTTP_LIMITS, create_async_client, is_banned_url

logger = logging.getLogger(__name__)

_DEFAULT_SERPER_BASE_URL = "https://google.serper.dev"


# ---------------------------------------------------------------------------
# URL decode helpers
# ---------------------------------------------------------------------------

_RESERVED_PERCENT_ENCODINGS = frozenset(
    {
        "%2f",
        "%2F",  # /
        "%3f",
        "%3F",  # ?
        "%23",  # #
        "%26",  # &
        "%3d",
        "%3D",  # =
        "%40",  # @
        "%3a",
        "%3A",  # :
        "%5b",
        "%5B",  # [
        "%5d",
        "%5D",  # ]
        "%21",  # !
        "%24",  # $
        "%27",  # '
        "%28",  # (
        "%29",  # )
        "%2a",
        "%2A",  # *
        "%2b",
        "%2B",  # +
        "%2c",
        "%2C",  # ,
        "%3b",
        "%3B",  # ;
        "%25",  # %
        "%20",  # space
    }
)


def _safe_unquote(url: str) -> str:
    """Decode percent-encoded characters that don't alter URL semantics."""
    if not url:
        return url

    result: list[str] = []
    i = 0
    n = len(url)

    while i < n:
        if url[i] == "%" and i + 2 < n:
            hex_chars = url[i + 1 : i + 3]
            if all(c in "0123456789ABCDEFabcdef" for c in hex_chars):
                percent_encoded = url[i : i + 3]
                if percent_encoded in _RESERVED_PERCENT_ENCODINGS:
                    result.append(percent_encoded)
                    i += 3
                    continue

                encoded_sequence = percent_encoded
                j = i + 3
                while j + 2 < n and url[j] == "%":
                    next_hex = url[j + 1 : j + 3]
                    if all(c in "0123456789ABCDEFabcdef" for c in next_hex):
                        next_encoded = url[j : j + 3]
                        if next_encoded in _RESERVED_PERCENT_ENCODINGS:
                            break
                        encoded_sequence += next_encoded
                        j += 3
                    else:
                        break

                try:
                    decoded = unquote(encoded_sequence)
                    result.append(decoded)
                    i = j
                    continue
                except Exception:
                    result.append(percent_encoded)
                    i += 3
                    continue

        result.append(url[i])
        i += 1

    return "".join(result)


def _decode_urls_in_data(data: Any) -> Any:
    """Recursively decode percent-encoded HTTP URLs in nested structures."""
    if isinstance(data, str):
        if "%" in data and "http" in data:
            return _safe_unquote(data)
        return data
    if isinstance(data, list):
        return [_decode_urls_in_data(item) for item in data]
    if isinstance(data, dict):
        return {key: _decode_urls_in_data(value) for key, value in data.items()}
    return data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_payload(
    query: str,
    *,
    num: int,
    gl: str,
    hl: str,
    location: str | None,
    time_range: str | None,
    page: int | None,
    autocorrect: bool | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "q": query.strip(),
        "gl": gl,
        "hl": hl,
        "num": num,
    }
    if location is not None:
        payload["location"] = location
    if time_range is not None:
        payload["tbs"] = time_range
    if page is not None:
        payload["page"] = page
    if autocorrect is not None:
        payload["autocorrect"] = autocorrect
    return payload


def _filter_organic(data: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    organic: list[dict[str, Any]] = []
    filtered = 0
    for item in data.get("organic", []):
        if is_banned_url(item.get("link", "")):
            filtered += 1
            continue
        organic.append(item)
    return organic, filtered


def _format_results_text(data: dict[str, Any]) -> str:
    """Format search results into a concise readable text block."""
    parts: list[str] = []

    kg = data.get("knowledgeGraph")
    if isinstance(kg, dict):
        title = kg.get("title", "")
        desc = kg.get("description", "")
        if title:
            parts.append(f"Knowledge Graph: {title}")
        if desc:
            parts.append(f"  {desc}")

    organic = data.get("organic", [])
    if organic:
        parts.append("Organic Results:")
        for idx, item in enumerate(organic, start=1):
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            parts.append(f"  [{idx}] {title}")
            if link:
                parts.append(f"      URL: {link}")
            if snippet:
                parts.append(f"      {snippet}")

    paa = data.get("peopleAlsoAsk", [])
    if paa:
        parts.append("People Also Ask:")
        for item in paa:
            q = item.get("question", "")
            s = item.get("snippet", "")
            if q:
                parts.append(f"  Q: {q}")
            if s:
                parts.append(f"  A: {s}")

    if not parts:
        return json.dumps(data, ensure_ascii=False, indent=2)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_DEFAULT_WEB_SEARCH_DESCRIPTION: str = (
    "Search the web using Google and return relevant results. "
    "Use this tool to find up-to-date information, verify facts, or discover URLs for further investigation.\n"
    "\n"
    "Returns structured search results including organic results, knowledge graph, and related questions."
)

DEFAULT_WEB_SEARCH_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query string.",
        },
        "num": {
            "type": "integer",
            "description": "Number of results to return (default: 10, max: 100).",
            "default": 10,
        },
        "gl": {
            "type": "string",
            "description": "Country code for localised results, e.g., 'us', 'gb', (ISO 3166-1 alpha-2, default: 'us').",
            "default": "us",
        },
        "hl": {
            "type": "string",
            "description": "Language code for results, e.g., 'en', 'zh', (ISO 639-1, default: 'en').",
            "default": "en",
        },
        "location": {
            "type": "string",
            "description": "Freeform location string, e.g. 'New York, United States'.",
        },
        "time_range": {
            "type": "string",
            "description": "Recency filter: 'qdr:h' (hour), 'qdr:d' (day), 'qdr:w' (week), 'qdr:m' (month), 'qdr:y' (year).",
        },
        "page": {
            "type": "integer",
            "description": "Page number for pagination (default: 1).",
        },
    },
    "required": ["query"],
}

SIMPLE_WEB_SEARCH_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query string.",
        },
        "num": {
            "type": "integer",
            "description": "Number of results to return (default: 10, max: 100).",
            "default": 10,
        },
        "gl": {
            "type": "string",
            "description": "Country code for localised results, e.g., 'us', 'gb', (ISO 3166-1 alpha-2, default: 'us').",
            "default": "us",
        },
        "hl": {
            "type": "string",
            "description": "Language code for results, e.g., 'en', 'zh', (ISO 639-1, default: 'en').",
            "default": "en",
        },
    },
    "required": ["query"],
}


def create_web_search_tool(
    *,
    name: str = "web_search",
    description: str = _DEFAULT_WEB_SEARCH_DESCRIPTION,
    parameters: dict[str, Any] = DEFAULT_WEB_SEARCH_PARAMETERS,
    server_name: str | None = None,
    serper_api_key: str | None = None,
    serper_base_url: str | None = None,
    max_calls_per_task: int | None = None,
    max_consecutive_calls_with_same_args: int | None = 2,
    timeout: float | TimeoutConfig | None = None,
    retry: RetryConfig | None = None,
    output_format: str = "json",
    include_search_parameters_in_content: bool = False,
) -> CallableTool:
    """Create a web search tool backed by the Serper Google Search API.

    Args:
        name: Tool name exposed to the model.
        description: Tool description exposed to the model.
        parameters: Tool parameters schema.
        server_name: Optional MCP server name for grouping.
        serper_api_key: Serper API key. Falls back to ``SERPER_API_KEY`` env.
        serper_base_url: Serper base URL. Falls back to ``SERPER_BASE_URL`` env.
        max_calls_per_task: Per-task call budget for this tool.
        max_consecutive_calls_with_same_args: Block repeated identical queries.
        timeout: Per-request timeout budget. Accepts a ``TimeoutConfig``, a bare
            number of seconds (legacy), or ``None`` (reproduces the prior 30s default).
        retry: Transient-error retry policy. None reproduces the prior behavior
            (3 attempts, exponential backoff) but only retries connection errors,
            timeouts, and the configured retryable status codes (no longer every 4xx).
        output_format: ``"json"`` returns structured JSON, ``"text"`` returns human-readable text.
        include_search_parameters_in_content: Include Serper's ``searchParameters`` in JSON output.
    """
    api_key = serper_api_key or os.environ.get("SERPER_API_KEY", "")
    base_url = serper_base_url or os.environ.get("SERPER_BASE_URL", _DEFAULT_SERPER_BASE_URL)
    resolved_timeout = coerce_timeout(timeout) or default_search_timeout()
    resolved_retry = retry or default_search_retry()
    shared_client = create_async_client(timeout=resolved_timeout.to_httpx(), limits=DEFAULT_HTTP_LIMITS)

    async def _serper_request(payload: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        for attempt in range(1, resolved_retry.max_attempts + 1):
            try:
                response = await shared_client.post(f"{base_url}/search", json=payload, headers=headers)
                response.raise_for_status()
                return response
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if should_retry_transport_error(exc, resolved_retry) and attempt < resolved_retry.max_attempts:
                    delay = compute_backoff(attempt, resolved_retry)
                    logger.info("web_search: %s, retry in %.1fs (attempt %d)", type(exc).__name__, delay, attempt + 1)
                    await asyncio.sleep(delay)
                    continue
                raise
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if resolved_retry.is_retryable_status(status) and attempt < resolved_retry.max_attempts:
                    delay = compute_backoff(attempt, resolved_retry, retry_after=parse_retry_after(e.response.headers))
                    logger.info("web_search: HTTP %d, retry in %.1fs", status, delay)
                    await asyncio.sleep(delay)
                    continue
                raise
        msg = "web_search: all retries exhausted"
        raise RuntimeError(msg)

    async def _do_search(
        query: str,
        *,
        num: int,
        gl: str,
        hl: str,
        location: str | None,
        time_range: str | None,
        page: int | None,
        autocorrect: bool | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        payload = _build_payload(
            query,
            num=num,
            gl=gl,
            hl=hl,
            location=location,
            time_range=time_range,
            page=page,
            autocorrect=autocorrect,
        )
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

        t0 = time.perf_counter()
        response = await _serper_request(payload, headers)
        request_ms = (time.perf_counter() - t0) * 1000.0

        data = response.json()
        organic, filtered = _filter_organic(data)

        return (
            organic,
            data.get("searchParameters", {}),
            {
                "query": query,
                "request_ms": round(request_ms, 1),
                "raw_organic_count": len(data.get("organic", [])),
                "organic_count": len(organic),
                "filtered_count": filtered,
            },
        )

    async def _search(
        query: str = "",
        q: str | None = None,
        num: int = 10,
        gl: str = "us",
        hl: str = "en",
        location: str | None = None,
        time_range: str | None = None,
        page: int | None = None,
        autocorrect: bool | None = None,  # noqa: FBT001
    ) -> ToolResult:
        search_started = time.perf_counter()
        if not query and q:
            query = q

        if not api_key:
            return ToolResult(
                content="SERPER_API_KEY not set",
                status=ToolResultStatus.FAILED,
                metadata={"success": False, "error": "SERPER_API_KEY not set"},
            )

        if not query or not query.strip():
            return ToolResult(
                content="Search query cannot be empty",
                status=ToolResultStatus.FAILED,
                metadata={"success": False, "error": "Search query cannot be empty"},
            )

        try:
            kwargs = dict(
                num=num,
                gl=gl,
                hl=hl,
                location=location,
                time_range=time_range,
                page=page,
                autocorrect=autocorrect,
            )
            attempts: list[dict[str, Any]] = []
            quote_retry = False

            organic, search_params, attempt_meta = await _do_search(query.strip(), **kwargs)
            attempts.append(attempt_meta)

            # Quote-retry: strip double-quotes and retry when no results
            if not organic and '"' in query:
                stripped = query.replace('"', "").strip()
                if stripped:
                    quote_retry = True
                    organic, search_params, retry_meta = await _do_search(stripped, **kwargs)
                    attempts.append(retry_meta)

            response_data: dict[str, Any] = {
                "organic": organic,
            }
            if include_search_parameters_in_content:
                response_data["searchParameters"] = search_params
            response_data = _decode_urls_in_data(response_data)
            search_params = _decode_urls_in_data(search_params)

            if output_format == "text":
                content = _format_results_text(response_data)
            else:
                content = json.dumps(response_data, ensure_ascii=False)

            total_ms = round((time.perf_counter() - search_started) * 1000.0, 1)
            return ToolResult(
                content=content,
                metadata={
                    "success": True,
                    "timing_ms": total_ms,
                    "search_parameters": search_params,
                    "request_count": len(attempts),
                    "quote_retry_used": quote_retry,
                    "attempts": attempts,
                },
            )

        except Exception as exc:
            logger.warning("Web search failed: %s", exc)
            return ToolResult(
                content=f"Search failed: {exc}",
                status=ToolResultStatus.FAILED,
                metadata={"success": False, "error": f"Search failed: {exc}"},
            )

    return CallableTool(
        name=name,
        description=description,
        server_name=server_name,
        parameters=parameters,
        strict_mode=False,
        fn=_search,
        max_calls_per_task=max_calls_per_task,
        max_consecutive_calls_with_same_args=max_consecutive_calls_with_same_args,
        emoji="🔍",
    )
