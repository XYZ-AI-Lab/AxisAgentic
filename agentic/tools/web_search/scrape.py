# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Web scraping and LLM extraction tools for agentic pipelines.

Provides three composable tool factories:

* :func:`create_jina_scrape_tool` — fetch raw content via Jina Reader API (with direct-HTTP fallback).
* :func:`create_llm_extract_tool` — extract specific information from raw text using an LLM (with caching).
* :func:`create_scrape_and_extract_tool` — convenience tool: Jina scrape → LLM extraction in a single call.

All three tools share the common helpers in ``_utils`` and ``_scrape_utils``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from agentic.contracts.messages import ToolResultStatus
from agentic.tools.base import CallableTool, ToolResult
from agentic.tools.web_search._scrape_utils import (
    DEFAULT_CHUNK_ENVELOPE_MODE,
    DEFAULT_GLOBAL_ANCHOR_ENABLED,
    DEFAULT_JINA_BASE_URL,
    DEFAULT_MAX_CONTENT_CHARS,
    DEFAULT_SUMMARY_LLM_CHUNK_MAX_CONCURRENT,
    DEFAULT_SUMMARY_LLM_CHUNK_OVERLAP_CHARS,
    DEFAULT_SUMMARY_LLM_CHUNK_STRATEGY,
    DEFAULT_SUMMARY_LLM_MAX_CHUNKS,
    DEFAULT_SUMMARY_LLM_MAX_INPUT_CHARS,
    DEFAULT_SUMMARY_LLM_MAX_RECURSION_DEPTH,
    extract_with_llm,
    scrape_direct_typed,
    scrape_with_http,
    scrape_with_jina,
)
from agentic.tools.web_search._utils import DEFAULT_HTTP_LIMITS, create_async_client, is_banned_url

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agentic.model_clients.request_logger import ModelRequestLogger
    from agentic.tools.web_search._retry import RetryConfig, TimeoutConfig


_RAW_SCRAPE_CACHE_KEY_VERSION = 1
_RAW_SCRAPE_CACHE_METADATA_SCHEMA_VERSION = 1
_RAW_SCRAPE_CACHE_PROVIDER = "web_search"
_RAW_SCRAPE_CACHE_COUNTER_SCOPE = "tool_instance_task"
_RAW_SCRAPE_CACHE_NORMALIZE_POLICY_NONE = "none"
_RAW_SCRAPE_CACHE_NORMALIZE_POLICY_SORT_QUERY = "lowercase_scheme_host_sort_query_preserve_fragment"
_OUTPUT_FORMAT_TEXT = "text"
_OUTPUT_FORMAT_STRUCTURED_JSON = "structured_json"


class RawScrapeCache:
    """In-memory cache for successful non-empty raw scrape results."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._store: dict[str, ToolResult] = {}
        self.hits = 0
        self.misses = 0
        self.bypasses = 0
        self.puts = 0

    @staticmethod
    def _copy_result(result: ToolResult) -> ToolResult:
        return ToolResult(
            content=result.content,
            status=result.status,
            reason=result.reason,
            metadata=dict(result.metadata),
        )

    def get(self, key_hash: str) -> ToolResult | None:
        with self._lock:
            result = self._store.get(key_hash)
            if result is None:
                self.misses += 1
                return None
            self.hits += 1
            return self._copy_result(result)

    def put(self, key_hash: str, result: ToolResult) -> None:
        with self._lock:
            self._store[key_hash] = self._copy_result(result)
            self.puts += 1

    def record_bypass(self) -> None:
        with self._lock:
            self.bypasses += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "bypasses": self.bypasses,
                "puts": self.puts,
            }


@dataclass(frozen=True)
class RawScrapeCacheOptions:
    """Options for caching raw scrape results inside a tool instance."""

    enabled: bool = False
    scope: str = "task"
    provider: str = _RAW_SCRAPE_CACHE_PROVIDER
    normalize_url: bool = False
    cache: RawScrapeCache | None = None


@dataclass(frozen=True)
class ScrapeBackendOptions:
    """Options for the raw page-fetching backend."""

    jina_api_key: str | None = None
    jina_base_url: str | None = None
    max_content_length: int = DEFAULT_MAX_CONTENT_CHARS
    # Timeout/retry for the Jina fetch and the direct-HTTP fallback. None falls
    # back to the prior hardcoded behavior, so stable traces are unchanged.
    timeout: TimeoutConfig | None = None
    retry: RetryConfig | None = None
    fallback_retry: RetryConfig | None = None
    # When True, a URL that the server answers with a structured-data Content-Type
    # (JSON/CSV/XML) is fetched directly (Jina renders such data APIs as empty HTML
    # forms). HTML/PDF/other still go through Jina. Routed purely by Content-Type.
    content_type_routing: bool = True


@dataclass(frozen=True)
class LLMExtractionOptions:
    """Options for optional LLM extraction after scraping."""

    base_url: str | None = None
    model_name: str | None = None
    api_key: str | None = None
    request_logger: ModelRequestLogger | None = None
    # Default-off: the persistent extraction cache uses file locks/fsyncs and
    # can add client-side latency in high-concurrency web-search rollouts.
    cache_enabled: bool = False
    # Timeout/retry for the summary endpoint. None falls back to the prior
    # hardcoded behavior, so stable traces are unchanged.
    timeout: TimeoutConfig | None = None
    retry: RetryConfig | None = None
    max_input_chars: int | None = None
    chunk_overlap_chars: int = DEFAULT_SUMMARY_LLM_CHUNK_OVERLAP_CHARS
    max_chunks: int = DEFAULT_SUMMARY_LLM_MAX_CHUNKS
    chunk_max_concurrent: int = DEFAULT_SUMMARY_LLM_CHUNK_MAX_CONCURRENT
    # Master switch for chunked map-reduce extraction of long scraped pages. When
    # False, long content uses the single-shot path (main behavior) regardless of
    # ``max_input_chars``. When True, ``chunk_strategy`` selects the method.
    chunked_extraction: bool = True
    # "single" — map every chunk then one reduce call; "recursive" — recursively
    # reduce the concatenated findings until they fit the input budget.
    chunk_strategy: str = DEFAULT_SUMMARY_LLM_CHUNK_STRATEGY
    # Recursive strategy only: max reduce-recursion depth before a lossless concat.
    max_recursion_depth: int = DEFAULT_SUMMARY_LLM_MAX_RECURSION_DEPTH
    # Core multi-schema extraction controls.
    global_anchor_enabled: bool = DEFAULT_GLOBAL_ANCHOR_ENABLED
    chunk_envelope_mode: str = DEFAULT_CHUNK_ENVELOPE_MODE
    csv_layer_b_enabled: bool = False
    # Extra request-body fields merged into every extraction LLM call (e.g. DeepSeek
    # reasoning: {"reasoning_effort": "max", "thinking": {"type": "enabled"}}).
    request_extra_body: dict[str, Any] | None = None


@dataclass(frozen=True)
class ScrapeAndExtractOutputOptions:
    """Options for formatting scrape-and-extract tool output."""

    format: str = _OUTPUT_FORMAT_TEXT


def _raw_scrape_cache_normalize_policy(*, normalize_url: bool) -> str:
    if not normalize_url:
        return _RAW_SCRAPE_CACHE_NORMALIZE_POLICY_NONE
    return _RAW_SCRAPE_CACHE_NORMALIZE_POLICY_SORT_QUERY


def _normalize_raw_scrape_cache_url(url: str, *, enabled: bool) -> str:
    text = (url or "").strip()
    if not enabled:
        return text
    try:
        split = urlsplit(text)
    except ValueError:
        return text
    query = urlencode(sorted(parse_qsl(split.query, keep_blank_values=True)), doseq=True)
    return urlunsplit((split.scheme.lower(), split.netloc.lower(), split.path or "", query, split.fragment))


def _raw_scrape_cache_key_hash(
    *,
    url: str,
    custom_headers: dict[str, str] | None,
    jina_base_url: str,
    max_content_length: int,
    normalize_url: bool,
) -> str:
    headers = sorted((str(key).casefold(), str(value)) for key, value in (custom_headers or {}).items())
    payload = {
        "custom_headers": headers,
        "jina_base_url": jina_base_url.rstrip("/"),
        "max_content_length": max_content_length,
        "normalize_url": normalize_url,
        "url": _normalize_raw_scrape_cache_url(url, enabled=normalize_url),
        "version": _RAW_SCRAPE_CACHE_KEY_VERSION,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _raw_scrape_cache_metadata(
    *,
    enabled: bool,
    status: str,
    raw_scrape_cache: RawScrapeCache | None,
    scope: str,
    provider: str,
    normalize_url: bool,
    key_hash: str | None = None,
    bypass_reason: str | None = None,
) -> dict[str, Any]:
    if not enabled:
        return {}
    counters = raw_scrape_cache.snapshot() if raw_scrape_cache is not None else {}
    metadata: dict[str, Any] = {
        "raw_scrape_cache_enabled": True,
        "raw_scrape_cache_provider": provider,
        "raw_scrape_cache_scope": scope,
        "raw_scrape_cache_schema_version": _RAW_SCRAPE_CACHE_METADATA_SCHEMA_VERSION,
        "raw_scrape_cache_key_version": _RAW_SCRAPE_CACHE_KEY_VERSION,
        "raw_scrape_cache_normalize_url": normalize_url,
        "raw_scrape_cache_normalize_policy": _raw_scrape_cache_normalize_policy(normalize_url=normalize_url),
        "raw_scrape_cache_status": status,
        "raw_scrape_cache_counter_scope": _RAW_SCRAPE_CACHE_COUNTER_SCOPE,
        "raw_scrape_cache_hit_count": counters.get("hits", 0),
        "raw_scrape_cache_miss_count": counters.get("misses", 0),
        "raw_scrape_cache_bypass_count": counters.get("bypasses", 0),
        "raw_scrape_cache_put_count": counters.get("puts", 0),
    }
    if key_hash:
        metadata["raw_scrape_cache_key_hash"] = key_hash
    if bypass_reason:
        metadata["raw_scrape_cache_bypass_reason"] = bypass_reason
    if status == "hit":
        metadata["raw_scrape_cache_saved_jina_request"] = True
    return metadata


def _is_cacheable_raw_scrape_result(result: ToolResult) -> bool:
    return result.status == ToolResultStatus.SUCCESS and result.metadata.get("success") is True and bool(result.content.strip())


def _validate_raw_scrape_cache_options(options: RawScrapeCacheOptions) -> None:
    if options.scope != "task":
        msg = f"Unsupported raw scrape cache scope: {options.scope!r}"
        raise ValueError(msg)
    if options.provider != _RAW_SCRAPE_CACHE_PROVIDER:
        msg = f"Unsupported raw scrape cache provider: {options.provider!r}"
        raise ValueError(msg)


def _normalize_output_format(output_format: str) -> str:
    if output_format in {_OUTPUT_FORMAT_TEXT, _OUTPUT_FORMAT_STRUCTURED_JSON}:
        return output_format
    msg = f"Unsupported scrape-and-extract output format: {output_format!r}"
    raise ValueError(msg)


def _resolve_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer value for %s=%r; using default %d", name, raw, default)
        return default


def _format_scrape_output(
    *,
    output_options: ScrapeAndExtractOutputOptions,
    success: bool,
    url: str,
    content: str,
    error: str,
) -> str:
    output_format = _normalize_output_format(output_options.format)
    if output_format == _OUTPUT_FORMAT_STRUCTURED_JSON:
        return json.dumps(
            {"success": success, "url": url, "extracted_info": content, "error": error},
            ensure_ascii=False,
        )
    return content


# ===========================================================================
# Tool 1: Jina Scrape (raw content fetching)  📄
# ===========================================================================

_JINA_SCRAPE_DESCRIPTION: str = (
    "Fetch the raw content of a web page, PDF, or other online document via the Jina Reader API.\n"
    "Returns clean Markdown text converted from the page content. Falls back to direct HTTP when Jina is unavailable.\n"
    "\n"
    "Use this tool when you need the full page content without LLM extraction — "
    "for example, to read a document, inspect source code, or feed content to another tool."
)

_JINA_SCRAPE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "The URL to scrape (web page, PDF, code file, etc.).",
        },
        "custom_headers": {
            "type": "object",
            "description": "Additional HTTP headers for the request.",
            "additionalProperties": {"type": "string"},
        },
    },
    "required": ["url"],
}


def create_jina_scrape_tool(
    *,
    name: str = "jina_scrape",
    description: str = _JINA_SCRAPE_DESCRIPTION,
    parameters: dict[str, Any] = _JINA_SCRAPE_PARAMETERS,
    server_name: str | None = None,
    jina_api_key: str | None = None,
    jina_base_url: str | None = None,
    max_content_length: int = DEFAULT_MAX_CONTENT_CHARS,
    max_calls_per_task: int | None = None,
    max_consecutive_calls_with_same_args: int | None = 2,
    timeout: TimeoutConfig | None = None,
    retry: RetryConfig | None = None,
    fallback_retry: RetryConfig | None = None,
    content_type_routing: bool = True,
) -> CallableTool:
    """Create a raw-content scraping tool backed by the Jina Reader API.

    The tool fetches web page content (HTML, PDF, code files) and converts it to clean Markdown text.
    When Jina fails it falls back to a direct HTTP GET with a browser-like User-Agent.

    Args:
        name: Tool name exposed to the model.
        description: Tool description.
        parameters: Tool parameters schema.
        server_name: Optional MCP server name for grouping.
        jina_api_key: Jina API key. Falls back to ``JINA_API_KEY`` env.
        jina_base_url: Jina base URL. Falls back to ``JINA_BASE_URL`` env.
        max_content_length: Truncate scraped content to this many characters.
        max_calls_per_task: Per-task call budget.
        max_consecutive_calls_with_same_args: Block consecutive identical requests.
        timeout: Scrape timeout configuration.
        retry: Primary scrape retry configuration.
        fallback_retry: Direct HTTP fallback retry configuration.
        content_type_routing: Try structured-data direct fetch before Jina.
    """
    api_key = jina_api_key or os.environ.get("JINA_API_KEY", "")
    base_url = jina_base_url or os.environ.get("JINA_BASE_URL", DEFAULT_JINA_BASE_URL)
    shared_client = create_async_client(follow_redirects=True, limits=DEFAULT_HTTP_LIMITS)

    async def _jina_scrape(
        url: str,
        custom_headers: dict[str, str] | None = None,
    ) -> ToolResult:
        t0 = time.perf_counter()
        timing: dict[str, float] = {}

        def _meta(*, backend: str | None = None) -> dict[str, Any]:
            timing["total"] = round((time.perf_counter() - t0) * 1000.0, 1)
            meta: dict[str, Any] = {"timing_ms": dict(timing)}
            if backend:
                meta["scrape_backend"] = backend
            return meta

        if is_banned_url(url):
            return ToolResult(
                content="This URL is blocked (benchmark data source).",
                status=ToolResultStatus.FAILED,
                metadata={
                    **_meta(),
                    "success": False,
                    "url": url,
                    "error": "This URL is blocked (benchmark data source).",
                },
            )

        # Content-type routing: data APIs (JSON/CSV/XML) are rendered by Jina as
        # empty HTML query forms; fetch them directly and keep the raw payload.
        # Decided from the server's Content-Type only (no URL allowlist). HTML/PDF
        # and any failure fall through to Jina below unchanged.
        if content_type_routing:
            direct_t0 = time.perf_counter()
            direct_result = await scrape_direct_typed(
                url,
                client=shared_client,
                custom_headers=custom_headers,
                max_chars=max_content_length,
            )
            timing["direct_probe"] = round((time.perf_counter() - direct_t0) * 1000.0, 1)
            if direct_result.get("is_structured") and direct_result.get("success"):
                raw_content = direct_result["content"]
                if raw_content.strip():
                    return ToolResult(
                        content=raw_content,
                        metadata={
                            **_meta(backend="direct"),
                            "success": True,
                            "url": url,
                            "content_type": direct_result.get("content_type", ""),
                            "total_chars": direct_result.get("total_chars", 0),
                            "total_lines": direct_result.get("total_lines", 0),
                            "truncated": direct_result.get("truncated", False),
                        },
                    )

        # Try Jina first
        jina_t0 = time.perf_counter()
        scrape_result = await scrape_with_jina(
            url,
            jina_api_key=api_key,
            jina_base_url=base_url,
            client=shared_client,
            custom_headers=custom_headers,
            max_chars=max_content_length,
            timeout=timeout,
            retry=retry,
        )
        timing["jina"] = round((time.perf_counter() - jina_t0) * 1000.0, 1)
        backend = "jina"

        # Fallback to direct HTTP
        if not scrape_result["success"]:
            logger.info("Jina failed (%s), trying direct HTTP", scrape_result["error"])
            http_t0 = time.perf_counter()
            scrape_result = await scrape_with_http(
                url,
                client=shared_client,
                custom_headers=custom_headers,
                max_chars=max_content_length,
                timeout=timeout,
                retry=fallback_retry,
            )
            timing["http_fallback"] = round((time.perf_counter() - http_t0) * 1000.0, 1)
            backend = "http_fallback"
            if not scrape_result["success"]:
                error = f"Scraping failed: {scrape_result['error']}"
                return ToolResult(
                    content=error,
                    status=ToolResultStatus.FAILED,
                    metadata={
                        **_meta(backend=backend),
                        "success": False,
                        "url": url,
                        "error": error,
                    },
                )

        raw_content = scrape_result["content"]
        if not raw_content.strip():
            return ToolResult(
                content="No content retrieved from the page.",
                metadata={
                    **_meta(backend=backend),
                    "success": False,
                    "url": url,
                    "error": "No content retrieved from the page.",
                },
            )

        return ToolResult(
            content=raw_content,
            metadata={
                **_meta(backend=backend),
                "success": True,
                "url": url,
                "total_chars": scrape_result.get("total_chars", 0),
                "total_lines": scrape_result.get("total_lines", 0),
                "truncated": scrape_result.get("truncated", False),
            },
        )

    return CallableTool(
        name=name,
        description=description,
        server_name=server_name,
        parameters=parameters,
        fn=_jina_scrape,
        max_calls_per_task=max_calls_per_task,
        max_consecutive_calls_with_same_args=max_consecutive_calls_with_same_args,
        strict_mode=False,
        emoji="📄",
    )


# ===========================================================================
# Tool 2: LLM Extraction (from raw text)  🧠
# ===========================================================================

_LLM_EXTRACT_DESCRIPTION: str = (
    "Extract specific information from raw text content using an LLM.\n"
    "Provide the raw content (e.g. from jina_scrape) and describe what information you need. "
    "Returns a focused extraction instead of the full text.\n"
    "\n"
    "Use this tool when you already have raw page content and want to extract particular facts, answers, or structured data from it."
)

_LLM_EXTRACT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "description": "The raw text content to extract information from.",
        },
        "info_to_extract": {
            "type": "string",
            "description": "What specific information to extract from the content.",
        },
    },
    "required": ["content", "info_to_extract"],
}


def _coerce_extraction_result(extracted: dict[str, Any]) -> dict[str, Any]:
    """Keep tool output string-typed if an extractor reports malformed success."""
    extracted_info = extracted.get("extracted_info", "")
    if isinstance(extracted_info, str) and extracted_info.strip():
        return extracted

    error = extracted.get("error") or (
        "LLM extraction returned empty content"
        if isinstance(extracted_info, str)
        else f"LLM extraction returned non-text content: {type(extracted_info).__name__}"
    )
    if extracted.get("success"):
        logger.warning("LLM extraction reported success with invalid content: %s", error)
    normalized = dict(extracted)
    normalized["success"] = False
    normalized["extracted_info"] = ""
    normalized["error"] = error
    return normalized


def create_llm_extract_tool(  # noqa: D417, PLR0913
    *,
    name: str = "llm_extract",
    description: str = _LLM_EXTRACT_DESCRIPTION,
    parameters: dict[str, Any] = _LLM_EXTRACT_PARAMETERS,
    server_name: str | None = None,
    llm_base_url: str | None = None,
    llm_model_name: str | None = None,
    llm_api_key: str | None = None,
    max_tokens: int = 8192,
    request_logger: ModelRequestLogger | None = None,
    cache_enabled: bool = False,
    max_calls_per_task: int | None = None,
    max_consecutive_calls_with_same_args: int | None = 2,
    timeout: TimeoutConfig | None = None,
    retry: RetryConfig | None = None,
    max_input_chars: int | None = None,
    chunk_overlap_chars: int = DEFAULT_SUMMARY_LLM_CHUNK_OVERLAP_CHARS,
    max_chunks: int = DEFAULT_SUMMARY_LLM_MAX_CHUNKS,
    chunk_max_concurrent: int = DEFAULT_SUMMARY_LLM_CHUNK_MAX_CONCURRENT,
    chunked_extraction: bool = True,
    chunk_strategy: str = DEFAULT_SUMMARY_LLM_CHUNK_STRATEGY,
    max_recursion_depth: int = DEFAULT_SUMMARY_LLM_MAX_RECURSION_DEPTH,
    global_anchor_enabled: bool = DEFAULT_GLOBAL_ANCHOR_ENABLED,
    chunk_envelope_mode: str = DEFAULT_CHUNK_ENVELOPE_MODE,
    csv_layer_b_enabled: bool = False,
    request_extra_body: dict[str, Any] | None = None,
) -> CallableTool:
    """Create an LLM extraction tool that distils raw text into focused answers.

    Cache is default-off because persistent cache writes can add client-side
    latency in high-concurrency runs. When enabled, successful results are
    cached by ``(content, info_to_extract, model, max_tokens)``.

    Args:
        name: Tool name exposed to the model.
        description: Tool description.
        parameters: Tool parameters schema.
        server_name: Optional MCP server name for grouping.
        llm_base_url: LLM API endpoint. Falls back to ``SUMMARY_LLM_BASE_URL`` env.
        llm_model_name: Model name. Falls back to ``SUMMARY_LLM_MODEL_NAME`` env.
        llm_api_key: API key. Falls back to ``SUMMARY_LLM_API_KEY`` env.
        max_tokens: Maximum tokens for the LLM response.
        cache_enabled: Whether to use the shared LLM extraction cache. This is
            disabled by default because disk-backed writes may add client-side latency.
        max_calls_per_task: Per-task call budget.
        max_consecutive_calls_with_same_args: Block consecutive identical requests.
        max_input_chars: Approximate prompt character budget before chunked extraction.
            Falls back to ``SUMMARY_LLM_MAX_INPUT_CHARS`` env, then a safe default.
            Use 0 or a negative value to disable chunking.
        chunk_overlap_chars: Character overlap between adjacent chunks.
        max_chunks: Soft target chunk count for telemetry/logging; per-chunk input budget is preserved.
        chunk_max_concurrent: Maximum concurrent chunk-map LLM calls.
        chunked_extraction: Master switch for chunked map-reduce extraction. When
            False, long content uses the single-shot path regardless of ``max_input_chars``.
        chunk_strategy: ``"single"`` (one reduce call) or ``"recursive"`` (recursive reduce).
        max_recursion_depth: Recursive strategy only: max reduce-recursion depth.
    """
    resolved_llm_base_url = llm_base_url or os.environ.get("SUMMARY_LLM_BASE_URL", "")
    resolved_llm_model = llm_model_name or os.environ.get("SUMMARY_LLM_MODEL_NAME", "LLM")
    resolved_llm_api_key = llm_api_key or os.environ.get("SUMMARY_LLM_API_KEY", "")
    resolved_max_input_chars = (
        max_input_chars if max_input_chars is not None else _resolve_env_int("SUMMARY_LLM_MAX_INPUT_CHARS", DEFAULT_SUMMARY_LLM_MAX_INPUT_CHARS)
    )

    shared_client = create_async_client(follow_redirects=True, limits=DEFAULT_HTTP_LIMITS)

    async def _llm_extract(
        content: str,
        info_to_extract: str,
    ) -> ToolResult:
        t0 = time.perf_counter()

        def _meta() -> dict[str, Any]:
            return {"timing_ms": {"total": round((time.perf_counter() - t0) * 1000.0, 1)}}

        if not resolved_llm_base_url or not resolved_llm_base_url.strip():
            return ToolResult(
                content="",
                status=ToolResultStatus.FAILED,
                metadata={
                    **_meta(),
                    "success": False,
                    "error": "LLM base URL not configured",
                    "tokens_used": 0,
                },
            )

        extract_kwargs: dict[str, Any] = {
            "content": content,
            "info_to_extract": info_to_extract,
            "llm_base_url": resolved_llm_base_url,
            "llm_api_key": resolved_llm_api_key or None,
            "client": shared_client,
            "model": resolved_llm_model,
            "max_tokens": max_tokens,
            "request_logger": request_logger,
            "timeout": timeout,
            "retry": retry,
            "max_input_chars": resolved_max_input_chars,
            "chunk_overlap_chars": chunk_overlap_chars,
            "max_chunks": max_chunks,
            "chunk_max_concurrent": chunk_max_concurrent,
            "chunked_extraction": chunked_extraction,
            "chunk_strategy": chunk_strategy,
            "max_recursion_depth": max_recursion_depth,
            "enable_global_anchor": global_anchor_enabled,
            "chunk_envelope_mode": chunk_envelope_mode,
            "csv_layer_b_enabled": csv_layer_b_enabled,
            "request_extra_body": request_extra_body,
        }
        if not cache_enabled:
            extract_kwargs["cache"] = None
        extracted = await extract_with_llm(**extract_kwargs)
        extracted = _coerce_extraction_result(extracted)

        status = ToolResultStatus.SUCCESS if extracted["success"] else ToolResultStatus.FAILED
        return ToolResult(
            content=extracted.get("extracted_info", ""),
            status=status,
            metadata={
                **_meta(),
                "success": extracted["success"],
                "error": extracted.get("error", ""),
                "tokens_used": extracted.get("tokens_used", 0),
                "extraction_strategy": extracted.get("strategy", "direct"),
                "chunk_count": extracted.get("chunk_count", 0),
                "relevant_chunk_count": extracted.get("relevant_chunk_count", 0),
            },
        )

    return CallableTool(
        name=name,
        description=description,
        server_name=server_name,
        parameters=parameters,
        fn=_llm_extract,
        max_calls_per_task=max_calls_per_task,
        max_consecutive_calls_with_same_args=max_consecutive_calls_with_same_args,
        strict_mode=False,
        emoji="🧠",
    )


# ===========================================================================
# Tool 3: Scrape & Extract (Jina + LLM combined)  🌐
# ===========================================================================

_SCRAPE_AND_EXTRACT_DESCRIPTION: str = (
    "Fetch and read the contents of a web page, PDF, or other online document. "
    "Returns the page content as clean text. When 'info_to_extract' is provided, "
    "uses an LLM to extract only the relevant information from the page.\n"
    "\n"
    "Use this tool to read web pages found via web_search, examine documents, or access any URL-addressable content."
    # "\n\n"
    # "This is a convenience tool that combines jina_scrape (raw content fetching) and llm_extract (LLM-based extraction) into a single call."
)

DEFAULT_SCRAPE_AND_EXTRACT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "The URL to scrape (web page, PDF, code file, etc.).",
        },
        "info_to_extract": {
            "type": "string",
            "description": (
                "What specific information to extract from the page. "
                "If provided, returns a focused extraction instead of the full page content. "
                "If omitted, returns the full page content."
            ),
        },
        "custom_headers": {
            "type": "object",
            "description": "Additional HTTP headers for the request.",
            "additionalProperties": {"type": "string"},
        },
    },
    "required": ["url"],
}

SIMPLE_SCRAPE_AND_EXTRACT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "The URL to scrape (web page, PDF, code file, etc.).",
        },
        "info_to_extract": {
            "type": "string",
            "description": (
                "What specific information to extract from the page. "
                "If provided, returns a focused extraction instead of the full page content. "
                "If omitted, returns the full page content."
            ),
        },
    },
    "required": ["url"],
}


def create_scrape_and_extract_tool(  # noqa: D417, PLR0913, PLR0915
    *,
    name: str = "scrape_and_extract_info",
    description: str = _SCRAPE_AND_EXTRACT_DESCRIPTION,
    parameters: dict[str, Any] = DEFAULT_SCRAPE_AND_EXTRACT_PARAMETERS,
    server_name: str | None = None,
    jina_api_key: str | None = None,
    jina_base_url: str | None = None,
    max_content_length: int = DEFAULT_MAX_CONTENT_CHARS,
    max_calls_per_task: int | None = None,
    max_consecutive_calls_with_same_args: int | None = 2,
    summary_llm_base_url: str | None = None,
    summary_llm_model_name: str | None = None,
    summary_llm_api_key: str | None = None,
    summary_llm_request_logger: ModelRequestLogger | None = None,
    output_format: str = "text",
    raw_scrape_cache_enabled: bool = False,
    raw_scrape_cache_scope: str = "task",
    raw_scrape_cache_provider: str = _RAW_SCRAPE_CACHE_PROVIDER,
    raw_scrape_cache_normalize_url: bool = False,
    raw_scrape_cache: RawScrapeCache | None = None,
    scrape_backend_options: ScrapeBackendOptions | None = None,
    llm_extraction_options: LLMExtractionOptions | None = None,
    raw_scrape_cache_options: RawScrapeCacheOptions | None = None,
    output_options: ScrapeAndExtractOutputOptions | None = None,
) -> CallableTool:
    """Create a combined scrape-then-extract tool.

    Internally creates ``jina_scrape`` and ``llm_extract`` sub-tools and invokes them sequentially
    so all scraping / extraction logic is reused without duplication.

    Args:
        name: Tool name exposed to the model.
        description: Tool description.
        parameters: Tool parameters schema.
        server_name: Optional MCP server name for grouping.
        jina_api_key: Jina API key. Falls back to ``JINA_API_KEY`` env.
        jina_base_url: Jina base URL. Falls back to ``JINA_BASE_URL`` env.
        max_content_length: Truncate scraped content to this many characters.
        max_calls_per_task: Per-task call budget.
        max_consecutive_calls_with_same_args: Block consecutive identical requests.
        summary_llm_base_url: LLM API endpoint for extraction. Falls back to ``SUMMARY_LLM_BASE_URL`` env.
        summary_llm_model_name: Model name for the extraction LLM. Falls back to ``SUMMARY_LLM_MODEL_NAME`` env.
        summary_llm_api_key: API key for the extraction LLM. Falls back to ``SUMMARY_LLM_API_KEY`` env.
        output_format: Legacy output selector. ``"text"`` returns extracted text directly;
            ``"structured_json"`` returns a JSON object with success/url/extracted_info/error.
        raw_scrape_cache_enabled: Legacy opt-in cache flag for successful non-empty raw scrape results.
        raw_scrape_cache_scope: Legacy cache scope. Currently only ``"task"`` is supported.
        raw_scrape_cache_provider: Legacy cache provider provenance. Currently only ``"web_search"`` is supported.
        raw_scrape_cache_normalize_url: Legacy opt-in URL case/query normalization for cache keys.
        raw_scrape_cache: Legacy optional cache object, mainly for tests or advanced wiring.
        scrape_backend_options: Grouped raw scraping backend options. Takes precedence over legacy scrape args when provided.
        llm_extraction_options: Grouped summary LLM options. Takes precedence over legacy summary LLM args when provided.
        raw_scrape_cache_options: Grouped raw scrape cache options. Takes precedence over legacy raw cache args when provided.
        output_options: Grouped output formatting options. Takes precedence over ``output_format`` when provided.
    """
    scrape_options = scrape_backend_options or ScrapeBackendOptions(
        jina_api_key=jina_api_key,
        jina_base_url=jina_base_url,
        max_content_length=max_content_length,
    )
    llm_options = llm_extraction_options or LLMExtractionOptions(
        base_url=summary_llm_base_url,
        model_name=summary_llm_model_name,
        api_key=summary_llm_api_key,
        request_logger=summary_llm_request_logger,
    )
    cache_options = raw_scrape_cache_options or RawScrapeCacheOptions(
        enabled=raw_scrape_cache_enabled,
        scope=raw_scrape_cache_scope,
        provider=raw_scrape_cache_provider,
        normalize_url=raw_scrape_cache_normalize_url,
        cache=raw_scrape_cache,
    )
    output_opts = output_options or ScrapeAndExtractOutputOptions(format=output_format)
    _validate_raw_scrape_cache_options(cache_options)
    output_opts = ScrapeAndExtractOutputOptions(format=_normalize_output_format(output_opts.format))

    # Build the two sub-tools whose ._fn we will invoke directly.
    resolved_jina_base_url = scrape_options.jina_base_url or os.environ.get("JINA_BASE_URL", DEFAULT_JINA_BASE_URL)
    _jina_tool = create_jina_scrape_tool(
        jina_api_key=scrape_options.jina_api_key,
        jina_base_url=resolved_jina_base_url,
        max_content_length=scrape_options.max_content_length,
        timeout=scrape_options.timeout,
        retry=scrape_options.retry,
        fallback_retry=scrape_options.fallback_retry,
        content_type_routing=scrape_options.content_type_routing,
    )
    _llm_tool = create_llm_extract_tool(
        llm_base_url=llm_options.base_url,
        llm_model_name=llm_options.model_name,
        llm_api_key=llm_options.api_key,
        request_logger=llm_options.request_logger,
        cache_enabled=llm_options.cache_enabled,
        timeout=llm_options.timeout,
        retry=llm_options.retry,
        max_input_chars=llm_options.max_input_chars,
        chunk_overlap_chars=llm_options.chunk_overlap_chars,
        max_chunks=llm_options.max_chunks,
        chunk_max_concurrent=llm_options.chunk_max_concurrent,
        chunked_extraction=llm_options.chunked_extraction,
        chunk_strategy=llm_options.chunk_strategy,
        max_recursion_depth=llm_options.max_recursion_depth,
        global_anchor_enabled=llm_options.global_anchor_enabled,
        chunk_envelope_mode=llm_options.chunk_envelope_mode,
        csv_layer_b_enabled=llm_options.csv_layer_b_enabled,
        request_extra_body=llm_options.request_extra_body,
    )

    resolved_llm_base_url = llm_options.base_url or os.environ.get("SUMMARY_LLM_BASE_URL", "")
    llm_enabled = bool(resolved_llm_base_url and resolved_llm_base_url.strip())
    raw_cache = cache_options.cache or RawScrapeCache()

    async def _scrape_and_extract(  # noqa: PLR0915
        url: str,
        info_to_extract: str = "",
        custom_headers: dict[str, str] | None = None,
    ) -> ToolResult:
        t0 = time.perf_counter()
        timing: dict[str, float] = {}

        def _meta(*, backend: str | None = None) -> dict[str, Any]:
            timing["total"] = round((time.perf_counter() - t0) * 1000.0, 1)
            meta: dict[str, Any] = {"timing_ms": dict(timing)}
            if backend:
                meta["scrape_backend"] = backend
            return meta

        # --- Step 1: call jina_scrape sub-tool ---
        raw_cache_key_hash: str | None = None
        raw_cache_status = ""
        raw_cache_bypass_reason: str | None = None
        if cache_options.enabled:
            raw_cache_key_hash = _raw_scrape_cache_key_hash(
                url=url,
                custom_headers=custom_headers,
                jina_base_url=resolved_jina_base_url,
                max_content_length=scrape_options.max_content_length,
                normalize_url=cache_options.normalize_url,
            )
            lookup_t0 = time.perf_counter()
            scrape_result = raw_cache.get(raw_cache_key_hash)
            timing["raw_scrape_cache_lookup"] = round((time.perf_counter() - lookup_t0) * 1000.0, 1)
            if scrape_result is None:
                raw_cache_status = "miss"
                jina_t0 = time.perf_counter()
                scrape_result = await _jina_tool._fn(url=url, custom_headers=custom_headers)
                timing["jina_scrape"] = round((time.perf_counter() - jina_t0) * 1000.0, 1)
                if _is_cacheable_raw_scrape_result(scrape_result):
                    raw_cache.put(raw_cache_key_hash, scrape_result)
                elif scrape_result.status == ToolResultStatus.FAILED:
                    raw_cache_status = "bypass"
                    raw_cache_bypass_reason = "raw_scrape_failed"
                    raw_cache.record_bypass()
                elif scrape_result.metadata.get("success") is not True:
                    raw_cache_status = "bypass"
                    error_text = str(scrape_result.metadata.get("error", "")).lower()
                    raw_cache_bypass_reason = "empty_content" if "no content" in error_text else "raw_scrape_unsuccessful"
                    raw_cache.record_bypass()
                elif not scrape_result.content.strip():
                    raw_cache_status = "bypass"
                    raw_cache_bypass_reason = "empty_content"
                    raw_cache.record_bypass()
            else:
                raw_cache_status = "hit"
                timing["jina_scrape"] = 0.0
        else:
            jina_t0 = time.perf_counter()
            scrape_result = await _jina_tool._fn(url=url, custom_headers=custom_headers)
            timing["jina_scrape"] = round((time.perf_counter() - jina_t0) * 1000.0, 1)

        raw_cache_meta = _raw_scrape_cache_metadata(
            enabled=cache_options.enabled,
            status=raw_cache_status,
            raw_scrape_cache=raw_cache,
            scope=cache_options.scope,
            provider=cache_options.provider,
            normalize_url=cache_options.normalize_url,
            key_hash=raw_cache_key_hash,
            bypass_reason=raw_cache_bypass_reason,
        )

        # Propagate failures directly
        if scrape_result.status == ToolResultStatus.FAILED:
            scrape_result.metadata.update({**_meta(), **raw_cache_meta})
            return scrape_result

        raw_content = scrape_result.content
        backend = scrape_result.metadata.get("scrape_backend")
        scrape_stats = {
            "total_chars": scrape_result.metadata.get("total_chars", 0),
            "total_lines": scrape_result.metadata.get("total_lines", 0),
            "truncated": scrape_result.metadata.get("truncated", False),
        }

        # --- Step 2: optionally call llm_extract sub-tool ---
        if llm_enabled and info_to_extract:
            llm_t0 = time.perf_counter()
            extract_result: ToolResult = await _llm_tool._fn(
                content=raw_content,
                info_to_extract=info_to_extract,
            )
            timing["llm_extraction"] = round((time.perf_counter() - llm_t0) * 1000.0, 1)

            extract_meta = extract_result.metadata
            status = ToolResultStatus.SUCCESS if extract_meta.get("success") else ToolResultStatus.FAILED
            content = _format_scrape_output(
                output_options=output_opts,
                success=bool(extract_meta.get("success")),
                url=url,
                content=extract_result.content,
                error=extract_meta.get("error", ""),
            )
            return ToolResult(
                content=content,
                status=status,
                metadata={
                    **_meta(backend=backend),
                    **raw_cache_meta,
                    "success": extract_meta.get("success", False),
                    "url": url,
                    "error": extract_meta.get("error", ""),
                    "tokens_used": extract_meta.get("tokens_used", 0),
                    "extraction_strategy": extract_meta.get("extraction_strategy", "direct"),
                    "chunk_count": extract_meta.get("chunk_count", 0),
                    "relevant_chunk_count": extract_meta.get("relevant_chunk_count", 0),
                    "scrape_stats": scrape_stats,
                },
            )

        # No extraction requested — return raw content
        if not raw_content.strip():
            content = _format_scrape_output(
                output_options=output_opts,
                success=False,
                url=url,
                content="" if output_opts.format == _OUTPUT_FORMAT_STRUCTURED_JSON else "No content retrieved from the page.",
                error="No content retrieved from the page.",
            )
            return ToolResult(
                content=content,
                metadata={
                    **_meta(backend=backend),
                    **raw_cache_meta,
                    "success": False,
                    "url": url,
                    "error": "No content retrieved from the page.",
                },
            )
        raw_content = _format_scrape_output(
            output_options=output_opts,
            success=True,
            url=url,
            content=raw_content,
            error="",
        )
        return ToolResult(
            content=raw_content,
            metadata={
                **_meta(backend=backend),
                **raw_cache_meta,
                "success": True,
                "url": url,
                "scrape_stats": scrape_stats,
            },
        )

    return CallableTool(
        name=name,
        description=description,
        server_name=server_name,
        parameters=parameters,
        fn=_scrape_and_extract,
        max_calls_per_task=max_calls_per_task,
        max_consecutive_calls_with_same_args=max_consecutive_calls_with_same_args,
        strict_mode=False,
        emoji="🌐",
    )
