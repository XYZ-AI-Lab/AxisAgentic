# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Internal helpers for the scrape / extraction tool family.

Houses the low-level building blocks (Jina scraping, HTTP fallback, LLM extraction with optional caching)
so that ``scrape.py`` can stay focused on the three public tool-factory functions.

.. note::

   This module is **not** part of the public API — import from
   ``agentic.tools.web_search`` instead.
"""

from __future__ import annotations

import asyncio
import csv
import fcntl
import hashlib
import io
import json
import logging
import os
import pathlib
import re
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import httpx

from agentic.tools.web_search._retry import (
    RetryConfig,
    TimeoutConfig,
    compute_backoff,
    default_scrape_fallback_retry,
    default_scrape_retry,
    default_scrape_timeout,
    default_summary_llm_retry,
    default_summary_llm_timeout,
    parse_retry_after,
    should_retry_transport_error,
)
from agentic.tools.web_search._utils import create_async_client  # noqa: F401 (re-export convenience)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agentic.model_clients.request_logger import ModelRequestLogger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_JINA_BASE_URL = "https://r.jina.ai"
DEFAULT_MAX_CONTENT_CHARS = 102_400 * 4  # ~400 KB
DEFAULT_SUMMARY_LLM_MAX_INPUT_CHARS = 120_000
DEFAULT_SUMMARY_LLM_CHUNK_OVERLAP_CHARS = 1_200
# ``max_chunks`` is a *soft reduce fan-in* (how many per-chunk extracts to merge
# per reduce call), NOT a content cap: the chunked path always maps EVERY chunk
# and recursively reduces, so arbitrarily large pages (tens of MB) are fully
# covered with no silent tail loss.
DEFAULT_SUMMARY_LLM_MAX_CHUNKS = 12
DEFAULT_SUMMARY_LLM_CHUNK_MAX_CONCURRENT = 4
# Bound on how many times the reduce step may recurse (map-reduce the
# concatenated per-chunk findings) before falling back to a lossless concat.
# With a ~120k char input budget and a fan-in of 12 this still covers tens of MB
# of input (12^4 reduce leaves before the bound bites).
DEFAULT_SUMMARY_LLM_MAX_RECURSION_DEPTH = 4
# Which chunk map-reduce strategy to use once the master ``chunked_extraction``
# switch is on and the prompt exceeds ``max_input_chars``:
#   * "single"    — map every chunk, then ONE reduce call (overflow handled by the
#                   reduce call's own length-retry halving). The original behavior. (DEFAULT)
#   * "recursive" — map every chunk, then recursively reduce the concatenated
#                   findings until they fit the input budget (no silent tail loss).
DEFAULT_SUMMARY_LLM_CHUNK_STRATEGY = "single"
# Only "recursive" is matched in logic (is_recursive = chunk_strategy == _CHUNK_STRATEGY_RECURSIVE);
# any other value (i.e. "single") takes the non-recursive branch. Valid values are enforced by the
# config Literal["single", "recursive"], so an unknown string is rejected at config load, not here.
_CHUNK_STRATEGY_RECURSIVE = "recursive"
_CHUNK_NO_RELEVANT_INFORMATION = "NO_RELEVANT_INFORMATION"
_CSV_QUERY_STOPWORDS = frozenset(
    {
        "all",
        "and",
        "are",
        "check",
        "csv",
        "data",
        "dataset",
        "download",
        "extract",
        "file",
        "find",
        "for",
        "from",
        "greater",
        "less",
        "line",
        "lines",
        "list",
        "row",
        "rows",
        "table",
        "than",
        "the",
        "value",
        "values",
        "where",
        "with",
    }
)

# ---------------------------------------------------------------------------
# Global anchor (chunk-aware context prelude)
# ---------------------------------------------------------------------------
# When chunked extraction is in use, each map step only sees its own slice of
# the document. Without a global view the model misses table headers, units,
# section paths, and the overall scope (row/page counts, time-range). The
# "global anchor" is a one-shot, cheap pre-pass over the document head that
# produces a compact (<= ``DEFAULT_GLOBAL_ANCHOR_MAX_CHARS``) structured summary
# which is then prepended to every map *and* reduce prompt. The per-chunk
# envelope is global_anchor + chunk_local + info_to_extract + content.
DEFAULT_GLOBAL_ANCHOR_ENABLED = False
# How many chars from the START of the (possibly very large) document to feed
# to the anchor pass. The anchor is just a structural summary so it does not
# need to see every byte; the head usually contains the title, TOC, and any
# table headers. Tail / middle structural cues are recovered by the per-chunk
# map step.
DEFAULT_GLOBAL_ANCHOR_SAMPLE_CHARS = 30_000
# Upper bound on the anchor itself so the per-chunk prompt does not bloat.
DEFAULT_GLOBAL_ANCHOR_MAX_CHARS = 1_800
# Token budget for the anchor LLM call. The anchor is short and structured.
DEFAULT_GLOBAL_ANCHOR_MAX_TOKENS = 800
# Chunk envelope mode used when a non-empty global anchor is available.
#   "strict_caveat" (default): rules 1-4 identical to
#       "strict" but rule 5 replaced with a caveat-aware abstain — partial-
#       match data (multi-year aggregates covering the asked year, broader
#       regions containing the asked sub-region, related entities) flows
#       through with a one-line caveat instead of triggering hard NO_REL.
#       Partial matches are retained with a caveat instead of being rejected
#       when the framing differs slightly from the request.
#   "strict": original five-rule envelope including "never invent
#       values" and "return exactly NO_RELEVANT_INFORMATION" — strong
#       abstain guidance that can drop useful partial matches when document
#       framing differs from the request. Kept for backward compatibility.
#   "soft": anchor block prepended verbatim, then the legacy no-anchor chunk
#       prompt (one-liner). Removes the strict abstain rules so the LLM can
#       still surface near-relevant content with caveats; preserves the
#       structural disambiguation benefit of the anchor.
DEFAULT_CHUNK_ENVELOPE_MODE = "strict_caveat"
_GLOBAL_ANCHOR_PROMPT = """\
You are summarizing a web-scraped document so that downstream extractors with \
only a partial view (one chunk) still know the document's GLOBAL structure.

Return a CONCISE structured summary, plain text, exactly the lines below, in \
order, each on its own line. Total length MUST be at most {max_chars} \
characters. If a field has no signal in the content, write "unknown" \
(or "none" for lists). Do NOT add commentary, code fences, or extra fields.

TITLE: <document title or main heading>
URL: <source URL if visible in the content>
DOC_TYPE: <one of: tabular_csv, tabular_html, tabular_pdf_text, structured_json, prose_html, prose_markdown, mixed, unknown>
PRIMARY_SCHEMA: <if the document contains MULTIPLE DISTINCT tables (e.g. two or more "Table N" / "Exhibit N" captions, \
or several markdown table header/separator pairs with different column sets), output EXACTLY: \
"NONE - document contains multiple distinct tables; every table is a valid data source. Do not constrain extraction to a single schema."; \
if single tabular: comma-separated column headers (row 1) or first 3 column names plus row-count hint; \
if json: top-level keys; if prose: top 3 section headings; else: n/a>
UNITS_HINTS: <currency, multipliers, scales, fiscal-vs-calendar year, date format, time-zone>
TOC: <up to 8 section/page headings in document order, comma-separated>
QUESTION_KEYWORDS: <comma-separated 3-8 keywords/entities from the INFO TO EXTRACT below, taken verbatim where possible>
GLOBAL_SCOPE_HINTS: <one short sentence about full-document scope: time-range, row/page count, geo-coverage>

INFO TO EXTRACT (use ONLY to derive QUESTION_KEYWORDS; do NOT answer it here):
{info_to_extract}

CONTENT HEAD (may be only the start of a long document — do not assume completeness):
{content}

DOCUMENT_SUMMARY:
"""

_GLOBAL_ANCHOR_MARKER_BEGIN = "<DOCUMENT_GLOBAL_ANCHOR>"
_GLOBAL_ANCHOR_MARKER_END = "</DOCUMENT_GLOBAL_ANCHOR>"

# Machine-readable data MIME types. Jina (an HTML reader) renders data-API
# endpoints that return these as empty HTML query forms, so we fetch them
# directly and keep the raw payload. Decided purely from the server's
# Content-Type — provider-agnostic, no URL allowlist.
_STRUCTURED_DATA_CONTENT_TYPES = frozenset(
    {
        "application/json",
        "application/geo+json",
        "text/csv",
        "application/csv",
        "text/tab-separated-values",
        "application/xml",
        "text/xml",
    }
)

# Default data directory for persistent caches (project root / data).
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"
DEFAULT_LLM_CACHE_PATH = DEFAULT_DATA_DIR / "llm_extract_cache.json"
_CACHE_THREAD_LOCKS: dict[pathlib.Path, Any] = {}
_CACHE_THREAD_LOCKS_GUARD = threading.Lock()

# MAP / single-shot extraction prompt. Used in two places:
#   1. single-shot mode (chunked_extraction: false): the whole page in one call.
#   2. chunked modes: the "map" step run on each chunk.
# {} placeholders are filled (in order) with: the info-to-extract request, then the content.
EXTRACT_PROMPT = """\
You are given a piece of content and the requirement of information to extract. \
Your task is to extract the information specifically requested. \
Be precise and focus exclusively on the requested information.

INFORMATION TO EXTRACT:
{}

INSTRUCTIONS:
1. Extract the information relevant to the focus above.
2. If the exact information is not found, extract the most closely related details.
3. Be specific and include exact details when available.
4. Clearly organize the extracted information for easy understanding.
5. Do not include general summaries or unrelated content.

CONTENT TO ANALYZE:
{}

EXTRACTED INFORMATION:\
"""


def _thread_lock_for_path(path: pathlib.Path) -> Any:
    normalized_path = path.expanduser().resolve(strict=False)
    with _CACHE_THREAD_LOCKS_GUARD:
        lock = _CACHE_THREAD_LOCKS.get(normalized_path)
        if lock is None:
            lock = threading.RLock()
            _CACHE_THREAD_LOCKS[normalized_path] = lock
        return lock


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


def empty_result(error: str) -> dict[str, Any]:
    """Return a canonical *failed* scrape-result dict."""
    return {"success": False, "content": "", "error": error}


def content_stats(content: str, max_chars: int) -> dict[str, Any]:
    """Truncate *content* to *max_chars* and return statistics."""
    total_chars = len(content)
    total_lines = content.count("\n") + 1 if content else 0
    displayed = content[:max_chars]
    return {
        "success": True,
        "content": displayed,
        "error": "",
        "total_chars": total_chars,
        "total_lines": total_lines,
        "truncated": total_chars > max_chars,
    }


# ---------------------------------------------------------------------------
# HTTP retry
# ---------------------------------------------------------------------------


async def retry_get(
    url: str,
    headers: dict[str, str],
    label: str,
    *,
    client: httpx.AsyncClient,
    timeout: TimeoutConfig,  # noqa: ASYNC109
    retry: RetryConfig,
) -> httpx.Response:
    """HTTP GET with retry on transient errors, driven by ``RetryConfig``."""
    request_timeout = timeout.to_httpx()
    for attempt in range(1, retry.max_attempts + 1):
        try:
            response = await client.get(url, headers=headers, timeout=request_timeout)
            response.raise_for_status()
            return response
        except (
            httpx.ConnectTimeout,
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.RequestError,
        ) as exc:
            retryable = should_retry_transport_error(exc, retry)
            if retryable and attempt < retry.max_attempts:
                delay = compute_backoff(attempt, retry)
                logger.info(
                    "%s: %s, retry in %.1fs (attempt %d)",
                    label,
                    type(exc).__name__,
                    delay,
                    attempt + 1,
                )
                await asyncio.sleep(delay)
                continue
            raise
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if retry.is_retryable_status(status) and attempt < retry.max_attempts:
                delay = compute_backoff(attempt, retry, retry_after=parse_retry_after(e.response.headers))
                logger.info("%s: HTTP %d, retry in %.1fs", label, status, delay)
                await asyncio.sleep(delay)
                continue
            raise
    msg = f"{label}: all retries exhausted for {url}"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Jina scraping
# ---------------------------------------------------------------------------


async def scrape_with_jina(
    url: str,
    jina_api_key: str,
    jina_base_url: str,
    *,
    client: httpx.AsyncClient,
    custom_headers: dict[str, str] | None = None,
    max_chars: int = DEFAULT_MAX_CONTENT_CHARS,
    timeout: TimeoutConfig | None = None,  # noqa: ASYNC109
    retry: RetryConfig | None = None,
) -> dict[str, Any]:
    """Scrape *url* through the Jina Reader API with retry logic."""
    if not url or not url.strip():
        return empty_result("URL cannot be empty")
    if not jina_api_key:
        return empty_result("JINA_API_KEY not set")

    # Avoid duplicate Jina prefix
    if url.startswith("https://r.jina.ai/") and url.count("http") >= 2:
        url = url[len("https://r.jina.ai/") :]

    jina_url = f"{jina_base_url}/{url}"
    try:
        headers: dict[str, str] = {"Authorization": f"Bearer {jina_api_key}"}
        if custom_headers:
            headers.update(custom_headers)
        response = await retry_get(
            jina_url,
            headers,
            "Jina",
            client=client,
            timeout=timeout or default_scrape_timeout(),
            retry=retry or default_scrape_retry(),
        )
    except Exception as e:
        return empty_result(f"Jina scrape failed: {e}")

    content = response.text
    if not content:
        return empty_result("No content returned from Jina API")

    # Handle insufficient balance
    try:
        cdict = json.loads(content)
    except json.JSONDecodeError:
        cdict = None
    if isinstance(cdict, dict) and cdict.get("name") == "InsufficientBalanceError":
        return empty_result("Jina: insufficient balance")

    return content_stats(content, max_chars)


# ---------------------------------------------------------------------------
# Direct HTTP fallback
# ---------------------------------------------------------------------------


async def scrape_with_http(
    url: str,
    *,
    client: httpx.AsyncClient,
    custom_headers: dict[str, str] | None = None,
    max_chars: int = DEFAULT_MAX_CONTENT_CHARS,
    timeout: TimeoutConfig | None = None,  # noqa: ASYNC109
    retry: RetryConfig | None = None,
) -> dict[str, Any]:
    """Direct HTTP GET fallback when Jina is unavailable."""
    if not url or not url.strip():
        return empty_result("URL cannot be empty")

    try:
        headers: dict[str, str] = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        }
        if custom_headers:
            headers.update(custom_headers)
        response = await retry_get(
            url,
            headers,
            "HTTP",
            client=client,
            timeout=timeout or default_scrape_timeout(),
            retry=retry or default_scrape_fallback_retry(),
        )
    except Exception as e:
        return empty_result(f"HTTP scrape failed: {e}")

    content = response.text
    if not content:
        return empty_result("No content returned")

    return content_stats(content, max_chars)


def _is_structured_data_content_type(content_type: str) -> bool:
    """True for machine-readable data MIME types (JSON/CSV/XML family).

    Provider-agnostic: decided purely from the server's ``Content-Type`` —
    no URL allowlist. Also matches vendor/structured suffixes such as
    ``application/vnd.api+json``, ``application/sdmx-json`` and ``*+xml``.
    """
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    if not ct:
        return False
    if ct in _STRUCTURED_DATA_CONTENT_TYPES:
        return True
    return ct.endswith(("+json", "+xml", "-json"))


async def scrape_direct_typed(
    url: str,
    *,
    client: httpx.AsyncClient,
    custom_headers: dict[str, str] | None = None,
    max_chars: int = DEFAULT_MAX_CONTENT_CHARS,
    probe_timeout: float = 60.0,
) -> dict[str, Any]:
    """Direct GET for structured-data Content-Type responses only.

    Data-API endpoints (e.g. ArcGIS ``/query?f=json``, Socrata ``/resource/*.json``,
    Eurostat dissemination) are rendered by Jina as empty HTML query forms; a direct
    GET returns the real machine-readable payload. For HTML/PDF/other responses this
    returns ``is_structured=False`` WITHOUT downloading the (possibly large) body, so
    the caller falls back to Jina. Any error is non-fatal and also routes back to Jina.
    """
    if not url or not url.strip():
        return {"success": False, "is_structured": False, "error": "URL cannot be empty"}
    headers: dict[str, str] = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    }
    if custom_headers:
        headers.update(custom_headers)
    try:
        async with client.stream("GET", url, headers=headers, timeout=httpx.Timeout(probe_timeout, connect=20.0)) as response:
            content_type = response.headers.get("Content-Type", "")
            if response.status_code >= 400 or not _is_structured_data_content_type(content_type):
                # HTML / PDF / error — let Jina handle it; do not download the body.
                return {
                    "success": False,
                    "is_structured": False,
                    "content_type": content_type,
                    "status_code": response.status_code,
                }
            await response.aread()
            text = response.text
    except Exception as e:
        return {"success": False, "is_structured": False, "error": f"direct fetch failed: {e}"}
    if not text or not text.strip():
        return {"success": False, "is_structured": True, "content_type": content_type, "error": "empty body"}
    decoded = _maybe_decode_jsonstat(text, content_type)
    result = content_stats(decoded if decoded is not None else text, max_chars)
    result["is_structured"] = True
    result["content_type"] = content_type
    if decoded is not None:
        result["decoded"] = "jsonstat"
    return result


def _maybe_decode_jsonstat(text: str, content_type: str, *, max_rows: int = 5000) -> str | None:  # noqa: PLR0915
    """Decode a JSON-stat 2.0 dataset into human-readable records, or return None.

    Official statistics APIs (Eurostat, StatCan, OECD, SDMX-JSON) return JSON-stat:
    values live in a flat ``{flat_index: number}`` map where the index encodes every
    dimension (freq x indicator x unit x geo x time ...) via mixed-radix strides, and
    the index->code/label tables sit in a separate ``dimension`` block. An LLM cannot
    reliably decode that index — it emits confident wrong numbers. Here we expand it
    deterministically into JSONL records ``{dim_label: category, ..., value}``.

    Only triggers on ``class == "dataset"`` (the JSON-stat signature); other JSON
    (Socrata rows, ArcGIS features — already labelled inline) is left untouched.
    """
    if "json" not in (content_type or "").lower():
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("class") != "dataset":
        return None
    dims = payload.get("id")
    size = payload.get("size")
    dimension = payload.get("dimension")
    value = payload.get("value")
    if not (isinstance(dims, list) and isinstance(size, list) and isinstance(dimension, dict) and value is not None):
        return None
    try:
        sizemap = {nm: int(sz) for nm, sz in zip(dims, size, strict=False)}
        stride: dict[str, int] = {}
        acc = 1
        for nm in reversed(dims):
            stride[nm] = acc
            acc *= sizemap[nm]
        pos_to_code: dict[str, dict[int, str]] = {}
        code_to_label: dict[str, dict[str, str]] = {}
        for nm in dims:
            cat = (dimension.get(nm) or {}).get("category", {})
            idx = cat.get("index", {})
            if isinstance(idx, dict):
                pos_to_code[nm] = {int(p): c for c, p in idx.items()}
            elif isinstance(idx, list):
                pos_to_code[nm] = dict(enumerate(idx))
            else:
                pos_to_code[nm] = {}
            lab = cat.get("label", {})
            code_to_label[nm] = lab if isinstance(lab, dict) else {}
        items = list(value.items()) if isinstance(value, dict) else list(enumerate(value))
    except Exception:
        return None
    lines = [f"# JSON-stat dataset decoded (dimensions: {', '.join(dims)}): {payload.get('label', '')}".strip()]
    n = 0
    for k, v in items:
        if v is None:
            continue
        try:
            flat = int(k)
        except Exception as exc:
            logger.debug("Skipping non-integer JSON-stat key %r: %s", k, exc)
            continue
        rec: dict[str, Any] = {}
        for nm in dims:
            pos = (flat // stride[nm]) % sizemap[nm]
            code = pos_to_code[nm].get(pos, str(pos))
            rec[nm] = code_to_label[nm].get(code, code)
        rec["value"] = v
        lines.append(json.dumps(rec, ensure_ascii=False))
        n += 1
        if n >= max_rows:
            lines.append(f"# ...decoded {max_rows} records (truncated; refine the query to filter)")
            break
    if n == 0:
        return None
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM extraction cache
# ---------------------------------------------------------------------------


class LLMExtractionCache:
    """Disk-persistent cache for LLM extraction results.

    Keys are SHA-256 hashes of ``(content, info_to_extract, model, max_tokens)``
    plus an optional extraction-strategy namespace; baseline calls keep the
    original four-field key. Only **successful** results are cached.

    The cache is stored as a JSON file at *cache_path*.  It is loaded from disk
    on first access and flushed back after every :meth:`put`.  This means the
    cache survives process restarts and is shared across independent runs. The
    flush path uses file locks and ``fsync``; enabling this cache can add
    client-side latency in high-concurrency scrape/extract workloads.
    """

    def __init__(self, cache_path: str | pathlib.Path | None = None) -> None:
        self._path = pathlib.Path(cache_path or DEFAULT_LLM_CACHE_PATH)
        self._thread_lock = _thread_lock_for_path(self._path)
        self._store: dict[str, dict[str, Any]] | None = None  # lazy-loaded
        self._dirty_keys: set[str] = set()
        self.hits: int = 0
        self.misses: int = 0

    # -- lazy disk I/O --

    def _ensure_loaded(self) -> dict[str, dict[str, Any]]:
        """Load from disk on first access (lazy init)."""
        if self._store is not None:
            return self._store
        self._store = self._read_disk_store()
        if self._store:
            logger.info("LLM cache loaded: %d entries from %s", len(self._store), self._path)
        return self._store

    def _read_disk_store(self) -> dict[str, dict[str, Any]]:
        if not self._path.is_file():
            return {}
        try:
            with self._path.open() as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(key): value for key, value in data.items() if isinstance(value, dict)}
        except Exception:
            logger.warning("LLM cache file corrupt, starting fresh: %s", self._path, exc_info=True)
        return {}

    @contextmanager
    def _file_lock(self) -> Any:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._path.with_name(f"{self._path.name}.lock")
        self._thread_lock.acquire()
        try:
            with lock_path.open("a+") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            self._thread_lock.release()

    def _unique_tmp_path(self) -> pathlib.Path:
        return self._path.with_name(f".{self._path.name}.{os.getpid()}.{threading.get_ident()}.{uuid4().hex}.tmp")

    def _flush(self) -> None:
        """Write current store to disk."""
        if self._store is None:
            return
        tmp_path: pathlib.Path | None = None
        try:
            with self._file_lock():
                disk_store = self._read_disk_store()
                dirty_store = {key: self._store[key] for key in self._dirty_keys if key in self._store}
                merged_store = {**disk_store, **dirty_store}
                tmp_path = self._unique_tmp_path()
                with tmp_path.open("w") as f:
                    json.dump(merged_store, f, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                tmp_path.replace(self._path)
                self._store = merged_store
                self._dirty_keys.clear()
        except Exception:
            logger.warning("Failed to flush LLM cache to %s", self._path, exc_info=True)
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    logger.debug("Failed to remove temporary LLM cache file %s", tmp_path, exc_info=True)

    # -- key helpers --

    @staticmethod
    def _make_key(
        content: str,
        info_to_extract: str,
        model: str,
        max_tokens: int,
        cache_namespace: str = "",
    ) -> str:
        raw = f"{content}\x00{info_to_extract}\x00{model}\x00{max_tokens}"
        # Keep the legacy key byte-for-byte when no namespace is supplied so
        # baseline runs can continue to use their existing cache entries.
        if cache_namespace:
            raw = f"{raw}\x00{cache_namespace}"
        return hashlib.sha256(raw.encode()).hexdigest()

    # -- public API --

    def get(
        self,
        content: str,
        info_to_extract: str,
        model: str,
        max_tokens: int,
        cache_namespace: str = "",
    ) -> dict[str, Any] | None:
        with self._thread_lock:
            store = self._ensure_loaded()
            key = self._make_key(content, info_to_extract, model, max_tokens, cache_namespace)
            result = store.get(key)
            if result is not None:
                self.hits += 1
            else:
                self.misses += 1
            return result

    def put(
        self,
        content: str,
        info_to_extract: str,
        model: str,
        max_tokens: int,
        result: dict[str, Any],
        cache_namespace: str = "",
    ) -> None:
        with self._thread_lock:
            store = self._ensure_loaded()
            key = self._make_key(content, info_to_extract, model, max_tokens, cache_namespace)
            store[key] = result
            self._dirty_keys.add(key)
            self._flush()


# Module-level singleton cache shared by all tool instances in the process.
_llm_cache = LLMExtractionCache()


# ---------------------------------------------------------------------------
# LLM extraction (with optional cache)
# ---------------------------------------------------------------------------


async def _run_cache_io(fn: Any, *args: Any) -> Any:
    """Run cache disk I/O outside the event loop without using the default executor."""
    state: dict[str, Any] = {"done": False, "result": None, "exception": None}

    def worker() -> None:
        try:
            state["result"] = fn(*args)
        except BaseException as exc:  # pragma: no cover - defensive bridge back to the loop
            state["exception"] = exc
        else:
            state["exception"] = None
        finally:
            state["done"] = True

    # Daemon thread keeps cache I/O from blocking event-loop progress without
    # making process shutdown depend on Python's default executor cleanup.
    threading.Thread(target=worker, name="llm-extraction-cache-io", daemon=True).start()
    while not state["done"]:  # noqa: ASYNC110
        await asyncio.sleep(0.001)
    if state["exception"] is not None:
        raise state["exception"]
    return state["result"]


def _build_llm_payload(model: str, max_tokens: int, prompt: str, extra_body: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
    }
    if extra_body:
        payload.update(extra_body)
    return payload


def _llm_response_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        body: Any = response.json()
    except json.JSONDecodeError:
        body = response.text
    return {
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "body": body,
    }


def _log_summary_llm_request(
    request_logger: ModelRequestLogger | None,
    *,
    request_id: str | None,
    started_at: str | None,
    request_start: float,
    endpoint_url: str,
    request: dict[str, Any],
    attempt: int,
    response: Any | None = None,
    error: dict[str, Any] | None = None,
) -> None:
    if request_logger is None or request_id is None or started_at is None:
        return
    request_logger.log(
        request_id=request_id,
        started_at=started_at,
        elapsed_ms=(request_logger.perf_counter() - request_start) * 1000.0,
        request=request,
        response=response,
        error=error,
        metadata={
            "client": "summary_llm",
            "recipe": "web_search",
            "tool": "scrape_and_extract_info",
            "endpoint_url": endpoint_url,
            "attempt": attempt,
            "cache_status": "miss",
        },
    )


def _ensure_chat_completions_path(url: str) -> str:
    """Append ``/chat/completions`` when *url* is a bare OpenAI-style base (e.g. ``.../v1``).

    Accepts already-qualified endpoints (``.../v1/chat/completions`` or the trailing-slash form) unchanged so both env conventions work.
    """
    if not url:
        return url
    stripped = url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    return f"{stripped}/chat/completions"


def _is_context_length_error(response_text: str) -> bool:
    text = response_text.lower()
    return "context length" in text and ("exceeds" in text or "exceed" in text or "longer than" in text or "too long" in text)


def _extract_stop_reason(choice: dict[str, Any]) -> str | None:
    for key in ("stop_reason", "finish_reason"):
        value = choice.get(key)
        if value:
            return str(value)

    provider_field_owners: list[Any] = [choice]
    message = choice.get("message")
    if isinstance(message, dict):
        provider_field_owners.append(message)
    for owner in provider_field_owners:
        provider_fields = owner.get("provider_specific_fields")
        if isinstance(provider_fields, dict):
            for key in ("stop_reason", "finish_reason"):
                value = provider_fields.get(key)
                if value:
                    return str(value)
    return None


def _is_length_stop_reason(choice: dict[str, Any]) -> bool:
    for key in ("stop_reason", "finish_reason"):
        value = choice.get(key)
        if isinstance(value, str) and value.lower() == "length":
            return True

    provider_field_owners: list[Any] = [choice]
    message = choice.get("message")
    if isinstance(message, dict):
        provider_field_owners.append(message)
    for owner in provider_field_owners:
        provider_fields = owner.get("provider_specific_fields")
        if isinstance(provider_fields, dict):
            for key in ("stop_reason", "finish_reason"):
                value = provider_fields.get(key)
                if isinstance(value, str) and value.lower() == "length":
                    return True
    return False


def _halve_content_for_length_retry(content: str) -> str | None:
    next_length = len(content) // 2
    if next_length <= 0 or next_length >= len(content):
        return None
    return content[:next_length]


def _prompt_char_len(info_to_extract: str, content: str) -> int:
    return len(EXTRACT_PROMPT.format(info_to_extract, content))


def _should_use_chunked_extraction(
    content: str,
    info_to_extract: str,
    max_input_chars: int | None,
    *,
    chunked_extraction: bool = True,
) -> bool:
    if not chunked_extraction:
        return False
    if max_input_chars is None or max_input_chars <= 0:
        return False
    return _prompt_char_len(info_to_extract, content) > max_input_chars


def _detect_csv_structure(content: str) -> tuple[str, str, int] | None:
    r"""Detect a header-row CSV / TSV layout in *content*.

    Returns ``(separator, header_line, body_start_offset)`` when the content
    looks like a header-row delimited file, otherwise ``None``. The detection
    is conservative: a false positive sends the structure-aware path down the wrong path and
    a false negative just falls back to the legacy char-window splitter, so
    we err on the side of False.

    Heuristic for structure-aware CSV detection:
      * First non-empty line has at least ``MIN_SEPS`` separators of one type
        (``,`` preferred over ``\\t``); markdown ``|``-delimited tables are
        rejected so they keep flowing through char-window splitting.
      * At least ``MIN_BODY_LINES`` subsequent non-empty lines have the same
        separator count within ±1 (allowing optional trailing empty field).
      * Header line and body lines together must cover at least 80% of the
        first ``MAX_PROBE`` non-empty lines so the document is "mostly CSV"
        rather than CSV embedded in prose.
    """
    if not content:
        return None
    MIN_SEPS = 2
    MIN_BODY_LINES = 3
    MAX_PROBE = 20

    # Reject markdown tables outright.
    head = content[:8_000]
    if "|---" in head or "| ---" in head:
        return None

    # Walk lines and pick first non-empty as candidate header.
    lines = content.splitlines(keepends=True)
    cumulative = 0
    header_idx = -1
    header_line: str = ""
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped:
            header_idx = i
            header_line = stripped
            break
        cumulative += len(ln)
    if header_idx < 0:
        return None

    # Pick the separator with the higher count in the header.
    comma_count = header_line.count(",")
    tab_count = header_line.count("\t")
    if tab_count >= MIN_SEPS and tab_count >= comma_count:
        separator = "\t"
        sep_count = tab_count
    elif comma_count >= MIN_SEPS:
        separator = ","
        sep_count = comma_count
    else:
        return None

    # Probe the next non-empty lines.
    matching = 0
    probed = 0
    for ln in lines[header_idx + 1 :]:
        stripped = ln.strip()
        if not stripped:
            continue
        probed += 1
        body_sep = stripped.count(separator)
        if abs(body_sep - sep_count) <= 1:
            matching += 1
        if probed >= MAX_PROBE:
            break

    if matching < MIN_BODY_LINES:
        return None
    # "Mostly CSV" gate: at least 80% of probed body lines match the header
    # separator count. Prose-with-some-CSV-rows fails this gate and routes
    # back to the char-window splitter.
    if probed > 0 and matching / probed < 0.80:
        return None

    # body_start_offset = first character AFTER the header line's newline.
    header_full_len = len(lines[header_idx])
    body_start = cumulative + header_full_len
    return separator, header_line, body_start


def _normalize_csv_hint(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _csv_query_tokens(info_to_extract: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_./%-]*|\d{4}", info_to_extract.casefold())
        if len(token) >= 3 and token not in _CSV_QUERY_STOPWORDS
    }


def _csv_query_row_terms(info_to_extract: str) -> set[str]:
    terms = {match.group(1).strip() for match in re.finditer(r'"([^"]{2,80})"', info_to_extract)}
    terms.update(match.group(1).strip() for match in re.finditer(r"'([^']{2,80})'", info_to_extract))
    terms.update(re.findall(r"\b(?:18|19|20)\d{2}\b", info_to_extract))
    for pattern in (
        r"\b(?:containing|contains|contain)\s+([A-Za-z0-9_.%/-]{2,80})",
        r"\b(?:where|with)\s+[A-Za-z0-9_ .()/-]{0,40}=\s*([A-Za-z0-9_.%/-]{2,80})",
        r"\b(?:for|of)\s+([A-Z][A-Za-z0-9_.%/-]{2,80})",
    ):
        terms.update(match.group(1).strip() for match in re.finditer(pattern, info_to_extract))
    return {term for term in terms if term and term.casefold() not in _CSV_QUERY_STOPWORDS}


def _find_csv_column_index(headers: list[str], hint: str) -> int | None:
    norm_hint = _normalize_csv_hint(hint)
    if not norm_hint:
        return None
    normalized_headers = [_normalize_csv_hint(header) for header in headers]
    for i, norm_header in enumerate(normalized_headers):
        if norm_header == norm_hint:
            return i
    for i, norm_header in enumerate(normalized_headers):
        if norm_hint in norm_header or norm_header in norm_hint:
            return i
    return None


def _csv_query_conditions(info_to_extract: str, headers: list[str]) -> list[tuple[int, str, str]]:
    conditions: list[tuple[int, str, str]] = []
    patterns = [
        r"\b([A-Za-z_][A-Za-z0-9_ .()/%-]{0,50}?)\s*(=|>=|<=|>|<)\s*\"?([A-Za-z0-9_.%/-]+)\"?",
        r"\b([A-Za-z_][A-Za-z0-9_ .()/%-]{0,50}?)\s+is\s+(greater than|less than|at least|at most|equal to)\s+\"?([A-Za-z0-9_.%/-]+)\"?",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, info_to_extract, flags=re.IGNORECASE):
            raw_col = match.group(1)
            if "(" in raw_col and "column" in raw_col.casefold():
                paren_match = re.search(r"\bthe\s+([A-Za-z_][A-Za-z0-9_ ]*)\s+column\b", raw_col, flags=re.IGNORECASE)
                if paren_match:
                    raw_col = paren_match.group(1)
            op = match.group(2).casefold()
            value = match.group(3)
            col_idx = _find_csv_column_index(headers, raw_col)
            if col_idx is not None:
                if op == "greater than":
                    op = ">"
                elif op == "less than":
                    op = "<"
                elif op == "at least":
                    op = ">="
                elif op == "at most":
                    op = "<="
                elif op == "equal to":
                    op = "="
                conditions.append((col_idx, op, value))
    return conditions


def _csv_cell_matches_condition(cell: str, op: str, value: str) -> bool:
    if op == "=":
        return cell.strip().casefold() == value.strip().casefold()
    try:
        cell_num = float(cell.replace(",", "").strip().rstrip("%"))
        value_num = float(value.replace(",", "").strip().rstrip("%"))
    except ValueError:
        return False
    if op == ">":
        return cell_num > value_num
    if op == ">=":
        return cell_num >= value_num
    if op == "<":
        return cell_num < value_num
    if op == "<=":
        return cell_num <= value_num
    return False


def _csv_row_matches_query(row: list[str], row_terms: set[str], conditions: list[tuple[int, str, str]]) -> bool:
    if conditions and not all(idx < len(row) and _csv_cell_matches_condition(row[idx], op, value) for idx, op, value in conditions):
        return False
    if conditions:
        return True
    row_text = " ".join(row).casefold()
    return any(term.casefold() in row_text for term in row_terms)


def _select_csv_columns(
    headers: list[str],
    info_to_extract: str,
    conditions: list[tuple[int, str, str]],
    matched_rows: list[tuple[int, list[str]]],
    row_terms: set[str],
) -> list[int]:
    if len(headers) <= 12:
        return list(range(len(headers)))

    query_tokens = _csv_query_tokens(info_to_extract)
    selected: set[int] = {0}
    if len(headers) > 1:
        selected.add(1)
    selected.update(idx for idx, _, _ in conditions)

    for i, header in enumerate(headers):
        header_norm = header.casefold()
        header_compact = _normalize_csv_hint(header)
        if any(token in header_norm or _normalize_csv_hint(token) in header_compact for token in query_tokens):
            selected.add(i)
        if any(word in header_norm for word in ("name", "country", "state", "province", "territory", "year", "date")):
            selected.add(i)

    lowered_terms = [term.casefold() for term in row_terms if len(term) >= 3]
    for _, row in matched_rows[:200]:
        for i, cell in enumerate(row):
            cell_text = cell.casefold()
            if any(term in cell_text for term in lowered_terms):
                selected.update(j for j in (i - 1, i, i + 1) if 0 <= j < len(headers))

    return sorted(selected)[:16]


def _build_query_focused_csv_excerpt(
    content: str,
    *,
    separator: str,
    header: str,
    body_start: int,
    info_to_extract: str,
) -> str | None:
    """Query-focused CSV filtering shrinks CSV to matching rows and useful columns.

    The excerpt is intentionally conservative: if query hints produce no row
    match, or match nearly the whole table, callers fall back to the existing
    structure-aware splitting. This keeps broad "download/extract CSV" requests from
    losing coverage while letting targeted row-filter queries feed the LLM a
    much smaller table.
    """
    row_terms = _csv_query_row_terms(info_to_extract)
    body = content[body_start:]
    try:
        reader = csv.reader(io.StringIO(header + "\n" + body), delimiter=separator)
        parsed = list(reader)
    except csv.Error:
        return None
    if len(parsed) < 2:
        return None

    headers = [cell.strip() for cell in parsed[0]]
    rows = parsed[1:]
    if not headers or not rows:
        return None
    conditions = _csv_query_conditions(info_to_extract, headers)
    if not row_terms and not conditions:
        return None

    matched_rows: list[tuple[int, list[str]]] = []
    for row_idx, row in enumerate(rows, start=1):
        if _csv_row_matches_query(row, row_terms, conditions):
            matched_rows.append((row_idx, row))
    if not matched_rows:
        return None
    if len(matched_rows) / len(rows) > 0.80:
        return None

    selected_cols = _select_csv_columns(headers, info_to_extract, conditions, matched_rows, row_terms)
    output = io.StringIO()
    writer = csv.writer(output, delimiter=separator, lineterminator="\n")
    writer.writerow(["__row_index", *(headers[i] for i in selected_cols if i < len(headers))])
    for row_idx, row in matched_rows:
        writer.writerow([row_idx, *(row[i] if i < len(row) else "" for i in selected_cols)])
    excerpt = output.getvalue().rstrip("\n")
    if len(content) > 10_000 and len(excerpt) > len(content) * 0.85:
        return None
    return excerpt


def _split_csv_content_for_chunked_extraction(
    content: str,
    *,
    header: str,
    body_start: int,
    info_to_extract: str,
    max_input_chars: int,
    max_chunks: int,
) -> list[str]:
    r"""Structure-aware CSV splitter.

    Each emitted chunk = ``header`` + ``"\\n"`` + a contiguous slice of body
    rows. Rows are NEVER split mid-record so every chunk-internal LLM call
    sees fully aligned columns even when the row is the FIRST line of the
    chunk (the legacy splitter would have shown the chunk LLM e.g.
    ``"2020,Texas,3194"`` with no column meaning — the gold piece in
    chunks 2..N effectively becomes unparsable). The header is replicated
    in every chunk so the schema is always anchored in-chunk.

    This splitter does NOT use char-overlap: header replication serves the
    same role (each chunk independently aligned), and overlap on row data
    would just duplicate rows the reducer already merges.
    """
    if not content or body_start >= len(content):
        return []
    prompt_overhead = _prompt_char_len(info_to_extract, "")
    chunk_size = max(1, max_input_chars - prompt_overhead)
    # Reserve room for header + newline in every chunk.
    header_block = header + "\n"
    body_budget = max(1, chunk_size - len(header_block))

    body = content[body_start:]
    chunks: list[str] = []
    pos = 0
    body_len = len(body)
    while pos < body_len:
        end = min(body_len, pos + body_budget)
        # Snap to a row boundary so we never split mid-record.
        if end < body_len:
            line_boundary = body.rfind("\n", pos, end)
            if line_boundary > pos:
                end = line_boundary + 1  # include the trailing newline
        chunk_body = body[pos:end]
        # If a single row is wider than the budget, accept the row whole
        # rather than truncate it — preserving row integrity is the whole
        # point of preserving row integrity. The downstream length-retry path will halve
        # the chunk if it overflows the LLM.
        if not chunk_body.strip():
            pos = end if end > pos else pos + 1
            continue
        if "\n" not in chunk_body and end < body_len:
            # Couldn't find a newline within the budget; advance to the
            # next newline so the row stays whole.
            next_nl = body.find("\n", end)
            if next_nl == -1:
                next_nl = body_len
            chunk_body = body[pos : next_nl + 1]
            end = next_nl + 1
        chunks.append(header_block + chunk_body.rstrip("\n"))
        pos = end

    if max_chunks > 0 and len(chunks) > max_chunks:
        logger.info(
            "Structure-aware CSV splitter produced %d chunks above target %d; preserving per-chunk input budget",
            len(chunks),
            max_chunks,
        )
    return chunks


def _split_content_for_chunked_extraction(
    content: str,
    *,
    info_to_extract: str,
    max_input_chars: int,
    overlap_chars: int,
    max_chunks: int,
    csv_layer_b_enabled: bool = False,
) -> list[str]:
    if not content:
        return []
    # Structure-aware CSV splitting takes precedence when the
    # content looks like a header-row delimited file. Falls back silently
    # to char-window splitting for prose / mixed / non-CSV content.
    csv_struct = _detect_csv_structure(content) if csv_layer_b_enabled else None
    if csv_struct is not None:
        separator, header, body_start = csv_struct
        focused_content = _build_query_focused_csv_excerpt(
            content,
            separator=separator,
            header=header,
            body_start=body_start,
            info_to_extract=info_to_extract,
        )
        if focused_content:
            focused_header, _, _ = focused_content.partition("\n")
            focused_body_start = len(focused_header) + 1 if "\n" in focused_content else len(focused_content)
            focused_chunks = _split_csv_content_for_chunked_extraction(
                focused_content,
                header=focused_header,
                body_start=focused_body_start,
                info_to_extract=info_to_extract,
                max_input_chars=max_input_chars,
                max_chunks=max_chunks,
            )
            if focused_chunks:
                logger.info(
                    "Query-focused CSV excerpt active: separator=%r header_cols=%d chunks=%d",
                    separator,
                    focused_header.count(separator) + 1,
                    len(focused_chunks),
                )
                return focused_chunks
        csv_chunks = _split_csv_content_for_chunked_extraction(
            content,
            header=header,
            body_start=body_start,
            info_to_extract=info_to_extract,
            max_input_chars=max_input_chars,
            max_chunks=max_chunks,
        )
        if csv_chunks:
            logger.info(
                "Structure-aware CSV splitter active: separator=%r header_cols=%d chunks=%d",
                separator,
                header.count(separator) + 1,
                len(csv_chunks),
            )
            return csv_chunks
    prompt_overhead = _prompt_char_len(info_to_extract, "")
    chunk_size = max(1, max_input_chars - prompt_overhead)
    overlap = max(0, min(overlap_chars, chunk_size // 3))
    if max_chunks > 0:
        estimated_chunks = max(1, (len(content) + max(chunk_size - overlap, 1) - 1) // max(chunk_size - overlap, 1))
        if estimated_chunks > max_chunks:
            logger.info(
                "LLM extraction estimated %d chunks above target %d; preserving per-chunk input budget",
                estimated_chunks,
                max_chunks,
            )

    chunks: list[str] = []
    start = 0
    while start < len(content):
        end = min(len(content), start + chunk_size)
        if end < len(content):
            search_from = start + max((end - start) // 2, 1)
            paragraph_boundary = content.rfind("\n\n", search_from, end)
            line_boundary = content.rfind("\n", search_from, end)
            boundary = max(paragraph_boundary, line_boundary)
            if boundary > start:
                end = boundary
        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(content):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _chunk_info_request(
    info_to_extract: str,
    *,
    chunk_index: int,
    chunk_count: int,
    global_anchor: str = "",
    envelope_mode: str = DEFAULT_CHUNK_ENVELOPE_MODE,
) -> str:
    """MAP-step prompt for one chunk.

    When ``global_anchor`` is non-empty, the per-chunk map prompt is wrapped in
    a 3-layer envelope so the LLM sees the document-level context (title,
    schema, units, TOC, question keywords) before its slice. Empty anchor
    falls back to the original behavior — bytewise-identical to the
    pre-anchor prompt for backwards compatibility with existing prompts and
    regression tests.

    ``envelope_mode`` controls the chunk prompt scaffolding when an anchor is
    present (no effect when anchor is empty):
      * ``"strict"`` — original five-rule envelope (default; tested by
        existing test suite).
      * ``"soft"`` — anchor block prepended to the legacy no-anchor prompt
        (1-line instruction). Removes the strict abstain rules ("never invent"
        + "return exactly NO_RELEVANT_INFORMATION") that empirically caused
        hard-abstain regressions on PDFs with framing mismatches.
      * ``"strict_caveat"`` — keeps rules 1-4 of the strict envelope (anchor
        schema/units, partial-row extraction, locator hint, no invention) but
        replaces rule 5 with a caveat-aware abstain: extract partial-match
        data (multi-year aggregates covering the asked year, broader regions
        containing the asked sub-region, related entities) with a one-line
        caveat. NO_RELEVANT_INFORMATION is reserved for truly unrelated
        chunks.
    """
    if global_anchor and envelope_mode == "soft":
        # Anchor-only mode — anchor as advisory context; chunk prompt is the
        # legacy 1-line instruction. No "never invent" / strict NO_REL rules.
        return (
            f"{_GLOBAL_ANCHOR_MARKER_BEGIN}\n{global_anchor.strip()}\n{_GLOBAL_ANCHOR_MARKER_END}\n\n"
            f"{info_to_extract}\n\n"
            f"You are reading chunk {chunk_index} of {chunk_count} from one scraped page. "
            "Use the global anchor above to disambiguate column meaning, units, and scope. "
            "Extract only information in this chunk that helps answer the request. "
            f"If this chunk contains no relevant information, return exactly {_CHUNK_NO_RELEVANT_INFORMATION}."
        )
    if global_anchor and envelope_mode == "strict_caveat":
        # Rules 1-4 identical to strict; rule 5 replaced with caveat-aware
        # abstain so partial-match data (multi-year aggregates, broader
        # regions, related entities) still flows through with a caveat
        # instead of triggering hard NO_REL.
        return (
            f"{_GLOBAL_ANCHOR_MARKER_BEGIN}\n{global_anchor.strip()}\n{_GLOBAL_ANCHOR_MARKER_END}\n\n"
            f"<CHUNK_CONTEXT>\nThis is chunk {chunk_index} of {chunk_count} from one scraped page. "
            f"Use the global anchor above to disambiguate column meaning, units, and scope before "
            f"reading the chunk content.\n</CHUNK_CONTEXT>\n\n"
            f"<TASK>\n{info_to_extract}\n</TASK>\n\n"
            f"<INSTRUCTIONS>\n"
            f"1. Extract only information IN THIS CHUNK that helps answer the TASK.\n"
            f"2. Use the global anchor's PRIMARY_SCHEMA and UNITS_HINTS to interpret bare numbers, "
            f"column references, and abbreviations; never invent values not present in this chunk.\n"
            f"3. If you see partial table rows or list items at the chunk boundary, still extract them "
            f"verbatim — they will be merged with adjacent chunks in the reduce step.\n"
            f"4. When citing numbers/entities, include a short locator hint where visible "
            f'(e.g. "[row label X]" or "[section: ...]") so the reduce step can de-duplicate.\n'
            f"5. If this chunk contains data that PARTIALLY addresses the TASK (e.g., a multi-year "
            f"aggregate covering the asked year, a broader region containing the asked sub-region, a "
            f"related entity, or the most-specific value available), extract it AND add a one-line "
            f"caveat noting the framing mismatch (e.g., '[caveat: 2012-2019 aggregate includes 2014]' "
            f"or '[caveat: state-level data; document scope is national]'). Return exactly "
            f"{_CHUNK_NO_RELEVANT_INFORMATION} ONLY when NO part of this chunk relates to the TASK in any way.\n"
            f"</INSTRUCTIONS>"
        )
    if global_anchor:
        return (
            f"{_GLOBAL_ANCHOR_MARKER_BEGIN}\n{global_anchor.strip()}\n{_GLOBAL_ANCHOR_MARKER_END}\n\n"
            f"<CHUNK_CONTEXT>\nThis is chunk {chunk_index} of {chunk_count} from one scraped page. "
            f"Use the global anchor above to disambiguate column meaning, units, and scope before "
            f"reading the chunk content.\n</CHUNK_CONTEXT>\n\n"
            f"<TASK>\n{info_to_extract}\n</TASK>\n\n"
            f"<INSTRUCTIONS>\n"
            f"1. Extract only information IN THIS CHUNK that helps answer the TASK.\n"
            f"2. Use the global anchor's PRIMARY_SCHEMA and UNITS_HINTS to interpret bare numbers, "
            f"column references, and abbreviations; never invent values not present in this chunk.\n"
            f"3. If you see partial table rows or list items at the chunk boundary, still extract them "
            f"verbatim — they will be merged with adjacent chunks in the reduce step.\n"
            f"4. When citing numbers/entities, include a short locator hint where visible "
            f'(e.g. "[row label X]" or "[section: ...]") so the reduce step can de-duplicate.\n'
            f"5. If this chunk contains no information relevant to the TASK, return exactly "
            f"{_CHUNK_NO_RELEVANT_INFORMATION}.\n"
            f"</INSTRUCTIONS>"
        )
    return (
        f"{info_to_extract}\n\n"
        f"You are reading chunk {chunk_index} of {chunk_count} from one scraped page. "
        "Extract only information in this chunk that helps answer the request. "
        f"If this chunk contains no relevant information, return exactly {_CHUNK_NO_RELEVANT_INFORMATION}."
    )


def _chunk_finding_is_relevant(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    return normalized.upper() != _CHUNK_NO_RELEVANT_INFORMATION


def _reduce_chunk_info_request(
    info_to_extract: str,
    *,
    tighten: bool = True,
    global_anchor: str = "",
) -> str:
    """REDUCE-step prompt: merge the per-chunk map findings into one answer.

    Two variants, selected by ``tighten`` (driven by chunk_strategy):
      * tighten=False -> chunk_strategy "single": the original, looser
        reduce prompt — "synthesize a single precise answer".
      * tighten=True  -> chunk_strategy "recursive": precision-first;
        allows single OR multiple items, and forbids substituting the closest-matching item
        when no exact answer exists (must say "not found" instead).

    ``global_anchor`` is prepended (wrapped in the same marker block as the map
    prompt) when non-empty so the reduce LLM also sees the document-level
    schema/units context. Empty anchor falls back to the pre-anchor behavior —
    bytewise-identical to the original prompt for backwards compatibility.
    """
    anchor_block = f"{_GLOBAL_ANCHOR_MARKER_BEGIN}\n{global_anchor.strip()}\n{_GLOBAL_ANCHOR_MARKER_END}\n\n" if global_anchor else ""
    if not tighten:
        # "single" strategy: one reduce call with looser wording.
        return (
            f"{anchor_block}"
            f"{info_to_extract}\n\n"
            "The content contains extracted findings from chunks of the same scraped page. "
            "Synthesize a single precise answer for the original request. "
            "Use only the chunk findings, preserve exact details, and remove duplicate overlap."
        )
    # "recursive" strategy: prefer precision and never
    # substitute the closest-matching item as if it were the answer.
    return (
        f"{anchor_block}"
        f"{info_to_extract}\n\n"
        "The content contains extracted findings from chunks of the same scraped page. "
        "Extract the precise information that satisfies the request — this may be a single "
        "item or several items, depending on what the request asks for. "
        "Use only the chunk findings, do not add anything not present in them, "
        "preserve exact details, and remove duplicate overlap. "
        "If the findings do not contain a precise answer, state explicitly that it "
        "was not found — do not substitute the closest-matching item as if it were the answer."
    )


def _format_chunk_findings(chunk_findings: list[tuple[int, str]]) -> str:
    return "\n\n".join(f"[Chunk {chunk_index}]\n{finding.strip()}" for chunk_index, finding in chunk_findings)


async def _extract_global_anchor(
    content: str,
    info_to_extract: str,
    llm_base_url: str,
    llm_api_key: str | None,
    *,
    client: httpx.AsyncClient,
    model: str,
    request_logger: ModelRequestLogger | None,
    request_timeout_config: TimeoutConfig | None,
    retry: RetryConfig | None,  # noqa: ARG001 — anchor is a soft pass; no transient retry
    request_extra_body: dict[str, Any] | None = None,
    sample_chars: int = DEFAULT_GLOBAL_ANCHOR_SAMPLE_CHARS,
    max_chars: int = DEFAULT_GLOBAL_ANCHOR_MAX_CHARS,
    max_tokens: int = DEFAULT_GLOBAL_ANCHOR_MAX_TOKENS,
) -> tuple[str, int, str]:
    """One-shot cheap LLM call producing a compact document-level summary.

    Returns ``(anchor_text, tokens_used, error)``. On any failure the anchor
    text is "" and ``error`` is non-empty; callers must treat this as a soft
    degradation (chunked extraction falls back to the legacy no-anchor
    prompt). This intentionally does NOT raise — anchor extraction is an
    advisory pass and must never block the main map-reduce.

    The anchor pass is single-shot (no chunking on the anchor itself): only
    the first ``sample_chars`` of the document are shown. Long documents
    typically carry their title/TOC/schema in the head; the per-chunk map
    step still recovers tail/middle structural cues independently.

    This bypasses ``extract_with_llm`` and posts directly to the LLM, because
    ``EXTRACT_PROMPT`` would otherwise wrap the anchor prompt in a generic
    "extract relevant information" framing that fights the strict, fielded
    output the anchor needs.
    """
    if not content or not content.strip():
        return "", 0, "empty content"
    if not llm_base_url or not llm_base_url.strip():
        return "", 0, "LLM base URL not configured"
    sample = content[: max(1, sample_chars)]
    prompt = _GLOBAL_ANCHOR_PROMPT.format(
        max_chars=max_chars,
        info_to_extract=info_to_extract,
        content=sample,
    )

    endpoint_url = _ensure_chat_completions_path(llm_base_url)
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if llm_api_key:
        headers["Authorization"] = f"Bearer {llm_api_key}"
    resolved_timeout = request_timeout_config or default_summary_llm_timeout()
    payload = _build_llm_payload(model, max_tokens, prompt, request_extra_body)

    try:
        response = await client.post(
            endpoint_url,
            headers=headers,
            json=payload,
            timeout=resolved_timeout.to_httpx(),
        )
    except (httpx.HTTPError, TimeoutError) as exc:
        if request_logger is not None:
            logger.info("global anchor pass failed (network): %s", exc)
        return "", 0, f"anchor http error: {exc!s}"

    if response.status_code != 200:
        if request_logger is not None:
            logger.info("global anchor pass failed: status=%d", response.status_code)
        return "", 0, f"anchor http status {response.status_code}"

    try:
        body = response.json()
    except json.JSONDecodeError:
        return "", 0, "anchor response not valid JSON"
    choices = body.get("choices") if isinstance(body, dict) else None
    tokens_used = 0
    if isinstance(body, dict):
        usage = body.get("usage")
        if isinstance(usage, dict):
            tokens_used = int(usage.get("total_tokens", 0) or 0)
    if not isinstance(choices, list) or not choices:
        return "", tokens_used, "anchor response missing choices"
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    raw_text = message.get("content") if isinstance(message, dict) else None
    ok, normalized, err = _normalize_extracted_info(raw_text)
    if not ok:
        return "", tokens_used, err or "anchor response normalize failed"
    text = normalized.strip()
    # Sanity gate: a valid anchor must contain at least one of the expected
    # field markers ("TITLE:", "DOC_TYPE:", "PRIMARY_SCHEMA:", "TOC:",
    # "QUESTION_KEYWORDS:"). If the LLM hallucinated the chunk-level sentinel
    # or produced free prose without the requested structure, treat the pass
    # as soft-failed and fall back to legacy no-anchor chunk prompts. This
    # avoids polluting every chunk with garbage that the model would then
    # have to ignore.
    upper = text.upper()
    if upper == _CHUNK_NO_RELEVANT_INFORMATION:
        return "", tokens_used, "anchor pass returned chunk-sentinel"
    if not any(marker in upper for marker in ("TITLE:", "DOC_TYPE:", "PRIMARY_SCHEMA:", "TOC:", "QUESTION_KEYWORDS:")):
        return "", tokens_used, "anchor pass returned unstructured text"
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text, tokens_used, ""


def _normalize_extracted_info(value: Any) -> tuple[bool, str, str]:
    if value is None:
        return False, "", "LLM response content is None"
    if not isinstance(value, str):
        return False, "", f"LLM response content is not text: {type(value).__name__}"
    if not value.strip():
        return False, "", "LLM response content is empty"
    return True, value, ""


def _length_stop_retry_detail(summary_error: str) -> str:
    return summary_error or "response may be truncated"


def _length_stop_exhausted_error(summary_error: str) -> str:
    if summary_error:
        return f"{summary_error}; stop_reason=length after content halving"
    return "LLM response stopped with stop_reason=length after content halving"


async def _get_cached_llm_extraction(
    cache: LLMExtractionCache | None,
    content: str,
    info_to_extract: str,
    model: str,
    max_tokens: int,
    cache_namespace: str = "",
) -> dict[str, Any] | None:
    if cache is None:
        return None
    cached = await _run_cache_io(cache.get, content, info_to_extract, model, max_tokens, cache_namespace)
    if cached is not None:
        logger.debug("LLM extraction cache hit (hits=%d, misses=%d)", cache.hits, cache.misses)
    return cached


async def _extract_with_llm_chunked(  # noqa: PLR0913, PLR0915
    content: str,
    info_to_extract: str,
    llm_base_url: str,
    llm_api_key: str | None,
    *,
    client: httpx.AsyncClient,
    model: str,
    max_tokens: int,
    request_logger: ModelRequestLogger | None,
    request_timeout_config: TimeoutConfig | None,
    retry: RetryConfig | None,
    max_input_chars: int,
    chunk_overlap_chars: int,
    max_chunks: int,
    chunk_max_concurrent: int,
    chunk_strategy: str = DEFAULT_SUMMARY_LLM_CHUNK_STRATEGY,
    depth: int = 0,
    max_recursion_depth: int = DEFAULT_SUMMARY_LLM_MAX_RECURSION_DEPTH,
    base_chunk_count: int = 0,
    enable_global_anchor: bool = DEFAULT_GLOBAL_ANCHOR_ENABLED,
    anchor_sample_chars: int = DEFAULT_GLOBAL_ANCHOR_SAMPLE_CHARS,
    anchor_max_chars: int = DEFAULT_GLOBAL_ANCHOR_MAX_CHARS,
    anchor_max_tokens: int = DEFAULT_GLOBAL_ANCHOR_MAX_TOKENS,
    global_anchor: str = "",
    chunk_envelope_mode: str = DEFAULT_CHUNK_ENVELOPE_MODE,
    csv_layer_b_enabled: bool = False,
    request_extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map-reduce extraction over arbitrarily large *content*.

    Every chunk is mapped (no hard cap). The reduce step depends on
    ``chunk_strategy``:

    * ``"single"`` — one reduce call over the concatenated per-chunk findings,
      using the original (untightened) reduce prompt. If that concatenation
      itself overflows the input budget it is handled by the reduce call's own
      length-retry halving (the faithful pre-recursion behavior).
    * ``"recursive"`` — the reduce step recurses on the concatenated per-chunk
      findings (with the tightened, precision-first prompt) until they fit the
      input budget, so the tail of very large pages is never silently dropped.

    ``max_chunks`` is only a soft reduce fan-in, not a content cap.

    Global anchor:
    when ``enable_global_anchor`` is true and we are at the top-level
    (``depth == 0`` with no inherited ``global_anchor``), one cheap LLM call
    over the document head produces a compact structured summary (title,
    schema, units, TOC, question keywords). That anchor is prepended to every
    map AND reduce prompt so each chunk sees the document-level context, not
    just its slice. Anchor failure is soft-degraded to no-anchor — the
    map-reduce never blocks on it. Recursive reduce levels reuse the anchor
    from depth 0 so it is only paid for once per document.

    Returned metadata keeps the existing ``strategy`` / ``chunk_count`` /
    ``relevant_chunk_count`` keys and adds:

    * ``recursion_depth``: the maximum reduce-recursion depth actually reached.
    * ``total_chunks``: the cumulative number of chunks processed across every
      recursion level (the real work), seeded by ``base_chunk_count`` so the
      top-level map chunks are counted alongside the reduce-level chunks.
    * ``truncated`` (+ ``truncated_chars`` when known): only set True when output
      was genuinely bounded; we always prefer a lossless concat over dropping.
    * ``anchor_used`` (bool) and ``anchor_tokens_used`` (int): whether a
      non-empty global anchor was injected into the chunk prompts at this
      level (or inherited from a parent recursion frame), and how many tokens
      that anchor pass consumed (0 if reused from depth 0 or disabled).
    """
    chunks = _split_content_for_chunked_extraction(
        content,
        info_to_extract=info_to_extract,
        max_input_chars=max_input_chars,
        overlap_chars=chunk_overlap_chars,
        max_chunks=max_chunks,
        csv_layer_b_enabled=csv_layer_b_enabled,
    )
    if not chunks:
        return {"success": False, "extracted_info": "", "error": "Chunked LLM extraction found no content chunks"}

    # ``total_chunks`` accumulates this level's chunks plus everything already
    # processed at shallower reduce levels (passed via ``base_chunk_count``).
    cumulative_chunks = base_chunk_count + len(chunks)
    semaphore = asyncio.Semaphore(max(1, chunk_max_concurrent))

    # --- Global anchor pre-pass (top-level only) ---
    # Anchor is computed once at depth 0 and threaded through recursive reduce
    # calls. A non-empty inherited ``global_anchor`` short-circuits the pass.
    anchor_tokens_used = 0
    anchor_error = ""
    if global_anchor:
        # Inherited from a parent recursion level; nothing to compute.
        pass
    elif enable_global_anchor and depth == 0:
        anchor_text, anchor_tokens_used, anchor_error = await _extract_global_anchor(
            content,
            info_to_extract,
            llm_base_url,
            llm_api_key,
            client=client,
            model=model,
            request_logger=request_logger,
            request_timeout_config=request_timeout_config,
            retry=retry,
            request_extra_body=request_extra_body,
            sample_chars=anchor_sample_chars,
            max_chars=anchor_max_chars,
            max_tokens=anchor_max_tokens,
        )
        if anchor_text:
            global_anchor = anchor_text
            logger.info(
                "global anchor extracted: %d chars, %d tokens (sample=%d/%d)",
                len(anchor_text),
                anchor_tokens_used,
                min(len(content), anchor_sample_chars),
                len(content),
            )
        else:
            logger.info(
                "global anchor pass skipped (no anchor produced): %s — falling back to legacy chunk prompt",
                anchor_error or "empty",
            )

    async def _extract_chunk(index: int, chunk: str) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            chunk_result = await extract_with_llm(
                chunk,
                _chunk_info_request(
                    info_to_extract,
                    chunk_index=index,
                    chunk_count=len(chunks),
                    global_anchor=global_anchor,
                    envelope_mode=chunk_envelope_mode,
                ),
                llm_base_url,
                llm_api_key,
                client=client,
                model=model,
                max_tokens=max_tokens,
                cache=None,
                request_logger=request_logger,
                timeout=request_timeout_config,
                retry=retry,
                max_input_chars=None,
                request_extra_body=request_extra_body,
            )
        return index, chunk_result

    chunk_findings: list[tuple[int, str]] = []
    chunk_errors: list[str] = []
    tokens_used = anchor_tokens_used
    chunk_results = await asyncio.gather(*[_extract_chunk(index, chunk) for index, chunk in enumerate(chunks, start=1)])
    for index, chunk_result in sorted(chunk_results, key=lambda item: item[0]):
        tokens_used += int(chunk_result.get("tokens_used", 0) or 0)
        if chunk_result.get("success") and _chunk_finding_is_relevant(str(chunk_result.get("extracted_info", ""))):
            chunk_findings.append((index, str(chunk_result["extracted_info"])))
        elif not chunk_result.get("success"):
            chunk_errors.append(f"chunk {index}: {chunk_result.get('error', '')}")

    logger.info(
        "chunked extraction: %d chars -> %d chunks, %d relevant, %d failed (depth=%d, anchor=%s)",
        len(content),
        len(chunks),
        len(chunk_findings),
        len(chunk_errors),
        depth,
        "on" if global_anchor else "off",
    )

    if not chunk_findings:
        if len(chunk_errors) == len(chunks):
            return {
                "success": False,
                "extracted_info": "",
                "error": "Chunked LLM extraction failed for every chunk: " + "; ".join(chunk_errors[:3]),
                "tokens_used": tokens_used,
                "strategy": "chunked",
                "chunk_count": len(chunks),
                "total_chunks": cumulative_chunks,
                "recursion_depth": depth,
                "anchor_used": bool(global_anchor),
                "anchor_tokens_used": anchor_tokens_used,
            }
        return {
            "success": True,
            "extracted_info": "No relevant information found in the scraped content.",
            "error": "",
            "tokens_used": tokens_used,
            "strategy": "chunked",
            "chunk_count": len(chunks),
            "total_chunks": cumulative_chunks,
            "relevant_chunk_count": 0,
            "recursion_depth": depth,
            "anchor_used": bool(global_anchor),
            "anchor_tokens_used": anchor_tokens_used,
        }

    if len(chunk_findings) == 1:
        return {
            "success": True,
            "extracted_info": chunk_findings[0][1],
            "error": "",
            "tokens_used": tokens_used,
            "strategy": "chunked",
            "chunk_count": len(chunks),
            "total_chunks": cumulative_chunks,
            "relevant_chunk_count": 1,
            "recursion_depth": depth,
            "anchor_used": bool(global_anchor),
            "anchor_tokens_used": anchor_tokens_used,
        }

    # --- REDUCE: concatenate the kept per-chunk findings ---
    combined = _format_chunk_findings(chunk_findings)
    is_recursive = chunk_strategy == _CHUNK_STRATEGY_RECURSIVE
    reduce_info = _reduce_chunk_info_request(info_to_extract, tighten=is_recursive, global_anchor=global_anchor)

    # Recursive strategy only: if the concatenation itself still exceeds the input
    # budget, RECURSE — treat it as fresh content and map-reduce it again until it
    # fits. This removes the cap and avoids the silent ``_halve_content_for_length_retry``
    # drop that a direct (max_input_chars=None) reduce would trigger. We seed the
    # recursion's ``base_chunk_count`` with this level's chunks so ``total_chunks``
    # stays cumulative. The single strategy skips this and does one reduce call.
    if is_recursive and depth < max_recursion_depth and _should_use_chunked_extraction(combined, reduce_info, max_input_chars):
        sub = await _extract_with_llm_chunked(
            combined,
            reduce_info,
            llm_base_url,
            llm_api_key,
            client=client,
            model=model,
            max_tokens=max_tokens,
            request_logger=request_logger,
            request_timeout_config=request_timeout_config,
            retry=retry,
            max_input_chars=max_input_chars,
            chunk_overlap_chars=chunk_overlap_chars,
            max_chunks=max_chunks,
            chunk_max_concurrent=chunk_max_concurrent,
            chunk_strategy=chunk_strategy,
            depth=depth + 1,
            max_recursion_depth=max_recursion_depth,
            base_chunk_count=cumulative_chunks,
            enable_global_anchor=enable_global_anchor,
            anchor_sample_chars=anchor_sample_chars,
            anchor_max_chars=anchor_max_chars,
            anchor_max_tokens=anchor_max_tokens,
            global_anchor=global_anchor,
            chunk_envelope_mode=chunk_envelope_mode,
            csv_layer_b_enabled=csv_layer_b_enabled,
            request_extra_body=request_extra_body,
        )
        sub["tokens_used"] = int(sub.get("tokens_used", 0) or 0) + tokens_used
        sub["strategy"] = "chunked_map_reduce"
        sub["chunk_count"] = len(chunks)
        sub["relevant_chunk_count"] = len(chunk_findings)
        # ``sub`` already seeded its total_chunks from cumulative_chunks; report
        # the deepest level reached and keep the cumulative chunk count.
        sub["total_chunks"] = max(cumulative_chunks, int(sub.get("total_chunks", cumulative_chunks) or cumulative_chunks))
        sub["recursion_depth"] = max(depth, int(sub.get("recursion_depth", depth) or depth))
        # Preserve anchor metadata from the top-level frame: the recursive child
        # received ``global_anchor`` verbatim and did not re-run the pass, so
        # its anchor_tokens_used is 0; surface this level's tokens instead.
        sub["anchor_used"] = bool(global_anchor)
        sub["anchor_tokens_used"] = max(int(sub.get("anchor_tokens_used", 0) or 0), anchor_tokens_used)
        return sub

    # Recursive strategy at the recursion-depth bound with content still over the
    # budget: return the lossless concatenation rather than dropping the tail, and
    # flag ``truncated``. (The single strategy never reaches here — it falls
    # through to the one reduce call below regardless of overflow.)
    at_depth_limit_overflow = is_recursive and depth >= max_recursion_depth and _should_use_chunked_extraction(combined, reduce_info, max_input_chars)
    if at_depth_limit_overflow:
        logger.warning(
            "Chunked LLM extraction hit reduce-recursion depth %d with %d chars; "
            "returning lossless concatenation of %d chunk findings (no silent truncation)",
            max_recursion_depth,
            len(combined),
            len(chunk_findings),
        )
        return {
            "success": True,
            "extracted_info": combined,
            "error": "",
            "tokens_used": tokens_used,
            "strategy": "chunked_map_concat",
            "chunk_count": len(chunks),
            "total_chunks": cumulative_chunks,
            "relevant_chunk_count": len(chunk_findings),
            "recursion_depth": depth,
            "truncated": True,
            "truncated_chars": 0,
            "anchor_used": bool(global_anchor),
            "anchor_tokens_used": anchor_tokens_used,
        }

    reduce_result = await extract_with_llm(
        combined,
        reduce_info,
        llm_base_url,
        llm_api_key,
        client=client,
        model=model,
        max_tokens=max_tokens,
        cache=None,
        request_logger=request_logger,
        timeout=request_timeout_config,
        retry=retry,
        max_input_chars=None,
        request_extra_body=request_extra_body,
    )
    tokens_used += int(reduce_result.get("tokens_used", 0) or 0)
    if reduce_result.get("success"):
        result = dict(reduce_result)
        result["tokens_used"] = tokens_used
        result["strategy"] = "chunked_map_reduce"
        result["chunk_count"] = len(chunks)
        result["total_chunks"] = cumulative_chunks
        result["relevant_chunk_count"] = len(chunk_findings)
        result["recursion_depth"] = depth
        result["anchor_used"] = bool(global_anchor)
        result["anchor_tokens_used"] = anchor_tokens_used
        return result

    # Merge failed: fall back to the lossless raw concatenation so no data is lost.
    return {
        "success": True,
        "extracted_info": combined,
        "error": f"Chunk reduce failed; returning chunk findings: {reduce_result.get('error', '')}",
        "tokens_used": tokens_used,
        "strategy": "chunked_map_concat",
        "chunk_count": len(chunks),
        "total_chunks": cumulative_chunks,
        "relevant_chunk_count": len(chunk_findings),
        "recursion_depth": depth,
        "anchor_used": bool(global_anchor),
        "anchor_tokens_used": anchor_tokens_used,
    }


async def extract_with_llm(  # noqa: C901, PLR0912, PLR0913, PLR0915
    content: str,
    info_to_extract: str,
    llm_base_url: str,
    llm_api_key: str | None,
    *,
    client: httpx.AsyncClient,
    model: str = "LLM",
    max_tokens: int = 8192,
    cache: LLMExtractionCache | None = _llm_cache,
    request_logger: ModelRequestLogger | None = None,
    timeout: TimeoutConfig | None = None,  # noqa: ASYNC109
    retry: RetryConfig | None = None,
    max_input_chars: int | None = None,
    chunk_overlap_chars: int = DEFAULT_SUMMARY_LLM_CHUNK_OVERLAP_CHARS,
    max_chunks: int = DEFAULT_SUMMARY_LLM_MAX_CHUNKS,
    chunk_max_concurrent: int = DEFAULT_SUMMARY_LLM_CHUNK_MAX_CONCURRENT,
    chunked_extraction: bool = True,
    chunk_strategy: str = DEFAULT_SUMMARY_LLM_CHUNK_STRATEGY,
    max_recursion_depth: int = DEFAULT_SUMMARY_LLM_MAX_RECURSION_DEPTH,
    enable_global_anchor: bool = DEFAULT_GLOBAL_ANCHOR_ENABLED,
    anchor_sample_chars: int = DEFAULT_GLOBAL_ANCHOR_SAMPLE_CHARS,
    anchor_max_chars: int = DEFAULT_GLOBAL_ANCHOR_MAX_CHARS,
    anchor_max_tokens: int = DEFAULT_GLOBAL_ANCHOR_MAX_TOKENS,
    chunk_envelope_mode: str = DEFAULT_CHUNK_ENVELOPE_MODE,
    csv_layer_b_enabled: bool = False,
    request_extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send scraped content to an LLM to extract requested information.

    Results can be cached by ``(content, info_to_extract, model, max_tokens)``.
    Enabled experimental extraction features are additionally namespaced so
    they cannot reuse or overwrite baseline results. Only successful extractions
    are stored. Pass ``cache=None`` to disable cache
    reads and writes for this call. Cache I/O is kept off the event loop, but
    the feature is still default-off at the recipe/tool layer because persistent
    flushes may add client-side latency under high concurrency.

    ``timeout`` / ``retry`` control the per-request budget and transient-error
    backoff for the summary endpoint. They default to the previous hardcoded
    behavior, so a stable endpoint produces an identical result. Content-length
    and ``stop_reason=length`` halving are independent of ``retry`` and are
    always applied.

    ``enable_global_anchor`` (default False) toggles the chunk-aware document
    anchor pre-pass — see ``_extract_with_llm_chunked`` for the details. It is
    only consulted when chunked extraction is actually triggered; single-shot
    callers are unaffected.
    """
    # --- cache lookup ---
    # The baseline (both experimental features disabled) intentionally keeps
    # the legacy cache key. Any enabled experimental behavior receives a
    # namespace so it cannot poison a later baseline/comparison run.
    cache_namespace = ""
    if enable_global_anchor or csv_layer_b_enabled:
        cache_namespace = (
            f"global_anchor_enabled={enable_global_anchor}\x00chunk_envelope_mode={chunk_envelope_mode}\x00csv_layer_b_enabled={csv_layer_b_enabled}"
        )
    cached = await _get_cached_llm_extraction(cache, content, info_to_extract, model, max_tokens, cache_namespace)
    if cached is not None:
        return cached

    if not content or not content.strip():
        return {"success": False, "extracted_info": "", "error": "Content is empty"}

    if not llm_base_url or not llm_base_url.strip():
        return {"success": False, "extracted_info": "", "error": "LLM base URL not configured"}

    if _should_use_chunked_extraction(content, info_to_extract, max_input_chars, chunked_extraction=chunked_extraction):
        result = await _extract_with_llm_chunked(
            content,
            info_to_extract,
            llm_base_url,
            llm_api_key,
            client=client,
            model=model,
            max_tokens=max_tokens,
            request_logger=request_logger,
            request_timeout_config=timeout,
            retry=retry,
            max_input_chars=max_input_chars or DEFAULT_SUMMARY_LLM_MAX_INPUT_CHARS,
            chunk_overlap_chars=chunk_overlap_chars,
            max_chunks=max_chunks,
            chunk_max_concurrent=chunk_max_concurrent,
            chunk_strategy=chunk_strategy,
            max_recursion_depth=max_recursion_depth,
            enable_global_anchor=enable_global_anchor,
            anchor_sample_chars=anchor_sample_chars,
            anchor_max_chars=anchor_max_chars,
            anchor_max_tokens=anchor_max_tokens,
            chunk_envelope_mode=chunk_envelope_mode,
            csv_layer_b_enabled=csv_layer_b_enabled,
            request_extra_body=request_extra_body,
        )
        if result.get("success") and cache is not None:
            await _run_cache_io(cache.put, content, info_to_extract, model, max_tokens, result, cache_namespace)
        return result

    endpoint_url = _ensure_chat_completions_path(llm_base_url)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if llm_api_key:
        headers["Authorization"] = f"Bearer {llm_api_key}"

    resolved_timeout = timeout or default_summary_llm_timeout()
    resolved_retry = retry or default_summary_llm_retry()
    request_timeout = resolved_timeout.to_httpx()
    transient_attempt = 0
    model_attempt = 0
    current_content = content
    prompt = EXTRACT_PROMPT.format(info_to_extract, current_content)
    payload = _build_llm_payload(model, max_tokens, prompt, request_extra_body)

    while True:
        model_attempt += 1
        request_id = request_logger.next_id() if request_logger is not None else None
        request_started_at = request_logger.now() if request_logger is not None else None
        request_start = request_logger.perf_counter() if request_logger is not None else 0.0
        try:
            response = await client.post(
                endpoint_url,
                headers=headers,
                json=payload,
                timeout=request_timeout,
            )
            # Handle context length exceeded by truncating
            if _is_context_length_error(response.text):
                _log_summary_llm_request(
                    request_logger,
                    request_id=request_id,
                    started_at=request_started_at,
                    request_start=request_start,
                    endpoint_url=endpoint_url,
                    request=payload,
                    response=_llm_response_payload(response),
                    attempt=model_attempt,
                )
                truncated = _halve_content_for_length_retry(current_content)
                if truncated is None:
                    return {
                        "success": False,
                        "extracted_info": "",
                        "error": "LLM context length exceeded after content halving",
                    }
                logger.info(
                    "LLM extraction context length exceeded; retrying with half content (%d -> %d chars)",
                    len(current_content),
                    len(truncated),
                )
                current_content = truncated
                prompt = EXTRACT_PROMPT.format(info_to_extract, current_content)
                payload["messages"][0]["content"] = prompt
                continue

            response.raise_for_status()
            _log_summary_llm_request(
                request_logger,
                request_id=request_id,
                started_at=request_started_at,
                request_start=request_start,
                endpoint_url=endpoint_url,
                request=payload,
                response=_llm_response_payload(response),
                attempt=model_attempt,
            )

            try:
                resp_data = response.json()
            except json.JSONDecodeError:
                return {"success": False, "extracted_info": "", "error": "Failed to parse LLM response"}

            if resp_data.get("choices"):
                choice = resp_data["choices"][0]
                try:
                    summary = choice["message"]["content"]
                except Exception as e:
                    return {"success": False, "extracted_info": "", "error": f"LLM response format error: {e}"}
                tokens_used = resp_data.get("usage", {}).get("total_tokens", 0)
                is_valid_summary, extracted_info, summary_error = _normalize_extracted_info(summary)
                if _is_length_stop_reason(choice):
                    truncated = _halve_content_for_length_retry(current_content)
                    if truncated is not None:
                        logger.warning(
                            "LLM extraction stopped with stop_reason=%s; retrying with half content (%d -> %d chars): %s",
                            _extract_stop_reason(choice),
                            len(current_content),
                            len(truncated),
                            _length_stop_retry_detail(summary_error),
                        )
                        current_content = truncated
                        prompt = EXTRACT_PROMPT.format(info_to_extract, current_content)
                        payload["messages"][0]["content"] = prompt
                        continue
                    summary_error = _length_stop_exhausted_error(summary_error)
                    is_valid_summary = False
                if not is_valid_summary:
                    logger.warning("LLM extraction returned invalid content: %s", summary_error)
                    return {
                        "success": False,
                        "extracted_info": "",
                        "error": summary_error,
                        "tokens_used": tokens_used,
                    }
                result = {"success": True, "extracted_info": extracted_info, "error": "", "tokens_used": tokens_used}
                if cache is not None:
                    await _run_cache_io(cache.put, content, info_to_extract, model, max_tokens, result, cache_namespace)
                return result

            if "error" in resp_data:
                return {"success": False, "extracted_info": "", "error": f"LLM error: {resp_data['error']}"}

            return {"success": False, "extracted_info": "", "error": f"Unexpected LLM response: {resp_data}"}

        except (httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout) as exc:
            _log_summary_llm_request(
                request_logger,
                request_id=request_id,
                started_at=request_started_at,
                request_start=request_start,
                endpoint_url=endpoint_url,
                request=payload,
                error={"type": type(exc).__name__, "message": str(exc)},
                attempt=model_attempt,
            )
            if should_retry_transport_error(exc, resolved_retry) and transient_attempt < resolved_retry.max_attempts - 1:
                delay = compute_backoff(transient_attempt + 1, resolved_retry)
                transient_attempt += 1
                logger.info("LLM extraction: %s, retry in %.1fs", type(exc).__name__, delay)
                await asyncio.sleep(delay)
                continue
            return {"success": False, "extracted_info": "", "error": f"LLM call failed: {exc}"}

        except httpx.HTTPStatusError as e:
            _log_summary_llm_request(
                request_logger,
                request_id=request_id,
                started_at=request_started_at,
                request_start=request_start,
                endpoint_url=endpoint_url,
                request=payload,
                response=_llm_response_payload(e.response),
                error={"type": type(e).__name__, "message": str(e)},
                attempt=model_attempt,
            )
            status = e.response.status_code
            if resolved_retry.is_retryable_status(status) and transient_attempt < resolved_retry.max_attempts - 1:
                delay = compute_backoff(transient_attempt + 1, resolved_retry, retry_after=parse_retry_after(e.response.headers))
                transient_attempt += 1
                await asyncio.sleep(delay)
                continue
            return {"success": False, "extracted_info": "", "error": f"LLM HTTP {status}: {e}"}

    return {"success": False, "extracted_info": "", "error": "LLM extraction: all retries exhausted"}
