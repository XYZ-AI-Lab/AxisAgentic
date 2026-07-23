# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for the web-search tool family.

Provides common helpers used by both the search and scrape tools to avoid code duplication:

* **Banned-URL filtering** — blocks known benchmark-leaking domains.
* **Async HTTP client factory** — creates an ``httpx.AsyncClient`` with optional HTTP/2 support and graceful fallback.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------

DEFAULT_HTTP_LIMITS = httpx.Limits(
    max_connections=100,
    max_keepalive_connections=20,
    keepalive_expiry=30.0,
)

# ---------------------------------------------------------------------------
# Banned-URL filter
# ---------------------------------------------------------------------------

BANNED_URL_SUBSTRINGS: list[str] = [
    "huggingface.co/datasets",
    "huggingface.co/posts",
    "huggingface.co/spaces",
]


def is_banned_url(url: str) -> bool:
    """Return ``True`` if *url* matches any banned pattern."""
    if not url:
        return False
    return any(banned in url for banned in BANNED_URL_SUBSTRINGS)


# ---------------------------------------------------------------------------
# Async HTTP client factory
# ---------------------------------------------------------------------------


def create_async_client(
    *,
    follow_redirects: bool = True,
    limits: httpx.Limits = DEFAULT_HTTP_LIMITS,
    timeout: float | httpx.Timeout | None = None,
) -> httpx.AsyncClient:
    """Create an ``httpx.AsyncClient`` with optional HTTP/2 support.

    Falls back to HTTP/1.1 if the ``h2`` package is not installed.
    """
    kwargs: dict[str, Any] = {
        "follow_redirects": follow_redirects,
        "limits": limits,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    try:
        return httpx.AsyncClient(http2=True, **kwargs)
    except ImportError:
        logger.info("h2 not installed; falling back to HTTP/1.1")
        return httpx.AsyncClient(**kwargs)
