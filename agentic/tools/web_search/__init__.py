# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""General-purpose web search and scraping tools for agentic pipelines.

Provides:

* :func:`create_web_search_tool` — Google search via Serper API.
* :func:`create_jina_scrape_tool` — Raw content fetching via Jina Reader API.
* :func:`create_llm_extract_tool` — LLM-based extraction from raw text (with caching).
* :func:`create_scrape_and_extract_tool` — Convenience tool: Jina scrape → LLM extraction in a single call.
"""

from agentic.tools.web_search._retry import RetryConfig, TimeoutConfig
from agentic.tools.web_search.scrape import (
    DEFAULT_SCRAPE_AND_EXTRACT_PARAMETERS,
    SIMPLE_SCRAPE_AND_EXTRACT_PARAMETERS,
    LLMExtractionOptions,
    RawScrapeCacheOptions,
    ScrapeAndExtractOutputOptions,
    ScrapeBackendOptions,
    create_jina_scrape_tool,
    create_llm_extract_tool,
    create_scrape_and_extract_tool,
)
from agentic.tools.web_search.search import DEFAULT_WEB_SEARCH_PARAMETERS, SIMPLE_WEB_SEARCH_PARAMETERS, create_web_search_tool

__all__ = [
    "DEFAULT_SCRAPE_AND_EXTRACT_PARAMETERS",
    "DEFAULT_WEB_SEARCH_PARAMETERS",
    "SIMPLE_SCRAPE_AND_EXTRACT_PARAMETERS",
    "SIMPLE_WEB_SEARCH_PARAMETERS",
    "LLMExtractionOptions",
    "RawScrapeCacheOptions",
    "RetryConfig",
    "ScrapeAndExtractOutputOptions",
    "ScrapeBackendOptions",
    "TimeoutConfig",
    "create_jina_scrape_tool",
    "create_llm_extract_tool",
    "create_scrape_and_extract_tool",
    "create_web_search_tool",
]
