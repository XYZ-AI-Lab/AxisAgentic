# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for the benchmark dashboard."""

from __future__ import annotations

from datetime import timedelta, timezone

_DEFAULT_WEB_SEARCH_LOG_DIR = "logs/web_search_infer"
_DEFAULT_WIDE_SEARCH_LOG_DIR = "logs/wide_search_infer"

ORIGINAL_COLOR = "#0068c9"
AGENTIC_COLOR = "#83c9ff"
DELTA_COLOR = "#d62728"

LLM_JUDGE_CORRECT_THRESHOLD = 0.5

MODEL_CLIENT_COLOR = "#5b7ab4"
USAGE_CHART_COLOR = "#ff7f0e"

SEARCH_COLOR = "#6dae5a"
SEARCH_OVERHEAD_COLOR = "#10c13a"
EXTRACT_COLOR = "#e6994a"
EXTRACT_JINA_COLOR = "#e6994a"
EXTRACT_LLM_COLOR = "#e074ec"
EXTRACT_FALLBACK_COLOR = "#eeaff0"
OTHER_TOOL_COLOR = "#c45b8e"

AGENTIC_OVERHEAD_COLOR = "#afc52e"

UTC8 = timezone(timedelta(hours=8))
