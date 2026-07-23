# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Timing data extraction and aggregation helpers for the dashboard."""

from __future__ import annotations

import contextlib
import re
from datetime import datetime
from typing import Any

from recipe.dashboard.constants import (
    AGENTIC_OVERHEAD_COLOR,
    EXTRACT_FALLBACK_COLOR,
    EXTRACT_JINA_COLOR,
    EXTRACT_LLM_COLOR,
    MODEL_CLIENT_COLOR,
    OTHER_TOOL_COLOR,
    SEARCH_COLOR,
    SEARCH_OVERHEAD_COLOR,
    UTC8,
)
from recipe.dashboard.discovery import _sorted_attempts
from recipe.dashboard.extraction import _ori_tool_calls_paired
from recipe.dashboard.loading import _extract_step_timing, _load_json_cached

# Consistent palette for non-tool categories.
_CAT_COLORS_BASE: dict[str, str] = {
    "Model Client": MODEL_CLIENT_COLOR,
    "Conversation Runtime": AGENTIC_OVERHEAD_COLOR,
    "Orchestrator": AGENTIC_OVERHEAD_COLOR,
    "Other": AGENTIC_OVERHEAD_COLOR,
}
# Softer, distinguishable palette for the 3 tool buckets.
_TOOL_COLORS: dict[str, str] = {
    "Tool: serper": SEARCH_COLOR,
    "Tool: google_search": SEARCH_COLOR,
    "Tool: web_search": SEARCH_COLOR,
    "Tool: google_search overhead": SEARCH_OVERHEAD_COLOR,
    "Tool: jina": EXTRACT_JINA_COLOR,
    "Tool: scrape_and_extract_info": EXTRACT_JINA_COLOR,
    "Tool: scrape_and_extract": EXTRACT_JINA_COLOR,
    "Tool: LLM extraction": EXTRACT_LLM_COLOR,
    "Tool: scrape_and_extract_info overhead": EXTRACT_FALLBACK_COLOR,
}


def _build_tool_color_map(tool_names: list[str]) -> dict[str, str]:
    """Assign colours to tool bucket labels."""
    return {name: _TOOL_COLORS.get(name, OTHER_TOOL_COLOR) for name in tool_names}


_KNOWN_TOOLS = {"google_search", "web_search", "scrape_and_extract_info", "scrape_and_extract"}
# Fixed display order for tool buckets.
_TOOL_ORDER = ["Tool: google_search", "Tool: scrape_and_extract_info", "Tool: Other Tools"]
_AGN_TOOL_PIE_ORDER = [
    "Tool: serper",
    "Tool: google_search overhead",
    "Tool: jina",
    "Tool: LLM extraction",
    "Tool: scrape_and_extract_info overhead",
    "Tool: Other Tools",
]


def _bucket_tool_name(raw_name: str) -> str:
    """Map a raw tool name to one of the 3 display buckets."""
    if raw_name == "web_search":
        return "Tool: google_search"
    if raw_name == "scrape_and_extract":
        return "Tool: scrape_and_extract_info"
    if raw_name in _KNOWN_TOOLS:
        return f"Tool: {raw_name}"
    return "Tool: Other Tools"


def _safe_ms(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(parsed, 0.0)


def _expand_agentic_tool_timing(tool_call: dict[str, Any]) -> dict[str, float]:
    """Expand one tool trace into timing buckets used by the dashboard."""
    tool_name = tool_call.get("tool_name", "unknown")
    latency_ms = _safe_ms(tool_call.get("latency_ms", 0))
    timing_ms = tool_call.get("metadata", {}).get("timing_ms", {})
    if tool_name in {"google_search", "web_search"}:
        if not isinstance(timing_ms, dict):
            return {_bucket_tool_name(tool_name): latency_ms}

        serper_ms = _safe_ms(timing_ms.get("serper_request") or timing_ms.get("request_ms"))
        residual_ms = max(latency_ms - serper_ms, 0.0)
        expanded: dict[str, float] = {}
        if serper_ms > 0:
            expanded["Tool: serper"] = serper_ms
        if residual_ms > 0:
            expanded["Tool: google_search overhead"] = residual_ms
        return expanded or {_bucket_tool_name(tool_name): latency_ms}

    if tool_name not in {"scrape_and_extract_info", "scrape_and_extract"}:
        return {_bucket_tool_name(tool_name): latency_ms}

    if not isinstance(timing_ms, dict):
        return {_bucket_tool_name(tool_name): latency_ms}

    expanded: dict[str, float] = {}
    jina_ms = _safe_ms(timing_ms.get("jina") or timing_ms.get("jina_scrape"))
    llm_ms = _safe_ms(timing_ms.get("llm_extraction"))
    residual_ms = max(latency_ms - jina_ms - llm_ms, 0.0)
    accounted = 0.0
    if jina_ms > 0:
        expanded["Tool: jina"] = jina_ms
        accounted += jina_ms
    if llm_ms > 0:
        expanded["Tool: LLM extraction"] = llm_ms
        accounted += llm_ms
    residual_ms = max(latency_ms - accounted, 0.0)
    if residual_ms > 0:
        expanded["Tool: scrape_and_extract_info overhead"] = residual_ms

    return expanded or {_bucket_tool_name(tool_name): latency_ms}


def _tool_timing_detail_rows(tool_call: dict[str, Any]) -> list[tuple[str, float]]:
    """Return per-tool timing rows for the tool-call detail panel."""
    tool_name = tool_call.get("tool_name", "unknown")
    timing_ms = tool_call.get("metadata", {}).get("timing_ms", {})
    if not isinstance(timing_ms, dict):
        return []

    label_map: dict[str, dict[str, str]] = {
        "google_search": {
            "serper_request": "Serper",
            "response_processing": "Response processing",
            "url_decode": "URL decode",
        },
        "web_search": {
            "request_ms": "Serper",
        },
        "scrape_and_extract_info": {
            "jina": "Jina",
            "llm_extraction": "LLM extraction",
            "python_fallback": "Python fallback",
        },
        "scrape_and_extract": {
            "jina_scrape": "Jina",
            "llm_extraction": "LLM extraction",
        },
    }
    ordered_keys = list(label_map.get(tool_name, {}).keys())
    rows: list[tuple[str, float]] = []
    for key in ordered_keys:
        ms = _safe_ms(timing_ms.get(key))
        if ms > 0:
            rows.append((label_map[tool_name][key], ms))
    return rows


def _bucket_per_tool(raw: dict[str, float]) -> dict[str, float]:
    """Collapse per-tool timing into the 3 buckets."""
    out: dict[str, float] = {}
    for name, ms in raw.items():
        bucket = _bucket_tool_name(name)
        out[bucket] = out.get(bucket, 0) + ms
    return out


def _bucket_per_tool_counts(raw: dict[str, int]) -> dict[str, int]:
    """Collapse per-tool counts into the same buckets as timing."""
    out: dict[str, int] = {}
    for name, count in raw.items():
        bucket = _bucket_tool_name(name)
        out[bucket] = out.get(bucket, 0) + count
    return out


def _parse_datetime_ms(dt_str: str | None) -> float | None:
    """Parse a datetime string and return epoch milliseconds, or *None*."""
    if not dt_str or dt_str == "?":
        return None

    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        with contextlib.suppress(ValueError):
            return datetime.strptime(dt_str, fmt).replace(tzinfo=UTC8).timestamp() * 1000
    with contextlib.suppress(ValueError, TypeError):
        parsed = datetime.fromisoformat(dt_str)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC8)
        return parsed.timestamp() * 1000
    return None


def _ori_infer_model_timing(data: dict[str, Any]) -> tuple[float, int]:
    """Infer original-pipeline model time from Message Retention -> Token Usage pairs.

    Upstream original traces do not store millisecond model-client durations.
    The best available proxy is the wall-clock gap between the per-call
    ``Message Retention`` step and the following ``Token Usage`` step. Those
    timestamps are second-resolution, so this is intentionally marked as an
    estimate in UI labels.
    """
    model_ms = 0.0
    model_call_count = 0
    pending_start_ms: float | None = None
    for step in data.get("step_logs", []) or []:
        step_name = step.get("step_name", "")
        if "Message Retention" in step_name:
            pending_start_ms = _parse_datetime_ms(step.get("timestamp"))
        elif "Token Usage" in step_name:
            end_ms = _parse_datetime_ms(step.get("timestamp"))
            if pending_start_ms is not None and end_ms is not None:
                model_ms += max(end_ms - pending_start_ms, 0.0)
                model_call_count += 1
            pending_start_ms = None
    return model_ms, model_call_count


def _ori_time_breakdown(data_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute time breakdown for original pipeline from tool-call latencies and wall-clock time.

    *data_list* should contain all attempt data dicts for a single task.
    Tool names are bucketed into google_search, scrape_and_extract_info, Other Tools.
    """
    raw_per_tool: dict[str, float] = {}
    raw_tool_counts: dict[str, int] = {}
    total_task_ms = 0.0
    inferred_model_ms = 0.0
    model_call_count = 0
    for data in data_list:
        start_ms = _parse_datetime_ms(data.get("start_time"))
        end_ms = _parse_datetime_ms(data.get("end_time"))
        if start_ms is None or end_ms is None:
            continue
        total_task_ms += end_ms - start_ms
        attempt_model_ms, attempt_model_calls = _ori_infer_model_timing(data)
        inferred_model_ms += attempt_model_ms
        model_call_count += attempt_model_calls
        for tc in _ori_tool_calls_paired(data):
            raw_per_tool[tc["tool_name"]] = raw_per_tool.get(tc["tool_name"], 0) + tc["latency_ms"]
            raw_tool_counts[tc["tool_name"]] = raw_tool_counts.get(tc["tool_name"], 0) + 1
    per_tool = _bucket_per_tool(raw_per_tool)
    tool_counts = _bucket_per_tool_counts(raw_tool_counts)
    tool_total = sum(per_tool.values())
    model_and_overhead = max(total_task_ms - tool_total, 0) if total_task_ms > 0 else 0
    # Clamp display buckets to the observed residual after tool time; the model
    # estimate is coarse and can exceed the residual on short/noisy attempts.
    model_client = min(inferred_model_ms, model_and_overhead)
    overhead = max(model_and_overhead - model_client, 0)
    return {
        "per_tool": per_tool,
        "tool_counts": tool_counts,
        "model_client_ms": model_client,
        "model_client_ms_estimated_raw": inferred_model_ms,
        "model_and_overhead_ms": model_and_overhead,
        "other_ms": overhead,
        "model_call_count": model_call_count,
        "total_ms": total_task_ms,
        "tool_total_ms": tool_total,
    }


def _agn_time_breakdown_aggregated(data_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregated time breakdown across all agentic attempts for a task.

    Returns: Model Client, bucketed Tool entries (google_search,
    scrape_and_extract_info, Other Tools), and Other
    (Other = total - model_client - sum_of_tools).
    """
    merged_by_cat: dict[str, float] = {}
    counts_by_cat: dict[str, int] = {}
    total_ms = 0.0
    raw_per_tool: dict[str, float] = {}
    tool_counts: dict[str, int] = {}
    for data in data_list:
        timing = _extract_step_timing(data.get("steps", []))
        for cat, ms in timing["by_category"].items():
            merged_by_cat[cat] = merged_by_cat.get(cat, 0) + ms
        for cat, count in timing.get("counts_by_category", {}).items():
            counts_by_cat[cat] = counts_by_cat.get(cat, 0) + count
        total_ms += timing["total_ms"]
        for tc in data.get("tool_calls", []):
            for label, ms in _expand_agentic_tool_timing(tc).items():
                raw_per_tool[label] = raw_per_tool.get(label, 0) + ms
                tool_counts[label] = tool_counts.get(label, 0) + 1

    all_per_tool = dict(raw_per_tool)
    tool_color_map = _build_tool_color_map(list(all_per_tool.keys()))

    model_client_ms = merged_by_cat.get("Model Client", 0)
    tool_total_ms = sum(all_per_tool.values())
    other_ms = max(total_ms - model_client_ms - tool_total_ms, 0) if total_ms > 0 else 0

    merged: dict[str, float] = {}
    merged_colors: dict[str, str] = {}

    # Bucketed tools in fixed order, skip empty "Other Tools"
    for label in _AGN_TOOL_PIE_ORDER:
        ms = all_per_tool.get(label, 0)
        if ms > 0 or label != "Tool: Other Tools":
            if ms > 0:
                merged[label] = ms
                merged_colors[label] = tool_color_map.get(label, OTHER_TOOL_COLOR)

    # Model Client
    if model_client_ms > 0:
        merged["Model Client"] = model_client_ms
        merged_colors["Model Client"] = _CAT_COLORS_BASE["Model Client"]

    # Other (same color as Model Client)
    if other_ms > 0:
        merged["Other"] = other_ms
        merged_colors["Other"] = _CAT_COLORS_BASE["Other"]

    return {
        "merged": merged,
        "merged_colors": merged_colors,
        "total_ms": total_ms,
        "by_category": merged_by_cat,
        "counts_by_category": counts_by_cat,
        "per_tool": all_per_tool,
        "tool_counts": tool_counts,
    }


def _ori_time_breakdown_aggregated(ori_index: dict[str, dict[int, str]]) -> dict[str, Any]:
    data_list: list[dict[str, Any]] = []
    for attempts in ori_index.values():
        for attempt in _sorted_attempts(attempts):
            data_list.append(_load_json_cached(attempts[attempt]))
    return _ori_time_breakdown(data_list)


def _agn_run_time_breakdown_aggregated(agn_index: dict[str, dict[int, str]]) -> dict[str, Any]:
    data_list: list[dict[str, Any]] = []
    for attempts in agn_index.values():
        for attempt in _sorted_attempts(attempts):
            data_list.append(_load_json_cached(attempts[attempt]))
    return _agn_time_breakdown_aggregated(data_list)


def _agentic_tool_execution_timestamps(data: dict[str, Any]) -> list[str]:
    timestamps: list[str] = []
    for step in data.get("steps", []) or []:
        if "tool.execution" not in step.get("step_name", ""):
            continue
        msg = step.get("message", "")
        match = re.search(r"processed\s+(\d+)\s+tool call", msg)
        count = int(match.group(1)) if match else 1
        timestamps.extend([step.get("timestamp", "")] * max(count, 1))
    return timestamps
