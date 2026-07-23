# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Trace extraction and query comparison helpers for the dashboard."""

from __future__ import annotations

import re
from typing import Any


def _ori_tool_calls_paired(data: dict[str, Any]) -> list[dict[str, Any]]:
    steps = data.get("step_logs", [])
    starts: list[dict[str, Any]] = []
    for s in steps:
        name = s.get("step_name", "")
        msg = s.get("message", "")
        if "Tool Call Start" in name:
            m_tool = re.search(r"tool '(\w+)'", msg)
            m_server = re.search(r"server '([^']+)'", msg)
            starts.append(
                {
                    "tool_name": m_tool.group(1) if m_tool else "unknown",
                    "server_name": m_server.group(1) if m_server else "",
                    "arguments": s.get("metadata", {}).get("arguments", {}),
                    "latency_ms": 0,
                    "status": "success",
                    "timestamp": s.get("timestamp", ""),
                }
            )
    turn_idx = 0
    for s in steps:
        name = s.get("step_name", "")
        msg = s.get("message", "")
        if "Turn:" in name and "Tool Call" in name:
            m_lat = re.search(r"completed in (\d+)ms", msg)
            if m_lat and turn_idx < len(starts):
                starts[turn_idx]["latency_ms"] = int(m_lat.group(1))
                starts[turn_idx]["timestamp"] = s.get("timestamp", starts[turn_idx].get("timestamp", ""))
                turn_idx += 1
    return starts


def _ori_per_step_tokens(data: dict[str, Any]) -> list[dict[str, int]]:
    steps = data.get("step_logs", [])
    cumulative: list[tuple[int, int]] = []
    for s in steps:
        if "Token Usage" in s.get("step_name", ""):
            m = re.search(r"Input: (\d+).*Output: (\d+)", s.get("message", ""))
            if m:
                cumulative.append((int(m.group(1)), int(m.group(2))))
    per_step: list[dict[str, int]] = []
    prev_in, prev_out = 0, 0
    for inp, out in cumulative:
        per_step.append({"input_tokens": inp - prev_in, "output_tokens": out - prev_out})
        prev_in, prev_out = inp, out
    return per_step


def _count_tool_calls_by_name(tool_calls: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tc in tool_calls:
        counts[tc.get("tool_name", "unknown")] = counts.get(tc.get("tool_name", "unknown"), 0) + 1
    return counts


def _extract_query_key(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Primary matching key per tool (single field only)."""
    _PRIMARY_FIELD: dict[str, str] = {
        "google_search": "q",
        "web_search": "query",
        "sogou_search": "Query",
        "search_and_browse": "subtask",
        "scrape_website": "url",
        "scrape_and_extract_info": "url",
        "scrape_and_extract": "url",
    }
    field = _PRIMARY_FIELD.get(tool_name)
    if field is None:
        return None
    val = str(arguments.get(field, ""))
    return val or None


def _build_query_comparison(
    left_tcs: list[dict[str, Any]],
    right_tcs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int, int]:
    """Build shared / left-only / right-only query lists grouped by tool.

    Returns (shared, left_only, right_only, left_excluded, right_excluded).
    Each entry has: Tool, Query, and only the index column(s) that apply.
    """
    left_map: dict[str, list[int]] = {}
    right_map: dict[str, list[int]] = {}
    left_excluded = right_excluded = 0
    for i, tc in enumerate(left_tcs):
        q = _extract_query_key(tc["tool_name"], tc.get("arguments", {}))
        if q is None:
            left_excluded += 1
            continue
        key = f"{tc['tool_name']}::{q}"
        left_map.setdefault(key, []).append(i)
    for i, tc in enumerate(right_tcs):
        q = _extract_query_key(tc.get("tool_name", ""), tc.get("arguments", {}))
        if q is None:
            right_excluded += 1
            continue
        key = f"{tc.get('tool_name', '')}::{q}"
        right_map.setdefault(key, []).append(i)
    all_keys = set(left_map) | set(right_map)
    shared, left_only, right_only = [], [], []
    for key in sorted(all_keys):
        tool, _, query = key.partition("::")
        left_idxs = left_map.get(key, [])
        right_idxs = right_map.get(key, [])
        if left_idxs and right_idxs:
            shared.append(
                {
                    "Tool": tool,
                    "Query": query[:120],
                    "Left #": ", ".join(f"[{i}]" for i in left_idxs),
                    "Right #": ", ".join(f"[{i}]" for i in right_idxs),
                }
            )
        elif left_idxs:
            left_only.append({"Tool": tool, "Query": query[:120], "Left #": ", ".join(f"[{i}]" for i in left_idxs)})
        else:
            right_only.append({"Tool": tool, "Query": query[:120], "Right #": ", ".join(f"[{i}]" for i in right_idxs)})
    return shared, left_only, right_only, left_excluded, right_excluded


def _build_exclusive_tables_by_tool(
    entries: list[dict[str, Any]],
    idx_col: str,
) -> dict[str, list[dict[str, Any]]]:
    """Group exclusive entries by tool. Returns {tool_name: sorted rows} with only Query and index."""
    by_tool: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        tool = entry["Tool"]
        by_tool.setdefault(tool, []).append({"#": entry[idx_col], "Query": entry["Query"]})
    for rows in by_tool.values():
        rows.sort(key=lambda r: r["Query"].lower())
    return by_tool
