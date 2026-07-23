# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Timing tab renderer for the benchmark dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

from recipe.dashboard.constants import AGENTIC_COLOR, MODEL_CLIENT_COLOR, ORIGINAL_COLOR, SEARCH_COLOR
from recipe.dashboard.timing_charts import (
    _render_all_task_timing_overview_sides,
    _render_one_breakdown_bar,
    _render_ten_minute_latency_curves_sides,
)

if TYPE_CHECKING:
    from recipe.dashboard.sides import DashboardSide

from recipe.dashboard.sides import side_run_title

_TOP_SLOW_FIELDS = ["task_elapsed_s", "model_client_elapsed_s", "non_model_overhead_s", "tool_latency_ms_sum"]
_PRIMARY_TOOL_LATENCY_KEYS = {"google_search", "scrape_and_extract_info"}


def _fmt_zero(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:,.0f}"
    return str(value)


def _wall_clock_s(payload: dict[str, Any] | None) -> float | None:
    if payload is None:
        return None
    elapsed = payload.get("start_to_end_elapsed_s")
    if elapsed is None:
        elapsed = payload.get("total_eval_elapsed_s")
    return float(elapsed) if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) else None


def _render_script_wall_clock_bar(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    *,
    left_label: str = "Original",
    right_label: str = "Agentic",
) -> None:
    """Show script-level wall-clock totals as one comparison bar."""
    st.subheader("Script-level wall-clock")
    rows: list[dict[str, Any]] = []
    for label, payload, color_group in ((left_label, left, "Left"), (right_label, right, "Right")):
        elapsed = _wall_clock_s(payload)
        if elapsed is None:
            continue
        rows.append(
            {
                "Metric": "Script wall-clock",
                "Pipeline": label,
                "Segment": label,
                "Value": elapsed,
                "Display": f"{elapsed:,.0f}s",
                "Color Group": color_group,
            }
        )
    _render_one_breakdown_bar("Script wall-clock", rows, value_kind="latency")
    # Emit start/end timestamps so users can cross-check wall-clock against their dispatch window.
    cols = st.columns(2)
    for col, label, payload in zip(cols, (left_label, right_label), (left, right), strict=False):
        with col:
            if payload is None:
                st.caption(f"{label}: no timing payload")
                continue
            st.caption(f"{label} started: `{payload.get('script_started_at') or '-'}`")
            st.caption(f"{label} ended:   `{payload.get('script_ended_at') or '-'}`")


def _render_task_wall_clock_curve(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    *,
    left_label: str = "Original",
    right_label: str = "Agentic",
) -> None:
    import altair as alt

    st.subheader("Task wall-clock cumulative")
    rows: list[dict[str, Any]] = []
    for label, payload in ((left_label, left), (right_label, right)):
        values = [
            (float(item["task_elapsed_s"]), str(item.get("task_id", "")))
            for item in (payload or {}).get("items", []) or []
            if isinstance(item.get("task_elapsed_s"), (int, float)) and not isinstance(item.get("task_elapsed_s"), bool)
        ]
        values.sort(key=lambda item: item[0])
        total = len(values)
        rows.extend(
            {
                "Pipeline": label,
                "Task": task_id,
                "Value": value,
                "Task Count": idx,
                "Total Tasks": total,
                "Cumulative Percentage": idx / total * 100,
            }
            for idx, (value, task_id) in enumerate(values, start=1)
        )
    if not rows:
        st.caption("No task wall-clock data.")
        return
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Value:Q", title="Task wall-clock (s)"),
            y=alt.Y("Cumulative Percentage:Q", scale=alt.Scale(domain=[0, 100]), title="Tasks (%)"),
            color=alt.Color("Pipeline:N", scale=alt.Scale(domain=[left_label, right_label], range=[ORIGINAL_COLOR, AGENTIC_COLOR])),
            tooltip=[
                "Pipeline:N",
                "Task:N",
                alt.Tooltip("Value:Q", title="Task wall-clock (s)", format=",.2f"),
                alt.Tooltip("Cumulative Percentage:Q", title="Tasks %", format=".1f"),
            ],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, width="stretch")


def _render_timing_overview_row(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    *,
    left_label: str,
    right_label: str,
) -> None:
    cols = st.columns(2)
    with cols[0]:
        _render_script_wall_clock_bar(left, right, left_label=left_label, right_label=right_label)
    with cols[1]:
        _render_task_wall_clock_curve(left, right, left_label=left_label, right_label=right_label)


def _render_top_slow(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    *,
    left_label: str = "Original",
    right_label: str = "Agentic",
) -> None:
    """Top-10 slow leaderboards, side-by-side per field."""
    st.subheader("Top slow tasks")
    selectable_fields = _TOP_SLOW_FIELDS
    selected = st.selectbox("Sort by", selectable_fields, index=0, key="timing_top_slow_field")

    col_o, col_a = st.columns(2)
    for col, label, payload in ((col_o, left_label, left), (col_a, right_label, right)):
        with col:
            st.markdown(f"**{label}**")
            top = (payload or {}).get("top_slow", {}).get(selected)
            if not isinstance(top, list) or not top:
                st.caption("No data.")
                continue
            rows = []
            for entry in top:
                rows.append(
                    {
                        "task_id": entry.get("task_id"),
                        "sort_value": _fmt_zero(entry.get("sort_value")),
                        "task_s": _fmt_zero(entry.get("task_elapsed_s")),
                        "model_s": _fmt_zero(entry.get("model_client_elapsed_s")),
                        "non_model_s": _fmt_zero(entry.get("non_model_overhead_s")),
                        "tool_ms": _fmt_zero(entry.get("tool_latency_ms_sum")),
                        "turns": entry.get("num_turns", "-"),
                        "score": entry.get("score", "-"),
                        "finish": entry.get("finish_reason", "-"),
                    }
                )
            st.dataframe(pd.DataFrame(rows).astype(str), width="stretch", hide_index=True)


def _render_per_task_breakdown(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    *,
    left_label: str = "Original",
    right_label: str = "Agentic",
) -> None:
    """Per-task stacked bars showing model-client / tool / residual time, side by side.

    Residual = task_elapsed_s - model_client_elapsed_s - tool_latency_s_sum (clamped at 0 so
    measurement noise on cached rows doesn't push bars below the axis).
    """
    import altair as alt

    st.subheader("Per-task time breakdown")

    def _rows(payload: dict[str, Any] | None, pipeline_label: str) -> list[dict[str, Any]]:
        if payload is None:
            return []
        out: list[dict[str, Any]] = []
        for item in payload.get("items", []) or []:
            tid = str(item.get("task_id", ""))
            if not tid:
                continue
            task_s = item.get("task_elapsed_s")
            model_s = item.get("model_client_elapsed_s") or 0.0
            tool_ms = item.get("tool_latency_ms_sum") or 0.0
            tool_s = tool_ms / 1000.0
            residual_s = max(float(task_s) - float(model_s) - tool_s, 0.0) if isinstance(task_s, (int, float)) else 0.0
            out.extend(
                [
                    {"Task": tid, "Pipeline": pipeline_label, "Category": "Model client", "Seconds": float(model_s)},
                    {"Task": tid, "Pipeline": pipeline_label, "Category": "Tool", "Seconds": float(tool_s)},
                    {"Task": tid, "Pipeline": pipeline_label, "Category": "Residual", "Seconds": float(residual_s)},
                ]
            )
        return out

    all_rows = _rows(left, left_label) + _rows(right, right_label)
    if not all_rows:
        st.info("No per-task timing data available.")
        return
    df = pd.DataFrame(all_rows)
    # Consistent task ordering: numeric task_id when possible, else string.
    task_ids = sorted(df["Task"].unique(), key=lambda t: (0, int(t)) if str(t).isdigit() else (1, str(t)))

    category_order = ["Model client", "Tool", "Residual"]
    category_colors = [MODEL_CLIENT_COLOR, SEARCH_COLOR, "#c45b8e"]
    pipeline_order = [left_label, right_label]
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Task:N", sort=task_ids, title="Task"),
            y=alt.Y("Seconds:Q", stack="zero"),
            color=alt.Color("Category:N", scale=alt.Scale(domain=category_order, range=category_colors)),
            xOffset=alt.XOffset("Pipeline:N", sort=pipeline_order),
            tooltip=["Task:N", "Pipeline:N", "Category:N", alt.Tooltip("Seconds:Q", format=".2f")],
        )
        .properties(height=350)
    )
    st.altair_chart(chart, width="stretch")


def _render_per_tool_latency(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    *,
    left_label: str = "Original",
    right_label: str = "Agentic",
) -> None:
    """Aggregate secondary per-tool latency across the whole run for each pipeline."""
    st.subheader("Tool latency totals")
    left_sum = (left or {}).get("tool_latency_ms_by_tool_sum", {}) or {}
    right_sum = (right or {}).get("tool_latency_ms_by_tool_sum", {}) or {}
    left_count = (left or {}).get("tool_count_by_tool_sum", {}) or {}
    right_count = (right or {}).get("tool_count_by_tool_sum", {}) or {}
    tools = sorted((set(left_sum) | set(right_sum)) - _PRIMARY_TOOL_LATENCY_KEYS)
    if not tools:
        st.info("No secondary per-tool latency data.")
        return

    def _bar_row(tool: str, label: str, total_ms: Any, count: Any, color_group: str, *, metric: str, value_kind: str) -> dict[str, Any] | None:
        if not isinstance(total_ms, (int, float)) or isinstance(total_ms, bool):
            return None
        if value_kind == "count":
            value = float(count) if isinstance(count, (int, float)) and not isinstance(count, bool) else 0.0
            display = f"{value:,.0f}"
        else:
            value = float(total_ms) / max(float(count), 1.0) / 1000 if isinstance(count, (int, float)) and count else 0.0
            display = f"{value:.2f}s"
        return {
            "Metric": metric,
            "Pipeline": label,
            "Segment": f"{label} {tool}",
            "Value": value,
            "Display": display,
            "Color Group": color_group,
        }

    count_rows_by_tool: dict[str, list[dict[str, Any]]] = {}
    avg_rows_by_tool: dict[str, list[dict[str, Any]]] = {}
    for tool in tools:
        count_rows_by_tool[tool] = [
            row
            for row in (
                _bar_row(tool, left_label, left_sum.get(tool), left_count.get(tool), "Left", metric=f"{tool} count", value_kind="count"),
                _bar_row(tool, right_label, right_sum.get(tool), right_count.get(tool), "Right", metric=f"{tool} count", value_kind="count"),
            )
            if row is not None
        ]
        avg_rows_by_tool[tool] = [
            row
            for row in (
                _bar_row(tool, left_label, left_sum.get(tool), left_count.get(tool), "Left", metric=f"{tool} avg.", value_kind="latency"),
                _bar_row(tool, right_label, right_sum.get(tool), right_count.get(tool), "Right", metric=f"{tool} avg.", value_kind="latency"),
            )
            if row is not None
        ]

    for start in range(0, len(tools), 3):
        tool_group = tools[start : start + 3]
        count_cols = st.columns(3)
        for col, tool in zip(count_cols, tool_group, strict=False):
            with col:
                _render_one_breakdown_bar(f"{tool} count", count_rows_by_tool[tool], value_kind="count")
        avg_cols = st.columns(3)
        for col, tool in zip(avg_cols, tool_group, strict=False):
            with col:
                _render_one_breakdown_bar(f"{tool} avg. latency", avg_rows_by_tool[tool], value_kind="latency")


def _render_timing_tab(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    *,
    left_label: str = "Original",
    right_label: str = "Agentic",
    left_side: DashboardSide | None = None,
    right_side: DashboardSide | None = None,
) -> None:
    st.header("Timing")
    if left_side is not None and right_side is not None:
        cols = st.columns(2)
        cols[0].markdown(f"#### {side_run_title(left_side)}")
        cols[1].markdown(f"#### {side_run_title(right_side)}")
    st.caption("Timing is read from `eval_results.json` when available, otherwise from completed task traces in the selected runs.")
    if left is None and right is None:
        st.info(
            "No timing payload found under the selected runs. "
            "The tab can read final `eval_results.json` files or completed task traces from an in-progress run."
        )
        return
    if left is None:
        st.caption(f"No timing data for {left_label} yet.")
    elif left.get("live_trace_fallback"):
        st.caption(f"{left_label}: using live task traces because `eval_results.json` is not available yet.")
    if right is None:
        st.caption(f"No timing data for {right_label} yet.")
    elif right.get("live_trace_fallback"):
        st.caption(f"{right_label}: using live task traces because `eval_results.json` is not available yet.")

    _render_timing_overview_row(left, right, left_label=left_label, right_label=right_label)
    if left_side is not None and right_side is not None:
        _render_all_task_timing_overview_sides(
            left_side,
            right_side,
            left_payload=left,
            right_payload=right,
            left_label=left_label,
            right_label=right_label,
        )
        _render_per_tool_latency(left, right, left_label=left_label, right_label=right_label)
    else:
        _render_per_tool_latency(left, right, left_label=left_label, right_label=right_label)
    if left_side is not None and right_side is not None:
        _render_ten_minute_latency_curves_sides(left_side, right_side, left_payload=left, right_payload=right)
    _render_per_task_breakdown(left, right, left_label=left_label, right_label=right_label)
    _render_top_slow(left, right, left_label=left_label, right_label=right_label)
