# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Timing chart helpers for the benchmark dashboard."""

from __future__ import annotations

import contextlib
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

from recipe.dashboard.constants import (
    AGENTIC_COLOR,
    DELTA_COLOR,
    ORIGINAL_COLOR,
    OTHER_TOOL_COLOR,
    UTC8,
)
from recipe.dashboard.discovery import _sorted_attempts
from recipe.dashboard.extraction import _ori_tool_calls_paired
from recipe.dashboard.loading import _load_json_cached
from recipe.dashboard.sides import load_side_summary, side_tool_calls
from recipe.dashboard.timing_processing import (
    _CAT_COLORS_BASE,
    _TOOL_COLORS,
    _TOOL_ORDER,
    _agentic_tool_execution_timestamps,
    _agn_run_time_breakdown_aggregated,
    _ori_time_breakdown_aggregated,
    _parse_datetime_ms,
    _safe_ms,
)

if TYPE_CHECKING:
    from recipe.dashboard.sides import DashboardSide

_TAIL_METRICS: list[tuple[str, str, str, int]] = [
    ("search_tool_latency_ms", "Search tool latency", "ms", 0),
    ("scrape_and_extract_latency_ms", "Scrape and extract latency", "ms", 0),
    ("model_client_elapsed_s", "Model client", "s", 2),
]


def _render_latency_curve(title: str, rows: list[dict[str, Any]]) -> None:
    st.markdown(f"**{title}**")
    if not rows:
        st.caption("No per-call latency data.")
        return
    df = pd.DataFrame(_aggregate_latency_curve_rows(rows, window="10min"))
    if df.empty:
        st.caption("No valid per-call latency data.")
        return
    # st.line_chart has no legend toggle; keep only compact series names and
    # avoid rendering any separate legend/table in the UI.
    series_order = list(dict.fromkeys(df["Series"].tolist()))
    chart_df = df.pivot_table(index="Time", columns="Series", values="Latency (s)", aggfunc="mean").sort_index().reset_index()
    st.line_chart(
        chart_df,
        x="Time",
        y=series_order,
        x_label="Execution time",
        y_label="Latency (s)",
        color=[AGENTIC_COLOR if str(series).startswith("Right") else ORIGINAL_COLOR for series in series_order],
        height=260,
        width="stretch",
    )


def _aggregate_latency_curve_rows(rows: list[dict[str, Any]], *, window: str) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce").dt.floor(window)
    df["Latency (s)"] = pd.to_numeric(df["Latency (s)"], errors="coerce")
    df = df.dropna(subset=["Time", "Latency (s)", "Series"])
    if df.empty:
        return []
    grouped = (
        df.groupby(["Series", "Time"], as_index=False)
        .agg(
            {
                "Latency (s)": "mean",
                "Task": "count",
            }
        )
        .rename(columns={"Task": "Calls"})
    )
    return grouped.sort_values(["Series", "Time"]).to_dict("records")


def _render_task_metric_cumulative_curve(
    title: str,
    rows: list[dict[str, Any]],
    left: DashboardSide,
    right: DashboardSide,
    *,
    x_field: str,
    x_title: str,
) -> None:
    import altair as alt

    st.markdown(f"**{title}**")
    if not rows:
        st.caption("No task data available.")
        return

    df = pd.DataFrame(rows).sort_values(["Run", x_field])
    color_scale = alt.Scale(domain=[left.label, right.label], range=[ORIGINAL_COLOR, AGENTIC_COLOR])
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{x_field}:Q", sort="ascending", title=x_title),
            y=alt.Y("Cumulative Percentage:Q", scale=alt.Scale(domain=[0, 100]), title="Tasks (%)"),
            color=alt.Color("Run:N", scale=color_scale, sort=[left.label, right.label]),
            tooltip=[
                "Run:N",
                "Task:N",
                alt.Tooltip(f"{x_field}:Q", title=x_title, format=","),
                alt.Tooltip("Task Count:Q", title="Tasks"),
                alt.Tooltip("Total Tasks:Q", title="Total"),
                alt.Tooltip("Cumulative Percentage:Q", title="Tasks %", format=".1f"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")


def _metric_curve_rows_from_task_rows(task_rows: list[dict[str, Any]], value_field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[tuple[float, str]]] = {}
    for row in task_rows:
        value = row.get(value_field)
        if isinstance(value, (int, float)):
            grouped.setdefault(str(row.get("Run", "")), []).append((float(value), str(row.get("Task", ""))))

    for run, values in grouped.items():
        values.sort(key=lambda item: item[0])
        total = len(values)
        if total <= 0:
            continue
        rows.extend(
            {
                "Run": run,
                "Task": task_id,
                value_field: value,
                "Task Count": idx,
                "Total Tasks": total,
                "Cumulative Percentage": idx / total * 100,
            }
            for idx, (value, task_id) in enumerate(values, start=1)
        )
    return rows


def _task_rows_from_payloads(
    left: DashboardSide,
    right: DashboardSide,
    left_payload: dict[str, Any] | None,
    right_payload: dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    rows: list[dict[str, Any]] = []
    for side, payload in ((left, left_payload), (right, right_payload)):
        items = (payload or {}).get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            task_id = str(item.get("task_id", ""))
            duration = _numeric(item.get("task_elapsed_s"))
            tokens = _numeric(item.get("total_tokens"))
            turns = _numeric(item.get("num_turns"))
            if duration is None and tokens is None and turns is None:
                continue
            row: dict[str, Any] = {"Run": side.label, "Task": task_id}
            if duration is not None:
                row["Execution Time (s)"] = duration
            if tokens is not None:
                row["Tokens"] = tokens
            if turns is not None:
                row["Turns"] = turns
            rows.append(row)
    return rows or None


def _render_task_cumulative_curves(left: DashboardSide, right: DashboardSide, task_rows: list[dict[str, Any]] | None = None) -> None:
    st.subheader("Per-task Cumulative Percentages")
    cols = st.columns(3)
    duration_rows = (
        _metric_curve_rows_from_task_rows(task_rows, "Execution Time (s)")
        if task_rows is not None
        else _side_task_duration_curve_rows(left) + _side_task_duration_curve_rows(right)
    )
    token_rows = (
        _metric_curve_rows_from_task_rows(task_rows, "Tokens")
        if task_rows is not None
        else _side_task_metric_curve_rows(left, "total_tokens", "Tokens") + _side_task_metric_curve_rows(right, "total_tokens", "Tokens")
    )
    turn_rows = (
        _metric_curve_rows_from_task_rows(task_rows, "Turns")
        if task_rows is not None
        else _side_task_metric_curve_rows(left, "turns", "Turns") + _side_task_metric_curve_rows(right, "turns", "Turns")
    )
    with cols[0]:
        _render_task_metric_cumulative_curve(
            "Execution Time",
            duration_rows,
            left,
            right,
            x_field="Execution Time (s)",
            x_title="Per-task execution time (s)",
        )
    with cols[1]:
        _render_task_metric_cumulative_curve(
            "Token Usage",
            token_rows,
            left,
            right,
            x_field="Tokens",
            x_title="Per-task tokens",
        )
    with cols[2]:
        _render_task_metric_cumulative_curve(
            "Turn Count",
            turn_rows,
            left,
            right,
            x_field="Turns",
            x_title="Per-task turns",
        )
    st.caption("Tasks are sorted ascending by each per-task metric; each point is the cumulative percentage of tasks at or below that value.")


def _task_duration_s(side: DashboardSide, data: dict[str, Any]) -> float | None:
    start_ms = _parse_datetime_ms(data.get("started_at"))
    end_ms = None
    for key in ("ended_at", "finished_at", "completed_at"):
        end_ms = _parse_datetime_ms(data.get(key))
        if end_ms is not None:
            break
    if start_ms is not None and end_ms is not None and end_ms >= start_ms:
        return (end_ms - start_ms) / 1000

    timestamps: list[float] = []
    for step in (data.get("steps") or []) + (data.get("step_logs") or []):
        if isinstance(step, dict):
            ts_ms = _parse_datetime_ms(step.get("timestamp"))
            if ts_ms is not None:
                timestamps.append(ts_ms)

    if side.kind == "original":
        for tool_call in _ori_tool_calls_paired(data):
            ts_ms = _parse_datetime_ms(tool_call.get("timestamp"))
            if ts_ms is not None:
                timestamps.append(ts_ms)

    if len(timestamps) >= 2:
        return max(max(timestamps) - min(timestamps), 0.0) / 1000
    return None


def _side_task_duration_curve_rows(side: DashboardSide) -> list[dict[str, Any]]:
    duration_rows: list[tuple[float, str]] = []
    for task_id, attempts in side.index.items():
        duration_s = 0.0
        has_duration = False
        for attempt in _sorted_attempts(attempts):
            attempt_duration_s = _task_duration_s(side, _load_json_cached(attempts[attempt]))
            if attempt_duration_s is not None:
                duration_s += attempt_duration_s
                has_duration = True
        if has_duration:
            duration_rows.append((duration_s, task_id))

    duration_rows.sort(key=lambda row: row[0])
    total = len(duration_rows)
    if total <= 0:
        return []

    return [
        {
            "Run": side.label,
            "Task": task_id,
            "Execution Time (s)": duration_s,
            "Task Count": idx,
            "Total Tasks": total,
            "Cumulative Percentage": idx / total * 100,
        }
        for idx, (duration_s, task_id) in enumerate(duration_rows, start=1)
    ]


def _side_task_metric_curve_rows(side: DashboardSide, summary_key: str, value_field: str) -> list[dict[str, Any]]:
    metric_rows: list[tuple[float, str]] = []
    for task_id, attempts in side.index.items():
        total = 0.0
        has_value = False
        for attempt in _sorted_attempts(attempts):
            value = load_side_summary(side, attempts[attempt]).get(summary_key)
            if isinstance(value, (int, float)):
                total += float(value)
                has_value = True
        if has_value:
            metric_rows.append((total, task_id))

    metric_rows.sort(key=lambda row: row[0])
    total_tasks = len(metric_rows)
    if total_tasks <= 0:
        return []

    return [
        {
            "Run": side.label,
            "Task": task_id,
            value_field: value,
            "Task Count": idx,
            "Total Tasks": total_tasks,
            "Cumulative Percentage": idx / total_tasks * 100,
        }
        for idx, (value, task_id) in enumerate(metric_rows, start=1)
    ]


def _tool_usage_color_map(tool_order: list[str]) -> dict[str, str]:
    import colorsys

    return {
        tool: "#{:02x}{:02x}{:02x}".format(*(round(channel * 255) for channel in colorsys.hsv_to_rgb((idx * 0.61803398875) % 1.0, 0.62, 0.78)))
        for idx, tool in enumerate(tool_order)
    }


def _tool_call_succeeded(tool_call: dict[str, Any]) -> bool:
    success = tool_call.get("success")
    if isinstance(success, bool):
        return success

    status = str(tool_call.get("status", "") or "").strip().lower()
    if status:
        return status in {"success", "successful", "succeeded", "ok", "completed"}

    return not tool_call.get("error")


def _clean_count_map(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    counts: dict[str, int] = {}
    for name, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        count = int(value)
        if count > 0:
            counts[str(name)] = counts.get(str(name), 0) + count
    return counts


def _merge_count_map(target: dict[str, int], raw: Any) -> None:
    for name, count in _clean_count_map(raw).items():
        target[name] = target.get(name, 0) + count


@st.cache_data(ttl=30, show_spinner=False)
def _load_eval_tool_count_groups(run_dir: str) -> dict[str, dict[str, int]] | None:
    path = Path(run_dir) / "eval_results.json"
    if not path.exists():
        return None
    with contextlib.suppress(UnicodeDecodeError, json.JSONDecodeError, OSError):
        payload = json.loads(path.read_text(encoding="utf-8"))
        usage = _clean_count_map(payload.get("tool_count_by_tool_sum"))
        success = _clean_count_map(payload.get("tool_success_by_tool_sum"))
        failed = _clean_count_map(payload.get("tool_failed_by_tool_sum"))
        has_usage_summary = bool(usage)
        has_success_summary = bool(success)
        has_failed_summary = bool(failed)

        items = payload.get("items", [])
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                if not has_usage_summary:
                    _merge_count_map(usage, item.get("tool_count_by_tool"))
                if not has_success_summary:
                    _merge_count_map(success, item.get("tool_success_by_tool"))
                if not has_failed_summary:
                    _merge_count_map(failed, item.get("tool_failed_by_tool"))

        if not usage:
            return None
        if not success and not failed:
            success = dict(usage)
        return {"usage": usage, "success": success, "failed": failed}
    return None


def _scan_tool_count_groups(side: DashboardSide) -> dict[str, dict[str, int]]:
    usage: dict[str, int] = {}
    success: dict[str, int] = {}
    failed: dict[str, int] = {}
    for attempts in side.index.values():
        for attempt in _sorted_attempts(attempts):
            for tc in side_tool_calls(side, _load_json_cached(attempts[attempt])):
                name = str(tc.get("tool_name", "unknown") or "unknown")
                usage[name] = usage.get(name, 0) + 1
                if _tool_call_succeeded(tc):
                    success[name] = success.get(name, 0) + 1
                else:
                    failed[name] = failed.get(name, 0) + 1
    return {"usage": usage, "success": success, "failed": failed}


def _tool_call_count_groups_by_side(sides: list[DashboardSide]) -> dict[str, dict[str, dict[str, int]]]:
    counts_by_side: dict[str, dict[str, dict[str, int]]] = {}
    for side in sides:
        artifact_counts = _load_eval_tool_count_groups(str(side.run)) if side.run else None
        counts_by_side[side.label] = artifact_counts or _scan_tool_count_groups(side)
    return counts_by_side


def _render_tool_distribution_pie(title: str, counts: dict[str, int], tool_colors: dict[str, str]) -> None:
    import altair as alt

    st.markdown(f"**{title}**")
    if not counts:
        st.caption("No tool calls.")
        return

    tool_order = sorted(tool for tool, count in counts.items() if count > 0)
    df = pd.DataFrame([{"Tool": tool, "Count": counts[tool]} for tool in tool_order])
    chart = (
        alt.Chart(df)
        .mark_arc(innerRadius=45)
        .encode(
            theta=alt.Theta("Count:Q", stack=True),
            color=alt.Color(
                "Tool:N",
                scale=alt.Scale(domain=tool_order, range=[tool_colors[tool] for tool in tool_order]),
                legend=alt.Legend(orient="right", title=None),
            ),
            tooltip=["Tool:N", "Count:Q"],
        )
        .properties(height=230)
    )
    st.altair_chart(chart, width="stretch")


def _render_tool_distribution_pies(
    sides: list[DashboardSide],
    usage_counts_by_side: dict[str, dict[str, int]],
    success_counts_by_side: dict[str, dict[str, int]],
    failed_counts_by_side: dict[str, dict[str, int]],
    tool_colors: dict[str, str],
) -> None:
    st.subheader("Tool Distributions")
    for side in sides:
        cols = st.columns(3)
        chart_specs = [
            (f"{side.label} total tools", usage_counts_by_side[side.label]),
            (f"{side.label} successful tools", success_counts_by_side[side.label]),
            (f"{side.label} failed/rejected tools", failed_counts_by_side[side.label]),
        ]
        for col, (title, counts) in zip(cols, chart_specs, strict=False):
            with col:
                _render_tool_distribution_pie(title, counts, tool_colors)


def _render_tool_usage_distribution(sides: list[DashboardSide]) -> None:
    count_groups_by_side = _tool_call_count_groups_by_side(sides)
    usage_counts_by_side = {side.label: count_groups_by_side[side.label]["usage"] for side in sides}
    success_counts_by_side = {side.label: count_groups_by_side[side.label]["success"] for side in sides}
    failed_counts_by_side = {side.label: count_groups_by_side[side.label]["failed"] for side in sides}
    global_tool_order = sorted({tool for counts in usage_counts_by_side.values() for tool in counts})
    tool_colors = _tool_usage_color_map(global_tool_order)

    _render_tool_distribution_pies(sides, usage_counts_by_side, success_counts_by_side, failed_counts_by_side, tool_colors)


def _render_time_breakdown_pie(title: str, rows: list[dict[str, Any]]) -> None:
    import altair as alt

    st.markdown(f"**{title}**")
    if not rows:
        st.caption("No timing data.")
        return
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_arc(innerRadius=45)
        .encode(
            theta=alt.Theta("Time (s):Q"),
            color=alt.Color("Category:N", scale=alt.Scale(domain=df["Category"].tolist(), range=df["color"].tolist())),
            tooltip=["Category:N", alt.Tooltip("Time (s):Q", format=".1f")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width="stretch")


def _avg_seconds(total_ms: float, count: int) -> float | None:
    if count <= 0:
        return None
    return total_ms / count / 1000


def _fmt_avg_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}s"


def _stacked_bar_value_row(metric: str, segment: str, value: float, display: str, color_group: str, pipeline: str | None = None) -> dict[str, Any]:
    return {
        "Metric": metric,
        "Pipeline": pipeline or color_group,
        "Segment": segment,
        "Value": value,
        "Display": display,
        "Color Group": color_group,
    }


def _render_one_breakdown_bar(title: str, rows: list[dict[str, Any]], *, value_kind: str) -> None:
    import altair as alt

    df = pd.DataFrame([row for row in rows if row["Value"] > 0])
    st.markdown(f"**{title}**")
    if df.empty:
        st.caption("No data.")
        return

    if "Display" not in df or df["Display"].isna().any() or (df["Display"].astype(str) == "").any():
        if value_kind == "latency":
            df["Display"] = df["Value"].map(_fmt_avg_seconds)
        else:
            df["Display"] = df["Value"].map(lambda v: f"{v:,.0f}")

    order_map = {"Original": 0, "Agentic": 1, "Left": 0, "Right": 1, "Delta": 2}
    df["Stack Order"] = df["Color Group"].map(order_map).fillna(99)
    df = df.sort_values("Stack Order").reset_index(drop=True)
    total_value = float(df["Value"].sum())
    df["Fraction"] = df["Value"] / total_value if total_value > 0 else 0.0
    cumulative = 0.0
    label_x: list[float] = []
    for fraction in df["Fraction"]:
        label_x.append(cumulative + float(fraction) / 2)
        cumulative += float(fraction)
    df["Label X"] = label_x
    color_domain = ["Original", "Agentic", "Left", "Right", "Delta"]
    color_range = [ORIGINAL_COLOR, AGENTIC_COLOR, ORIGINAL_COLOR, AGENTIC_COLOR, DELTA_COLOR]
    bars = (
        alt.Chart(df)
        .mark_bar(size=38)
        .encode(
            x=alt.X("Fraction:Q", stack="zero", scale=alt.Scale(domain=[0, 1]), axis=None, title=None),
            y=alt.Y("Metric:N", axis=None, title=None, scale=alt.Scale(paddingInner=0.05, paddingOuter=0.05)),
            color=alt.Color("Color Group:N", scale=alt.Scale(domain=color_domain, range=color_range), legend=None),
            order=alt.Order("Stack Order:Q", sort="ascending"),
            tooltip=[
                alt.Tooltip("Segment:N"),
                alt.Tooltip("Display:N", title="Value"),
                alt.Tooltip("Value:Q", title="Raw", format=".2f"),
            ],
        )
    )
    labels = (
        alt.Chart(df)
        .mark_text(dy=-25, fontSize=12)
        .encode(
            x=alt.X("Label X:Q", scale=alt.Scale(domain=[0, 1]), axis=None, title=None),
            y=alt.Y("Metric:N"),
            text=alt.Text("Display:N"),
            order=alt.Order("Stack Order:Q", sort="ascending"),
        )
    )
    chart = (bars + labels).properties(height=76).configure_view(strokeOpacity=0)
    st.altair_chart(chart, width="stretch")


def _side_time_breakdown(side: DashboardSide) -> dict[str, Any]:
    if side.kind == "original":
        return _ori_time_breakdown_aggregated(side.index) if side.index else {}
    return _agn_run_time_breakdown_aggregated(side.index) if side.index else {}


def _tool_label(raw_name: str) -> str:
    if raw_name.startswith("Tool: "):
        return raw_name
    if raw_name == "web_search":
        return "Tool: google_search"
    if raw_name == "scrape_and_extract":
        return "Tool: scrape_and_extract_info"
    if raw_name in {"google_search", "scrape_and_extract_info"}:
        return f"Tool: {raw_name}"
    return "Tool: Other Tools"


def _numeric(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _numeric_sum_map(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for name, value in raw.items():
        numeric = _numeric(value)
        if numeric is not None and numeric > 0:
            label = _tool_label(str(name))
            out[label] = out.get(label, 0.0) + numeric
    return out


def _numeric_count_map(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for name, value in raw.items():
        numeric = _numeric(value)
        if numeric is not None and numeric > 0:
            label = _tool_label(str(name))
            out[label] = out.get(label, 0) + int(numeric)
    return out


def _merge_agentic_breakdown_display(bd: dict[str, Any]) -> dict[str, Any]:
    per_tool = _numeric_sum_map(bd.get("per_tool"))
    by_category = bd.get("by_category") if isinstance(bd.get("by_category"), dict) else {}
    model_ms = _numeric(by_category.get("Model Client")) or _numeric(bd.get("model_client_ms")) or 0.0
    total_ms = _numeric(bd.get("total_ms")) or 0.0
    tool_total = sum(per_tool.values())
    other_ms = max(total_ms - model_ms - tool_total, 0.0) if total_ms > 0 else 0.0

    merged: dict[str, float] = {}
    merged_colors: dict[str, str] = {}
    for label in [
        "Tool: serper",
        "Tool: google_search",
        "Tool: google_search overhead",
        "Tool: jina",
        "Tool: scrape_and_extract_info",
        "Tool: LLM extraction",
        "Tool: scrape_and_extract_info overhead",
        "Tool: Other Tools",
    ]:
        value = per_tool.get(label, 0.0)
        if value > 0:
            merged[label] = value
            merged_colors[label] = _TOOL_COLORS.get(label, OTHER_TOOL_COLOR)
    if model_ms > 0:
        merged["Model Client"] = model_ms
        merged_colors["Model Client"] = _CAT_COLORS_BASE["Model Client"]
    if other_ms > 0:
        merged["Other"] = other_ms
        merged_colors["Other"] = _CAT_COLORS_BASE["Other"]

    out = dict(bd)
    out["per_tool"] = per_tool
    out["merged"] = merged
    out["merged_colors"] = merged_colors
    out.setdefault("by_category", {"Model Client": model_ms} if model_ms > 0 else {})
    out.setdefault("counts_by_category", {})
    out["total_ms"] = total_ms
    out.setdefault("tool_counts", _numeric_count_map(bd.get("tool_counts")))
    return out


def _payload_time_breakdown(side: DashboardSide, payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None

    payload_bd = payload.get("time_breakdown")
    if isinstance(payload_bd, dict) and (_numeric(payload_bd.get("total_ms")) or payload_bd.get("per_tool")):
        if side.kind == "original":
            out = dict(payload_bd)
            out["per_tool"] = _numeric_sum_map(out.get("per_tool"))
            out["tool_counts"] = _numeric_count_map(out.get("tool_counts"))
            return out
        return _merge_agentic_breakdown_display(payload_bd)

    items = payload.get("items")
    if not isinstance(items, list):
        return None

    total_ms = 0.0
    model_ms = 0.0
    model_count = 0
    per_tool: dict[str, float] = {}
    tool_counts: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        task_s = _numeric(item.get("task_elapsed_s"))
        model_s = _numeric(item.get("model_client_elapsed_s"))
        turns = _numeric(item.get("num_turns"))
        if task_s is not None:
            total_ms += task_s * 1000
        if model_s is not None:
            model_ms += model_s * 1000
        if turns is not None:
            model_count += int(turns)
        for label, ms in _numeric_sum_map(item.get("tool_latency_ms_by_tool")).items():
            per_tool[label] = per_tool.get(label, 0.0) + ms
        for label, count in _numeric_count_map(item.get("tool_count_by_tool")).items():
            tool_counts[label] = tool_counts.get(label, 0) + count

    if total_ms <= 0 and not per_tool:
        return None
    tool_total = sum(per_tool.values())
    other_ms = max(total_ms - model_ms - tool_total, 0.0) if total_ms > 0 else 0.0
    bd: dict[str, Any] = {
        "per_tool": per_tool,
        "tool_counts": tool_counts,
        "model_client_ms": model_ms,
        "other_ms": other_ms,
        "model_call_count": model_count,
        "total_ms": total_ms,
        "tool_total_ms": tool_total,
        "by_category": {"Model Client": model_ms} if model_ms > 0 else {},
        "counts_by_category": {"Model Client": model_count} if model_count > 0 else {},
    }
    if side.kind == "original":
        return bd
    return _merge_agentic_breakdown_display(bd)


def _side_time_breakdown_title(side: DashboardSide, bd: dict[str, Any]) -> str:
    task_count = len(side.index)
    total_ms = float(bd.get("total_ms", 0) or 0)
    if task_count <= 0 or total_ms <= 0:
        return side.label
    avg_time = total_ms / task_count / 1000
    return f"{side.label}: {avg_time:,.0f}s"


def _side_pie_rows(side: DashboardSide, bd: dict[str, Any]) -> list[dict[str, Any]]:
    if side.kind == "original":
        rows: list[dict[str, Any]] = []
        for label in _TOOL_ORDER:
            ms = (bd.get("per_tool") or {}).get(label, 0)
            if ms > 0:
                rows.append({"Category": label, "Time (s)": ms / 1000, "color": _TOOL_COLORS.get(label, OTHER_TOOL_COLOR)})
        if bd.get("model_client_ms", 0) > 0:
            rows.append({"Category": "Model Client (estimated)", "Time (s)": bd["model_client_ms"] / 1000, "color": _CAT_COLORS_BASE["Model Client"]})
        if bd.get("other_ms", 0) > 0:
            rows.append({"Category": "Other / overhead", "Time (s)": bd["other_ms"] / 1000, "color": _CAT_COLORS_BASE["Other"]})
        return rows

    return [
        {"Category": label, "Time (s)": ms / 1000, "color": (bd.get("merged_colors") or {}).get(label, OTHER_TOOL_COLOR)}
        for label, ms in (bd.get("merged") or {}).items()
        if round(ms / 1000, 1) > 0
    ]


def _side_position(side: DashboardSide) -> str:
    return "Left" if side.label == "Left" else "Right"


def _side_color_group(side: DashboardSide) -> str:
    return _side_position(side)


def _original_breakdown_rows(side: DashboardSide, bd: dict[str, Any], *, value_kind: str) -> dict[str, dict[str, Any]]:
    per_tool = bd.get("per_tool") or {}
    tool_counts = bd.get("tool_counts") or {}
    side_name = _side_position(side)
    color_group = _side_color_group(side)

    search_count = int(tool_counts.get("Tool: google_search", 0) or 0)
    extract_count = int(tool_counts.get("Tool: scrape_and_extract_info", 0) or 0)
    model_count = int(bd.get("model_call_count", 0) or 0)
    search_avg = _avg_seconds(float(per_tool.get("Tool: google_search", 0) or 0), search_count) or 0.0
    extract_avg = _avg_seconds(float(per_tool.get("Tool: scrape_and_extract_info", 0) or 0), extract_count) or 0.0
    model_avg = _avg_seconds(float(bd.get("model_client_ms", 0) or 0), model_count) or 0.0

    if value_kind == "count":
        return {
            "search": _stacked_bar_value_row(
                "Count: google_search v.s. serper", f"{side_name} Original Tool: google_search", search_count, f"{search_count:,}", color_group
            ),
            "extract": _stacked_bar_value_row(
                "Count: scrape_and_extraction v.s. jina / LLM extraction",
                f"{side_name} Original Tool: scrape_and_extract_info",
                extract_count,
                f"{extract_count:,}",
                color_group,
            ),
            "model": _stacked_bar_value_row(
                "Count: Assistant Call", f"{side_name} Original inferred model-call count", model_count, f"{model_count:,}", color_group
            ),
        }
    return {
        "search": _stacked_bar_value_row(
            "Latency: google_search v.s. serper",
            f"{side_name} Original Tool: google_search avg.",
            search_avg,
            _fmt_avg_seconds(search_avg),
            color_group,
        ),
        "extract": _stacked_bar_value_row(
            "Latency: scrape_and_extraction v.s. jina + LLM extraction",
            f"{side_name} Original Tool: scrape_and_extract_info avg.",
            extract_avg,
            _fmt_avg_seconds(extract_avg),
            color_group,
        ),
        "model": _stacked_bar_value_row(
            "Latency: inferred model v.s. Model Client",
            f"{side_name} Original inferred model avg.",
            model_avg,
            _fmt_avg_seconds(model_avg),
            color_group,
        ),
    }


def _agentic_breakdown_rows(side: DashboardSide, bd: dict[str, Any], *, value_kind: str) -> dict[str, list[dict[str, Any]]]:
    per_tool = bd.get("per_tool") or {}
    tool_counts = bd.get("tool_counts") or {}
    by_cat = bd.get("by_category") or {}
    counts_by_cat = bd.get("counts_by_category") or {}
    side_name = _side_position(side)
    color_group = _side_color_group(side)

    serper_count = int(tool_counts.get("Tool: serper", 0) or tool_counts.get("Tool: google_search", 0) or 0)
    jina_count = int(tool_counts.get("Tool: jina", 0) or 0)
    llm_count = int(tool_counts.get("Tool: LLM extraction", 0) or 0)
    model_count = int(counts_by_cat.get("Model Client", 0) or 0)
    serper_avg = _avg_seconds(float(per_tool.get("Tool: serper", 0) or per_tool.get("Tool: google_search", 0) or 0), serper_count) or 0.0
    jina_avg = _avg_seconds(float(per_tool.get("Tool: jina", 0) or 0), jina_count) or 0.0
    llm_avg = _avg_seconds(float(per_tool.get("Tool: LLM extraction", 0) or 0), llm_count) or 0.0
    model_avg = _avg_seconds(float(by_cat.get("Model Client", 0) or 0), model_count) or 0.0

    if value_kind == "count":
        extract_count = max(jina_count, llm_count)
        return {
            "search": [
                _stacked_bar_value_row(
                    "Count: google_search v.s. serper", f"{side_name} Agentic Tool: serper", serper_count, f"{serper_count:,}", color_group
                )
            ],
            "extract": [
                _stacked_bar_value_row(
                    "Count: scrape_and_extraction v.s. jina / LLM extraction",
                    f"{side_name} Agentic Tool: max(jina, LLM extraction)",
                    extract_count,
                    f"{extract_count:,}",
                    color_group,
                ),
            ],
            "model": [
                _stacked_bar_value_row(
                    "Count: Assistant Call", f"{side_name} Agentic Model Client count", model_count, f"{model_count:,}", color_group
                )
            ],
        }
    return {
        "search": [
            _stacked_bar_value_row(
                "Latency: google_search v.s. serper", f"{side_name} Agentic Tool: serper avg.", serper_avg, _fmt_avg_seconds(serper_avg), color_group
            )
        ],
        "extract": [
            _stacked_bar_value_row(
                "Latency: scrape_and_extraction v.s. jina + LLM extraction",
                f"{side_name} Agentic Tool: jina avg.",
                jina_avg,
                _fmt_avg_seconds(jina_avg),
                color_group,
            ),
            _stacked_bar_value_row(
                "Latency: scrape_and_extraction v.s. jina + LLM extraction",
                f"{side_name} Agentic Tool: LLM extraction avg.",
                llm_avg,
                _fmt_avg_seconds(llm_avg),
                color_group,
            ),
        ],
        "model": [
            _stacked_bar_value_row(
                "Latency: inferred model v.s. Model Client",
                f"{side_name} Agentic Model Client avg.",
                model_avg,
                _fmt_avg_seconds(model_avg),
                color_group,
            )
        ],
    }


def _breakdown_rows_for_side(side: DashboardSide, bd: dict[str, Any], *, value_kind: str) -> dict[str, list[dict[str, Any]]]:
    if side.kind == "original":
        rows = _original_breakdown_rows(side, bd, value_kind=value_kind)
        return {key: [value] for key, value in rows.items()}
    return _agentic_breakdown_rows(side, bd, value_kind=value_kind)


def _render_time_breakdown_bar_comparisons(left: DashboardSide, right: DashboardSide, left_bd: dict[str, Any], right_bd: dict[str, Any]) -> None:
    count_bar_specs = [
        ("Count: google_search v.s. serper", "search"),
        ("Count: scrape_and_extraction v.s. jina / LLM extraction", "extract"),
        ("Count: Assistant Call", "model"),
    ]
    latency_bar_specs = [
        ("Latency: google_search v.s. serper", "search"),
        ("Latency: scrape_and_extraction v.s. jina + LLM extraction", "extract"),
        ("Latency: inferred model v.s. Model Client", "model"),
    ]
    left_count_rows = _breakdown_rows_for_side(left, left_bd, value_kind="count")
    right_count_rows = _breakdown_rows_for_side(right, right_bd, value_kind="count")
    left_latency_rows = _breakdown_rows_for_side(left, left_bd, value_kind="latency")
    right_latency_rows = _breakdown_rows_for_side(right, right_bd, value_kind="latency")

    count_cols = st.columns(3)
    for idx, (title, key) in enumerate(count_bar_specs):
        with count_cols[idx]:
            _render_one_breakdown_bar(title, left_count_rows[key] + right_count_rows[key], value_kind="count")

    avg_cols = st.columns(3)
    for idx, (title, key) in enumerate(latency_bar_specs):
        with avg_cols[idx]:
            _render_one_breakdown_bar(title, left_latency_rows[key] + right_latency_rows[key], value_kind="latency")


def _tail_metric_value(item: dict[str, Any], metric_key: str) -> float | None:
    if metric_key == "search_tool_latency_ms":
        by_tool = item.get("tool_latency_ms_by_tool") or {}
        if not isinstance(by_tool, dict):
            return None
        value = by_tool.get("serper")
        if value is None:
            value = by_tool.get("google_search")
        if value is None:
            value = by_tool.get("web_search")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    if metric_key == "scrape_and_extract_latency_ms":
        by_tool = item.get("tool_latency_ms_by_tool") or {}
        if not isinstance(by_tool, dict):
            return None
        value = by_tool.get("scrape_and_extract_info")
        if value is None:
            value = by_tool.get("scrape_and_extract")
        if value is None:
            parts = [by_tool.get("jina"), by_tool.get("llm_extraction")]
            numeric = [float(part) for part in parts if isinstance(part, (int, float)) and not isinstance(part, bool)]
            value = sum(numeric) if numeric else None
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    value = item.get(metric_key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _render_tail_curves(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    *,
    left_label: str = "Original",
    right_label: str = "Agentic",
) -> None:
    """Per-pipeline cumulative curves for every timing metric."""
    import altair as alt

    rows: list[dict[str, Any]] = []
    for metric_key, display_name, unit, _digits in _TAIL_METRICS:
        for label, payload in ((left_label, left), (right_label, right)):
            values = [value for item in (payload or {}).get("items", []) or [] if (value := _tail_metric_value(item, metric_key)) is not None]
            values.sort()
            total = len(values)
            rows.extend(
                {
                    "Metric": f"{display_name} ({unit})",
                    "Pipeline": label,
                    "Value": value,
                    "Task Count": idx,
                    "Total Tasks": total,
                    "Cumulative Percentage": idx / total * 100,
                }
                for idx, value in enumerate(values, start=1)
            )

    if not rows:
        st.info("No per-task timing values available.")
        return

    df = pd.DataFrame(rows)
    metrics = [f"{display_name} ({unit})" for _, display_name, unit, _ in _TAIL_METRICS]
    color_scale = alt.Scale(domain=[left_label, right_label], range=[ORIGINAL_COLOR, AGENTIC_COLOR])
    cols = st.columns(3)
    for idx, metric in enumerate(metrics):
        with cols[idx % len(cols)]:
            st.markdown(f"**{metric}**")
            metric_df = df[df["Metric"] == metric]
            chart = (
                alt.Chart(metric_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Value:Q", title=metric),
                    y=alt.Y("Cumulative Percentage:Q", scale=alt.Scale(domain=[0, 100]), title="Tasks (%)"),
                    color=alt.Color("Pipeline:N", scale=color_scale, sort=[left_label, right_label]),
                    tooltip=[
                        "Pipeline:N",
                        alt.Tooltip("Value:Q", title=metric, format=",.2f"),
                        alt.Tooltip("Task Count:Q", title="Tasks"),
                        alt.Tooltip("Total Tasks:Q", title="Total"),
                        alt.Tooltip("Cumulative Percentage:Q", title="Tasks %", format=".1f"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, width="stretch")


def _render_all_task_timing_overview_sides(
    left: DashboardSide,
    right: DashboardSide,
    *,
    left_payload: dict[str, Any] | None = None,
    right_payload: dict[str, Any] | None = None,
    left_label: str = "Original",
    right_label: str = "Agentic",
    task_rows: list[dict[str, Any]] | None = None,
) -> None:
    st.subheader("All-task Time Breakdown")
    left_bd = _payload_time_breakdown(left, left_payload) or _side_time_breakdown(left)
    right_bd = _payload_time_breakdown(right, right_payload) or _side_time_breakdown(right)
    col_left, col_right = st.columns(2)
    with col_left:
        _render_time_breakdown_pie(_side_time_breakdown_title(left, left_bd), _side_pie_rows(left, left_bd))
    with col_right:
        _render_time_breakdown_pie(_side_time_breakdown_title(right, right_bd), _side_pie_rows(right, right_bd))

    _render_time_breakdown_bar_comparisons(left, right, left_bd, right_bd)
    _render_tail_curves(left_payload, right_payload, left_label=left_label, right_label=right_label)
    _render_task_cumulative_curves(left, right, task_rows or _task_rows_from_payloads(left, right, left_payload, right_payload))


def _latency_curve_row(side: DashboardSide, task_id: str, attempt: int, call_idx: int, ts_ms: float, latency_ms: float) -> dict[str, Any]:
    return {
        "Time": datetime.fromtimestamp(ts_ms / 1000, tz=UTC8),
        "Latency (s)": latency_ms / 1000,
        "Series": side.label,
        "Task": task_id,
        "Attempt": attempt,
        "Call": call_idx,
    }


def _add_original_latency_rows(rows: dict[str, list[dict[str, Any]]], side: DashboardSide, task_id: str, attempt: int, data: dict[str, Any]) -> None:
    for call_idx, tc in enumerate(_ori_tool_calls_paired(data)):
        ts_ms = _parse_datetime_ms(tc.get("timestamp"))
        if ts_ms is None:
            continue
        row = _latency_curve_row(side, task_id, attempt, call_idx, ts_ms, _safe_ms(tc.get("latency_ms")))
        if tc.get("tool_name") == "google_search":
            rows["search"].append(row)
        elif tc.get("tool_name") == "scrape_and_extract_info":
            rows["extract"].append(row)

    pending_start_ms: float | None = None
    model_call_idx = 0
    for step in data.get("step_logs", []) or []:
        step_name = step.get("step_name", "")
        if "Message Retention" in step_name:
            pending_start_ms = _parse_datetime_ms(step.get("timestamp"))
        elif "Token Usage" in step_name:
            end_ms = _parse_datetime_ms(step.get("timestamp"))
            if pending_start_ms is not None and end_ms is not None:
                rows["model"].append(_latency_curve_row(side, task_id, attempt, model_call_idx, end_ms, max(end_ms - pending_start_ms, 0.0)))
                model_call_idx += 1
            pending_start_ms = None


def _add_agentic_latency_rows(rows: dict[str, list[dict[str, Any]]], side: DashboardSide, task_id: str, attempt: int, data: dict[str, Any]) -> None:
    tool_timestamps = _agentic_tool_execution_timestamps(data)
    for call_idx, tc in enumerate(data.get("tool_calls", []) or []):
        ts_ms = _parse_datetime_ms(tool_timestamps[call_idx] if call_idx < len(tool_timestamps) else data.get("ended_at"))
        if ts_ms is None:
            continue
        timing_ms = tc.get("metadata", {}).get("timing_ms", {})
        if not isinstance(timing_ms, dict):
            timing_ms = {}
        if tc.get("tool_name") in {"google_search", "web_search"}:
            latency_ms = _safe_ms(timing_ms.get("serper_request") or tc.get("latency_ms"))
            rows["search"].append(_latency_curve_row(side, task_id, attempt, call_idx, ts_ms, latency_ms))
        elif tc.get("tool_name") in {"scrape_and_extract_info", "scrape_and_extract"}:
            latency_ms = _safe_ms(timing_ms.get("jina") or timing_ms.get("jina_scrape")) + _safe_ms(timing_ms.get("llm_extraction"))
            if latency_ms <= 0:
                latency_ms = _safe_ms(tc.get("latency_ms"))
            rows["extract"].append(_latency_curve_row(side, task_id, attempt, call_idx, ts_ms, latency_ms))

    model_call_idx = 0
    for step in data.get("steps", []) or []:
        if "model_client.inference" not in step.get("step_name", ""):
            continue
        elapsed_ms = step.get("metadata", {}).get("elapsed_ms")
        ts_ms = _parse_datetime_ms(step.get("timestamp"))
        if not isinstance(elapsed_ms, (int, float)) or ts_ms is None:
            continue
        rows["model"].append(_latency_curve_row(side, task_id, attempt, model_call_idx, ts_ms, max(float(elapsed_ms), 0.0)))
        model_call_idx += 1


def _payload_latency_curve_rows(side: DashboardSide, payload: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]] | None:
    raw_rows = (payload or {}).get("latency_curve_rows")
    if not isinstance(raw_rows, list):
        return None
    rows: dict[str, list[dict[str, Any]]] = {"search": [], "extract": [], "model": []}
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        metric = str(raw.get("metric", ""))
        if metric not in rows:
            continue
        latency_s = _numeric(raw.get("latency_s"))
        if latency_s is None:
            continue
        time_value = raw.get("time")
        if not time_value:
            continue
        rows[metric].append(
            {
                "Time": time_value,
                "Latency (s)": latency_s,
                "Series": side.label,
                "Task": str(raw.get("task_id", "")),
                "Attempt": int(raw.get("attempt", 0) or 0),
                "Call": int(raw.get("call", 0) or 0),
            }
        )
    for metric_rows in rows.values():
        metric_rows.sort(key=lambda row: str(row["Time"]))
    return rows


def _side_latency_curve_rows(side: DashboardSide) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {"search": [], "extract": [], "model": []}
    for task_id, attempts in side.index.items():
        for attempt in _sorted_attempts(attempts):
            data = _load_json_cached(attempts[attempt])
            if side.kind == "original":
                _add_original_latency_rows(rows, side, task_id, attempt, data)
            else:
                _add_agentic_latency_rows(rows, side, task_id, attempt, data)

    for metric_rows in rows.values():
        metric_rows.sort(key=lambda row: row["Time"])
    return rows


def _render_ten_minute_latency_curves_sides(
    left: DashboardSide,
    right: DashboardSide,
    *,
    left_payload: dict[str, Any] | None = None,
    right_payload: dict[str, Any] | None = None,
) -> None:
    st.subheader("10-minute Average Latency")
    left_rows = _payload_latency_curve_rows(left, left_payload) or _side_latency_curve_rows(left)
    right_rows = _payload_latency_curve_rows(right, right_payload) or _side_latency_curve_rows(right)
    chart_specs = [
        ("Search Tool", left_rows["search"] + right_rows["search"]),
        ("Extraction Tool", left_rows["extract"] + right_rows["extract"]),
        ("Model Client", left_rows["model"] + right_rows["model"]),
    ]
    cols = st.columns(3)
    for idx, (title, rows) in enumerate(chart_specs):
        with cols[idx]:
            _render_latency_curve(title, rows)
