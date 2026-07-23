# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Assistant-message statistics tab for the benchmark dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import altair as alt
import pandas as pd
import streamlit as st

from recipe.common.log_processing.assistant_message_stats import (
    ARTIFACT_NAME,
    ASSISTANT_OUTPUT_LENGTH_DISTRIBUTION_KEYS,
    TRACE_TOOL_CALL_CATEGORY_LABELS,
    TRACE_TOOL_CALL_CATEGORY_ORDER,
    VALIDITY_LABELS,
    VALIDITY_ORDER,
    AttemptScope,
    build_assistant_message_stats_payload,
    load_assistant_message_stats_artifact,
)
from recipe.common.log_processing.trace_distributions import scan_trace_index
from recipe.dashboard.sides import DashboardSide, side_run_title


def _trace_dir_names(side: DashboardSide) -> tuple[str, ...]:
    if side.agentic_trace_dir:
        return (side.agentic_trace_dir,)
    if side.run_type == "web_search":
        return ("web-search-benchmark",)
    return ("wide-search",)


@st.cache_data(ttl=30, show_spinner=False)
def _build_assistant_message_stats_cached(
    trace_index: dict[str, dict[int, str]],
    *,
    attempt_scope: AttemptScope,
) -> dict[str, Any]:
    return build_assistant_message_stats_payload(trace_index, attempt_scope=attempt_scope)


@st.cache_data(ttl=300, show_spinner=False)
def _load_assistant_message_stats_artifact_cached(run_dir: str, artifact_mtime_ns: int) -> dict[str, Any] | None:
    _ = artifact_mtime_ns
    return load_assistant_message_stats_artifact(run_dir)


@st.cache_data(ttl=30, show_spinner=False)
def _scan_agentic_trace_index_cached(
    run_dir: str,
    *,
    trace_dir_names: tuple[str, ...],
    source_signature: tuple[int, int],
) -> dict[str, dict[int, str]]:
    _ = source_signature
    return scan_trace_index(run_dir, kind="agentic", trace_dir_names=trace_dir_names)


def _payload_for_side(side: DashboardSide, *, attempt_scope: AttemptScope) -> tuple[dict[str, Any] | None, str]:
    if side.kind != "agentic":
        return None, "unsupported"
    precomputed = _precomputed_payload_for_side(side, attempt_scope=attempt_scope)
    if precomputed is not None:
        return precomputed, "precomputed artifact"
    live = _live_payload_for_side(side, attempt_scope=attempt_scope)
    return live, "live trace scan"


def _precomputed_payload_for_side(side: DashboardSide, *, attempt_scope: AttemptScope) -> dict[str, Any] | None:
    if side.run is None:
        return None
    run_dir = str(side.run)
    trace_dir_names = _trace_dir_names(side)
    artifact_mtime_ns = _artifact_mtime_ns(side.run)
    if artifact_mtime_ns is None:
        return None
    artifact = _load_assistant_message_stats_artifact_cached(run_dir, artifact_mtime_ns)
    if not isinstance(artifact, dict):
        return None
    if not _same_run_dir(artifact.get("run_dir"), run_dir):
        return None
    if not _same_trace_dir_names(artifact.get("trace_dir_names"), trace_dir_names):
        return None
    source_signature = _trace_source_signature(side.run, trace_dir_names)
    if not _artifact_is_current(artifact_mtime_ns, source_signature):
        return None
    payloads = artifact.get("payloads")
    if not isinstance(payloads, dict):
        return None
    payload = payloads.get(attempt_scope)
    if not isinstance(payload, dict):
        return None
    if payload.get("kind") != "agentic" or payload.get("attempt_scope") != attempt_scope:
        return None
    if not _payload_has_assistant_output_length_distributions(payload):
        return None
    return payload


def _live_payload_for_side(side: DashboardSide, *, attempt_scope: AttemptScope) -> dict[str, Any] | None:
    trace_index = side.index
    if not trace_index and side.run is not None:
        trace_dir_names = _trace_dir_names(side)
        trace_index = _scan_agentic_trace_index_cached(
            str(side.run),
            trace_dir_names=trace_dir_names,
            source_signature=_trace_source_signature(side.run, trace_dir_names),
        )
    if not trace_index:
        return None
    return _build_assistant_message_stats_cached(trace_index, attempt_scope=attempt_scope)


def _artifact_mtime_ns(run_dir: Path) -> int | None:
    try:
        return (run_dir / ARTIFACT_NAME).stat().st_mtime_ns
    except OSError:
        return None


def _trace_source_signature(run_dir: Path, trace_dir_names: tuple[str, ...]) -> tuple[int, int]:
    count = 0
    latest_mtime_ns = 0
    for name in trace_dir_names:
        trace_dir = run_dir / name
        if not trace_dir.exists():
            continue
        for path in trace_dir.glob("*.json"):
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size == 0:
                continue
            count += 1
            latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
    return count, latest_mtime_ns


def _artifact_is_current(artifact_mtime_ns: int, source_signature: tuple[int, int]) -> bool:
    _trace_count, latest_trace_mtime_ns = source_signature
    return latest_trace_mtime_ns <= artifact_mtime_ns


def _same_trace_dir_names(artifact_trace_dir_names: Any, selected_trace_dir_names: tuple[str, ...]) -> bool:
    if not isinstance(artifact_trace_dir_names, list):
        return False
    return tuple(str(name) for name in artifact_trace_dir_names) == selected_trace_dir_names


def _same_run_dir(artifact_run_dir: Any, selected_run_dir: str) -> bool:
    if not isinstance(artifact_run_dir, str) or not artifact_run_dir:
        return False
    if artifact_run_dir == selected_run_dir:
        return True
    try:
        return Path(artifact_run_dir).resolve() == Path(selected_run_dir).resolve()
    except OSError:
        return False


def _count_rows(rows: Any, *, order: tuple[str, ...], labels: dict[str, str]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            category = row.get("category")
            count = row.get("count")
            if isinstance(category, str) and isinstance(count, (int, float)) and not isinstance(count, bool):
                counts[category] = int(count)
    output = [{"Category": labels.get(category, category), "Count": counts.get(category, 0)} for category in order]
    extras = sorted(category for category in counts if category not in set(order))
    output.extend({"Category": labels.get(category, category), "Count": counts[category]} for category in extras)
    total = sum(row["Count"] for row in output)
    for row in output:
        row["Percent"] = 0.0 if total == 0 else row["Count"] / total * 100.0
    return output


def _value_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    output: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get("value")
        count = row.get("count")
        if isinstance(count, (int, float)) and not isinstance(count, bool):
            output.append({"Value": value, "Count": int(count)})
    return output


def _payload_has_assistant_output_length_distributions(payload: dict[str, Any]) -> bool:
    length = payload.get("assistant_output_length")
    if not isinstance(length, dict):
        return False
    distributions = length.get("distributions")
    if not isinstance(distributions, dict):
        return False
    return all(isinstance(distributions.get(key), list) for key in ASSISTANT_OUTPUT_LENGTH_DISTRIBUTION_KEYS)


def _assistant_output_average_chars(payload: dict[str, Any]) -> float | None:
    length = payload.get("assistant_output_length")
    if not isinstance(length, dict):
        return None
    average = length.get("average")
    if isinstance(average, bool) or not isinstance(average, (int, float)):
        return None
    return float(average)


def _format_average_chars(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:,.1f}"


def _assistant_output_length_cdf_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    length = payload.get("assistant_output_length")
    distributions = length.get("distributions") if isinstance(length, dict) else None
    if not isinstance(distributions, dict):
        return []
    series_specs = (
        ("all", "All messages"),
        ("with_tool_calls", "With tool calls"),
        ("without_tool_calls", "Without tool calls"),
    )
    rows: list[dict[str, Any]] = []
    for key, label in series_specs:
        distribution = _value_rows(distributions.get(key, []))
        total = sum(row["Count"] for row in distribution)
        if total <= 0:
            continue
        cumulative = 0
        for row in sorted(distribution, key=lambda item: int(item["Value"] or 0)):
            value = row.get("Value")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            cumulative += int(row["Count"])
            rows.append(
                {
                    "Length": int(value),
                    "Percent": cumulative / total * 100.0,
                    "Series": label,
                    "Messages": cumulative,
                    "Total": total,
                }
            )
    return rows


def _render_assistant_output_length_curve(payload: dict[str, Any]) -> None:
    st.markdown("**Assistant Output Length CDF**")
    rows = _assistant_output_length_cdf_rows(payload)
    df = pd.DataFrame(rows)
    if df.empty:
        st.caption("No assistant output length distribution available.")
        return
    chart = (
        alt.Chart(df)
        .mark_line(interpolate="step-after")
        .encode(
            x=alt.X("Length:Q", title="Assistant message length (chars)"),
            y=alt.Y("Percent:Q", title="Messages <= length (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Series:N", title=None),
            tooltip=[
                "Series:N",
                alt.Tooltip("Length:Q", format=",d"),
                alt.Tooltip("Percent:Q", format=".1f"),
                alt.Tooltip("Messages:Q", format=",d"),
                alt.Tooltip("Total:Q", format=",d"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")


def _render_count_chart(title: str, rows: list[dict[str, Any]]) -> None:
    st.markdown(f"**{title}**")
    df = pd.DataFrame(rows)
    if df.empty or int(df["Count"].sum()) <= 0:
        st.caption("No data.")
        st.dataframe(df.astype(str), width="stretch", hide_index=True)
        return
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Count:Q", title="Messages"),
            y=alt.Y("Category:N", sort=list(df["Category"]), title=None),
            tooltip=["Category:N", "Count:Q", alt.Tooltip("Percent:Q", format=".1f")],
        )
        .properties(height=max(180, min(360, 34 * len(df))))
    )
    st.altair_chart(chart, width="stretch")
    st.dataframe(df.assign(Percent=df["Percent"].map(lambda value: f"{value:.1f}%")).astype(str), width="stretch", hide_index=True)


def _render_value_distribution(title: str, rows: list[dict[str, Any]], *, x_title: str) -> None:
    st.markdown(f"**{title}**")
    df = pd.DataFrame(rows)
    if df.empty or int(df["Count"].sum()) <= 0:
        st.caption("No data.")
        st.dataframe(df.astype(str), width="stretch", hide_index=True)
        return
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Value:Q", title=x_title),
            y=alt.Y("Count:Q", title="Traces"),
            tooltip=["Value", "Count"],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, width="stretch")
    st.dataframe(df.astype(str), width="stretch", hide_index=True)


def _render_validity(payload: dict[str, Any]) -> None:
    validity = payload.get("message_validity")
    rows = _count_rows(validity.get("counts", []) if isinstance(validity, dict) else [], order=VALIDITY_ORDER, labels=VALIDITY_LABELS)
    _render_count_chart("Validity Composition", rows)


def _render_tool_calls(payload: dict[str, Any]) -> None:
    tool_call_messages = payload.get("tool_call_messages")
    message_level = tool_call_messages.get("message_level", {}) if isinstance(tool_call_messages, dict) else {}
    trace_level = tool_call_messages.get("trace_level", {}) if isinstance(tool_call_messages, dict) else {}

    cols = st.columns(2)
    with cols[0]:
        message_rows = _count_rows(
            message_level.get("counts", []),
            order=("single_tool_call", "multiple_tool_calls"),
            labels={"single_tool_call": "Single tool call", "multiple_tool_calls": "Multiple tool calls"},
        )
        _render_count_chart("Message Level", message_rows)
    with cols[1]:
        trace_rows = _count_rows(
            trace_level.get("counts", []),
            order=TRACE_TOOL_CALL_CATEGORY_ORDER,
            labels=TRACE_TOOL_CALL_CATEGORY_LABELS,
        )
        _render_count_chart("Trace Level", trace_rows)

    dist_tabs = st.tabs(["Single per trace", "Multi per trace", "All per trace", "Max per message"])
    distributions = [
        ("Single-call assistant messages per trace", trace_level.get("single_tool_call_messages_per_trace", []), "Single-call messages"),
        ("Multi-call assistant messages per trace", trace_level.get("multi_tool_call_messages_per_trace", []), "Multi-call messages"),
        ("Tool-call assistant messages per trace", trace_level.get("tool_call_messages_per_trace", []), "Tool-call messages"),
        ("Max tool calls in one assistant message", trace_level.get("max_tool_calls_per_message_per_trace", []), "Max tool calls"),
    ]
    for tab, (title, rows, x_title) in zip(dist_tabs, distributions, strict=True):
        with tab:
            _render_value_distribution(title, _value_rows(rows), x_title=x_title)


def _render_records(payload: dict[str, Any]) -> None:
    records = payload.get("records", [])
    if not isinstance(records, list) or not records:
        st.caption("No trace records.")
        return
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        rows.append(
            {
                "task_id": record.get("task_id"),
                "attempt": record.get("attempt"),
                "assistant_messages": record.get("assistant_message_count"),
                "assistant_output_chars": record.get("assistant_output_chars"),
                "avg_assistant_output_chars": record.get("avg_assistant_output_chars"),
                "tool_call_messages": record.get("tool_call_message_count"),
                "single_tool_call_messages": record.get("single_tool_call_message_count"),
                "multi_tool_call_messages": record.get("multi_tool_call_message_count"),
                "max_tool_calls_in_message": record.get("max_tool_calls_in_message"),
                "trace_tool_call_category": record.get("trace_tool_call_category"),
            }
        )
    st.dataframe(pd.DataFrame(rows).astype(str), width="stretch", hide_index=True)


def _render_side(side: DashboardSide, *, payload: dict[str, Any] | None, source: str) -> None:
    st.subheader(side_run_title(side))
    if side.kind != "agentic":
        st.info("Assistant-message stats are available for agentic and web_search runs.")
        return
    if payload is None:
        st.info("No agentic traces found for this side.")
        return
    st.caption(f"Source: {source}")
    tool_call_messages = payload.get("tool_call_messages")
    trace_level = tool_call_messages.get("trace_level", {}) if isinstance(tool_call_messages, dict) else {}
    message_level = tool_call_messages.get("message_level", {}) if isinstance(tool_call_messages, dict) else {}
    trace_counts = _count_rows(trace_level.get("counts", []), order=TRACE_TOOL_CALL_CATEGORY_ORDER, labels=TRACE_TOOL_CALL_CATEGORY_LABELS)
    multi_trace_count = next((row["Count"] for row in trace_counts if row["Category"] == TRACE_TOOL_CALL_CATEGORY_LABELS["multi_tool_call_trace"]), 0)
    cols = st.columns(5)
    cols[0].metric("Traces", int(payload.get("trace_count", 0) or 0))
    cols[1].metric("Assistant messages", int(payload.get("assistant_message_count", 0) or 0))
    cols[2].metric("Avg output chars", _format_average_chars(_assistant_output_average_chars(payload)))
    cols[3].metric("Tool-call messages", int(message_level.get("total", 0) or 0) if isinstance(message_level, dict) else 0)
    cols[4].metric("Multi-call traces", multi_trace_count)

    tabs = st.tabs(["Length CDF", "Validity", "Tool Calls", "Trace Records"])
    with tabs[0]:
        _render_assistant_output_length_curve(payload)
    with tabs[1]:
        _render_validity(payload)
    with tabs[2]:
        _render_tool_calls(payload)
    with tabs[3]:
        _render_records(payload)


def _render_assistant_message_tab(left: DashboardSide, right: DashboardSide) -> None:
    st.header("Assistant Message")
    attempt_scope_label = st.radio(
        "Trace scope",
        ["Latest attempt per task", "All attempts"],
        horizontal=True,
        key="assistant_message_attempt_scope",
    )
    attempt_scope = cast("AttemptScope", "latest" if attempt_scope_label == "Latest attempt per task" else "all")
    left_payload, left_source = _payload_for_side(left, attempt_scope=attempt_scope)
    right_payload, right_source = _payload_for_side(right, attempt_scope=attempt_scope)

    left_col, right_col = st.columns(2)
    with left_col:
        _render_side(left, payload=left_payload, source=left_source)
    with right_col:
        _render_side(right, payload=right_payload, source=right_source)
