# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Trace distribution tab for benchmark dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import altair as alt
import pandas as pd
import streamlit as st

from recipe.common.log_processing import load_dashboard_summary
from recipe.common.log_processing.trace_distributions import (
    GROUP_LABELS,
    GROUP_ORDER,
    METRIC_LABELS,
    SCHEMA_VERSION,
    AttemptScope,
    TraceKind,
    build_trace_distribution_payload,
    load_llm_judge_rows,
    load_trace_distribution_artifact,
    scan_trace_index,
)
from recipe.dashboard.sides import DashboardSide, side_run_title

_DONE_REASON_COLORS = [
    "#17BECF",
    "#9467BD",
    "#E377C2",
    "#AEC7E8",
    "#FFBB78",
    "#98DF8A",
    "#C5B0D5",
    "#8C564B",
    "#7F7F7F",
    "#BCBD22",
]


def _trace_kind(side: DashboardSide) -> TraceKind:
    return "original" if side.kind == "original" else "agentic"


def _trace_dir_names(side: DashboardSide) -> tuple[str, ...]:
    if side.kind == "original":
        return ()
    if side.agentic_trace_dir:
        return (side.agentic_trace_dir,)
    if side.run_type == "web_search":
        return ("web-search-benchmark",)
    return ("wide-search",)


@st.cache_data(ttl=30, show_spinner=False)
def _build_trace_distribution_cached(
    trace_index: dict[str, dict[int, str]],
    llm_evals: dict[str, dict[str, Any]],
    *,
    kind: TraceKind,
    attempt_scope: AttemptScope,
) -> dict[str, Any]:
    return build_trace_distribution_payload(trace_index, llm_evals, kind=kind, attempt_scope=attempt_scope)


@st.cache_data(ttl=30, show_spinner=False)
def _load_dashboard_summary_cached(run_dir: str) -> dict[str, Any] | None:
    return load_dashboard_summary(run_dir)


@st.cache_data(ttl=300, show_spinner=False)
def _load_trace_distribution_artifact_cached(run_dir: str) -> dict[str, Any] | None:
    return load_trace_distribution_artifact(run_dir)


@st.cache_data(ttl=30, show_spinner=False)
def _scan_trace_index_cached(run_dir: str, *, kind: TraceKind, trace_dir_names: tuple[str, ...]) -> dict[str, dict[int, str]]:
    return scan_trace_index(run_dir, kind=kind, trace_dir_names=trace_dir_names)


@st.cache_data(ttl=30, show_spinner=False)
def _load_llm_judge_rows_cached(run_dir: str) -> dict[str, dict[str, Any]]:
    return load_llm_judge_rows(run_dir)


def _payload_for_side(side: DashboardSide, *, attempt_scope: AttemptScope) -> tuple[dict[str, Any] | None, str]:
    precomputed = _precomputed_payload_for_side(side, attempt_scope=attempt_scope)
    if precomputed is not None:
        return precomputed, "precomputed artifact"
    live = _live_payload_for_side(side, attempt_scope=attempt_scope)
    return live, "live trace scan"


def _precomputed_payload_for_side(side: DashboardSide, *, attempt_scope: AttemptScope) -> dict[str, Any] | None:
    if side.run is None:
        return None
    run_dir = str(side.run)
    summary = _load_dashboard_summary_cached(run_dir)
    if not _is_completed_summary(summary):
        return None
    artifact = _load_trace_distribution_artifact_cached(run_dir)
    if not isinstance(artifact, dict):
        return None
    if artifact.get("schema_version") != SCHEMA_VERSION:
        return None
    if not _same_run_dir(artifact.get("run_dir"), run_dir):
        return None
    payloads = artifact.get("payloads")
    if not isinstance(payloads, dict):
        return None
    payload = payloads.get(attempt_scope)
    if not isinstance(payload, dict):
        return None
    if payload.get("kind") != _trace_kind(side) or payload.get("attempt_scope") != attempt_scope:
        return None
    if payload.get("schema_version") != SCHEMA_VERSION:
        return None
    if not _payload_has_current_metrics(payload):
        return None
    return payload


def _same_run_dir(artifact_run_dir: Any, selected_run_dir: str) -> bool:
    if not isinstance(artifact_run_dir, str) or not artifact_run_dir:
        return False
    if artifact_run_dir == selected_run_dir:
        return True
    try:
        return Path(artifact_run_dir).resolve() == Path(selected_run_dir).resolve()
    except OSError:
        return False


def _payload_has_current_metrics(payload: dict[str, Any]) -> bool:
    groups = payload.get("groups")
    if not isinstance(groups, dict):
        return False
    all_group = groups.get("all")
    if not isinstance(all_group, dict):
        return False
    return all(metric in all_group for metric in ("discard_all_resets", "discard_all_reset_turns", "hidden_turns"))


def _live_payload_for_side(side: DashboardSide, *, attempt_scope: AttemptScope) -> dict[str, Any] | None:
    trace_index = side.index
    llm_evals = side.llm_evals
    if not trace_index and side.run is not None:
        trace_index = _scan_trace_index_cached(str(side.run), kind=_trace_kind(side), trace_dir_names=_trace_dir_names(side))
    if not llm_evals and side.run is not None:
        llm_evals = _load_llm_judge_rows_cached(str(side.run))
    if not trace_index:
        return None
    return _build_trace_distribution_cached(trace_index, llm_evals, kind=_trace_kind(side), attempt_scope=attempt_scope)


def _is_completed_summary(summary: dict[str, Any] | None) -> bool:
    if not isinstance(summary, dict):
        return False
    progress_ratio = _numeric_float(summary.get("progress_ratio"))
    if progress_ratio is not None:
        return progress_ratio >= 1.0
    expected = _numeric_int(summary.get("expected_tasks"))
    if expected is None or expected <= 0:
        return False
    llm_summary = summary.get("llm_judge")
    judged = _numeric_int(llm_summary.get("judged_task_count")) if isinstance(llm_summary, dict) else None
    return judged is not None and judged >= expected


def _numeric_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _numeric_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _distribution_rows(payload: dict[str, Any], metric: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = payload.get("groups", {})
    if not isinstance(groups, dict):
        return rows
    for group in GROUP_ORDER:
        group_payload = groups.get(group, {})
        if not isinstance(group_payload, dict):
            continue
        metric_payload = group_payload.get(metric, {})
        counts = metric_payload.get("counts") if isinstance(metric_payload, dict) else []
        if not isinstance(counts, list):
            continue
        for item in counts:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "Group": GROUP_LABELS.get(group, group),
                    "Value": item.get("value"),
                    "Count": int(item.get("count", 0) or 0),
                }
            )
    return rows


def _done_reason_order_for_payloads(payloads: list[dict[str, Any] | None]) -> list[str]:
    values: set[str] = set()
    for payload in payloads:
        if payload is None:
            continue
        for row in _distribution_rows(payload, "done_reason"):
            values.add(str(row["Value"]))
    return sorted(values)


def _cumulative_percentage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(str(row["Group"]), []).append(row)

    cumulative_rows: list[dict[str, Any]] = []
    for group, group_rows in by_group.items():
        sorted_rows = sorted(group_rows, key=lambda row: float(row["Value"]))
        total = sum(int(row["Count"]) for row in sorted_rows)
        running = 0
        for row in sorted_rows:
            running += int(row["Count"])
            cumulative_rows.append(
                {
                    "Group": group,
                    "Value": row["Value"],
                    "Cumulative %": 0.0 if total == 0 else running / total * 100.0,
                    "Cumulative Count": running,
                    "Total": total,
                }
            )
    return cumulative_rows


def _summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = payload.get("groups", {})
    if not isinstance(groups, dict):
        return rows
    for group in GROUP_ORDER:
        group_payload = groups.get(group, {})
        if not isinstance(group_payload, dict):
            continue
        done_counts = (group_payload.get("done_reason") or {}).get("counts") or []
        top_done = "-"
        if isinstance(done_counts, list) and done_counts:
            top = max((item for item in done_counts if isinstance(item, dict)), key=lambda item: int(item.get("count", 0) or 0), default=None)
            if top:
                top_done = f"{top.get('value')} ({top.get('count')})"
        visible_summary = (group_payload.get("visible_turns") or {}).get("summary") or {}
        actual_summary = (group_payload.get("actual_turns") or {}).get("summary") or {}
        attempts_summary = (group_payload.get("actual_attempts") or {}).get("summary") or {}
        discard_summary = (group_payload.get("discard_all_resets") or {}).get("summary") or {}
        reset_turn_summary = (group_payload.get("discard_all_reset_turns") or {}).get("summary") or {}
        hidden_summary = (group_payload.get("hidden_turns") or {}).get("summary") or {}
        rows.append(
            {
                "Group": GROUP_LABELS.get(group, group),
                "Traces": int(group_payload.get("count", 0) or 0),
                "Top done_reason": top_done,
                "Visible turns mean": _fmt_float(visible_summary.get("mean")),
                "Visible turns median": _fmt_float(visible_summary.get("median")),
                "Actual turns mean": _fmt_float(actual_summary.get("mean")),
                "Actual turns median": _fmt_float(actual_summary.get("median")),
                "Actual attempts mean": _fmt_float(attempts_summary.get("mean")),
                "Actual attempts median": _fmt_float(attempts_summary.get("median")),
                "Discard traces": _positive_count(group_payload.get("discard_all_resets")),
                "Discard resets mean": _fmt_float(discard_summary.get("mean")),
                "Discard resets max": _fmt_float(discard_summary.get("max")),
                "Reset visible turns median": _fmt_float(reset_turn_summary.get("median")),
                "Reset visible turns max": _fmt_float(reset_turn_summary.get("max")),
                "Hidden turns mean": _fmt_float(hidden_summary.get("mean")),
                "Hidden turns median": _fmt_float(hidden_summary.get("median")),
            }
        )
    return rows


def _positive_count(metric_payload: Any) -> int:
    counts = metric_payload.get("counts") if isinstance(metric_payload, dict) else None
    if not isinstance(counts, list):
        return 0
    total = 0
    for item in counts:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            continue
        total += int(item.get("count", 0) or 0)
    return total


def _fmt_float(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.2f}"
    return "-"


def _render_metric_distribution(payload: dict[str, Any], *, metric: str, done_reason_order: list[str]) -> None:
    rows = _distribution_rows(payload, metric)
    if not rows:
        st.caption("No trace distribution data.")
        return
    df = pd.DataFrame(rows)
    if metric == "done_reason":
        _render_done_reason_pies(rows, reason_order=done_reason_order)
        st.dataframe(df.astype(str), width="stretch", hide_index=True)
        return

    cumulative_rows = _cumulative_percentage_rows(rows)
    cumulative_df = pd.DataFrame(cumulative_rows)
    chart = (
        alt.Chart(cumulative_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Value:Q", title=METRIC_LABELS.get(metric, metric), sort=None),
            y=alt.Y("Cumulative %:Q", title="Cumulative percentage", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Group:N", sort=[GROUP_LABELS[group] for group in GROUP_ORDER]),
            tooltip=["Group", "Value", alt.Tooltip("Cumulative %:Q", format=".1f"), "Cumulative Count", "Total"],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width="stretch")
    st.dataframe(df.astype(str), width="stretch", hide_index=True)


def _render_done_reason_pies(rows: list[dict[str, Any]], *, reason_order: list[str]) -> None:
    rows_by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_group.setdefault(str(row["Group"]), []).append(row)

    color_scale = alt.Scale(
        domain=reason_order,
        range=[_DONE_REASON_COLORS[idx % len(_DONE_REASON_COLORS)] for idx in range(len(reason_order))],
    )
    ordered_group_labels = [GROUP_LABELS[group] for group in GROUP_ORDER]
    for start in range(0, len(ordered_group_labels), 2):
        cols = st.columns(2)
        for col, group_label in zip(cols, ordered_group_labels[start : start + 2], strict=False):
            with col:
                group_rows = rows_by_group.get(group_label, [])
                st.caption(group_label)
                if not group_rows:
                    st.caption("No data.")
                    continue
                total = sum(int(row["Count"]) for row in group_rows)
                chart_rows = [
                    {
                        **row,
                        "Percent": 0.0 if total == 0 else int(row["Count"]) / total * 100.0,
                    }
                    for row in group_rows
                ]
                chart = (
                    alt.Chart(pd.DataFrame(chart_rows))
                    .mark_arc(outerRadius=68)
                    .encode(
                        theta=alt.Theta("Count:Q"),
                        color=alt.Color("Value:N", scale=color_scale, legend=None),
                        tooltip=["Value", "Count", alt.Tooltip("Percent:Q", format=".1f")],
                    )
                    .properties(width=180, height=180)
                )
                st.altair_chart(chart, width=180)
    _render_done_reason_legend(reason_order, color_scale)


def _render_done_reason_legend(reason_order: list[str], color_scale: alt.Scale) -> None:
    if not reason_order:
        return
    legend_df = pd.DataFrame({"Value": reason_order, "x": [0] * len(reason_order), "y": [0] * len(reason_order)})
    legend = (
        alt.Chart(legend_df)
        .mark_point(opacity=0)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            color=alt.Color(
                "Value:N",
                scale=color_scale,
                legend=alt.Legend(title="done_reason", orient="bottom", columns=2),
            ),
        )
        .properties(height=1)
    )
    st.altair_chart(legend, width="stretch")


def _render_side_distribution(
    side: DashboardSide,
    *,
    payload: dict[str, Any] | None,
    source: str,
    done_reason_order: list[str],
) -> None:
    st.subheader(side_run_title(side))
    if payload is None:
        st.info("No traces found for this side.")
        return
    st.caption(f"Source: {source}")
    groups = payload.get("groups", {})
    all_count = int(groups.get("all", {}).get("count", 0) if isinstance(groups, dict) else 0)
    correct_count = int(groups.get("correct", {}).get("count", 0) if isinstance(groups, dict) else 0)
    wrong_count = int(groups.get("wrong", {}).get("count", 0) if isinstance(groups, dict) else 0)
    empty_count = int(groups.get("empty", {}).get("count", 0) if isinstance(groups, dict) else 0)
    cols = st.columns(4)
    cols[0].metric("All", all_count)
    cols[1].metric("Correct", correct_count)
    cols[2].metric("Wrong", wrong_count)
    cols[3].metric("Empty", empty_count)
    st.dataframe(pd.DataFrame(_summary_rows(payload)).astype(str), width="stretch", hide_index=True)

    metrics = (
        "done_reason",
        "visible_turns",
        "actual_turns",
        "actual_attempts",
        "discard_all_resets",
        "discard_all_reset_turns",
        "hidden_turns",
    )
    metric_tabs = st.tabs([METRIC_LABELS[metric] for metric in metrics])
    for tab, metric in zip(metric_tabs, metrics, strict=True):
        with tab:
            _render_metric_distribution(payload, metric=metric, done_reason_order=done_reason_order)


def _render_trace_distribution_tab(left: DashboardSide, right: DashboardSide) -> None:
    st.header("Trace Distributions")
    st.caption(
        "Correct/wrong groups use numeric `llm_judge_score`; pending judge rows appear only in All unless their output is empty. "
        "Visible turns reconstruct runtime visibility by applying rollback, compaction, and discard-all markers."
    )
    attempt_scope_label = st.radio(
        "Trace scope",
        ["Latest attempt per task", "All attempts"],
        horizontal=True,
        key="trace_distribution_attempt_scope",
    )
    attempt_scope = cast("AttemptScope", "latest" if attempt_scope_label == "Latest attempt per task" else "all")
    left_payload, left_source = _payload_for_side(left, attempt_scope=attempt_scope)
    right_payload, right_source = _payload_for_side(right, attempt_scope=attempt_scope)
    done_reason_order = _done_reason_order_for_payloads([left_payload, right_payload])

    left_col, right_col = st.columns(2)
    with left_col:
        _render_side_distribution(left, payload=left_payload, source=left_source, done_reason_order=done_reason_order)
    with right_col:
        _render_side_distribution(right, payload=right_payload, source=right_source, done_reason_order=done_reason_order)
