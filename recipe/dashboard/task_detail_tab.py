# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Task-detail tab renderer for the benchmark dashboard."""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from recipe.dashboard.constants import MODEL_CLIENT_COLOR, OTHER_TOOL_COLOR, USAGE_CHART_COLOR
from recipe.dashboard.discovery import _sorted_attempts
from recipe.dashboard.extraction import _build_exclusive_tables_by_tool, _build_query_comparison, _count_tool_calls_by_name
from recipe.dashboard.loading import _extract_step_timing, _load_json_cached
from recipe.dashboard.rendering import _render_side_messages, _render_side_tool_call_list, _render_wrapped_text
from recipe.dashboard.sides import (
    DashboardSide,
    load_side_summary,
    side_answer_correct,
    side_llm_correct,
    side_per_step_tokens,
    side_run_title,
    side_tool_calls,
)
from recipe.dashboard.timing_processing import (
    _AGN_TOOL_PIE_ORDER,
    _CAT_COLORS_BASE,
    _TOOL_COLORS,
    _TOOL_ORDER,
    _agn_time_breakdown_aggregated,
    _expand_agentic_tool_timing,
    _ori_time_breakdown,
)

_MCP_TOOL_RE = re.compile(
    r"<use_mcp_tool>\s*<server_name>(.*?)</server_name>\s*<tool_name>(.*?)</tool_name>\s*<arguments>\s*(.*?)\s*</arguments>\s*</use_mcp_tool>",
    re.DOTALL,
)


def _message_text(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _parse_tool_call_name(name: str) -> tuple[str, str]:
    if name.startswith("agent-"):
        split_at = name.rfind("-")
        if split_at > len("agent-"):
            return name[:split_at], name[split_at + 1 :]
        return name, ""
    if name.startswith("tool-"):
        parts = name.split("-", 2)
        if len(parts) == 3:
            return parts[1], parts[2]
    return "unknown", name


def _parse_structured_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        raw_name = str(function.get("name", ""))
        arguments: Any = function.get("arguments", {})
        if isinstance(arguments, str):
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                arguments = json.loads(arguments)
        server_name, tool_name = _parse_tool_call_name(raw_name)
        parsed.append(
            {
                "server_name": server_name,
                "tool_name": tool_name,
                "arguments": arguments,
                "id": tool_call.get("id", "") if isinstance(tool_call, dict) else "",
                "format": "structured",
            }
        )
    return parsed


def _parse_mcp_tool_calls(text: str) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for match in _MCP_TOOL_RE.finditer(text):
        raw_args = match.group(3).strip()
        try:
            arguments: Any = json.loads(raw_args)
        except (json.JSONDecodeError, ValueError):
            arguments = raw_args
        parsed.append(
            {
                "server_name": match.group(1).strip(),
                "tool_name": match.group(2).strip(),
                "arguments": arguments,
                "id": "",
                "format": "mcp",
            }
        )
    return parsed


def _sub_agent_sessions(data: dict[str, Any]) -> dict[str, Any]:
    sessions = data.get("browser_agent_message_history_sessions") or {}
    if not sessions:
        sessions = data.get("sub_agent_message_history_sessions") or {}
    return sessions if isinstance(sessions, dict) else {}


def _session_messages(session: Any) -> list[dict[str, Any]]:
    if isinstance(session, dict):
        messages = session.get("message_history", [])
        return messages if isinstance(messages, list) else []
    return []


def _side_main_flow_messages(side: DashboardSide, data: dict[str, Any]) -> list[dict[str, Any]]:
    if side.kind == "original":
        history = data.get("main_agent_message_history", {})
        if isinstance(history, dict):
            messages = history.get("message_history", [])
            return messages if isinstance(messages, list) else []
        return []
    return data.get("conversation", []) or []


def _execution_flow_steps(side: DashboardSide, data: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    sessions = _sub_agent_sessions(data)
    sub_agent_call_count = 0

    for idx, message in enumerate(_side_main_flow_messages(side, data)):
        content = _message_text(message.get("content"))
        tool_calls = _parse_structured_tool_calls(message)
        tool_calls.extend(_parse_mcp_tool_calls(content))
        session_id = None
        session_flow: list[dict[str, Any]] = []

        for tool_call in tool_calls:
            server_name = str(tool_call.get("server_name", ""))
            if server_name.startswith("agent-"):
                sub_agent_call_count += 1
                candidate = f"{server_name}_{sub_agent_call_count}"
                if candidate in sessions:
                    session_id = candidate
                    session_flow = _execution_flow_from_messages(_session_messages(sessions[candidate]), agent=session_id)
                    break
            elif server_name.startswith("browsing-agent"):
                sub_agent_call_count += 1
                candidate = f"browser_agent_{sub_agent_call_count}"
                if candidate in sessions:
                    session_id = candidate
                    session_flow = _execution_flow_from_messages(_session_messages(sessions[candidate]), agent=session_id)
                    break

        steps.append(
            {
                "step_id": idx,
                "agent": "main_agent",
                "role": message.get("role", "?"),
                "content": content,
                "tool_calls": tool_calls,
                "session_id": session_id,
                "session_flow": session_flow,
                "timestamp": message.get("timestamp", ""),
            }
        )
    return steps


def _execution_flow_from_messages(messages: list[dict[str, Any]], *, agent: str) -> list[dict[str, Any]]:
    flow: list[dict[str, Any]] = []
    for idx, message in enumerate(messages):
        content = _message_text(message.get("content"))
        tool_calls = _parse_structured_tool_calls(message)
        tool_calls.extend(_parse_mcp_tool_calls(content))
        flow.append(
            {
                "step_id": idx,
                "agent": agent,
                "role": message.get("role", "?"),
                "content": content,
                "tool_calls": tool_calls,
                "session_id": None,
                "session_flow": [],
                "timestamp": message.get("timestamp", ""),
            }
        )
    return flow


def _tool_call_label(tool_call: dict[str, Any]) -> str:
    server = tool_call.get("server_name") or "unknown"
    tool = tool_call.get("tool_name") or "unknown"
    fmt = tool_call.get("format") or "unknown"
    return f"{server}.{tool} ({fmt})"


def _render_flow_steps(steps: list[dict[str, Any]], *, prefix: str = "") -> None:
    if not steps:
        st.info("No flow messages.")
        return
    for step in steps:
        content = str(step.get("content", ""))
        preview = content[:100].replace("\n", " ") if content else "(empty)"
        label = f"{prefix}[{step['step_id']}] {step.get('agent', '?')} / {step.get('role', '?')}: {preview}"
        with st.expander(label, expanded=False):
            if step.get("timestamp"):
                st.caption(str(step["timestamp"]))
            _render_wrapped_text(content)
            tool_calls = step.get("tool_calls") or []
            if tool_calls:
                st.markdown("**Tool Calls**")
                for tool_idx, tool_call in enumerate(tool_calls):
                    st.markdown(f"`{tool_idx}` `{_tool_call_label(tool_call)}`")
                    st.json(tool_call.get("arguments", {}))
            if step.get("session_id"):
                st.markdown(f"**Nested Session:** `{step['session_id']}`")
                _render_flow_steps(step.get("session_flow") or [], prefix=f"{step['session_id']} ")


def _side_time_breakdown(side: DashboardSide, data_list: list[dict[str, Any]]) -> tuple[dict[str, Any], list[tuple[str, float]], dict[str, str]]:
    if side.kind == "original":
        bd = _ori_time_breakdown(data_list)
        items: list[tuple[str, float]] = []
        colors: dict[str, str] = {}
        for label in _TOOL_ORDER:
            ms = (bd.get("per_tool") or {}).get(label, 0)
            if ms > 0:
                items.append((label, ms))
                colors[label] = _TOOL_COLORS.get(label, OTHER_TOOL_COLOR)
        if bd.get("model_client_ms", 0) > 0:
            items.append(("Model Client (estimated)", bd["model_client_ms"]))
            colors["Model Client (estimated)"] = _CAT_COLORS_BASE["Model Client"]
        if bd.get("other_ms", 0) > 0:
            items.append(("Other / overhead", bd["other_ms"]))
            colors["Other / overhead"] = _CAT_COLORS_BASE["Other"]
        return bd, items, colors

    bd = _agn_time_breakdown_aggregated(data_list)
    items = [(label, ms) for label, ms in (bd.get("merged") or {}).items() if round(ms / 1000, 1) > 0]
    colors = {label: (bd.get("merged_colors") or {}).get(label, OTHER_TOOL_COLOR) for label, _ in items}
    return bd, items, colors


def _render_side_time_breakdown_pie(side: DashboardSide, data_list: list[dict[str, Any]]) -> None:
    import altair as alt

    st.markdown(f"#### {side.label}")
    if not data_list:
        st.info("No log.")
        return
    bd, items, colors = _side_time_breakdown(side, data_list)
    total_ms = bd.get("total_ms", 0)
    if total_ms <= 0:
        st.info("No timing data.")
        return
    cols = st.columns(min(len(items) + 1, 8))
    for i, (label, ms) in enumerate(items):
        if i < len(cols) - 1:
            cols[i].metric(label, f"{ms / 1000:.1f}s", f"{ms / total_ms * 100:.0f}%")
    cols[min(len(items), len(cols) - 1)].metric("Total", f"{total_ms / 1000:.1f}s")

    pie_data = [{"Category": label, "Time (s)": ms / 1000, "color": colors.get(label, OTHER_TOOL_COLOR)} for label, ms in items]
    if not pie_data:
        return
    df_pie = pd.DataFrame(pie_data)
    chart = (
        alt.Chart(df_pie)
        .mark_arc(innerRadius=40)
        .encode(
            theta=alt.Theta("Time (s):Q"),
            color=alt.Color("Category:N", scale=alt.Scale(domain=[r["Category"] for r in pie_data], range=[r["color"] for r in pie_data])),
            tooltip=["Category:N", alt.Tooltip("Time (s):Q", format=".1f")],
        )
        .properties(height=250)
    )
    st.altair_chart(chart, width="stretch")


def _render_agentic_turn_timing_chart(side: DashboardSide, data_list: list[dict[str, Any]]) -> None:
    import altair as alt

    st.markdown(f"**{side.label}**")
    if side.kind != "agentic" or not data_list:
        st.caption("No agentic timing data.")
        return

    bar_rows = []
    attempt_boundaries: list[float] = []
    global_turn = 0
    bd = _agn_time_breakdown_aggregated(data_list)
    for att_idx, data in enumerate(data_list):
        timing = _extract_step_timing(data.get("steps", []))
        timed_steps = [s for s in timing["steps"] if s["elapsed_ms"] is not None]
        tool_exec_queue = [_expand_agentic_tool_timing(tc) for tc in data.get("tool_calls", [])]
        tool_exec_idx = 0
        seen_turns: dict[int, int] = {}
        for step in timed_steps:
            orig_turn = step["turn"]
            if orig_turn not in seen_turns:
                seen_turns[orig_turn] = global_turn
                global_turn += 1
            cat = step["category"]
            if cat == "Tool Execution" and tool_exec_idx < len(tool_exec_queue):
                for label, phase_ms in tool_exec_queue[tool_exec_idx].items():
                    bar_rows.append(
                        {
                            "turn": seen_turns[orig_turn],
                            "step_name": step["step_name"],
                            "category": label,
                            "elapsed_ms": phase_ms,
                            "attempt": att_idx + 1,
                        }
                    )
                tool_exec_idx += 1
                continue
            bar_rows.append(
                {
                    "turn": seen_turns[orig_turn],
                    "step_name": step["step_name"],
                    "category": cat,
                    "elapsed_ms": step["elapsed_ms"],
                    "attempt": att_idx + 1,
                }
            )
        if att_idx < len(data_list) - 1 and global_turn > 0:
            attempt_boundaries.append(global_turn - 0.5)

    if not bar_rows:
        st.caption("No per-turn timing data.")
        return

    df = pd.DataFrame(bar_rows)
    keep_cats = {*_AGN_TOOL_PIE_ORDER, "Model Client"}
    df = df[df["category"].isin(keep_cats)]
    if df.empty:
        st.caption("No per-turn timing data.")
        return
    cats_ordered = [label for label in _AGN_TOOL_PIE_ORDER if label in set(df["category"].unique())]
    if "Model Client" in set(df["category"].unique()):
        cats_ordered.append("Model Client")
    bar_colors = {cat: (bd.get("merged_colors") or {}).get(cat, _CAT_COLORS_BASE.get(cat, "#999")) for cat in cats_ordered}
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("turn:Q", title="Turn (global)"),
            y=alt.Y("elapsed_ms:Q", title="Time (ms)"),
            color=alt.Color("category:N", title="Category", scale=alt.Scale(domain=cats_ordered, range=[bar_colors[c] for c in cats_ordered])),
            tooltip=["turn:Q", "step_name:N", "category:N", alt.Tooltip("elapsed_ms:Q", format=",.0f"), "attempt:N"],
        )
        .properties(height=300)
    )
    if attempt_boundaries:
        rules = (
            alt.Chart(pd.DataFrame({"x": attempt_boundaries})).mark_rule(strokeDash=[4, 4], color="red", opacity=0.7, strokeWidth=2).encode(x="x:Q")
        )
        chart = alt.layer(chart, rules)
    st.altair_chart(chart, width="stretch")


def _render_side_step_model_time_and_token_chart(side: DashboardSide, data_list: list[dict[str, Any]]) -> None:
    import altair as alt

    st.markdown(f"**{side.label}**")
    if not data_list:
        st.caption("No per-step data.")
        return

    rows: list[dict[str, Any]] = []
    boundaries: list[float] = []
    global_step = 0
    for att_idx, data in enumerate(data_list):
        model_steps = []
        if side.kind == "agentic":
            model_steps = [
                s for s in data.get("steps", []) if "model_client" in s.get("step_name", "") and s.get("metadata", {}).get("elapsed_ms") is not None
            ]
        per_step_tokens = side_per_step_tokens(side, data)
        for i in range(max(len(model_steps), len(per_step_tokens))):
            rows.append(
                {
                    "Step": global_step,
                    "Model Time (ms)": model_steps[i].get("metadata", {}).get("elapsed_ms", 0) if i < len(model_steps) else 0,
                    "Total Tokens": per_step_tokens[i].get("total_tokens", 0) if i < len(per_step_tokens) else 0,
                }
            )
            global_step += 1
        if att_idx < len(data_list) - 1 and global_step > 0:
            boundaries.append(global_step - 0.5)
    if not rows:
        st.caption("No per-step model/token data.")
        return

    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(x=alt.X("Step:Q", title="Step (global)"))
    line_time = base.mark_line(color=MODEL_CLIENT_COLOR, strokeWidth=2, point=alt.OverlayMarkDef(color=MODEL_CLIENT_COLOR)).encode(
        y=alt.Y("Model Time (ms):Q", axis=alt.Axis(title="Model Time (ms)", titleColor=MODEL_CLIENT_COLOR)),
        tooltip=["Step:Q", alt.Tooltip("Model Time (ms):Q", format=",.0f"), alt.Tooltip("Total Tokens:Q", format=",")],
    )
    line_tokens = base.mark_line(color=USAGE_CHART_COLOR, strokeWidth=2, strokeDash=[5, 3], point=alt.OverlayMarkDef(color=USAGE_CHART_COLOR)).encode(
        y=alt.Y("Total Tokens:Q", axis=alt.Axis(title="Total Tokens", titleColor=USAGE_CHART_COLOR)),
        tooltip=["Step:Q", alt.Tooltip("Total Tokens:Q", format=","), alt.Tooltip("Model Time (ms):Q", format=",.0f")],
    )
    layers = [line_time, line_tokens]
    if boundaries:
        layers.append(
            alt.Chart(pd.DataFrame({"x": boundaries})).mark_rule(strokeDash=[4, 4], color="red", opacity=0.7, strokeWidth=2).encode(x="x:Q")
        )
    st.altair_chart(alt.layer(*layers).resolve_scale(y="independent").properties(height=300), width="stretch")


def _side_display_score(side: DashboardSide, task_id: str) -> tuple[bool | None, str]:
    llm_correct = side_llm_correct(side.llm_evals.get(task_id))
    if llm_correct is not None:
        return llm_correct, "LLM judge"
    em_correct = side_answer_correct(side.evals.get(task_id))
    if em_correct is not None:
        return em_correct, "EM"
    return None, ""


@st.cache_data(ttl=30, show_spinner=False)
def _load_eval_items_by_task(run_dir: str) -> dict[str, dict[str, Any]]:
    path = Path(run_dir) / "eval_results.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return {}
    items = payload.get("items", [])
    if not isinstance(items, list):
        return {}
    by_task: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id")
        if task_id is None and item.get("idx") is not None:
            task_id = item.get("idx")
        if task_id is not None:
            by_task[str(task_id)] = item
    return by_task


def _sum_mapping_values(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None
    total = 0.0
    seen = False
    for raw in value.values():
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            total += float(raw)
            seen = True
    return total if seen else None


def _artifact_stats_for_task(side: DashboardSide, task_id: str) -> dict[str, Any]:
    item: dict[str, Any] = {}
    if side.run is not None:
        item = _load_eval_items_by_task(str(side.run)).get(str(task_id), {})

    correct, score_source = _side_display_score(side, task_id)
    attempts = len(side.index.get(str(task_id), {}))
    tool_latency_ms_sum = item.get("tool_latency_ms_sum")
    if tool_latency_ms_sum is None:
        tool_latency_ms_sum = _sum_mapping_values(item.get("tool_latency_ms_by_tool"))
    tool_count = item.get("tool_count")
    if tool_count is None:
        tool_count = _sum_mapping_values(item.get("tool_count_by_tool"))

    return {
        "attempts": attempts or item.get("attempts"),
        "turns": item.get("num_turns") or item.get("turns"),
        "input_tokens": item.get("input_tokens"),
        "output_tokens": item.get("output_tokens"),
        "total_tokens": item.get("total_tokens"),
        "tool_count": tool_count,
        "task_elapsed_s": item.get("task_elapsed_s"),
        "model_client_elapsed_s": item.get("model_client_elapsed_s") or item.get("llm_elapsed_s"),
        "tool_latency_ms_sum": tool_latency_ms_sum,
        "finish_reason": item.get("finish_reason") or item.get("status"),
        "correct": correct,
        "score_source": score_source,
        "artifact_found": bool(item),
    }


def _fmt_metric(value: Any, *, suffix: str = "", precision: int = 1) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}{suffix}"
        return f"{value:,.{precision}f}{suffix}"
    return str(value)


def _render_side_step_logs(side: DashboardSide, task_id: str) -> None:
    st.markdown(f"**{side.label}**")
    if task_id not in side.index:
        st.info("No log.")
        return

    attempts = _sorted_attempts(side.index[task_id])
    selected = st.selectbox(f"{side.label} attempt", attempts, index=len(attempts) - 1, key=f"steps_{side.label}_{task_id}")
    data = _load_json_cached(side.index[task_id][selected])
    if side.kind == "agentic":
        timing = _extract_step_timing(data.get("steps", []))
        rows = [
            {
                "Turn": step["turn"],
                "Category": step["category"],
                "Step": step["step_name"],
                "Message": step["message"][:160],
                "Time": f"{step['elapsed_ms']:.0f} ms" if step["elapsed_ms"] is not None else "-",
                "Timestamp": step["timestamp"],
            }
            for step in timing["steps"]
        ]
    else:
        rows = [
            {
                "Step": step.get("step_name", ""),
                "Message": str(step.get("message", ""))[:160],
                "Timestamp": step.get("timestamp", ""),
            }
            for step in data.get("step_logs", []) or []
        ]
    st.dataframe(pd.DataFrame(rows).astype(str), width="stretch", hide_index=True) if rows else st.write("No steps.")


def _task_attempt_data(task_id: str, sides: list[DashboardSide]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    data_by_side: dict[str, list[dict[str, Any]]] = {}
    paths_by_side: dict[str, list[str]] = {}
    for side in sides:
        attempts = side.index.get(task_id)
        if not attempts:
            data_by_side[side.label] = []
            paths_by_side[side.label] = []
            continue
        ordered = _sorted_attempts(attempts)
        paths_by_side[side.label] = [attempts[a] for a in ordered]
        data_by_side[side.label] = [_load_json_cached(attempts[a]) for a in ordered]
    return data_by_side, paths_by_side


def _render_artifact_overview_summary(task_id: str, sides: list[DashboardSide]) -> None:
    cols = st.columns(2)
    for col, side in zip(cols, sides, strict=False):
        with col:
            stats = _artifact_stats_for_task(side, task_id)
            if not stats["artifact_found"] and task_id not in side.index:
                st.info("No log.")
                continue
            correct = stats["correct"]
            score_source = stats["score_source"]
            score_text = "Score pending" if correct is None else f"{'✅' if correct else '❌'} ({score_source})"
            st.markdown(f"#### {side.label} · {score_text} | {_fmt_metric(stats['attempts'])} attempts | {_fmt_metric(stats['turns'])} turns")
            metric_cols = st.columns(3)
            metric_cols[0].metric("Task Time", _fmt_metric(stats["task_elapsed_s"], suffix="s"))
            metric_cols[1].metric("Model Time", _fmt_metric(stats["model_client_elapsed_s"], suffix="s"))
            tool_time_s = stats["tool_latency_ms_sum"] / 1000 if isinstance(stats["tool_latency_ms_sum"], (int, float)) else None
            metric_cols[2].metric("Tool Time", _fmt_metric(tool_time_s, suffix="s"))
            token_cols = st.columns(3)
            token_cols[0].metric("Input Tokens", _fmt_metric(stats["input_tokens"]))
            token_cols[1].metric("Output Tokens", _fmt_metric(stats["output_tokens"]))
            token_cols[2].metric("Total Tokens", _fmt_metric(stats["total_tokens"]))
            st.caption(f"Finish: {_fmt_metric(stats['finish_reason'])} · Tool calls: {_fmt_metric(stats['tool_count'])}")


def _render_detail_overview(
    task_id: str,
    sides: list[DashboardSide],
) -> None:
    _render_artifact_overview_summary(task_id, sides)

    load_trace_timing = st.checkbox(
        "Load trace timing details",
        value=False,
        key=f"overview_trace_timing_{task_id}",
        help="Loads raw task traces for timing pies, per-turn charts, and per-step model/token charts.",
    )
    if not load_trace_timing:
        st.caption("Raw-trace timing charts are deferred to keep task detail navigation fast.")
        return

    data_by_side, paths_by_side = _task_attempt_data(task_id, sides)

    st.subheader("Time Breakdown")
    cols = st.columns(2)
    for col, side in zip(cols, sides, strict=False):
        with col:
            _render_side_time_breakdown_pie(side, data_by_side[side.label])

    cols = st.columns(2)
    for col, side in zip(cols, sides, strict=False):
        with col:
            all_tcs = [tc for data in data_by_side[side.label] for tc in side_tool_calls(side, data)]
            counts = _count_tool_calls_by_name(all_tcs) if all_tcs else {}
            tool_cols = st.columns(3)
            tool_cols[0].metric("Tool Total", f"{len(all_tcs):,}")
            search_count = counts.get("google_search", 0) + counts.get("web_search", 0)
            extract_count = counts.get("scrape_and_extract_info", 0) + counts.get("scrape_and_extract", 0)
            tool_cols[1].metric("search", f"{search_count:,}")
            tool_cols[2].metric("scrape/extract", f"{extract_count:,}")

    st.subheader("Per-Turn Timing")
    cols = st.columns(2)
    for col, side in zip(cols, sides, strict=False):
        with col:
            _render_agentic_turn_timing_chart(side, data_by_side[side.label])

    cols = st.columns(2)
    for col, side in zip(cols, sides, strict=False):
        with col:
            summaries = [load_side_summary(side, p) for p in paths_by_side[side.label]]
            token_cols = st.columns(3)
            token_cols[0].metric("Input Tokens", f"{sum(int(s.get('input_tokens', 0) or 0) for s in summaries):,}")
            token_cols[1].metric("Output Tokens", f"{sum(int(s.get('output_tokens', 0) or 0) for s in summaries):,}")
            token_cols[2].metric("Total Tokens", f"{sum(int(s.get('total_tokens', 0) or 0) for s in summaries):,}")

    st.subheader("Per-Step Model Time & Token Usage")
    cols = st.columns(2)
    for col, side in zip(cols, sides, strict=False):
        with col:
            _render_side_step_model_time_and_token_chart(side, data_by_side[side.label])


def _select_attempt_data(task_id: str, sides: list[DashboardSide], *, key_prefix: str) -> dict[str, dict[str, Any] | None]:
    selector_cols = st.columns(2)
    selected_data: dict[str, dict[str, Any] | None] = {}
    for col, side in zip(selector_cols, sides, strict=False):
        with col:
            attempts = side.index.get(task_id)
            if not attempts:
                selected_data[side.label] = None
                continue
            attempt_nums = _sorted_attempts(attempts)
            selected_attempt = st.selectbox(
                f"{side.label} attempt",
                attempt_nums,
                index=len(attempt_nums) - 1,
                key=f"{key_prefix}_{side.label}_{task_id}",
            )
            selected_data[side.label] = _load_json_cached(attempts[selected_attempt])
    return selected_data


def _render_query_comparison(left: DashboardSide, right: DashboardSide, selected_data: dict[str, dict[str, Any] | None]) -> None:
    left_tcs = side_tool_calls(left, selected_data[left.label]) if selected_data[left.label] else []
    right_tcs = side_tool_calls(right, selected_data[right.label]) if selected_data[right.label] else []
    shared_tc, left_only_tc, right_only_tc, _, _ = _build_query_comparison(left_tcs, right_tcs)
    if not (shared_tc or left_only_tc or right_only_tc):
        return
    st.subheader("Query Comparison")
    if shared_tc:
        st.markdown(f"**Both runs ({len(shared_tc)}):**")
        st.dataframe(pd.DataFrame(shared_tc).astype(str), width="stretch", hide_index=True)
    left_by_tool = _build_exclusive_tables_by_tool(left_only_tc, "Left #")
    right_by_tool = _build_exclusive_tables_by_tool(right_only_tc, "Right #")
    for tool in sorted(set(left_by_tool) | set(right_by_tool)):
        left_rows = left_by_tool.get(tool, [])
        right_rows = right_by_tool.get(tool, [])
        st.markdown(f"*{tool}*  (left: {len(left_rows)}, right: {len(right_rows)})")
        col_l, col_r = st.columns(2)
        with col_l:
            st.dataframe(pd.DataFrame(left_rows).astype(str), width="stretch", hide_index=True) if left_rows else st.caption("(none)")
        with col_r:
            st.dataframe(pd.DataFrame(right_rows).astype(str), width="stretch", hide_index=True) if right_rows else st.caption("(none)")


def _render_detail_tool_calls(task_id: str, left: DashboardSide, right: DashboardSide) -> None:
    sides = [left, right]
    selected_data = _select_attempt_data(task_id, sides, key_prefix="tc")
    _render_query_comparison(left, right, selected_data)
    st.subheader("Detailed Tool Calls")
    cols = st.columns(2)
    for col, side in zip(cols, sides, strict=False):
        with col:
            st.markdown(f"**{side.label}**")
            _render_side_tool_call_list(side, selected_data[side.label])


def _render_detail_conversation(task_id: str, left: DashboardSide, right: DashboardSide) -> None:
    sides = [left, right]
    selected_data = _select_attempt_data(task_id, sides, key_prefix="cv")
    options = ["Side by Side", f"{left.label} Only", f"{right.label} Only"]
    view_mode = st.radio("View", options, horizontal=True, key=f"cv_generic_{task_id}")
    if view_mode == "Side by Side":
        cols = st.columns(2)
        for col, side in zip(cols, sides, strict=False):
            with col:
                st.markdown(f"##### {side.label}")
                _render_side_messages(side, selected_data[side.label])
    elif view_mode == options[1]:
        _render_side_messages(left, selected_data[left.label])
    else:
        _render_side_messages(right, selected_data[right.label])


def _render_detail_step_logs(task_id: str, sides: list[DashboardSide]) -> None:
    cols = st.columns(2)
    for col, side in zip(cols, sides, strict=False):
        with col:
            _render_side_step_logs(side, task_id)


def _render_detail_execution_flow(task_id: str, left: DashboardSide, right: DashboardSide) -> None:
    sides = [left, right]
    selected_data = _select_attempt_data(task_id, sides, key_prefix="flow")
    options = ["Side by Side", f"{left.label} Only", f"{right.label} Only"]
    view_mode = st.radio("View", options, horizontal=True, key=f"flow_generic_{task_id}")

    def _render_side(side: DashboardSide) -> None:
        st.markdown(f"##### {side.label}")
        data = selected_data[side.label]
        if data is None:
            st.info("No data.")
            return
        _render_flow_steps(_execution_flow_steps(side, data))

    if view_mode == "Side by Side":
        cols = st.columns(2)
        for col, side in zip(cols, sides, strict=False):
            with col:
                _render_side(side)
    elif view_mode == options[1]:
        _render_side(left)
    else:
        _render_side(right)


def _raw_trace_sections(side: DashboardSide, data: dict[str, Any]) -> dict[str, Any]:
    if side.kind == "original":
        return {
            "basic": {
                key: data.get(key)
                for key in (
                    "status",
                    "task_id",
                    "final_boxed_answer",
                    "ground_truth",
                    "final_judge_result",
                    "judge_type",
                    "error",
                )
            },
            "main_agent_message_history": data.get("main_agent_message_history", {}),
            "sub_agent_message_history_sessions": data.get("sub_agent_message_history_sessions", {}),
            "browser_agent_message_history_sessions": data.get("browser_agent_message_history_sessions", {}),
            "step_logs": data.get("step_logs", []),
            "trace_data": data.get("trace_data", {}),
            "env_info": data.get("env_info", {}),
        }
    return {
        "basic": {key: data.get(key) for key in ("task_id", "task_input", "status", "started_at", "ended_at", "log_dir")},
        "metadata": data.get("metadata", {}),
        "conversation": data.get("conversation", []),
        "steps": data.get("steps", []),
        "tool_calls": data.get("tool_calls", []),
        "token_usage": data.get("token_usage", {}),
    }


def _render_detail_raw_trace(task_id: str, left: DashboardSide, right: DashboardSide) -> None:
    sides = [left, right]
    selected_data = _select_attempt_data(task_id, sides, key_prefix="raw")
    options = ["Curated Sections", "Full JSON"]
    mode = st.radio("Raw trace mode", options, horizontal=True, key=f"raw_trace_mode_{task_id}")
    cols = st.columns(2)
    for col, side in zip(cols, sides, strict=False):
        with col:
            st.markdown(f"##### {side.label}")
            data = selected_data[side.label]
            if data is None:
                st.info("No data.")
                continue
            if mode == "Full JSON":
                st.json(data)
                continue
            sections = _raw_trace_sections(side, data)
            section_names = list(sections)
            selected_section = st.selectbox(f"{side.label} section", section_names, key=f"raw_section_{side.label}_{task_id}")
            st.json(sections[selected_section])


def _render_task_detail_sides(task_id: str, left: DashboardSide, right: DashboardSide) -> None:
    st.header(f"Task {task_id}")
    header_cols = st.columns(2)
    header_cols[0].markdown(f"#### {side_run_title(left)}")
    header_cols[1].markdown(f"#### {side_run_title(right)}")
    sides = [left, right]
    detail_view = st.radio(
        "Detail view",
        ["Overview", "Tool Calls", "Conversation", "Execution Flow", "Step Logs", "Raw Trace"],
        horizontal=True,
        key=f"detail_view_generic_{task_id}",
    )

    if detail_view == "Overview":
        _render_detail_overview(task_id, sides)
    elif detail_view == "Tool Calls":
        _render_detail_tool_calls(task_id, left, right)
    elif detail_view == "Conversation":
        _render_detail_conversation(task_id, left, right)
    elif detail_view == "Execution Flow":
        _render_detail_execution_flow(task_id, left, right)
    elif detail_view == "Step Logs":
        _render_detail_step_logs(task_id, sides)
    elif detail_view == "Raw Trace":
        _render_detail_raw_trace(task_id, left, right)
