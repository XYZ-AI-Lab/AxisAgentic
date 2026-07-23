# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Reusable Streamlit rendering primitives for the dashboard."""

from __future__ import annotations

import contextlib
import json
import re
from typing import Any

import streamlit as st

from recipe.dashboard.extraction import _ori_tool_calls_paired
from recipe.dashboard.loading import _ori_messages, _ori_system_prompt
from recipe.dashboard.sides import DashboardSide, side_messages, side_tool_calls
from recipe.dashboard.timing_processing import _tool_timing_detail_rows

_MSG_CSS = (
    "white-space:pre-wrap; word-wrap:break-word; font-family:monospace; font-size:13px;"
    "background:#f6f8fa; padding:8px 12px; border-radius:6px; max-height:600px; overflow-y:auto;"
)


def _render_wrapped_text(text: str) -> None:
    import html as _html

    st.markdown(f'<div style="{_MSG_CSS}">{_html.escape(text, quote=False)}</div>', unsafe_allow_html=True)


def _render_json_result(value: Any) -> None:
    if isinstance(value, str):
        with contextlib.suppress(json.JSONDecodeError, ValueError):
            value = json.loads(value)
    if isinstance(value, (dict, list, int, float, bool)) or value is None:
        st.json(value)
        return
    _render_wrapped_text(str(value))


def _fmt_answer_pair(
    left_answer: str,
    right_answer: str,
    *,
    left_em_correct: bool | None,
    left_llm_correct: bool | None,
    right_em_correct: bool | None,
    right_llm_correct: bool | None,
) -> tuple[str, str]:
    """Format answers with two leading evaluation emojis: EM then LLM Judge."""

    def _e(*, correct: bool | None) -> str:
        if correct is None:
            return "\u2754"  # ❔
        return "\u2705" if correct else "\U0001f4a2"  # ✅ or 💢

    def _f(ans: str, *, em_correct: bool | None, llm_correct: bool | None) -> str:
        return f"{_e(correct=em_correct)}{_e(correct=llm_correct)}{ans}" if ans and ans != "-" else "-"

    return (
        _f(left_answer, em_correct=left_em_correct, llm_correct=left_llm_correct),
        _f(right_answer, em_correct=right_em_correct, llm_correct=right_llm_correct),
    )


def _render_message_bubble(role: str, content: str, idx: int, *, tool_calls: list[dict[str, Any]] | None = None) -> None:
    short = content[:120].replace("\n", " ") if content else "(empty)"
    with st.expander(f"**[{idx}] {role}**: {short}...", expanded=False):
        if "<think>" in content:
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if think_match:
                st.markdown("**Thinking:**")
                _render_wrapped_text(think_match.group(1))
                rest = content[think_match.end() :].strip()
                if rest:
                    st.markdown("**Response:**")
                    _render_wrapped_text(rest)
            else:
                _render_wrapped_text(content)
        else:
            _render_wrapped_text(content)
        if tool_calls:
            st.markdown("**Tool Call:**")
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    with contextlib.suppress(json.JSONDecodeError, ValueError):
                        args = json.loads(args)
                st.markdown(f"`{fn.get('name', '?')}`")
                st.json(args)


def _render_ori_messages(ori: dict[str, Any] | None) -> None:
    if ori is None:
        st.info("No data.")
        return
    sys_prompt = _ori_system_prompt(ori)
    if sys_prompt:
        with st.expander("System Prompt", expanded=False):
            _render_wrapped_text(sys_prompt)
    for i, m in enumerate(_ori_messages(ori)):
        _render_message_bubble(m.get("role", "?"), str(m.get("content", "")), i)


def _render_side_messages(side: DashboardSide, data: dict[str, Any] | None) -> None:
    if data is None:
        st.info("No data.")
        return
    if side.kind == "original":
        _render_ori_messages(data)
        return
    for i, m in enumerate(side_messages(side, data)):
        role = m.get("role", "?")
        tool_calls = m.get("tool_calls", [])
        label = role
        if role == "tool" and m.get("name"):
            label = f"tool ({m['name']})"
        if tool_calls:
            fn_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
            label = f"assistant -> {', '.join(fn_names)}"
        _render_message_bubble(label, str(m.get("content", "")), i, tool_calls=tool_calls)


def _render_ori_tool_call_list(data: dict[str, Any]) -> None:
    paired = _ori_tool_calls_paired(data)
    if not paired:
        st.info("No tool calls.")
        return
    _EMOJI = {"google_search": "\U0001f50d", "scrape_and_extract_info": "\U0001f310"}
    for i, tc in enumerate(paired):
        name = tc["tool_name"]
        with st.expander(f"[{i}] {_EMOJI.get(name, '')} {name} ({tc['latency_ms']}ms)", expanded=False):
            st.markdown(f"**Tool:** `{name}`  **Server:** `{tc['server_name']}`  **Latency:** {tc['latency_ms']}ms")
            st.json(tc["arguments"])


def _render_agn_tool_call_list(tool_calls: list[dict[str, Any]]) -> None:
    if not tool_calls:
        st.info("No tool calls.")
        return
    for i, tc in enumerate(tool_calls):
        name = tc.get("tool_name", "?")
        latency = tc.get("latency_ms", 0)
        with st.expander(f"[{i}] {tc.get('emoji', '')} {name} ({latency:.0f}ms)", expanded=False):
            st.markdown(f"**Tool:** `{name}`  **Status:** `{tc.get('status', '?')}`  **Latency:** {latency:.0f}ms")
            if tc.get("reason"):
                st.markdown(f"**Reason:** `{tc['reason']}`")
            timing_details = _tool_timing_detail_rows(tc)
            if timing_details:
                st.markdown("**Timing Breakdown:**")
                for label, ms in timing_details:
                    st.markdown(f"- `{label}`: {ms / 1000:.1f}s")
            request_count = tc.get("metadata", {}).get("request_count")
            quote_retry_used = tc.get("metadata", {}).get("quote_retry_used")
            if request_count is not None or quote_retry_used:
                summary_parts = []
                if request_count is not None:
                    summary_parts.append(f"requests={request_count}")
                if quote_retry_used:
                    summary_parts.append("quote_retry=True")
                st.markdown(f"**Metadata:** `{', '.join(summary_parts)}`")
            st.json(tc.get("arguments", {}))
            if tc.get("content"):
                st.markdown("**Result:**")
                _render_json_result(tc["content"])


def _render_side_tool_call_list(side: DashboardSide, data: dict[str, Any] | None) -> None:
    if data is None:
        st.info("No data.")
        return
    if side.kind == "original":
        _render_ori_tool_call_list(data)
        return
    _render_agn_tool_call_list(side_tool_calls(side, data))
