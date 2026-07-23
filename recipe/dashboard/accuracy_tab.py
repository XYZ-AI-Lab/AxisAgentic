# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy tab renderer for the benchmark dashboard."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from recipe.dashboard.constants import LLM_JUDGE_CORRECT_THRESHOLD
from recipe.dashboard.discovery import _all_task_ids, _sorted_attempts
from recipe.dashboard.metrics import _format_score_accuracy
from recipe.dashboard.rendering import _fmt_answer_pair
from recipe.dashboard.sides import (
    DashboardSide,
    load_side_summary,
    side_answer_correct,
    side_llm_correct,
    side_short_label,
)

_RESULT_TABLE_FILTER_LABELS = {
    "all": "All",
    "left_right_right_wrong": "Left✅ Right💢",
    "left_wrong_right_right": "Left💢 Right✅",
    "both_right": "Both✅",
    "both_wrong": "Both💢",
    "left_finished": "Only Left finished",
    "right_finished": "Only Right finished",
    "eval_pending": "Eval⏳",
}


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _task_id_from_timing_row(row: dict[str, Any]) -> str:
    task_id = row.get("task_id")
    if task_id is not None and str(task_id) != "":
        return str(task_id)
    idx = row.get("idx")
    if idx is not None and str(idx) != "":
        return str(idx)
    return ""


@st.cache_data(ttl=30, show_spinner=False)
def _load_eval_timing_stats(run_dir: str) -> dict[str, dict[str, Any]]:
    path = Path(run_dir) / "eval_results.json"
    if not path.exists():
        return {}
    with contextlib.suppress(UnicodeDecodeError, json.JSONDecodeError, OSError):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("items", [])
        if not isinstance(rows, list):
            return {}
        stats: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            task_id = _task_id_from_timing_row(row)
            if not task_id:
                continue
            turns = _safe_int(row.get("num_turns"))
            stats[task_id] = {
                "turns": turns,
                "turns_by_attempt": [turns] if turns else [],
                "tokens": _safe_int(row.get("total_tokens")),
                "tools": _safe_int(row.get("tool_count")),
            }
        return stats
    return {}


def _stats_from_artifacts(side: DashboardSide, tid: str, timing_stats: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    eval_row = side.evals.get(tid) or {}
    llm_row = side.llm_evals.get(tid) or {}
    timing_row = timing_stats.get(tid) or {}
    if not eval_row and not llm_row and not timing_row:
        return None

    timing_present = bool(timing_row) or any(
        _safe_int(row.get(key)) > 0 for row in (eval_row, llm_row) for key in ("num_turns", "total_tokens", "tool_count")
    )
    turns = _safe_int(timing_row.get("turns") or eval_row.get("num_turns") or llm_row.get("num_turns"))
    turns_by_attempt = timing_row.get("turns_by_attempt")
    if not isinstance(turns_by_attempt, list):
        turns_by_attempt = [turns] if turns else []

    return {
        "answer": _first_text(eval_row, ("prediction", "output", "answer", "extracted_answer", "final_boxed_answer"))
        or _first_text(llm_row, ("prediction", "output", "answer", "extracted_answer", "final_boxed_answer")),
        "ground_truth": _first_text(eval_row, ("ground_truth", "label", "correct_answer"))
        or _first_text(llm_row, ("ground_truth", "label", "correct_answer")),
        "turns": turns,
        "turns_by_attempt": [_safe_int(value) for value in turns_by_attempt],
        "tokens": _safe_int(timing_row.get("tokens") or eval_row.get("total_tokens") or llm_row.get("total_tokens")),
        "tools": _safe_int(timing_row.get("tools") or eval_row.get("tool_count") or llm_row.get("tool_count")),
        "_timing_present": timing_present,
    }


def _stats_from_trace(side: DashboardSide, attempts: dict[int, str]) -> dict[str, Any]:
    summaries = [load_side_summary(side, attempts[a]) for a in _sorted_attempts(attempts)]
    latest = summaries[-1] if summaries else {}
    return {
        "answer": str(latest.get("answer", "") or ""),
        "ground_truth": str(latest.get("ground_truth", "") or ""),
        "turns": sum(int(s.get("turns", 0) or 0) for s in summaries),
        "turns_by_attempt": [int(s.get("turns", 0) or 0) for s in summaries],
        "tokens": sum(int(s.get("total_tokens", 0) or 0) for s in summaries),
        "tools": sum(int(s.get("num_tool_calls", 0) or 0) for s in summaries),
        "_timing_present": True,
    }


def _merge_artifact_and_trace_stats(artifact_stats: dict[str, Any], trace_stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": artifact_stats.get("answer") or trace_stats.get("answer") or "",
        "ground_truth": artifact_stats.get("ground_truth") or trace_stats.get("ground_truth") or "",
        "turns": artifact_stats.get("turns") or trace_stats.get("turns") or 0,
        "turns_by_attempt": artifact_stats.get("turns_by_attempt") or trace_stats.get("turns_by_attempt") or [],
        "tokens": artifact_stats.get("tokens") or trace_stats.get("tokens") or 0,
        "tools": artifact_stats.get("tools") or trace_stats.get("tools") or 0,
    }


def _display_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in stats.items() if not key.startswith("_")}


def _build_side_task_stats(side: DashboardSide, task_ids: list[str]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    timing_stats = _load_eval_timing_stats(str(side.run)) if side.run else {}
    for tid in task_ids:
        attempts = side.index.get(tid)
        artifact_stats = _stats_from_artifacts(side, tid, timing_stats)
        if artifact_stats is not None:
            if not artifact_stats.get("_timing_present") and attempts:
                stats[tid] = _display_stats(_merge_artifact_and_trace_stats(artifact_stats, _stats_from_trace(side, attempts)))
                continue
            stats[tid] = _display_stats(artifact_stats)
            continue

        if not attempts:
            continue
        stats[tid] = _display_stats(_stats_from_trace(side, attempts))
    return stats


def _format_empty_answer_ratio_from_stats(stats: dict[str, dict[str, Any]], *, task_ids: set[str]) -> str:
    total = 0
    empty = 0
    for tid in task_ids:
        row = stats.get(tid)
        if row is None:
            continue
        total += 1
        if not str(row.get("answer", "") or "").strip():
            empty += 1
    if total == 0:
        return "-"
    return f"{empty}/{total} ({empty / total * 100:.0f}%)"


def _format_eval_source_status(evals: dict[str, dict[str, Any]]) -> str:
    summary = evals.get("_summary") or {}
    if not summary:
        return ""
    total = summary.get("num_items")
    source_total = summary.get("source_num_items")
    dataset_total = summary.get("dataset_num_items")
    source_kind = summary.get("source_kind") or "unknown"
    complete = summary.get("source_complete")
    if complete is True:
        status = "complete"
    elif complete is False:
        status = "partial/live"
    else:
        status = "unknown"
    bits = [f"{status}", f"source={source_kind}"]
    if total is not None:
        bits.append(f"judged={total}")
    if source_total is not None:
        bits.append(f"source_rows={source_total}")
    if dataset_total is not None:
        bits.append(f"dataset={dataset_total}")
    return " | ".join(bits)


def _side_judged_correct(side: DashboardSide, task_id: str) -> bool | None:
    llm_correct = side_llm_correct(side.llm_evals.get(task_id))
    if llm_correct is not None:
        return llm_correct
    return side_answer_correct(side.evals.get(task_id))


def _result_filter_key_for_task(left: DashboardSide, right: DashboardSide, task_id: str) -> str:
    left_finished = task_id in left.index
    right_finished = task_id in right.index
    if left_finished and not right_finished:
        return "left_finished"
    if right_finished and not left_finished:
        return "right_finished"

    left_correct = _side_judged_correct(left, task_id)
    right_correct = _side_judged_correct(right, task_id)
    if left_correct is None or right_correct is None:
        return "eval_pending"
    if left_correct and right_correct:
        return "both_right"
    if left_correct and not right_correct:
        return "left_right_right_wrong"
    if not left_correct and right_correct:
        return "left_wrong_right_right"
    return "both_wrong"


def _task_matches_result_filter(filter_key: str, left: DashboardSide, right: DashboardSide, task_id: str) -> bool:
    return filter_key == "all" or _result_filter_key_for_task(left, right, task_id) == filter_key


def _render_accuracy_summary(
    left: DashboardSide,
    right: DashboardSide,
    stats_by_side: dict[str, dict[str, dict[str, Any]]],
    all_ids: list[str],
) -> None:
    visible_task_ids = set(all_ids)
    st.caption(
        f"All started tasks: `{len(all_ids)}` (`{len(set(left.index) & set(right.index))}` shared)"
        "  |  "
        f"Left started tasks: `{len(left.index)}`"
        "  |  "
        f"Right started tasks: `{len(right.index)}`"
    )

    name_cols = st.columns(2)
    with name_cols[0]:
        st.markdown(f"#### {left.label}: {left.exp_name or '-'} / {left.run_name or '-'}")
        metric_cols1 = st.columns(3)
        metric_cols1[0].metric("Exact Match Accuracy", _format_score_accuracy(left.evals, bool_key="is_correct", task_ids=visible_task_ids))
        metric_cols1[1].metric(
            "LLM Judge Accuracy",
            _format_score_accuracy(
                left.llm_evals,
                score_key="llm_judge_score",
                score_threshold=LLM_JUDGE_CORRECT_THRESHOLD,
                task_ids=visible_task_ids,
            ),
        )
        metric_cols1[2].metric("Empty Answer Ratio", _format_empty_answer_ratio_from_stats(stats_by_side[left.label], task_ids=visible_task_ids))
        left_eval_status = _format_eval_source_status(left.llm_evals) or _format_eval_source_status(left.evals)
        if left_eval_status:
            st.caption(f"Eval source: {left_eval_status}")
    with name_cols[1]:
        st.markdown(f"#### {right.label}: {right.exp_name or '-'} / {right.run_name or '-'}")
        metric_cols2 = st.columns(3)
        metric_cols2[0].metric("Exact Match Accuracy", _format_score_accuracy(right.evals, bool_key="is_correct", task_ids=visible_task_ids))
        metric_cols2[1].metric(
            "LLM Judge Accuracy",
            _format_score_accuracy(
                right.llm_evals,
                score_key="llm_judge_score",
                score_threshold=LLM_JUDGE_CORRECT_THRESHOLD,
                task_ids=visible_task_ids,
            ),
        )
        metric_cols2[2].metric("Empty Answer Ratio", _format_empty_answer_ratio_from_stats(stats_by_side[right.label], task_ids=visible_task_ids))
        right_eval_status = _format_eval_source_status(right.llm_evals) or _format_eval_source_status(right.evals)
        if right_eval_status:
            st.caption(f"Eval source: {right_eval_status}")

    st.caption("Accuracy excludes pending or missing evaluations. Empty ratio is empty extracted latest answers / latest task results.")


def _ground_truth_by_task(sides: list[DashboardSide], stats_by_side: dict[str, dict[str, dict[str, Any]]], all_ids: list[str]) -> dict[str, str]:
    gt_map: dict[str, str] = {}
    for tid in all_ids:
        for side in sides:
            evl = side.evals.get(tid)
            if evl and evl.get("ground_truth"):
                gt_map[tid] = str(evl["ground_truth"])
                break
            stats = stats_by_side[side.label].get(tid)
            if stats is not None:
                gt_map[tid] = str(stats.get("ground_truth", ""))
                break
    return gt_map


def _format_token_total(value: int, max_value: int) -> str:
    if max_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    return f"{value / 1_000:.1f}K" if max_value >= 1_000 else f"{value:,}"


def _build_accuracy_table_rows(
    left: DashboardSide,
    right: DashboardSide,
    stats_by_side: dict[str, dict[str, dict[str, Any]]],
    table_task_ids: list[str],
    gt_map: dict[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    sides = [left, right]
    rows: list[dict[str, Any]] = []
    token_totals_by_side: dict[str, list[int]] = {left.label: [], right.label: []}
    turn_sums = {left.label: 0, right.label: 0}
    tool_sums = {left.label: 0, right.label: 0}

    for tid in table_task_ids:
        row: dict[str, Any] = {"Task ID": tid, "GT": (gt_map.get(tid) or "-")[:60]}
        answers: list[tuple[str, bool | None, bool | None]] = []
        for side in sides:
            short = side_short_label(side)
            token_total = 0
            stats = stats_by_side[side.label].get(tid)
            if stats is not None:
                answer = str(stats.get("answer", "") or "")[:60]
                turns = int(stats.get("turns", 0) or 0)
                token_total = int(stats.get("tokens", 0) or 0)
                tools = int(stats.get("tools", 0) or 0)
                row[f"{short} Turns"] = ", ".join(str(turns) for turns in stats.get("turns_by_attempt", []))
                row[f"{short} #TC"] = str(tools)
                turn_sums[side.label] += turns
                tool_sums[side.label] += tools
            else:
                answer = "-"
                row[f"{short} Turns"] = "-"
                row[f"{short} #TC"] = "-"
            token_totals_by_side[side.label].append(token_total)
            answers.append((answer, side_answer_correct(side.evals.get(tid)), side_llm_correct(side.llm_evals.get(tid))))

        row[f"{side_short_label(left)} Answer"], row[f"{side_short_label(right)} Answer"] = _fmt_answer_pair(
            answers[0][0],
            answers[1][0],
            left_em_correct=answers[0][1],
            left_llm_correct=answers[0][2],
            right_em_correct=answers[1][1],
            right_llm_correct=answers[1][2],
        )
        rows.append(row)

    all_tokens = [v for values in token_totals_by_side.values() for v in values if v > 0]
    mx = max(all_tokens) if all_tokens else 0

    for i, row in enumerate(rows):
        for side in sides:
            tokens = token_totals_by_side[side.label][i]
            row[f"{side_short_label(side)} Tokens"] = _format_token_total(tokens, mx) if tokens else "-"

    total_row: dict[str, Any] = {"Task ID": "Total", "GT": ""}
    for side in sides:
        short = side_short_label(side)
        total_row.update(
            {
                f"{short} Answer": "",
                f"{short} Turns": str(turn_sums[side.label]),
                f"{short} #TC": str(tool_sums[side.label]),
                f"{short} Tokens": _format_token_total(sum(token_totals_by_side[side.label]), mx),
            }
        )
    rows.append(total_row)

    col_order = ["Task ID", "GT"]
    for side in sides:
        short = side_short_label(side)
        col_order.extend([f"{short} Answer", f"{short} Turns", f"{short} #TC", f"{short} Tokens"])
    return rows, col_order


def _render_accuracy_tab(left: DashboardSide, right: DashboardSide) -> None:
    st.header("Accuracy")
    sides = [left, right]
    all_ids = _all_task_ids(left.index, right.index)
    stats_by_side = {side.label: _build_side_task_stats(side, all_ids) for side in sides}

    _render_accuracy_summary(left, right, stats_by_side, all_ids)
    filter_key = st.selectbox(
        "Result table filter",
        list(_RESULT_TABLE_FILTER_LABELS),
        format_func=_RESULT_TABLE_FILTER_LABELS.__getitem__,
    )
    table_task_ids = [tid for tid in all_ids if _task_matches_result_filter(filter_key, left, right, tid)]
    st.caption(f"Showing `{len(table_task_ids)}` / `{len(all_ids)}` tasks in the result table.")

    rows, col_order = _build_accuracy_table_rows(left, right, stats_by_side, table_task_ids, _ground_truth_by_task(sides, stats_by_side, all_ids))
    st.dataframe(pd.DataFrame(rows)[col_order].astype(str), width="stretch", hide_index=True)
