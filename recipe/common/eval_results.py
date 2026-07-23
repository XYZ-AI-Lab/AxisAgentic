# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Timing aggregation shared by benchmark recipe pipelines.

Both pipelines emit an ``eval_results.json`` under their run directory with the
same schema so :mod:`recipe.dashboard.app` can compare them side-by-side:

``TaskTimingRow`` holds one row per task.  ``build_eval_results_payload`` turns a
list of rows plus script-level fields into the full JSON payload
(per-task rows, aggregate tails, top-slow items, script totals).

Script-level fields are optional so the offline extractor can emit what it can read
from disk (task wall-clock, tool latencies, aggregated LLM estimates) and the
agentic runner can emit the full superset it tracks in-process.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TaskTimingRow:
    """Timing and outcome fields for one evaluated task.

    The same row shape is emitted by the agentic runner, which measures wall-clock
    timing directly, and by the offline extractor, which reads timing from
    ``task_*.json`` files. Fields the offline path cannot observe stay ``None``.
    """

    task_id: str
    idx: int | None = None
    score: float | None = None
    # Turns used by the final attempt for this original task.
    num_turns: int | None = None
    # Actual max attempt number observed for this original task.
    num_attempts: int | None = None
    # Turns used across all observed attempts for this original task.
    total_turns: int | None = None
    finish_reason: str | None = None
    cached: bool = False

    # Wall-clock timing.
    task_elapsed_s: float | None = None
    orchestrator_build_elapsed_s: float | None = None
    model_client_elapsed_s: float | None = None
    group_completion_elapsed_s: float | None = None
    non_model_overhead_s: float | None = None

    # Tool timing aggregated across all attempts of this task.
    tool_latency_ms_sum: float | None = None
    tool_latency_ms_mean: float | None = None
    tool_latency_ms_max: float | None = None
    tool_count: int | None = None
    tool_latency_ms_by_tool: dict[str, float] = field(default_factory=dict)
    tool_count_by_tool: dict[str, int] = field(default_factory=dict)
    tool_success_by_tool: dict[str, int] = field(default_factory=dict)
    tool_failed_by_tool: dict[str, int] = field(default_factory=dict)

    # Token usage aggregated across all attempts.
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    sorted_v = sorted(values)
    pos = q * (len(sorted_v) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_v) - 1)
    frac = pos - lo
    return float(sorted_v[lo] * (1.0 - frac) + sorted_v[hi] * frac)


def _tail_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None, "mean": None, "sum": None}
    return {
        "count": len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
        "mean": sum(values) / len(values),
        "sum": sum(values),
    }


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _count_distribution(values: list[int], *, fill_from: int | None = None) -> dict[str, int]:
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if fill_from is not None and values:
        for value in range(fill_from, max(values) + 1):
            counts.setdefault(value, 0)
    return {str(value): counts[value] for value in sorted(counts)}


# Per-task timing fields we surface tail (p50/p95/p99/max) stats for.
_TAIL_FIELDS: tuple[str, ...] = (
    "task_elapsed_s",
    "orchestrator_build_elapsed_s",
    "model_client_elapsed_s",
    "group_completion_elapsed_s",
    "non_model_overhead_s",
    "tool_latency_ms_sum",
    "tool_latency_ms_mean",
    "tool_latency_ms_max",
)

# Fields we surface top-slow leaderboards for — subset of the tail fields.
_TOP_SLOW_FIELDS: tuple[str, ...] = (
    "task_elapsed_s",
    "model_client_elapsed_s",
    "non_model_overhead_s",
    "tool_latency_ms_sum",
)


def _slow_item(row: TaskTimingRow, field_name: str) -> dict[str, Any]:
    """Project a row into the leaderboard record shown for top-slow tables."""
    value = getattr(row, field_name, None)
    return {
        "task_id": row.task_id,
        "idx": row.idx,
        "score": row.score,
        "num_turns": row.num_turns,
        "finish_reason": row.finish_reason,
        "sort_field": field_name,
        "sort_value": value,
        "task_elapsed_s": row.task_elapsed_s,
        "model_client_elapsed_s": row.model_client_elapsed_s,
        "non_model_overhead_s": row.non_model_overhead_s,
        "tool_latency_ms_sum": row.tool_latency_ms_sum,
        "tool_count": row.tool_count,
    }


def _top_slow(rows: list[TaskTimingRow], field_name: str, *, limit: int = 10) -> list[dict[str, Any]]:
    ranked = [row for row in rows if isinstance(getattr(row, field_name, None), (int, float)) and not isinstance(getattr(row, field_name), bool)]
    ranked.sort(key=lambda row: float(getattr(row, field_name)), reverse=True)
    return [_slow_item(row, field_name) for row in ranked[:limit]]


def build_eval_results_payload(
    rows: list[TaskTimingRow],
    *,
    source: str,
    model: str | None = None,
    num_items: int | None = None,
    # Script-level wall-clock timing (all optional; the offline path fills what it can).
    script_started_at: str | None = None,
    script_ended_at: str | None = None,
    start_to_end_elapsed_s: float | None = None,
    total_eval_elapsed_s: float | None = None,
    model_client_build_elapsed_s: float | None = None,
    task_logger_close_drain_elapsed_s: float | None = None,
    trace_file_writes_disabled: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the ``eval_results.json`` payload for a run.

    Both pipelines call this with the same row shape so the dashboard doesn't
    have to branch by source when reading tail stats.
    """
    total = num_items if num_items is not None else len(rows)
    scored = [row.score for row in rows if isinstance(row.score, (int, float)) and not isinstance(row.score, bool)]
    correct = sum(1 for score in scored if score > 0.5)
    accuracy = correct / len(scored) if scored else None

    payload: dict[str, Any] = {
        "source": source,
        "model": model,
        "num_items": total,
        "correct": correct if scored else None,
        "accuracy": accuracy,
        "script_started_at": script_started_at,
        "script_ended_at": script_ended_at,
        "start_to_end_elapsed_s": start_to_end_elapsed_s,
        "total_eval_elapsed_s": total_eval_elapsed_s,
        "model_client_build_elapsed_s": model_client_build_elapsed_s,
        "task_logger_close_drain_elapsed_s": task_logger_close_drain_elapsed_s,
        "trace_file_writes_disabled": trace_file_writes_disabled,
    }

    # Aggregate tail stats per timing field, leaving the summary dict as-is so
    # the dashboard can pull any subset it wants without re-walking rows.
    tails: dict[str, dict[str, float | int | None]] = {}
    for name in _TAIL_FIELDS:
        values = [
            float(getattr(row, name))
            for row in rows
            if isinstance(getattr(row, name, None), (int, float)) and not isinstance(getattr(row, name), bool)
        ]
        tails[name] = _tail_summary(values)
    payload["tails"] = tails

    # Token totals (sum across tasks).
    payload["input_tokens_sum"] = sum(row.input_tokens for row in rows if isinstance(row.input_tokens, int)) or None
    payload["output_tokens_sum"] = sum(row.output_tokens for row in rows if isinstance(row.output_tokens, int)) or None
    payload["total_tokens_sum"] = sum(row.total_tokens for row in rows if isinstance(row.total_tokens, int)) or None

    final_turn_values = [value for row in rows if (value := _int_value(row.num_turns)) is not None]
    attempt_values = [value for row in rows if (value := _int_value(row.num_attempts)) is not None]
    total_turn_values: list[int] = []
    for row in rows:
        total_turns = _int_value(row.total_turns)
        if total_turns is not None:
            total_turn_values.append(total_turns)
            continue
        num_turns = _int_value(row.num_turns)
        if num_turns is not None:
            total_turn_values.append(num_turns)
    retry_task_count = sum(1 for value in attempt_values if value > 1)
    payload["avg_num_turns"] = sum(final_turn_values) / len(final_turn_values) if final_turn_values else None
    payload["turn_counted_task_count"] = len(final_turn_values)
    payload["avg_num_attempts"] = sum(attempt_values) / len(attempt_values) if attempt_values else None
    payload["attempt_counted_task_count"] = len(attempt_values)
    payload["retry_task_count"] = retry_task_count if attempt_values else None
    payload["retry_ratio"] = retry_task_count / len(attempt_values) if attempt_values else None
    payload["attempts_distribution"] = _count_distribution(attempt_values, fill_from=1)
    payload["total_turn_counted_task_count"] = len(total_turn_values)
    payload["avg_total_turns"] = sum(total_turn_values) / len(total_turn_values) if total_turn_values else None
    payload["turns_distribution"] = _count_distribution(total_turn_values)

    # Aggregate per-tool latency across all tasks.
    tool_latency_totals: dict[str, float] = {}
    tool_count_totals: dict[str, int] = {}
    tool_success_totals: dict[str, int] = {}
    tool_failed_totals: dict[str, int] = {}
    for row in rows:
        for tool, ms in row.tool_latency_ms_by_tool.items():
            tool_latency_totals[tool] = tool_latency_totals.get(tool, 0.0) + float(ms)
        for tool, count in row.tool_count_by_tool.items():
            tool_count_totals[tool] = tool_count_totals.get(tool, 0) + int(count)
        for tool, count in row.tool_success_by_tool.items():
            tool_success_totals[tool] = tool_success_totals.get(tool, 0) + int(count)
        for tool, count in row.tool_failed_by_tool.items():
            tool_failed_totals[tool] = tool_failed_totals.get(tool, 0) + int(count)
    payload["tool_latency_ms_by_tool_sum"] = tool_latency_totals
    payload["tool_count_by_tool_sum"] = tool_count_totals
    payload["tool_success_by_tool_sum"] = tool_success_totals
    payload["tool_failed_by_tool_sum"] = tool_failed_totals

    payload["top_slow"] = {name: _top_slow(rows, name) for name in _TOP_SLOW_FIELDS}
    payload["items"] = [row.to_dict() for row in rows]

    if extra:
        payload.update(extra)
    return payload


def compute_non_model_overhead_s(
    task_elapsed_s: float | None,
    model_client_elapsed_s: float | None,
    tool_latency_ms_sum: float | None,
) -> float | None:
    """Return ``task - model - tool`` when every input is available, else ``None``."""
    if task_elapsed_s is None or model_client_elapsed_s is None:
        return None
    tool_s = (tool_latency_ms_sum or 0.0) / 1000.0
    return task_elapsed_s - model_client_elapsed_s - tool_s
