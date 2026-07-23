# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Build compact dashboard artifacts from run logs."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recipe.common.log_processing.expected_tasks import ExpectedTaskCount, resolve_expected_task_count
from recipe.common.log_processing.trace_distributions import (
    TraceKind,
    write_trace_distribution_artifact,
)

SCHEMA_VERSION = 1
SUMMARY_NAME = "dashboard_summary.json"
EVENTS_NAME = "dashboard_events.jsonl"

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def load_dashboard_summary(run_dir: str | Path) -> dict[str, Any] | None:
    path = Path(run_dir) / SUMMARY_NAME
    if not path.exists():
        return None
    loaded = _load_json(path)
    return loaded or None


def _numeric_score(row: dict[str, Any], key: str) -> float | None:
    score = row.get(key)
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return float(score)
    return None


def _numeric_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _integer_value(row: dict[str, Any], key: str) -> int | None:
    value = row.get(key)
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


def _llm_judge_stats(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _iter_jsonl(run_dir / "llm_judge_results.jsonl")
    by_task: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if task_id and _numeric_score(row, "llm_judge_score") is not None:
            by_task[task_id] = row

    scored = [_numeric_score(row, "llm_judge_score") for row in by_task.values()]
    scored_values = [score for score in scored if score is not None]
    correct = sum(1 for score in scored_values if score > 0.5)
    summary = _load_json(run_dir / "llm_judge_summary.json")
    if summary:
        correct = int(summary.get("llm_judge_correct", correct) or 0)
        accuracy = summary.get("llm_judge_accuracy")
        if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool):
            accuracy = correct / len(scored_values) if scored_values else None
    else:
        accuracy = correct / len(scored_values) if scored_values else None

    stats = {
        "judged_task_count": len(scored_values),
        "llm_judge_correct": correct,
        "llm_judge_accuracy": accuracy,
        "summary_available": bool(summary),
    }
    compact_rows = [
        {
            "task_id": str(row.get("task_id", "")),
            "llm_judge_score": _numeric_score(row, "llm_judge_score"),
            "llm_judge_result": row.get("llm_judge_result"),
            "em_score": _numeric_score(row, "em_score"),
            "em_result": row.get("em_result"),
        }
        for row in by_task.values()
    ]
    return stats, compact_rows


def _eval_results_stats(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    payload = _load_json(run_dir / "eval_results.json")
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    compact_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        compact_items.append(
            {
                "task_id": str(item.get("task_id", "")),
                "idx": item.get("idx"),
                "score": _numeric_score(item, "score"),
                "num_turns": item.get("num_turns"),
                "num_attempts": item.get("num_attempts"),
                "total_turns": item.get("total_turns"),
                "finish_reason": item.get("finish_reason"),
                "task_elapsed_s": item.get("task_elapsed_s"),
                "model_client_elapsed_s": item.get("model_client_elapsed_s"),
                "tool_latency_ms_sum": item.get("tool_latency_ms_sum"),
                "tool_count": item.get("tool_count"),
                "input_tokens": item.get("input_tokens"),
                "output_tokens": item.get("output_tokens"),
                "total_tokens": item.get("total_tokens"),
                "tool_latency_ms_by_tool": item.get("tool_latency_ms_by_tool", {}),
                "tool_count_by_tool": item.get("tool_count_by_tool", {}),
            }
        )
    turn_values = [_numeric_value(item, "num_turns") for item in compact_items]
    numeric_turn_values = [value for value in turn_values if value is not None]
    attempt_values = [value for item in compact_items if (value := _integer_value(item, "num_attempts")) is not None]
    retry_task_count = sum(1 for value in attempt_values if value > 1)
    total_turn_values: list[int] = []
    for item in compact_items:
        total_turns = _integer_value(item, "total_turns")
        if total_turns is not None:
            total_turn_values.append(total_turns)
            continue
        num_turns = _integer_value(item, "num_turns")
        if num_turns is not None:
            total_turn_values.append(num_turns)

    stats = {
        "source": payload.get("source"),
        "model": payload.get("model"),
        "num_items": payload.get("num_items"),
        "completed_task_count": len(compact_items),
        "avg_num_turns": payload.get("avg_num_turns")
        if isinstance(payload.get("avg_num_turns"), (int, float)) and not isinstance(payload.get("avg_num_turns"), bool)
        else (sum(numeric_turn_values) / len(numeric_turn_values) if numeric_turn_values else None),
        "turn_counted_task_count": payload.get("turn_counted_task_count")
        if isinstance(payload.get("turn_counted_task_count"), int)
        else len(numeric_turn_values),
        "avg_num_attempts": payload.get("avg_num_attempts")
        if isinstance(payload.get("avg_num_attempts"), (int, float)) and not isinstance(payload.get("avg_num_attempts"), bool)
        else (sum(attempt_values) / len(attempt_values) if attempt_values else None),
        "attempt_counted_task_count": payload.get("attempt_counted_task_count")
        if isinstance(payload.get("attempt_counted_task_count"), int)
        else len(attempt_values),
        "retry_task_count": payload.get("retry_task_count")
        if isinstance(payload.get("retry_task_count"), int)
        else (retry_task_count if attempt_values else None),
        "retry_ratio": payload.get("retry_ratio")
        if isinstance(payload.get("retry_ratio"), (int, float)) and not isinstance(payload.get("retry_ratio"), bool)
        else (retry_task_count / len(attempt_values) if attempt_values else None),
        "attempts_distribution": payload.get("attempts_distribution")
        if isinstance(payload.get("attempts_distribution"), dict)
        else _count_distribution(attempt_values, fill_from=1),
        "avg_total_turns": payload.get("avg_total_turns")
        if isinstance(payload.get("avg_total_turns"), (int, float)) and not isinstance(payload.get("avg_total_turns"), bool)
        else (sum(total_turn_values) / len(total_turn_values) if total_turn_values else None),
        "total_turn_counted_task_count": payload.get("total_turn_counted_task_count")
        if isinstance(payload.get("total_turn_counted_task_count"), int)
        else len(total_turn_values),
        "turns_distribution": payload.get("turns_distribution")
        if isinstance(payload.get("turns_distribution"), dict)
        else _count_distribution(total_turn_values),
        "correct": payload.get("correct"),
        "accuracy": payload.get("accuracy"),
        "script_started_at": payload.get("script_started_at"),
        "script_ended_at": payload.get("script_ended_at"),
        "tails": payload.get("tails", {}),
        "tool_latency_ms_by_tool_sum": payload.get("tool_latency_ms_by_tool_sum", {}),
        "tool_count_by_tool_sum": payload.get("tool_count_by_tool_sum", {}),
        "tool_success_by_tool_sum": payload.get("tool_success_by_tool_sum", {}),
        "tool_failed_by_tool_sum": payload.get("tool_failed_by_tool_sum", {}),
        "time_breakdown": payload.get("time_breakdown", {}),
        "latency_curve_row_count": len(payload.get("latency_curve_rows", [])) if isinstance(payload.get("latency_curve_rows"), list) else 0,
    }
    latency_rows = payload.get("latency_curve_rows", [])
    if not isinstance(latency_rows, list):
        latency_rows = []
    return stats, compact_items, [row for row in latency_rows if isinstance(row, dict)]


def _expected_task_fields(expected: ExpectedTaskCount) -> dict[str, Any]:
    return {
        "expected_tasks": expected.count,
        "expected_tasks_source": expected.source,
        "expected_tasks_warning": expected.warning,
        "dataset_path": expected.dataset_path,
        "dataset_max_tasks": expected.dataset_max_tasks,
    }


def build_dashboard_artifacts_payload(
    run_dir: str | Path,
    *,
    run_type: str | None = None,
    assistant_message_stats_artifact: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_path = Path(run_dir)
    expected = resolve_expected_task_count(run_path)
    eval_stats, eval_items, latency_rows = _eval_results_stats(run_path)
    judge_stats, judge_rows = _llm_judge_stats(run_path)
    generated_at = datetime.now(tz=UTC).isoformat()

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "run_dir": str(run_path),
        "run_type": run_type,
        **_expected_task_fields(expected),
        "eval_results": eval_stats,
        "llm_judge": judge_stats,
    }
    expected_count = expected.count
    judged = int(judge_stats["judged_task_count"])
    summary["progress_ratio"] = judged / expected_count if expected_count else None
    assistant_config = _assistant_message_stats_config(run_type)
    if assistant_config is not None:
        from recipe.common.log_processing.assistant_message_stats import (
            build_assistant_message_stats_summary,
            summarize_assistant_message_stats_artifact,
        )

        summary["assistant_message_stats"] = (
            summarize_assistant_message_stats_artifact(assistant_message_stats_artifact)
            if assistant_message_stats_artifact is not None
            else build_assistant_message_stats_summary(run_path, trace_dir_names=assistant_config)
        )

    events: list[dict[str, Any]] = [
        {
            "event_type": "run_summary",
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "run_dir": str(run_path),
            "run_type": run_type,
            **_expected_task_fields(expected),
        }
    ]
    events.extend({"event_type": "task_timing", **item} for item in eval_items)
    events.extend({"event_type": "latency_sample", **row} for row in latency_rows)
    events.extend({"event_type": "eval_result", **row} for row in judge_rows)
    return summary, events


def write_dashboard_artifacts(run_dir: str | Path, *, run_type: str | None = None) -> tuple[Path, Path]:
    run_path = Path(run_dir)
    assistant_config = _assistant_message_stats_config(run_type)
    assistant_message_stats_artifact: dict[str, Any] | None = None
    if assistant_config is not None:
        from recipe.common.log_processing.assistant_message_stats import build_assistant_message_stats_artifact

        assistant_message_stats_artifact = build_assistant_message_stats_artifact(run_path, trace_dir_names=assistant_config)
    summary, events = build_dashboard_artifacts_payload(
        run_path,
        run_type=run_type,
        assistant_message_stats_artifact=assistant_message_stats_artifact,
    )
    summary_path = run_path / SUMMARY_NAME
    events_path = run_path / EVENTS_NAME
    trace_distribution_path: Path | None = None
    assistant_message_stats_path: Path | None = None
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    events_path.write_text("\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n", encoding="utf-8")
    trace_config = _trace_distribution_config(run_type)
    if trace_config is not None:
        kind, trace_dir_names = trace_config
        trace_distribution_path = write_trace_distribution_artifact(run_path, kind=kind, trace_dir_names=trace_dir_names)
    if assistant_config is not None and assistant_message_stats_artifact is not None:
        from recipe.common.log_processing.assistant_message_stats import write_assistant_message_stats_artifact

        assistant_message_stats_path = write_assistant_message_stats_artifact(
            run_path,
            trace_dir_names=assistant_config,
            artifact=assistant_message_stats_artifact,
        )
    written_paths = [summary_path, events_path]
    if trace_distribution_path is not None:
        written_paths.append(trace_distribution_path)
    if assistant_message_stats_path is not None:
        written_paths.append(assistant_message_stats_path)
    logger.info("Wrote %s", ", ".join(str(path) for path in written_paths))
    return summary_path, events_path


def _trace_distribution_config(run_type: str | None) -> tuple[TraceKind, tuple[str, ...]] | None:
    if run_type == "web_search":
        return "agentic", ("web-search-benchmark",)
    if run_type == "wide_search":
        return "agentic", ("wide-search",)
    return None


def _assistant_message_stats_config(run_type: str | None) -> tuple[str, ...] | None:
    if run_type == "web_search":
        return ("web-search-benchmark",)
    if run_type == "wide_search":
        return ("wide-search",)
    return None


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build compact dashboard artifacts for completed or in-progress runs")
    parser.add_argument("--run-dir", type=Path, action="append", required=True, help="Run directory. Pass multiple times to process several runs.")
    parser.add_argument("--run-type", default=None, choices=["web_search", "wide_search"], help="Optional run type label to record")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for run_dir in args.run_dir:
        write_dashboard_artifacts(run_dir, run_type=args.run_type)


if __name__ == "__main__":
    _main()
