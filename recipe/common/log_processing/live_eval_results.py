# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct eval-results payloads from existing task traces."""

from __future__ import annotations

import json
import re
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from recipe.common.eval_results import TaskTimingRow, build_eval_results_payload, compute_non_model_overhead_s
from recipe.common.log_processing.trace_refs import add_trace_refs, is_attempt_budget_branch_stem

UTC8 = timezone(timedelta(hours=8))


def _sorted_attempts(attempts: dict[int, str]) -> list[int]:
    return sorted(attempts)


def _all_task_ids(*indexes: dict[str, dict[int, str]]) -> list[str]:
    merged: set[str] = set()
    for index in indexes:
        merged.update(index)
    return sorted(merged, key=lambda value: int(value) if value.isdigit() else value)


def _parse_agentic_trace_stem(stem: str) -> tuple[str, int] | None:
    match = re.match(r"(?P<base>.+)_attempt-(?P<attempt>\d+)(?:$|_)", stem)
    if not match:
        return None
    return match.group("base"), int(match.group("attempt"))


def scan_agentic_trace_index(run_dir: str | Path, trace_dir_names: tuple[str, ...]) -> dict[str, dict[int, str]]:
    run_path = Path(run_dir)
    index: dict[str, dict[int, str]] = {}
    for name in trace_dir_names:
        trace_dir = run_path / name
        if not trace_dir.exists():
            continue
        add_trace_refs(index, trace_dir)
        for path in sorted(trace_dir.glob("*.json")):
            if path.name == "trace_refs.json":
                continue
            if path.stat().st_size == 0:
                continue
            if is_attempt_budget_branch_stem(path.stem):
                continue
            parsed = _parse_agentic_trace_stem(path.stem)
            if parsed is None:
                continue
            task_id, attempt = parsed
            index.setdefault(task_id, {})[attempt] = str(path)
    return index


def _load_json_cached(path: str) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
        return {"_load_error": f"{type(exc).__name__}: {exc}", "_path": path}


def _ori_message_history(data: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
    history = data.get("main_agent_message_history", {})
    if isinstance(history, (dict, list)):
        return history
    return {}


def _ori_messages(data: dict[str, Any]) -> list[dict[str, Any]]:
    history = _ori_message_history(data)
    if isinstance(history, list):
        return [m for m in history if isinstance(m, dict)]
    msgs = history.get("message_history", [])
    if isinstance(msgs, list):
        return [m for m in msgs if isinstance(m, dict)]
    return []


def _ori_tool_calls_paired(data: dict[str, Any]) -> list[dict[str, Any]]:
    steps = data.get("step_logs", [])
    starts: list[dict[str, Any]] = []
    for step in steps:
        name = step.get("step_name", "")
        msg = step.get("message", "")
        if "Tool Call Start" in name:
            m_tool = re.search(r"tool '(\w+)'", msg)
            m_server = re.search(r"server '([^']+)'", msg)
            starts.append(
                {
                    "tool_name": m_tool.group(1) if m_tool else "unknown",
                    "server_name": m_server.group(1) if m_server else "",
                    "arguments": step.get("metadata", {}).get("arguments", {}),
                    "latency_ms": 0,
                    "status": "success",
                    "timestamp": step.get("timestamp", ""),
                }
            )
    turn_idx = 0
    for step in steps:
        name = step.get("step_name", "")
        msg = step.get("message", "")
        if "Turn:" in name and "Tool Call" in name:
            m_lat = re.search(r"completed in (\d+)ms", msg)
            if m_lat and turn_idx < len(starts):
                starts[turn_idx]["latency_ms"] = int(m_lat.group(1))
                starts[turn_idx]["timestamp"] = step.get("timestamp", starts[turn_idx].get("timestamp", ""))
                turn_idx += 1
    return starts


def _ori_per_step_tokens(data: dict[str, Any]) -> list[dict[str, int]]:
    steps = data.get("step_logs", [])
    cumulative: list[tuple[int, int]] = []
    for step in steps:
        if "Token Usage" in step.get("step_name", ""):
            match = re.search(r"Input: (\d+).*Output: (\d+)", step.get("message", ""))
            if match:
                cumulative.append((int(match.group(1)), int(match.group(2))))
    per_step: list[dict[str, int]] = []
    prev_in, prev_out = 0, 0
    for inp, out in cumulative:
        per_step.append({"input_tokens": inp - prev_in, "output_tokens": out - prev_out})
        prev_in, prev_out = inp, out
    return per_step


def _safe_ms(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(parsed, 0.0)


def _parse_datetime_ms(dt_str: str | None) -> float | None:
    if not dt_str or dt_str == "?":
        return None

    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        with suppress(ValueError):
            return datetime.strptime(dt_str, fmt).replace(tzinfo=UTC8).timestamp() * 1000
    with suppress(ValueError, TypeError):
        parsed = datetime.fromisoformat(dt_str)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC8)
        return parsed.timestamp() * 1000
    return None


_STEP_CATEGORIES: dict[str, str] = {
    "model_client": "Model Client",
    "tool": "Tool Execution",
    "runtime": "Conversation Runtime",
    "orchestrator": "Orchestrator",
}


def _classify_step(step_name: str) -> str:
    bare = re.sub(r"^[^\w]+", "", step_name)
    for prefix, label in _STEP_CATEGORIES.items():
        if bare.startswith(prefix):
            return label
    return "Other"


def _extract_step_timing(steps: list[dict[str, Any]]) -> dict[str, Any]:
    by_cat: dict[str, float] = {}
    counts_by_cat: dict[str, int] = {}
    for step in steps:
        cat = _classify_step(step.get("step_name", ""))
        elapsed = step.get("metadata", {}).get("elapsed_ms")
        if elapsed is not None:
            by_cat[cat] = by_cat.get(cat, 0) + elapsed
            counts_by_cat[cat] = counts_by_cat.get(cat, 0) + 1
    return {
        "by_category": by_cat,
        "counts_by_category": counts_by_cat,
        "total_ms": sum(by_cat.values()),
    }


_KNOWN_TOOLS = {"google_search", "web_search", "scrape_and_extract_info", "scrape_and_extract"}


def _bucket_tool_name(raw_name: str) -> str:
    if raw_name == "web_search":
        return "Tool: google_search"
    if raw_name == "scrape_and_extract":
        return "Tool: scrape_and_extract_info"
    if raw_name in _KNOWN_TOOLS:
        return f"Tool: {raw_name}"
    return "Tool: Other Tools"


def _bucket_per_tool(raw: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, ms in raw.items():
        bucket = _bucket_tool_name(name)
        out[bucket] = out.get(bucket, 0) + ms
    return out


def _bucket_per_tool_counts(raw: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, count in raw.items():
        bucket = _bucket_tool_name(name)
        out[bucket] = out.get(bucket, 0) + count
    return out


def _expand_agentic_tool_timing(tool_call: dict[str, Any]) -> dict[str, float]:
    tool_name = tool_call.get("tool_name", "unknown")
    latency_ms = _safe_ms(tool_call.get("latency_ms", 0))
    timing_ms = tool_call.get("metadata", {}).get("timing_ms", {})
    if tool_name in {"google_search", "web_search"}:
        if not isinstance(timing_ms, dict):
            return {_bucket_tool_name(tool_name): latency_ms}
        serper_ms = _safe_ms(timing_ms.get("serper_request") or timing_ms.get("request_ms"))
        residual_ms = max(latency_ms - serper_ms, 0.0)
        expanded: dict[str, float] = {}
        if serper_ms > 0:
            expanded["Tool: serper"] = serper_ms
        if residual_ms > 0:
            expanded["Tool: google_search overhead"] = residual_ms
        return expanded or {_bucket_tool_name(tool_name): latency_ms}

    if tool_name not in {"scrape_and_extract_info", "scrape_and_extract"}:
        return {_bucket_tool_name(tool_name): latency_ms}
    if not isinstance(timing_ms, dict):
        return {_bucket_tool_name(tool_name): latency_ms}

    expanded = {}
    jina_ms = _safe_ms(timing_ms.get("jina") or timing_ms.get("jina_scrape"))
    llm_ms = _safe_ms(timing_ms.get("llm_extraction"))
    accounted = 0.0
    if jina_ms > 0:
        expanded["Tool: jina"] = jina_ms
        accounted += jina_ms
    if llm_ms > 0:
        expanded["Tool: LLM extraction"] = llm_ms
        accounted += llm_ms
    residual_ms = max(latency_ms - accounted, 0.0)
    if residual_ms > 0:
        expanded["Tool: scrape_and_extract_info overhead"] = residual_ms
    return expanded or {_bucket_tool_name(tool_name): latency_ms}


def _ori_infer_model_timing(data: dict[str, Any]) -> tuple[float, int]:
    model_ms = 0.0
    model_call_count = 0
    pending_start_ms: float | None = None
    for step in data.get("step_logs", []) or []:
        step_name = step.get("step_name", "")
        if "Message Retention" in step_name:
            pending_start_ms = _parse_datetime_ms(step.get("timestamp"))
        elif "Token Usage" in step_name:
            end_ms = _parse_datetime_ms(step.get("timestamp"))
            if pending_start_ms is not None and end_ms is not None:
                model_ms += max(end_ms - pending_start_ms, 0.0)
                model_call_count += 1
            pending_start_ms = None
    return model_ms, model_call_count


def _ori_time_breakdown(data_list: list[dict[str, Any]]) -> dict[str, Any]:
    raw_per_tool: dict[str, float] = {}
    raw_tool_counts: dict[str, int] = {}
    total_task_ms = 0.0
    inferred_model_ms = 0.0
    model_call_count = 0
    for data in data_list:
        start_ms = _parse_datetime_ms(data.get("start_time"))
        end_ms = _parse_datetime_ms(data.get("end_time"))
        if start_ms is None or end_ms is None:
            continue
        total_task_ms += end_ms - start_ms
        attempt_model_ms, attempt_model_calls = _ori_infer_model_timing(data)
        inferred_model_ms += attempt_model_ms
        model_call_count += attempt_model_calls
        for tc in _ori_tool_calls_paired(data):
            raw_per_tool[tc["tool_name"]] = raw_per_tool.get(tc["tool_name"], 0) + tc["latency_ms"]
            raw_tool_counts[tc["tool_name"]] = raw_tool_counts.get(tc["tool_name"], 0) + 1
    per_tool = _bucket_per_tool(raw_per_tool)
    tool_total = sum(per_tool.values())
    model_and_overhead = max(total_task_ms - tool_total, 0) if total_task_ms > 0 else 0
    model_client = min(inferred_model_ms, model_and_overhead)
    return {
        "per_tool": per_tool,
        "tool_counts": _bucket_per_tool_counts(raw_tool_counts),
        "model_client_ms": model_client,
        "model_client_ms_estimated_raw": inferred_model_ms,
        "model_and_overhead_ms": model_and_overhead,
        "other_ms": max(model_and_overhead - model_client, 0),
        "model_call_count": model_call_count,
        "total_ms": total_task_ms,
        "tool_total_ms": tool_total,
    }


def _agn_time_breakdown_aggregated(data_list: list[dict[str, Any]]) -> dict[str, Any]:
    merged_by_cat: dict[str, float] = {}
    counts_by_cat: dict[str, int] = {}
    total_ms = 0.0
    raw_per_tool: dict[str, float] = {}
    tool_counts: dict[str, int] = {}
    for data in data_list:
        timing = _extract_step_timing(data.get("steps", []))
        for cat, ms in timing["by_category"].items():
            merged_by_cat[cat] = merged_by_cat.get(cat, 0) + ms
        for cat, count in timing.get("counts_by_category", {}).items():
            counts_by_cat[cat] = counts_by_cat.get(cat, 0) + count
        total_ms += timing["total_ms"]
        for tc in data.get("tool_calls", []):
            for label, ms in _expand_agentic_tool_timing(tc).items():
                raw_per_tool[label] = raw_per_tool.get(label, 0) + ms
                tool_counts[label] = tool_counts.get(label, 0) + 1

    return {
        "total_ms": total_ms,
        "by_category": merged_by_cat,
        "counts_by_category": counts_by_cat,
        "per_tool": dict(raw_per_tool),
        "tool_counts": tool_counts,
    }


def _agentic_tool_execution_timestamps(data: dict[str, Any]) -> list[str]:
    timestamps: list[str] = []
    for step in data.get("steps", []) or []:
        if "tool.execution" not in step.get("step_name", ""):
            continue
        msg = step.get("message", "")
        match = re.search(r"processed\s+(\d+)\s+tool call", msg)
        count = int(match.group(1)) if match else 1
        timestamps.extend([step.get("timestamp", "")] * max(count, 1))
    return timestamps


def _latency_event(metric: str, task_id: str, attempt: int, call_idx: int, ts_ms: float, latency_ms: float) -> dict[str, Any]:
    return {
        "metric": metric,
        "time": datetime.fromtimestamp(ts_ms / 1000, tz=UTC8).isoformat(),
        "latency_s": latency_ms / 1000,
        "task_id": task_id,
        "attempt": attempt,
        "call": call_idx,
    }


def _aggregate_latency_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, datetime], dict[str, Any]] = {}
    for event in events:
        metric = str(event.get("metric", ""))
        latency_s = event.get("latency_s")
        if metric not in {"search", "extract", "model"} or not isinstance(latency_s, (int, float)) or isinstance(latency_s, bool):
            continue
        with suppress(ValueError, TypeError):
            parsed = datetime.fromisoformat(str(event.get("time")))
            bucket = parsed.replace(minute=(parsed.minute // 10) * 10, second=0, microsecond=0)
            key = (metric, bucket)
            row = grouped.setdefault(key, {"metric": metric, "time": bucket.isoformat(), "latency_s_sum": 0.0, "calls": 0})
            row["latency_s_sum"] += float(latency_s)
            row["calls"] += 1

    rows: list[dict[str, Any]] = []
    for row in grouped.values():
        calls = int(row["calls"])
        if calls <= 0:
            continue
        rows.append(
            {
                "metric": row["metric"],
                "time": row["time"],
                "latency_s": float(row["latency_s_sum"]) / calls,
                "calls": calls,
            }
        )
    rows.sort(key=lambda row: (str(row["metric"]), str(row["time"])))
    return rows


def _add_original_latency_events(events: list[dict[str, Any]], task_id: str, attempt: int, data: dict[str, Any]) -> None:
    for call_idx, tc in enumerate(_ori_tool_calls_paired(data)):
        ts_ms = _parse_datetime_ms(tc.get("timestamp"))
        if ts_ms is None:
            continue
        if tc.get("tool_name") == "google_search":
            events.append(_latency_event("search", task_id, attempt, call_idx, ts_ms, _safe_ms(tc.get("latency_ms"))))
        elif tc.get("tool_name") == "scrape_and_extract_info":
            events.append(_latency_event("extract", task_id, attempt, call_idx, ts_ms, _safe_ms(tc.get("latency_ms"))))

    pending_start_ms: float | None = None
    model_call_idx = 0
    for step in data.get("step_logs", []) or []:
        step_name = step.get("step_name", "")
        if "Message Retention" in step_name:
            pending_start_ms = _parse_datetime_ms(step.get("timestamp"))
        elif "Token Usage" in step_name:
            end_ms = _parse_datetime_ms(step.get("timestamp"))
            if pending_start_ms is not None and end_ms is not None:
                events.append(_latency_event("model", task_id, attempt, model_call_idx, end_ms, max(end_ms - pending_start_ms, 0.0)))
                model_call_idx += 1
            pending_start_ms = None


def _add_agentic_latency_events(events: list[dict[str, Any]], task_id: str, attempt: int, data: dict[str, Any]) -> None:
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
            events.append(_latency_event("search", task_id, attempt, call_idx, ts_ms, latency_ms))
        elif tc.get("tool_name") in {"scrape_and_extract_info", "scrape_and_extract"}:
            latency_ms = _safe_ms(timing_ms.get("jina") or timing_ms.get("jina_scrape")) + _safe_ms(timing_ms.get("llm_extraction"))
            if latency_ms <= 0:
                latency_ms = _safe_ms(tc.get("latency_ms"))
            events.append(_latency_event("extract", task_id, attempt, call_idx, ts_ms, latency_ms))

    model_call_idx = 0
    for step in data.get("steps", []) or []:
        if "model_client.inference" not in step.get("step_name", ""):
            continue
        elapsed_ms = step.get("metadata", {}).get("elapsed_ms")
        ts_ms = _parse_datetime_ms(step.get("timestamp"))
        if not isinstance(elapsed_ms, (int, float)) or ts_ms is None:
            continue
        events.append(_latency_event("model", task_id, attempt, model_call_idx, ts_ms, max(float(elapsed_ms), 0.0)))
        model_call_idx += 1


def _build_agentic_timing_extras_from_data(all_data: list[dict[str, Any]], latency_events: list[dict[str, Any]]) -> dict[str, Any]:
    if not all_data:
        return {}
    return {
        "time_breakdown": _agn_time_breakdown_aggregated(all_data),
        "latency_curve_rows": _aggregate_latency_events(latency_events),
    }


def build_agentic_timing_extras_from_index(agn_index: dict[str, dict[int, str]]) -> dict[str, Any]:
    """Build dashboard-only aggregate timing fields from agentic trace files."""
    all_data: list[dict[str, Any]] = []
    latency_events: list[dict[str, Any]] = []
    for task_id in _all_task_ids(agn_index):
        attempts = agn_index.get(task_id, {})
        for attempt in _sorted_attempts(attempts):
            data = _load_json_cached(attempts[attempt])
            if data.get("_load_error"):
                continue
            all_data.append(data)
            _add_agentic_latency_events(latency_events, task_id, attempt, data)
    return {
        **_build_agentic_timing_extras_from_data(all_data, latency_events),
        **_agentic_attempt_usage_extras_from_index(agn_index),
    }


def _load_eval_results(run_dir: str) -> dict[str, Any] | None:
    """Load ``eval_results.json`` from a run directory (returns ``None`` if absent).

    Both pipelines now emit this file:
    * agentic: written directly by ``runners/evaluate_benchmark.py``
    * offline traces: written by an offline post-processing step

    The payload shape is the superset defined by ``recipe.common.eval_results``.
    """
    path = Path(run_dir) / "eval_results.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return None


def _live_run_bounds(rows: list[dict[str, Any]]) -> tuple[str | None, str | None, float | None]:
    starts = [r["start_ms"] for r in rows if isinstance(r.get("start_ms"), (int, float))]
    ends = [r["end_ms"] for r in rows if isinstance(r.get("end_ms"), (int, float))]
    if not starts:
        return None, None, None
    start_ms = min(starts)
    end_ms = max(ends) if ends else None
    started_at = datetime.fromtimestamp(start_ms / 1000, tz=UTC8).isoformat()
    ended_at = datetime.fromtimestamp(end_ms / 1000, tz=UTC8).isoformat() if end_ms is not None else None
    elapsed_s = (end_ms - start_ms) / 1000 if end_ms is not None else None
    return started_at, ended_at, elapsed_s


def _tool_stats_from_calls(
    tool_calls: list[dict[str, Any]],
) -> tuple[
    dict[str, float],
    dict[str, int],
    dict[str, int],
    dict[str, int],
    float | None,
    float | None,
    float | None,
    int | None,
]:
    latencies: list[float] = []
    by_tool: dict[str, float] = {}
    counts: dict[str, int] = {}
    successes: dict[str, int] = {}
    failures: dict[str, int] = {}
    for tc in tool_calls:
        tool = str(tc.get("tool_name") or "unknown")
        latency = _safe_ms(tc.get("latency_ms"))
        by_tool[tool] = by_tool.get(tool, 0.0) + latency
        counts[tool] = counts.get(tool, 0) + 1
        success = tc.get("success")
        if isinstance(success, bool):
            succeeded = success
        else:
            status = str(tc.get("status", "") or "").strip().lower()
            succeeded = status in {"success", "successful", "succeeded", "ok", "completed"} if status else not tc.get("error")
        if succeeded:
            successes[tool] = successes.get(tool, 0) + 1
        else:
            failures[tool] = failures.get(tool, 0) + 1
        latencies.append(latency)
    if not latencies:
        return {}, {}, {}, {}, None, None, None, None
    total = sum(latencies)
    return by_tool, counts, successes, failures, total, total / len(latencies), max(latencies), len(latencies)


def _agentic_task_elapsed_s(data_list: list[dict[str, Any]], breakdown: dict[str, Any]) -> float | None:
    elapsed_ms_values = [
        max((end_ms or 0) - (start_ms or 0), 0)
        for data in data_list
        if (start_ms := _parse_datetime_ms(data.get("started_at"))) is not None and (end_ms := _parse_datetime_ms(data.get("ended_at"))) is not None
    ]
    if elapsed_ms_values:
        return sum(elapsed_ms_values) / 1000
    total_ms = breakdown.get("total_ms", 0)
    return total_ms / 1000 if total_ms > 0 else None


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _agentic_attempt_turn_count(data: dict[str, Any]) -> int | None:
    meta_turns = _int_value(data.get("metadata", {}).get("turn_used"))
    if meta_turns is not None:
        return meta_turns
    conv = data.get("conversation", [])
    if not isinstance(conv, list) or not conv:
        return None
    assistant_turns = sum(1 for m in conv if isinstance(m, dict) and m.get("role") == "assistant")
    # Match by marker name, not content substring: compaction markers carry
    # free-form summary text that may mention "rollback". Fall back to the
    # substring heuristic only for legacy traces without a name field.
    rollbacks = sum(
        1
        for m in conv
        if isinstance(m, dict)
        and m.get("role") == "ConversationRuntime"
        and (m.get("name") == "rollback" or (m.get("name") is None and "rollback" in str(m.get("content", "")).lower()))
    )
    return max(assistant_turns - rollbacks, 0)


def _agentic_total_turn_count(data_list: list[dict[str, Any]]) -> int | None:
    turns = [_agentic_attempt_turn_count(data) for data in data_list]
    counted = [turn for turn in turns if turn is not None]
    return sum(counted) if counted else None


def _agentic_last_attempt_turn_count(data_list: list[dict[str, Any]]) -> int | None:
    if not data_list:
        return None
    return _agentic_attempt_turn_count(data_list[-1])


def _ori_attempt_turn_count(data: dict[str, Any]) -> int | None:
    meta_turns = _int_value(data.get("metadata", {}).get("turn_used"))
    if meta_turns is not None:
        return meta_turns
    msgs = _ori_messages(data)
    return sum(1 for m in msgs if m.get("role") == "assistant")


def _count_distribution(values: list[int], *, fill_from: int | None = None) -> dict[str, int]:
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if fill_from is not None and values:
        for value in range(fill_from, max(values) + 1):
            counts.setdefault(value, 0)
    return {str(value): counts[value] for value in sorted(counts)}


def _agentic_attempt_usage_extras_from_index(agn_index: dict[str, dict[int, str]]) -> dict[str, Any]:
    final_turn_values: list[int] = []
    attempt_values: list[int] = []
    total_turn_values: list[int] = []

    for task_id in _all_task_ids(agn_index):
        attempts = agn_index.get(task_id, {})
        valid_attempt_numbers: list[int] = []
        turn_counts: list[int] = []
        for attempt in _sorted_attempts(attempts):
            data = _load_json_cached(attempts[attempt])
            if data.get("_load_error"):
                continue
            valid_attempt_numbers.append(attempt)
            turn_count = _agentic_attempt_turn_count(data)
            if turn_count is not None:
                turn_counts.append(turn_count)
        if valid_attempt_numbers:
            attempt_values.append(max(valid_attempt_numbers))
        if turn_counts:
            final_turn_values.append(turn_counts[-1])
            total_turn_values.append(sum(turn_counts))

    retry_task_count = sum(1 for value in attempt_values if value > 1)
    return {
        "avg_num_turns": sum(final_turn_values) / len(final_turn_values) if final_turn_values else None,
        "turn_counted_task_count": len(final_turn_values),
        "avg_num_attempts": sum(attempt_values) / len(attempt_values) if attempt_values else None,
        "attempt_counted_task_count": len(attempt_values),
        "retry_task_count": retry_task_count if attempt_values else None,
        "retry_ratio": retry_task_count / len(attempt_values) if attempt_values else None,
        "attempts_distribution": _count_distribution(attempt_values, fill_from=1),
        "total_turn_counted_task_count": len(total_turn_values),
        "avg_total_turns": sum(total_turn_values) / len(total_turn_values) if total_turn_values else None,
        "turns_distribution": _count_distribution(total_turn_values),
    }


def _agentic_timing_row(
    task_id: str,
    data_list: list[dict[str, Any]],
    attempt_numbers: list[int],
    agn_evals: dict[str, dict[str, Any]],
) -> TaskTimingRow:
    breakdown = _agn_time_breakdown_aggregated(data_list)
    tool_calls: list[dict[str, Any]] = []
    for data in data_list:
        tool_calls.extend(data.get("tool_calls", []) or [])
    by_tool, counts, successes, failures, tool_total, tool_mean, tool_max, tool_count = _tool_stats_from_calls(tool_calls)
    task_elapsed_s = _agentic_task_elapsed_s(data_list, breakdown)
    model_ms = (breakdown.get("by_category") or {}).get("Model Client", 0)
    model_s = model_ms / 1000 if model_ms > 0 else None
    input_tokens = sum(int(d.get("token_usage", {}).get("input_tokens") or 0) for d in data_list)
    output_tokens = sum(int(d.get("token_usage", {}).get("output_tokens") or 0) for d in data_list)
    total_tokens = sum(int(d.get("token_usage", {}).get("total_tokens") or 0) for d in data_list) or input_tokens + output_tokens
    try:
        idx = int(task_id)
    except ValueError:
        idx = None
    return TaskTimingRow(
        task_id=task_id,
        idx=idx,
        score=agn_evals.get(task_id, {}).get("score"),
        num_turns=_agentic_last_attempt_turn_count(data_list),
        num_attempts=max(attempt_numbers) if attempt_numbers else None,
        total_turns=_agentic_total_turn_count(data_list),
        finish_reason=(data_list[-1].get("metadata") or {}).get("done_reason") or data_list[-1].get("status"),
        cached=False,
        task_elapsed_s=task_elapsed_s,
        model_client_elapsed_s=model_s,
        group_completion_elapsed_s=None,
        non_model_overhead_s=compute_non_model_overhead_s(task_elapsed_s, model_s, tool_total),
        tool_latency_ms_sum=tool_total,
        tool_latency_ms_mean=tool_mean,
        tool_latency_ms_max=tool_max,
        tool_count=tool_count,
        tool_latency_ms_by_tool=by_tool,
        tool_count_by_tool=counts,
        tool_success_by_tool=successes,
        tool_failed_by_tool=failures,
        input_tokens=input_tokens or None,
        output_tokens=output_tokens or None,
        total_tokens=total_tokens or None,
    )


def _build_live_ori_eval_results(run_dir: str, ori_index: dict[str, dict[int, str]], ori_evals: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    del run_dir  # The index already contains the trace paths.
    rows: list[TaskTimingRow] = []
    bounds: list[dict[str, Any]] = []
    all_data: list[dict[str, Any]] = []
    latency_events: list[dict[str, Any]] = []
    for task_id in _all_task_ids(ori_index):
        attempts = ori_index.get(task_id, {})
        data_list: list[dict[str, Any]] = []
        attempt_numbers: list[int] = []
        start_ms_values: list[float] = []
        end_ms_values: list[float] = []
        for attempt in _sorted_attempts(attempts):
            data = _load_json_cached(attempts[attempt])
            if data.get("_load_error"):
                continue
            data_list.append(data)
            attempt_numbers.append(attempt)
            all_data.append(data)
            _add_original_latency_events(latency_events, task_id, attempt, data)
            start_ms = _parse_datetime_ms(data.get("start_time"))
            end_ms = _parse_datetime_ms(data.get("end_time"))
            if start_ms is not None:
                start_ms_values.append(start_ms)
            if end_ms is not None:
                end_ms_values.append(end_ms)
        if not data_list:
            continue

        breakdown = _ori_time_breakdown(data_list)
        tool_calls: list[dict[str, Any]] = []
        for data in data_list:
            tool_calls.extend(_ori_tool_calls_paired(data))
        by_tool, counts, successes, failures, tool_total, tool_mean, tool_max, tool_count = _tool_stats_from_calls(tool_calls)
        input_tokens = output_tokens = total_tokens = 0
        for data in data_list:
            per_step_tokens = _ori_per_step_tokens(data)
            input_tokens += sum(int(row.get("input_tokens") or 0) for row in per_step_tokens)
            output_tokens += sum(int(row.get("output_tokens") or 0) for row in per_step_tokens)
            total_tokens = input_tokens + output_tokens
        turn_counts = [_ori_attempt_turn_count(data) for data in data_list]
        counted_turns = [turn for turn in turn_counts if turn is not None]

        task_elapsed_s = breakdown["total_ms"] / 1000 if breakdown.get("total_ms", 0) > 0 else None
        model_s = breakdown["model_client_ms"] / 1000 if breakdown.get("model_client_ms", 0) > 0 else None
        score = ori_evals.get(task_id, {}).get("score")
        try:
            idx = int(task_id)
        except ValueError:
            idx = None
        rows.append(
            TaskTimingRow(
                task_id=task_id,
                idx=idx,
                score=score,
                num_turns=counted_turns[-1] if counted_turns else None,
                num_attempts=max(attempt_numbers) if attempt_numbers else None,
                total_turns=sum(counted_turns) if counted_turns else None,
                finish_reason=data_list[-1].get("status") or data_list[-1].get("final_judge_result"),
                cached=False,
                task_elapsed_s=task_elapsed_s,
                model_client_elapsed_s=model_s,
                group_completion_elapsed_s=None,
                non_model_overhead_s=compute_non_model_overhead_s(task_elapsed_s, model_s, tool_total),
                tool_latency_ms_sum=tool_total,
                tool_latency_ms_mean=tool_mean,
                tool_latency_ms_max=tool_max,
                tool_count=tool_count,
                tool_latency_ms_by_tool=by_tool,
                tool_count_by_tool=counts,
                tool_success_by_tool=successes,
                tool_failed_by_tool=failures,
                input_tokens=input_tokens or None,
                output_tokens=output_tokens or None,
                total_tokens=total_tokens or None,
            )
        )
        if start_ms_values:
            bounds.append({"start_ms": min(start_ms_values), "end_ms": max(end_ms_values) if end_ms_values else None})

    if not rows:
        return None
    started_at, ended_at, elapsed_s = _live_run_bounds(bounds)
    payload = build_eval_results_payload(
        rows,
        source="ori-live-traces",
        num_items=len(rows),
        script_started_at=started_at,
        script_ended_at=ended_at,
        start_to_end_elapsed_s=elapsed_s,
        total_eval_elapsed_s=elapsed_s,
        extra={
            "live_trace_fallback": True,
            "time_breakdown": _ori_time_breakdown(all_data),
            "latency_curve_rows": _aggregate_latency_events(latency_events),
        },
    )
    return payload


def _build_live_agentic_eval_results(
    run_dir: str,
    agn_index: dict[str, dict[int, str]],
    agn_evals: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    del run_dir  # The index already contains the trace paths.
    rows: list[TaskTimingRow] = []
    bounds: list[dict[str, Any]] = []
    all_data: list[dict[str, Any]] = []
    latency_events: list[dict[str, Any]] = []
    for task_id in _all_task_ids(agn_index):
        attempts = agn_index.get(task_id, {})
        data_list: list[dict[str, Any]] = []
        attempt_numbers: list[int] = []
        start_ms_values: list[float] = []
        end_ms_values: list[float] = []
        for attempt in _sorted_attempts(attempts):
            data = _load_json_cached(attempts[attempt])
            if data.get("_load_error"):
                continue
            data_list.append(data)
            attempt_numbers.append(attempt)
            all_data.append(data)
            _add_agentic_latency_events(latency_events, task_id, attempt, data)
            start_ms = _parse_datetime_ms(data.get("started_at"))
            end_ms = _parse_datetime_ms(data.get("ended_at"))
            if start_ms is not None:
                start_ms_values.append(start_ms)
            if end_ms is not None:
                end_ms_values.append(end_ms)
        if not data_list:
            continue

        rows.append(_agentic_timing_row(task_id, data_list, attempt_numbers, agn_evals))
        if start_ms_values:
            bounds.append({"start_ms": min(start_ms_values), "end_ms": max(end_ms_values) if end_ms_values else None})

    if not rows:
        return None
    started_at, ended_at, elapsed_s = _live_run_bounds(bounds)
    return build_eval_results_payload(
        rows,
        source="agentic-live-traces",
        num_items=len(rows),
        script_started_at=started_at,
        script_ended_at=ended_at,
        start_to_end_elapsed_s=elapsed_s,
        total_eval_elapsed_s=elapsed_s,
        extra={
            "live_trace_fallback": True,
            **_build_agentic_timing_extras_from_data(all_data, latency_events),
        },
    )
