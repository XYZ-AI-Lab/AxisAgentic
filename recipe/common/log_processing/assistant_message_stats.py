# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Assistant-message statistics for agentic trace logs."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recipe.common.log_processing.trace_distributions import (
    DEFAULT_AGENTIC_TRACE_DIRS,
    AttemptScope,
    load_json,
    scan_trace_index,
)

SCHEMA_VERSION = 3
ARTIFACT_NAME = "assistant_message_stats.json"
DEFAULT_ATTEMPT_SCOPES: tuple[AttemptScope, ...] = ("latest", "all")
ASSISTANT_OUTPUT_LENGTH_UNIT = "chars"
ASSISTANT_OUTPUT_LENGTH_DISTRIBUTION_KEYS: tuple[str, ...] = ("all", "with_tool_calls", "without_tool_calls")

VALIDITY_ORDER: tuple[str, ...] = (
    "none_valid",
    "content_only",
    "reasoning_content_only",
    "tool_calls_only",
    "content_and_reasoning_content",
    "content_and_tool_calls",
    "reasoning_content_and_tool_calls",
    "content_reasoning_content_and_tool_calls",
)
VALIDITY_LABELS: dict[str, str] = {
    "none_valid": "None valid",
    "content_only": "Content only",
    "reasoning_content_only": "Reasoning content only",
    "tool_calls_only": "Tool calls only",
    "content_and_reasoning_content": "Content + reasoning content",
    "content_and_tool_calls": "Content + tool calls",
    "reasoning_content_and_tool_calls": "Reasoning content + tool calls",
    "content_reasoning_content_and_tool_calls": "Content + reasoning content + tool calls",
}

TOOL_CALL_MESSAGE_ORDER: tuple[str, ...] = ("single_tool_call", "multiple_tool_calls")
TOOL_CALL_MESSAGE_LABELS: dict[str, str] = {
    "single_tool_call": "Single tool call",
    "multiple_tool_calls": "Multiple tool calls",
}

TRACE_TOOL_CALL_CATEGORY_ORDER: tuple[str, ...] = ("no_tool_calls", "single_tool_call_trace", "multi_tool_call_trace")
TRACE_TOOL_CALL_CATEGORY_LABELS: dict[str, str] = {
    "no_tool_calls": "No tool calls",
    "single_tool_call_trace": "Only single-call assistant messages",
    "multi_tool_call_trace": "Has multi-call assistant message",
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssistantTraceStatsRecord:
    task_id: str
    attempt: int
    path: str
    assistant_message_count: int
    assistant_output_chars: int
    avg_assistant_output_chars: float | None
    assistant_output_length_distribution: dict[str, int]
    tool_call_assistant_output_length_distribution: dict[str, int]
    non_tool_call_assistant_output_length_distribution: dict[str, int]
    validity_counts: dict[str, int]
    tool_call_message_count: int
    single_tool_call_message_count: int
    multi_tool_call_message_count: int
    max_tool_calls_in_message: int
    trace_tool_call_category: str
    tool_call_count_distribution: dict[str, int]


def build_assistant_message_stats_records(
    trace_index: Mapping[str, Mapping[int, str]],
    *,
    attempt_scope: AttemptScope = "latest",
) -> list[AssistantTraceStatsRecord]:
    records: list[AssistantTraceStatsRecord] = []
    for task_id in _sorted_task_ids(trace_index):
        attempts = trace_index.get(task_id, {})
        for attempt, path in _selected_attempts(attempts, attempt_scope=attempt_scope):
            data = load_json(path)
            if not data:
                continue
            records.append(_build_trace_record(task_id=task_id, attempt=attempt, path=path, data=data))
    return records


def build_assistant_message_stats_payload(
    trace_index: Mapping[str, Mapping[int, str]],
    *,
    attempt_scope: AttemptScope = "latest",
) -> dict[str, Any]:
    records = build_assistant_message_stats_records(trace_index, attempt_scope=attempt_scope)
    assistant_message_count = sum(record.assistant_message_count for record in records)
    assistant_output_chars = sum(record.assistant_output_chars for record in records)
    assistant_output_length_distributions = _assistant_output_length_distributions(records)
    validity_counts = _merge_record_count_maps(records, "validity_counts", order=VALIDITY_ORDER)
    single_message_count = sum(record.single_tool_call_message_count for record in records)
    multi_message_count = sum(record.multi_tool_call_message_count for record in records)
    tool_call_message_count = single_message_count + multi_message_count
    trace_category_counts = Counter(record.trace_tool_call_category for record in records)
    tool_call_count_values: list[int] = []
    for record in records:
        for raw_count, count in record.tool_call_count_distribution.items():
            tool_call_count_values.extend([int(raw_count)] * count)

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "agentic",
        "attempt_scope": attempt_scope,
        "trace_count": len(records),
        "assistant_message_count": assistant_message_count,
        "assistant_output_length": {
            "unit": ASSISTANT_OUTPUT_LENGTH_UNIT,
            "total": assistant_output_chars,
            "average": _average(assistant_output_chars, assistant_message_count),
            "message_count": assistant_message_count,
            "distributions": assistant_output_length_distributions,
        },
        "message_validity": {
            "total": sum(validity_counts.values()),
            "counts": _category_rows(validity_counts, order=VALIDITY_ORDER, labels=VALIDITY_LABELS),
        },
        "tool_call_messages": {
            "message_level": {
                "total": tool_call_message_count,
                "counts": _category_rows(
                    {
                        "single_tool_call": single_message_count,
                        "multiple_tool_calls": multi_message_count,
                    },
                    order=TOOL_CALL_MESSAGE_ORDER,
                    labels=TOOL_CALL_MESSAGE_LABELS,
                ),
                "tool_call_count_distribution": _value_distribution_rows(tool_call_count_values, fill_from=1),
            },
            "trace_level": {
                "total": len(records),
                "trace_with_tool_calls": sum(1 for record in records if record.tool_call_message_count > 0),
                "counts": _category_rows(
                    dict(trace_category_counts),
                    order=TRACE_TOOL_CALL_CATEGORY_ORDER,
                    labels=TRACE_TOOL_CALL_CATEGORY_LABELS,
                ),
                "single_tool_call_messages_per_trace": _value_distribution_rows(
                    [record.single_tool_call_message_count for record in records],
                    fill_from=0,
                ),
                "multi_tool_call_messages_per_trace": _value_distribution_rows(
                    [record.multi_tool_call_message_count for record in records],
                    fill_from=0,
                ),
                "tool_call_messages_per_trace": _value_distribution_rows(
                    [record.tool_call_message_count for record in records],
                    fill_from=0,
                ),
                "max_tool_calls_per_message_per_trace": _value_distribution_rows(
                    [record.max_tool_calls_in_message for record in records],
                    fill_from=0,
                ),
            },
        },
        "records": [_record_payload(record) for record in records],
    }


def build_assistant_message_stats_artifact(
    run_dir: str | Path,
    *,
    trace_dir_names: tuple[str, ...] = DEFAULT_AGENTIC_TRACE_DIRS,
    attempt_scopes: tuple[AttemptScope, ...] = DEFAULT_ATTEMPT_SCOPES,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    trace_index = scan_trace_index(run_path, kind="agentic", trace_dir_names=trace_dir_names)
    payloads = {scope: build_assistant_message_stats_payload(trace_index, attempt_scope=scope) for scope in attempt_scopes}
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "assistant_message_stats",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "run_dir": str(run_path),
        "kind": "agentic",
        "trace_dir_names": list(trace_dir_names),
        "attempt_scopes": list(attempt_scopes),
        "payloads": payloads,
    }


def build_assistant_message_stats_summary(
    run_dir: str | Path,
    *,
    trace_dir_names: tuple[str, ...] = DEFAULT_AGENTIC_TRACE_DIRS,
    attempt_scopes: tuple[AttemptScope, ...] = DEFAULT_ATTEMPT_SCOPES,
) -> dict[str, Any]:
    artifact = build_assistant_message_stats_artifact(run_dir, trace_dir_names=trace_dir_names, attempt_scopes=attempt_scopes)
    return summarize_assistant_message_stats_artifact(artifact)


def summarize_assistant_message_stats_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    payloads = artifact.get("payloads", {})
    trace_dir_names = artifact.get("trace_dir_names", [])
    attempt_scopes = artifact.get("attempt_scopes", [])
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "assistant_message_stats",
        "kind": "agentic",
        "trace_dir_names": trace_dir_names if isinstance(trace_dir_names, list) else [],
        "attempt_scopes": attempt_scopes if isinstance(attempt_scopes, list) else [],
        "payloads": {scope: summarize_assistant_message_stats_payload(payload) for scope, payload in payloads.items() if isinstance(payload, dict)},
    }


def summarize_assistant_message_stats_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    tool_call_messages = payload.get("tool_call_messages")
    message_level = tool_call_messages.get("message_level", {}) if isinstance(tool_call_messages, dict) else {}
    trace_level = tool_call_messages.get("trace_level", {}) if isinstance(tool_call_messages, dict) else {}
    validity = payload.get("message_validity")
    return {
        "trace_count": int(payload.get("trace_count", 0) or 0),
        "assistant_message_count": int(payload.get("assistant_message_count", 0) or 0),
        "assistant_output_length": _assistant_output_length_summary(payload.get("assistant_output_length")),
        "message_validity_counts": _rows_to_count_map(validity.get("counts", []) if isinstance(validity, dict) else []),
        "tool_call_message_counts": _rows_to_count_map(message_level.get("counts", []) if isinstance(message_level, dict) else []),
        "trace_tool_call_category_counts": _rows_to_count_map(trace_level.get("counts", []) if isinstance(trace_level, dict) else []),
    }


def write_assistant_message_stats_artifact(
    run_dir: str | Path,
    *,
    trace_dir_names: tuple[str, ...] = DEFAULT_AGENTIC_TRACE_DIRS,
    artifact: Mapping[str, Any] | None = None,
) -> Path:
    run_path = Path(run_dir)
    payload = artifact if artifact is not None else build_assistant_message_stats_artifact(run_path, trace_dir_names=trace_dir_names)
    path = run_path / ARTIFACT_NAME
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def load_assistant_message_stats_artifact(run_dir: str | Path) -> dict[str, Any] | None:
    path = Path(run_dir) / ARTIFACT_NAME
    if not path.exists():
        return None
    loaded = load_json(path)
    return loaded or None


def _build_trace_record(*, task_id: str, attempt: int, path: str, data: Mapping[str, Any]) -> AssistantTraceStatsRecord:
    validity_counts = {category: 0 for category in VALIDITY_ORDER}
    tool_call_count_distribution: dict[str, int] = {}
    assistant_output_length_distribution: dict[str, int] = {}
    tool_call_assistant_output_length_distribution: dict[str, int] = {}
    non_tool_call_assistant_output_length_distribution: dict[str, int] = {}
    single_tool_call_message_count = 0
    multi_tool_call_message_count = 0
    max_tool_calls_in_message = 0
    assistant_output_chars = 0

    messages = _assistant_messages(data)
    for message in messages:
        assistant_output_char_count = _assistant_output_char_count(message)
        assistant_output_chars += assistant_output_char_count
        _add_distribution_value(assistant_output_length_distribution, assistant_output_char_count)
        category = _validity_category(message)
        validity_counts[category] += 1

        tool_call_count = _tool_call_count(message)
        if tool_call_count > 0:
            _add_distribution_value(tool_call_assistant_output_length_distribution, assistant_output_char_count)
        else:
            _add_distribution_value(non_tool_call_assistant_output_length_distribution, assistant_output_char_count)
        max_tool_calls_in_message = max(max_tool_calls_in_message, tool_call_count)
        if tool_call_count <= 0:
            continue
        tool_call_count_distribution[str(tool_call_count)] = tool_call_count_distribution.get(str(tool_call_count), 0) + 1
        if tool_call_count == 1:
            single_tool_call_message_count += 1
        else:
            multi_tool_call_message_count += 1

    return AssistantTraceStatsRecord(
        task_id=task_id,
        attempt=attempt,
        path=str(path),
        assistant_message_count=len(messages),
        assistant_output_chars=assistant_output_chars,
        avg_assistant_output_chars=_average(assistant_output_chars, len(messages)),
        assistant_output_length_distribution=assistant_output_length_distribution,
        tool_call_assistant_output_length_distribution=tool_call_assistant_output_length_distribution,
        non_tool_call_assistant_output_length_distribution=non_tool_call_assistant_output_length_distribution,
        validity_counts=validity_counts,
        tool_call_message_count=single_tool_call_message_count + multi_tool_call_message_count,
        single_tool_call_message_count=single_tool_call_message_count,
        multi_tool_call_message_count=multi_tool_call_message_count,
        max_tool_calls_in_message=max_tool_calls_in_message,
        trace_tool_call_category=_trace_tool_call_category(single_tool_call_message_count, multi_tool_call_message_count),
        tool_call_count_distribution=tool_call_count_distribution,
    )


def _record_payload(record: AssistantTraceStatsRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload.pop("assistant_output_length_distribution", None)
    payload.pop("tool_call_assistant_output_length_distribution", None)
    payload.pop("non_tool_call_assistant_output_length_distribution", None)
    return payload


def _assistant_messages(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    conversation = data.get("conversation", [])
    if not isinstance(conversation, list):
        return []
    return [message for message in conversation if isinstance(message, dict) and str(message.get("role", "")).lower() == "assistant"]


def _assistant_output_char_count(message: Mapping[str, Any]) -> int:
    total = 0
    for field_name in ("content", "reasoning_content", "reasoning", "visible_thought"):
        total += _output_field_char_count(message.get(field_name))
    total += _tool_calls_char_count(message.get("tool_calls"))
    return total


def _output_field_char_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    return len(_stable_json_or_string(value))


def _tool_calls_char_count(value: Any) -> int:
    if not isinstance(value, list) or not value:
        return 0
    return len(_stable_json_or_string(value))


def _stable_json_or_string(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _average(total: int, count: int) -> float | None:
    return total / count if count else None


def _add_distribution_value(counts: dict[str, int], value: int) -> None:
    key = str(max(int(value), 0))
    counts[key] = counts.get(key, 0) + 1


def _assistant_output_length_distributions(records: list[AssistantTraceStatsRecord]) -> dict[str, list[dict[str, int]]]:
    count_maps = {
        "all": (record.assistant_output_length_distribution for record in records),
        "with_tool_calls": (record.tool_call_assistant_output_length_distribution for record in records),
        "without_tool_calls": (record.non_tool_call_assistant_output_length_distribution for record in records),
    }
    return {key: _distribution_rows_from_count_maps(count_maps[key]) for key in ASSISTANT_OUTPUT_LENGTH_DISTRIBUTION_KEYS}


def _distribution_rows_from_count_maps(count_maps: Any) -> list[dict[str, int]]:
    merged: dict[int, int] = {}
    for counts in count_maps:
        if not isinstance(counts, Mapping):
            continue
        for raw_value, count in counts.items():
            if isinstance(count, bool) or not isinstance(count, (int, float)):
                continue
            try:
                value = int(raw_value)
            except (TypeError, ValueError):
                continue
            merged[value] = merged.get(value, 0) + int(count)
    return [{"value": value, "count": merged[value]} for value in sorted(merged)]


def _assistant_output_length_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {
            "unit": ASSISTANT_OUTPUT_LENGTH_UNIT,
            "total": 0,
            "average": None,
            "message_count": 0,
        }
    total = value.get("total")
    average = value.get("average")
    message_count = value.get("message_count")
    return {
        "unit": value.get("unit") if isinstance(value.get("unit"), str) else ASSISTANT_OUTPUT_LENGTH_UNIT,
        "total": int(total) if isinstance(total, (int, float)) and not isinstance(total, bool) else 0,
        "average": float(average) if isinstance(average, (int, float)) and not isinstance(average, bool) else None,
        "message_count": int(message_count) if isinstance(message_count, (int, float)) and not isinstance(message_count, bool) else 0,
    }


def _validity_category(message: Mapping[str, Any]) -> str:
    content_valid = _field_valid(message.get("content"))
    reasoning_valid = _field_valid(message.get("reasoning_content")) or _field_valid(message.get("reasoning"))
    tool_calls_valid = _tool_call_count(message) > 0
    if content_valid and reasoning_valid and tool_calls_valid:
        return "content_reasoning_content_and_tool_calls"
    if content_valid and reasoning_valid:
        return "content_and_reasoning_content"
    if content_valid and tool_calls_valid:
        return "content_and_tool_calls"
    if reasoning_valid and tool_calls_valid:
        return "reasoning_content_and_tool_calls"
    if content_valid:
        return "content_only"
    if reasoning_valid:
        return "reasoning_content_only"
    if tool_calls_valid:
        return "tool_calls_only"
    return "none_valid"


def _field_valid(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _tool_call_count(message: Mapping[str, Any]) -> int:
    tool_calls = message.get("tool_calls")
    return len(tool_calls) if isinstance(tool_calls, list) else 0


def _trace_tool_call_category(single_count: int, multi_count: int) -> str:
    if multi_count > 0:
        return "multi_tool_call_trace"
    if single_count > 0:
        return "single_tool_call_trace"
    return "no_tool_calls"


def _sorted_task_ids(index: Mapping[str, Mapping[int, str]]) -> list[str]:
    return sorted(index, key=_task_id_sort_key)


def _task_id_sort_key(task_id: str) -> tuple[int, int, str]:
    text = str(task_id)
    if text.isdigit():
        return (0, int(text), text)
    return (1, 0, text)


def _selected_attempts(attempts: Mapping[int, str], *, attempt_scope: AttemptScope) -> list[tuple[int, str]]:
    if not attempts:
        return []
    if attempt_scope == "latest":
        latest = max(attempts)
        return [(latest, attempts[latest])]
    return [(attempt, attempts[attempt]) for attempt in sorted(attempts)]


def _merge_record_count_maps(records: list[AssistantTraceStatsRecord], field_name: str, *, order: tuple[str, ...]) -> dict[str, int]:
    merged = {key: 0 for key in order}
    for record in records:
        raw = getattr(record, field_name)
        for key, value in raw.items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def _category_rows(counts: Mapping[str, int], *, order: tuple[str, ...], labels: Mapping[str, str]) -> list[dict[str, Any]]:
    rows = [{"category": category, "label": labels.get(category, category), "count": int(counts.get(category, 0) or 0)} for category in order]
    extra = sorted(category for category in counts if category not in set(order))
    rows.extend({"category": category, "label": labels.get(category, category), "count": int(counts.get(category, 0) or 0)} for category in extra)
    return rows


def _value_distribution_rows(values: list[int], *, fill_from: int) -> list[dict[str, int]]:
    if not values:
        return []
    counts = Counter(values)
    for value in range(fill_from, max(values) + 1):
        counts.setdefault(value, 0)
    return [{"value": value, "count": counts[value]} for value in sorted(counts)]


def _rows_to_count_map(rows: Any) -> dict[str, int]:
    if not isinstance(rows, list):
        return {}
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        category = row.get("category")
        if not isinstance(category, str):
            continue
        count = row.get("count")
        if isinstance(count, bool) or not isinstance(count, (int, float)):
            continue
        counts[category] = int(count)
    return counts


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build assistant-message statistics for agentic run traces")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing agentic trace subdirectories")
    parser.add_argument(
        "--trace-dir-name",
        action="append",
        default=None,
        help="Trace subdirectory to scan. Can be passed multiple times. Defaults to agentic trace directories.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(name)s %(levelname)s %(message)s")
    trace_dir_names = tuple(args.trace_dir_name) if args.trace_dir_name else DEFAULT_AGENTIC_TRACE_DIRS
    path = write_assistant_message_stats_artifact(args.run_dir, trace_dir_names=trace_dir_names)
    print(path)


if __name__ == "__main__":
    _main()
