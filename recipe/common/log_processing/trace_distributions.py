# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Trace-level distribution helpers for benchmark run logs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from agentic.contracts.markers import parse_compaction_marker_dict, parse_discard_all_marker_dict
from recipe.common.log_processing.trace_refs import add_trace_refs, is_attempt_budget_branch_stem

TraceKind = Literal["agentic", "original"]
AttemptScope = Literal["latest", "all"]

SCHEMA_VERSION = 6
ARTIFACT_NAME = "trace_distributions.json"
LLM_JUDGE_CORRECT_THRESHOLD = 0.5
GROUP_ORDER: tuple[str, ...] = ("all", "correct", "wrong", "empty")
GROUP_LABELS: dict[str, str] = {
    "all": "All traces",
    "correct": "Correct",
    "wrong": "Wrong",
    "empty": "Empty output",
}
METRIC_LABELS: dict[str, str] = {
    "done_reason": "done_reason",
    "visible_turns": "Visible turns",
    "actual_turns": "Actual turns",
    "actual_attempts": "Actual attempts",
    "discard_all_resets": "Discard-all resets",
    "discard_all_reset_turns": "Reset visible turns",
    "hidden_turns": "Hidden turns",
}
DEFAULT_AGENTIC_TRACE_DIRS: tuple[str, ...] = ("web-search-benchmark", "wide-search")
DEFAULT_ATTEMPT_SCOPES: tuple[AttemptScope, ...] = ("latest", "all")


@dataclass(frozen=True)
class TraceDistributionRecord:
    task_id: str
    attempt: int
    path: str
    done_reason: str
    visible_turns: int
    actual_turns: int
    actual_attempts: int
    discard_all_resets: int
    discard_all_reset_turns: list[int]
    hidden_turns: int
    output: str
    is_empty: bool
    llm_judge_score: float | None
    llm_judge_correct: bool | None


def load_json(path: str | Path) -> dict[str, Any]:
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def load_llm_judge_rows(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    run_path = Path(run_dir)
    rows: dict[str, dict[str, Any]] = {}
    jsonl_path = run_path / "llm_judge_results.jsonl"
    if jsonl_path.exists():
        try:
            lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            lines = []
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("task_id") is not None:
                rows[str(row["task_id"])] = row

    sidecar_dir = run_path / "llm_judge"
    if sidecar_dir.exists():
        for path in sorted(sidecar_dir.glob("*.json")):
            row = load_json(path)
            task_id = str(row.get("task_id", path.stem))
            if task_id and task_id not in rows:
                rows[task_id] = row
    return rows


def scan_trace_index(
    run_dir: str | Path,
    *,
    kind: TraceKind,
    trace_dir_names: tuple[str, ...] = DEFAULT_AGENTIC_TRACE_DIRS,
) -> dict[str, dict[int, str]]:
    run_path = Path(run_dir)
    index: dict[str, dict[int, str]] = {}
    if kind == "original":
        for path in sorted(run_path.glob("task_*.json")):
            if path.stat().st_size == 0:
                continue
            match = re.match(r"task_(?P<base>[^_]+)_attempt-(?P<attempt>\d+)", path.name)
            if not match:
                continue
            index.setdefault(match.group("base"), {})[int(match.group("attempt"))] = str(path)
        return index

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


def build_trace_distribution_payload(
    trace_index: Mapping[str, Mapping[int, str]],
    llm_judge_rows: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    kind: TraceKind,
    attempt_scope: AttemptScope = "latest",
    correct_threshold: float = LLM_JUDGE_CORRECT_THRESHOLD,
) -> dict[str, Any]:
    records = build_trace_distribution_records(
        trace_index,
        llm_judge_rows or {},
        kind=kind,
        attempt_scope=attempt_scope,
        correct_threshold=correct_threshold,
    )
    groups = {group: _build_group_distribution(_records_for_group(records, group), group=group) for group in GROUP_ORDER}
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "attempt_scope": attempt_scope,
        "record_count": len(records),
        "groups": groups,
        "records": [asdict(record) for record in records],
    }


def build_trace_distribution_artifact(
    run_dir: str | Path,
    *,
    kind: TraceKind,
    trace_dir_names: tuple[str, ...] = DEFAULT_AGENTIC_TRACE_DIRS,
    attempt_scopes: tuple[AttemptScope, ...] = DEFAULT_ATTEMPT_SCOPES,
    correct_threshold: float = LLM_JUDGE_CORRECT_THRESHOLD,
) -> dict[str, Any]:
    """Build the precomputed dashboard artifact for trace distributions.

    The dashboard can render completed runs from this compact artifact instead of
    rescanning every raw trace JSON. Running runs still use the live path so new
    traces and judge sidecars appear without a refresh step.
    """
    run_path = Path(run_dir)
    trace_index = scan_trace_index(run_path, kind=kind, trace_dir_names=trace_dir_names)
    llm_rows = load_llm_judge_rows(run_path)
    payloads = {
        scope: build_trace_distribution_payload(
            trace_index,
            llm_rows,
            kind=kind,
            attempt_scope=scope,
            correct_threshold=correct_threshold,
        )
        for scope in attempt_scopes
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "trace_distributions",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "run_dir": str(run_path),
        "kind": kind,
        "trace_dir_names": list(trace_dir_names),
        "attempt_scopes": list(attempt_scopes),
        "payloads": payloads,
    }


def write_trace_distribution_artifact(
    run_dir: str | Path,
    *,
    kind: TraceKind,
    trace_dir_names: tuple[str, ...] = DEFAULT_AGENTIC_TRACE_DIRS,
) -> Path:
    run_path = Path(run_dir)
    artifact = build_trace_distribution_artifact(run_path, kind=kind, trace_dir_names=trace_dir_names)
    path = run_path / ARTIFACT_NAME
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_trace_distribution_artifact(run_dir: str | Path) -> dict[str, Any] | None:
    path = Path(run_dir) / ARTIFACT_NAME
    if not path.exists():
        return None
    loaded = load_json(path)
    return loaded or None


def build_trace_distribution_records(
    trace_index: Mapping[str, Mapping[int, str]],
    llm_judge_rows: Mapping[str, Mapping[str, Any]],
    *,
    kind: TraceKind,
    attempt_scope: AttemptScope = "latest",
    correct_threshold: float = LLM_JUDGE_CORRECT_THRESHOLD,
) -> list[TraceDistributionRecord]:
    records: list[TraceDistributionRecord] = []
    for task_id in _sorted_task_ids(trace_index):
        attempts = trace_index.get(task_id, {})
        for attempt, path in _selected_attempts(attempts, attempt_scope=attempt_scope):
            data = load_json(path)
            if not data:
                continue
            output = _trace_output(data, kind=kind)
            score = _numeric_score(llm_judge_rows.get(task_id, {}), "llm_judge_score")
            visible_turns = _visible_turn_count(data, kind=kind)
            actual_turns = _actual_turn_count(data, kind=kind)
            reset_turns = _discard_all_reset_turns(data, kind=kind)
            records.append(
                TraceDistributionRecord(
                    task_id=task_id,
                    attempt=attempt,
                    path=str(path),
                    done_reason=_trace_done_reason(data, kind=kind),
                    visible_turns=visible_turns,
                    actual_turns=actual_turns,
                    actual_attempts=attempt,
                    discard_all_resets=_discard_all_reset_count(data, kind=kind),
                    discard_all_reset_turns=reset_turns,
                    hidden_turns=max(actual_turns - visible_turns, 0),
                    output=output,
                    is_empty=not output.strip(),
                    llm_judge_score=score,
                    llm_judge_correct=None if score is None else score > correct_threshold,
                )
            )
    return records


def _parse_agentic_trace_stem(stem: str) -> tuple[str, int] | None:
    match = re.match(r"(?P<base>.+)_attempt-(?P<attempt>\d+)(?:$|_)", stem)
    if not match:
        return None
    return match.group("base"), int(match.group("attempt"))


def _sorted_task_ids(index: Mapping[str, Mapping[int, str]]) -> list[str]:
    return sorted(index, key=lambda item: int(item) if str(item).isdigit() else str(item))


def _selected_attempts(attempts: Mapping[int, str], *, attempt_scope: AttemptScope) -> list[tuple[int, str]]:
    if not attempts:
        return []
    if attempt_scope == "latest":
        latest = max(attempts)
        return [(latest, attempts[latest])]
    return [(attempt, attempts[attempt]) for attempt in sorted(attempts)]


def _numeric_score(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _trace_done_reason(data: Mapping[str, Any], *, kind: TraceKind) -> str:
    metadata = data.get("metadata")
    if isinstance(metadata, Mapping):
        done_reason = metadata.get("done_reason")
        if done_reason:
            return str(done_reason)
    if kind == "original":
        return str(data.get("status") or data.get("final_judge_result") or "(missing)")
    return str(data.get("status") or "(missing)")


def _trace_output(data: Mapping[str, Any], *, kind: TraceKind) -> str:
    metadata = data.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("output") is not None:
        return str(metadata.get("output") or "")
    if kind == "original":
        return str(data.get("final_boxed_answer") or data.get("output") or "")
    return str(data.get("output") or "")


def _original_messages(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    history = data.get("main_agent_message_history", {})
    if isinstance(history, list):
        return [message for message in history if isinstance(message, dict)]
    if isinstance(history, Mapping):
        messages = history.get("message_history", [])
        if isinstance(messages, list):
            return [message for message in messages if isinstance(message, dict)]
    return []


def _conversation_messages(data: Mapping[str, Any], *, kind: TraceKind) -> list[dict[str, Any]]:
    if kind == "original":
        return _original_messages(data)
    conversation = data.get("conversation", [])
    return [message for message in conversation if isinstance(message, dict)] if isinstance(conversation, list) else []


def _visible_messages(data: Mapping[str, Any], *, kind: TraceKind) -> list[dict[str, Any]]:
    if kind == "original":
        return _original_messages(data)
    visible = data.get("visible_conversation", [])
    if isinstance(visible, list) and visible:
        return [message for message in visible if isinstance(message, dict)]
    return _reconstruct_visible_messages(_conversation_messages(data, kind=kind))


def _reconstruct_visible_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visible: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "ConversationRuntime":
            compaction = parse_compaction_marker_dict(dict(message))
            if compaction is not None:
                boundary = compaction.prefix_len + compaction.compressed_count
                if boundary <= len(visible):
                    summary_message: dict[str, Any] = {"role": "assistant", "content": compaction.summary}
                    visible = [*visible[: compaction.prefix_len], summary_message, *visible[boundary:]]
                continue
            discard = parse_discard_all_marker_dict(dict(message))
            if discard is not None:
                if discard.prefix_len <= len(visible):
                    visible = visible[: discard.prefix_len]
                continue
            rollback_count = _rollback_message_count(message)
            if rollback_count > 0:
                del visible[-rollback_count:]
            continue
        visible.append(message)
    return visible


def _rollback_message_count(message: Mapping[str, Any]) -> int:
    name = str(message.get("name") or "")
    content = str(message.get("content") or "")
    if name == "compaction":
        # Compaction markers carry free-form summary text that may well contain
        # the word "rollback"; never count them as rollbacks.
        return 0
    if name != "rollback" and "rollback" not in content.lower():
        return 0
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return 1
    if not isinstance(payload, Mapping):
        return 1
    value = payload.get("rollback_message_count")
    if isinstance(value, bool):
        return 1
    if isinstance(value, int):
        return max(value, 1)
    return 1


def _assistant_count(messages: list[dict[str, Any]]) -> int:
    return sum(1 for message in messages if message.get("role") == "assistant")


def _visible_turn_count(data: Mapping[str, Any], *, kind: TraceKind) -> int:
    return _assistant_count(_visible_messages(data, kind=kind))


def _actual_turn_count(data: Mapping[str, Any], *, kind: TraceKind) -> int:
    return _assistant_count(_conversation_messages(data, kind=kind))


def _discard_all_reset_count(data: Mapping[str, Any], *, kind: TraceKind) -> int:
    if kind == "original":
        return 0
    count = 0
    for message in _conversation_messages(data, kind=kind):
        if message.get("role") != "ConversationRuntime":
            continue
        if parse_discard_all_marker_dict(dict(message)) is not None:
            count += 1
    return count


def _discard_all_reset_turns(data: Mapping[str, Any], *, kind: TraceKind) -> list[int]:
    if kind == "original":
        return []
    turns: list[int] = []
    visible: list[dict[str, Any]] = []
    for message in _conversation_messages(data, kind=kind):
        if message.get("role") == "ConversationRuntime":
            compaction = parse_compaction_marker_dict(dict(message))
            if compaction is not None:
                boundary = compaction.prefix_len + compaction.compressed_count
                if boundary <= len(visible):
                    summary_message: dict[str, Any] = {"role": "assistant", "content": compaction.summary}
                    visible = [*visible[: compaction.prefix_len], summary_message, *visible[boundary:]]
                continue
            discard = parse_discard_all_marker_dict(dict(message))
            if discard is not None:
                turns.append(_assistant_count(visible))
                if discard.prefix_len <= len(visible):
                    visible = visible[: discard.prefix_len]
                continue
            rollback_count = _rollback_message_count(message)
            if rollback_count > 0:
                del visible[-rollback_count:]
            continue
        visible.append(message)
    return turns


def _records_for_group(records: list[TraceDistributionRecord], group: str) -> list[TraceDistributionRecord]:
    if group == "all":
        return records
    if group == "correct":
        return [record for record in records if record.llm_judge_correct is True]
    if group == "wrong":
        return [record for record in records if record.llm_judge_correct is False]
    if group == "empty":
        return [record for record in records if record.is_empty]
    return []


def _build_group_distribution(records: list[TraceDistributionRecord], *, group: str) -> dict[str, Any]:
    return {
        "label": GROUP_LABELS.get(group, group),
        "count": len(records),
        "done_reason": _categorical_distribution([record.done_reason for record in records]),
        "visible_turns": _numeric_distribution([record.visible_turns for record in records]),
        "actual_turns": _numeric_distribution([record.actual_turns for record in records]),
        "actual_attempts": _numeric_distribution([record.actual_attempts for record in records]),
        "discard_all_resets": _numeric_distribution([record.discard_all_resets for record in records]),
        "discard_all_reset_turns": _numeric_distribution([turn for record in records for turn in record.discard_all_reset_turns]),
        "hidden_turns": _numeric_distribution([record.hidden_turns for record in records]),
    }


def _categorical_distribution(values: list[str]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for value in values:
        label = value or "(missing)"
        counts[label] = counts.get(label, 0) + 1
    return {
        "counts": [{"value": value, "count": counts[value]} for value in sorted(counts)],
        "summary": {"distinct": len(counts)},
    }


def _numeric_distribution(values: list[int]) -> dict[str, Any]:
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {
        "counts": [{"value": value, "count": counts[value]} for value in sorted(counts)],
        "summary": _numeric_summary(values),
    }


def _numeric_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
    }


def _build_from_args(args: argparse.Namespace) -> dict[str, Any]:
    index = scan_trace_index(args.run_dir, kind=args.kind, trace_dir_names=tuple(args.trace_dir))
    llm_rows = load_llm_judge_rows(args.run_dir)
    return build_trace_distribution_payload(index, llm_rows, kind=args.kind, attempt_scope=args.attempt_scope)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build trace distribution statistics for a run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--kind", choices=["agentic", "original"], default="agentic")
    parser.add_argument("--attempt-scope", choices=["latest", "all"], default="latest")
    parser.add_argument("--trace-dir", action="append", default=list(DEFAULT_AGENTIC_TRACE_DIRS))
    args = parser.parse_args(argv)
    print(json.dumps(_build_from_args(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
