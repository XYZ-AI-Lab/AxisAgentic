# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic.sft_export import (
    SwiftAgentExportConfig,
    SwiftAgentExportResult,
    SwiftAgentExportWarning,
    conversation_to_swift_agent_sample,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

# Persisted statuses considered acceptable SFT material by default. The
# WebSearch orchestrator's ``_done_reason_for_stage`` writes
# ``assistant_final_answer`` for the success path and one of the
# ``terminated_*`` strings for forced finalizations (turn-limit, context-limit,
# tools-exhausted, generic force-completed). The string ``assistant_force_completed``
# is only the ``ConversationStage`` enum value — it is never persisted as a
# trace status, so listing it here would silently drop every forced-final trace.
DEFAULT_ACCEPTED_STATUSES: tuple[str, ...] = (
    "assistant_final_answer",
    "terminated_force_completed",
    "terminated_turn_limit",
    "terminated_tools_exhausted",
    "terminated_context_limit",
)


@dataclass(frozen=True)
class WebSearchSFTExportConfig:
    format: str = "swift_agent"
    trace_dir_names: tuple[str, ...] = ("web-search-benchmark",)
    accepted_statuses: tuple[str, ...] = DEFAULT_ACCEPTED_STATUSES
    max_traces: int | None = None
    include_metadata: bool = False
    strict: bool = False
    swift: SwiftAgentExportConfig = field(default_factory=SwiftAgentExportConfig)


@dataclass(frozen=True)
class WebSearchSFTExportRecord:
    trace_path: str
    sample_id: str
    status: str | None
    warnings: list[SwiftAgentExportWarning]


@dataclass(frozen=True)
class WebSearchSFTExportSummary:
    run_dir: str
    output_path: str
    manifest_path: str
    format: str
    total_trace_files: int
    exported: int
    skipped: int
    failed: int
    records: list[WebSearchSFTExportRecord]
    errors: list[dict[str, str]]


def export_web_search_run_to_sft(
    *,
    run_dir: str | Path,
    output_path: str | Path | None = None,
    config: WebSearchSFTExportConfig | None = None,
) -> WebSearchSFTExportSummary:
    cfg = config or WebSearchSFTExportConfig()
    if cfg.format != "swift_agent":
        msg = f"Unsupported SFT export format: {cfg.format!r}"
        raise ValueError(msg)
    run_path = Path(run_dir)
    out_path = Path(output_path) if output_path is not None else run_path / "sft_exports" / "swift_agent.jsonl"
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trace_files = _keep_latest_attempt_per_task(iter_web_search_trace_files(run_path, cfg.trace_dir_names))
    records: list[WebSearchSFTExportRecord] = []
    errors: list[dict[str, str]] = []
    exported = 0
    skipped = 0
    failed = 0
    accepted_statuses = set(cfg.accepted_statuses)

    with out_path.open("w", encoding="utf-8") as handle:
        for trace_path in trace_files:
            if cfg.max_traces is not None and exported >= cfg.max_traces:
                skipped += 1
                continue
            try:
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                status = trace.get("status")
                if accepted_statuses and status not in accepted_statuses:
                    skipped += 1
                    continue
                result = trace_to_swift_agent_sample(trace, trace_path=trace_path, run_path=run_path, config=cfg)
                handle.write(json.dumps(result.sample, ensure_ascii=False, separators=(",", ":")) + "\n")
                records.append(WebSearchSFTExportRecord(str(trace_path), str(result.sample.get("id") or trace_path.stem), status, result.warnings))
                exported += 1
            except Exception as exc:
                failed += 1
                errors.append({"trace_path": str(trace_path), "error_type": type(exc).__name__, "message": str(exc)})
                if cfg.strict:
                    raise

    summary = WebSearchSFTExportSummary(
        run_dir=str(run_path),
        output_path=str(out_path),
        manifest_path=str(manifest_path),
        format=cfg.format,
        total_trace_files=len(trace_files),
        exported=exported,
        skipped=skipped,
        failed=failed,
        records=records,
        errors=errors,
    )
    manifest_path.write_text(json.dumps(_summary_to_json(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def trace_to_swift_agent_sample(
    trace: dict[str, Any],
    *,
    trace_path: Path,
    run_path: Path,
    config: WebSearchSFTExportConfig,
) -> SwiftAgentExportResult:
    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    tools = metadata.get("tools") if isinstance(metadata, dict) else None
    sample_metadata = {
        "source_trace": _relative_or_absolute(trace_path, run_path),
        "task_id": trace.get("task_id"),
        "status": trace.get("status"),
        "task_input": trace.get("task_input"),
        "output": metadata.get("output") if isinstance(metadata, dict) else None,
    }
    swift_config = SwiftAgentExportConfig(
        include_reasoning=config.swift.include_reasoning,
        skip_empty_reasoning=config.swift.skip_empty_reasoning,
        wrap_non_json_tool_response=config.swift.wrap_non_json_tool_response,
        include_visible_thought=config.swift.include_visible_thought,
        apply_runtime_visibility=config.swift.apply_runtime_visibility,
        include_metadata=config.include_metadata,
        ensure_ascii=config.swift.ensure_ascii,
    )
    return conversation_to_swift_agent_sample(
        conversation=trace.get("conversation") or [],
        tools=tools,
        sample_id=str(trace.get("task_id") or trace_path.stem),
        metadata=sample_metadata,
        config=swift_config,
    )


def iter_web_search_trace_files(run_path: Path, trace_dir_names: Iterable[str]) -> Iterable[Path]:
    for trace_dir_name in trace_dir_names:
        trace_dir = run_path / trace_dir_name
        if not trace_dir.exists():
            continue
        yield from sorted(trace_dir.glob("*.json"), key=_trace_sort_key)


def _keep_latest_attempt_per_task(trace_files: Iterable[Path]) -> list[Path]:
    """Keep only the highest-numbered ``_attempt-N`` file per base task id.

    The WebSearch evaluator only treats the latest attempt as the canonical
    result (see ``_build_completed_task_cache`` in ``evaluate_benchmark.py``,
    which sorts by attempt number and loads ``attempt_ids[-1]``). Earlier
    attempts on disk are intermediate retries; exporting them as SFT samples
    duplicates the same task with rejected/retried conversations.
    """
    paths = list(trace_files)
    latest: dict[str, int] = {}
    for path in paths:
        base, sep, suffix = path.stem.rpartition("_attempt-")
        if sep and suffix.isdigit():
            n = int(suffix)
            if n > latest.get(base, -1):
                latest[base] = n
    kept: list[Path] = []
    for path in paths:
        base, sep, suffix = path.stem.rpartition("_attempt-")
        if not sep or not suffix.isdigit():
            kept.append(path)
            continue
        if int(suffix) == latest.get(base):
            kept.append(path)
    return kept


def _trace_sort_key(path: Path) -> tuple[int, str]:
    prefix = path.stem.split("_", 1)[0]
    if prefix.isdigit():
        return (int(prefix), path.name)
    return (10**12, path.name)


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _summary_to_json(summary: WebSearchSFTExportSummary) -> dict[str, Any]:
    return {
        "run_dir": summary.run_dir,
        "output_path": summary.output_path,
        "manifest_path": summary.manifest_path,
        "format": summary.format,
        "total_trace_files": summary.total_trace_files,
        "exported": summary.exported,
        "skipped": summary.skipped,
        "failed": summary.failed,
        "records": [
            {
                "trace_path": record.trace_path,
                "sample_id": record.sample_id,
                "status": record.status,
                "warnings": [asdict(warning) for warning in record.warnings],
            }
            for record in summary.records
        ],
        "errors": summary.errors,
    }
