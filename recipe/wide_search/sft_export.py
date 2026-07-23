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

WIDE_SEARCH_TRACE_DIR = "wide-search"
WIDE_SEARCH_SCORES_DIR = "widesearch_scores"

# Persisted statuses considered acceptable SFT material by default. The
# WideSearch orchestrator inherits ``TaskOrchestrator._done_reason_for_stage``,
# which writes ``assistant_final_answer`` for the success path and one of the
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
class WideSearchSFTExportConfig:
    format: str = "swift_agent"
    trace_dir_names: tuple[str, ...] = (WIDE_SEARCH_TRACE_DIR,)
    accepted_statuses: tuple[str, ...] = DEFAULT_ACCEPTED_STATUSES
    max_traces: int | None = None
    include_metadata: bool = False
    strict: bool = False
    min_score: float | None = None
    swift: SwiftAgentExportConfig = field(default_factory=SwiftAgentExportConfig)


@dataclass(frozen=True)
class WideSearchSFTExportRecord:
    trace_path: str
    sample_id: str
    status: str | None
    score: float | None
    warnings: list[SwiftAgentExportWarning]


@dataclass(frozen=True)
class WideSearchSFTExportSummary:
    run_dir: str
    output_path: str
    manifest_path: str
    format: str
    total_trace_files: int
    exported: int
    skipped: int
    failed: int
    records: list[WideSearchSFTExportRecord]
    errors: list[dict[str, str]]


def export_wide_search_run_to_sft(
    *,
    run_dir: str | Path,
    output_path: str | Path | None = None,
    config: WideSearchSFTExportConfig | None = None,
) -> WideSearchSFTExportSummary:
    cfg = config or WideSearchSFTExportConfig()
    if cfg.format != "swift_agent":
        msg = f"Unsupported SFT export format: {cfg.format!r}"
        raise ValueError(msg)
    run_path = Path(run_dir)
    out_path = Path(output_path) if output_path is not None else run_path / "sft_exports" / "swift_agent.jsonl"
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trace_files = _keep_latest_attempt_per_trial(iter_wide_search_trace_files(run_path, cfg.trace_dir_names))
    score_index = _load_score_index(run_path)
    records: list[WideSearchSFTExportRecord] = []
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
                score = _lookup_score(trace, score_index)
                if cfg.min_score is not None and (score is None or score < cfg.min_score):
                    skipped += 1
                    continue
                result = trace_to_swift_agent_sample(trace, trace_path=trace_path, run_path=run_path, config=cfg, score=score)
                handle.write(json.dumps(result.sample, ensure_ascii=False, separators=(",", ":")) + "\n")
                records.append(
                    WideSearchSFTExportRecord(
                        trace_path=str(trace_path),
                        sample_id=str(result.sample.get("id") or trace_path.stem),
                        status=status,
                        score=score,
                        warnings=result.warnings,
                    )
                )
                exported += 1
            except Exception as exc:
                failed += 1
                errors.append({"trace_path": str(trace_path), "error_type": type(exc).__name__, "message": str(exc)})
                if cfg.strict:
                    raise

    summary = WideSearchSFTExportSummary(
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
    config: WideSearchSFTExportConfig,
    score: float | None = None,
) -> SwiftAgentExportResult:
    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    tools = metadata.get("tools") if isinstance(metadata, dict) else None
    sample_metadata = {
        "source_trace": _relative_or_absolute(trace_path, run_path),
        "task_id": trace.get("task_id"),
        "status": trace.get("status"),
        "task_input": trace.get("task_input"),
        "output": metadata.get("output") if isinstance(metadata, dict) else None,
        "score": score,
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


def iter_wide_search_trace_files(run_path: Path, trace_dir_names: Iterable[str]) -> Iterable[Path]:
    for trace_dir_name in trace_dir_names:
        trace_dir = run_path / trace_dir_name
        if not trace_dir.exists():
            continue
        yield from sorted(trace_dir.glob("*.json"), key=_trace_sort_key)


def _keep_latest_attempt_per_trial(trace_files: Iterable[Path]) -> list[Path]:
    """Keep only the highest-numbered ``_attempt-N`` file per base trial id.

    The WideSearch evaluator only scores the latest attempt for each trial
    (see ``_load_completed_result`` in ``evaluate_widesearch.py``). The score
    sidecar is written once per trial, keyed by the base id. If retries
    produced earlier attempt traces on disk, exporting them alongside the
    latest attempt would attach the final attempt's score to intermediate
    failed traces, polluting the SFT corpus with mislabeled samples.
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


def _load_score_index(run_path: Path) -> dict[str, float]:
    """Load per-trial scores from ``widesearch_scores/<task_id>.json`` sidecars.

    Keyed by the trace's ``task_id`` (e.g. ``ws_en_001__trial-0``) so the
    ``_attempt-{N}`` suffix on trace filenames does not affect the lookup.
    """
    index: dict[str, float] = {}
    scores_dir = run_path / WIDE_SEARCH_SCORES_DIR
    if not scores_dir.exists():
        return index
    for path in sorted(scores_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        task_id = str(payload.get("task_id", path.stem))
        score = payload.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            index[task_id] = float(score)
    return index


def _lookup_score(trace: dict[str, Any], score_index: dict[str, float]) -> float | None:
    task_id = trace.get("task_id")
    if not isinstance(task_id, str):
        return None
    # The trace's ``task_id`` is the attempt id (``<base>_attempt-N``) but the
    # WideSearch evaluator writes score sidecars keyed by the trial id
    # (``<base>``), so strip the trailing ``_attempt-N`` segment before lookup.
    # Use ``in`` rather than ``or`` so a legitimate score of 0.0 is not lost
    # to falsey-coalescing — strict-match WideSearch scoring produces 0.0 often.
    base, sep, _ = task_id.rpartition("_attempt-")
    base_id = base if sep else task_id
    if base_id in score_index:
        return score_index[base_id]
    if task_id in score_index:
        return score_index[task_id]
    return None


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


def _summary_to_json(summary: WideSearchSFTExportSummary) -> dict[str, Any]:
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
                "score": record.score,
                "warnings": [asdict(warning) for warning in record.warnings],
            }
            for record in summary.records
        ],
        "errors": summary.errors,
    }
