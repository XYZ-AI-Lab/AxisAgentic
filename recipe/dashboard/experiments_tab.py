# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Experiment overview tab renderer for the benchmark dashboard."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from recipe.common.log_processing import ExpectedTaskCount, load_dashboard_summary, resolve_expected_task_count
from recipe.dashboard.constants import LLM_JUDGE_CORRECT_THRESHOLD
from recipe.dashboard.discovery import _discover_experiments, _discover_runs, _scan_agentic_index
from recipe.dashboard.live_results import _build_live_agentic_eval_results
from recipe.dashboard.loading import (
    _load_llm_judge_results,
    _load_widesearch_summary,
    _load_widesearch_trial_scores,
)

_RUNNING_LOG_STALE_SECONDS = 30 * 60
_JOB_DONE = "✅"
_JOB_RUNNING = "⏳"
_JOB_STOPPED = "⏹️"
_EXPERIMENT_PREFIXES: tuple[str, ...] = ()
_EXPERIMENT_STATUS_SORT_ORDER = {
    _JOB_RUNNING: 0,
    _JOB_DONE: 1,
    _JOB_STOPPED: 2,
}
_EXPERIMENT_TYPE_SORT_ORDER = {
    "web_search": 0,
    "wide_search": 1,
}
_LOG_TIMESTAMP_RE = re.compile(r"^\[(?P<timestamp>[^\]]+)\]")
_LOG_TAIL_BYTES = 64 * 1024
_MAIN_LOG_BY_RUN_TYPE = {
    "web_search": "agentic.log",
    "wide_search": "agentic.log",
}
_TRACE_DIR_BY_RUN_TYPE = {
    "web_search": "web-search-benchmark",
    "wide_search": "wide-search",
}


@dataclass(frozen=True)
class ExperimentOverviewSource:
    label: str
    log_dir: Path
    run_type: str


@dataclass(frozen=True)
class WideSearchProgress:
    judged_trials: int
    expected: ExpectedTaskCount
    num_trials: int | None
    total_instances: int | None


def _run_sort_key(run_name: str) -> tuple[int, str]:
    match = re.fullmatch(r"run_(\d+)", run_name)
    return (int(match.group(1)), run_name) if match else (-1, run_name)


def _latest_run(exp_root: Path) -> str | None:
    runs = _discover_runs(exp_root)
    if not runs:
        return None
    return max(runs, key=_run_sort_key)


def _judged_task_count(llm_evals: dict[str, dict[str, Any]]) -> int:
    count = 0
    for task_id, row in llm_evals.items():
        if str(task_id).startswith("_"):
            continue
        score = row.get("llm_judge_score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            count += 1
    return count


def _main_run_log_path(run_dir: Path, run_type: str) -> Path | None:
    preferred = _MAIN_LOG_BY_RUN_TYPE.get(run_type)
    if preferred:
        path = run_dir / preferred
        if path.exists():
            return path
    logs = sorted(run_dir.glob("*.log"))
    return logs[0] if logs else None


def _parse_log_timestamp(line: str) -> datetime | None:
    match = _LOG_TIMESTAMP_RE.match(line)
    if not match:
        return None
    raw = match.group("timestamp").replace(",", ".")
    with_timezone = datetime.fromisoformat(raw)
    if with_timezone.tzinfo is not None:
        return with_timezone
    return with_timezone.replace(tzinfo=datetime.now().astimezone().tzinfo)


def _read_log_tail(path: Path) -> list[str]:
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - _LOG_TAIL_BYTES))
        return f.read().decode("utf-8", errors="replace").splitlines()


@st.cache_data(ttl=30, show_spinner=False)
def _latest_main_log_update_at(run_dir: str, run_type: str) -> datetime | None:
    log_path = _main_run_log_path(Path(run_dir), run_type)
    if log_path is None:
        return None
    try:
        lines = _read_log_tail(log_path)
    except OSError:
        return None
    for line in reversed(lines):
        try:
            timestamp = _parse_log_timestamp(line)
        except ValueError:
            timestamp = None
        if timestamp is not None:
            return timestamp
    try:
        return datetime.fromtimestamp(log_path.stat().st_mtime, tz=UTC)
    except OSError:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def _latest_widesearch_artifact_update_at(run_dir: str) -> datetime | None:
    run_path = Path(run_dir)
    latest_mtime: float | None = None
    candidates = [
        run_path / "widesearch_summary.json",
        run_path / "widesearch_per_task.json",
        run_path / "run_metadata.json",
    ]
    for path in candidates:
        try:
            if path.exists():
                mtime = path.stat().st_mtime
                latest_mtime = mtime if latest_mtime is None else max(latest_mtime, mtime)
        except OSError:
            continue
    for directory_name in ("widesearch_scores", "wide-search"):
        directory = run_path / directory_name
        if not directory.exists():
            continue
        for path in directory.glob("*.json"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            latest_mtime = mtime if latest_mtime is None else max(latest_mtime, mtime)
    return datetime.fromtimestamp(latest_mtime, tz=UTC) if latest_mtime is not None else None


def _latest_status_update_at(run_dir: Path | None, run_type: str) -> datetime | None:
    if run_dir is None:
        return None
    log_update = _latest_main_log_update_at(str(run_dir), run_type)
    if run_type == "wide_search":
        artifact_update = _latest_widesearch_artifact_update_at(str(run_dir))
        if log_update is None:
            return artifact_update
        if artifact_update is None:
            return log_update
        log_ts = log_update if log_update.tzinfo is not None else log_update.replace(tzinfo=UTC)
        artifact_ts = artifact_update if artifact_update.tzinfo is not None else artifact_update.replace(tzinfo=UTC)
        return artifact_update if artifact_ts.astimezone(UTC) > log_ts.astimezone(UTC) else log_update
    if log_update is not None:
        return log_update
    return None


@st.cache_data(ttl=30, show_spinner=False)
def _resolve_expected_task_count_cached(run_dir: str) -> ExpectedTaskCount:
    return resolve_expected_task_count(run_dir)


@st.cache_data(ttl=30, show_spinner=False)
def _load_dashboard_summary_cached(run_dir: str) -> dict[str, Any] | None:
    return load_dashboard_summary(run_dir)


@st.cache_data(ttl=30, show_spinner=False)
def _load_aggregate_summary_cached(exp_root: str) -> dict[str, Any] | None:
    """Read the cross-run avg@N summary written to a multi-run experiment dir.

    Only LiveBrowseComp runs (run_eval_config with num_runs>1) emit this file.
    """
    path = Path(exp_root) / "aggregate_summary.json"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _accuracy_value_text(accuracy: float | None, std: float | None) -> str | None:
    """Score-value cell text; appends '+/- std' when a cross-run std is present."""
    if accuracy is None:
        return None
    if std is None:
        return f"{float(accuracy):.2%}"
    return f"{float(accuracy):.2%} ± {float(std):.2%}"


@st.cache_data(ttl=30, show_spinner=False)
def _load_eval_results_cached(run_dir: str) -> dict[str, Any] | None:
    path = Path(run_dir) / "eval_results.json"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


@st.cache_data(ttl=30, show_spinner=False)
def _load_live_eval_results_cached(run_dir: str, run_type: str) -> dict[str, Any] | None:
    trace_dir = _TRACE_DIR_BY_RUN_TYPE.get(run_type)
    if trace_dir is None:
        return None
    index = _scan_agentic_index(run_dir, (trace_dir,))
    return _build_live_agentic_eval_results(run_dir, index, {})


def _numeric_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    return None


def _metric_denominator(explicit_denominator: int | None, observed_count: int) -> int:
    if explicit_denominator is not None and explicit_denominator > 0:
        return max(explicit_denominator, observed_count)
    return observed_count


def _turn_values_from_items(items: Any) -> list[float]:
    if not isinstance(items, list):
        return []
    return [value for item in items if isinstance(item, dict) if (value := _numeric_float(item.get("num_turns"))) is not None]


def _attempt_values_from_items(items: Any) -> list[float]:
    if not isinstance(items, list):
        return []
    return [value for item in items if isinstance(item, dict) if (value := _numeric_float(item.get("num_attempts"))) is not None]


def _average_finished_task_turns_from_items(items: Any, *, denominator: int | None = None) -> float | None:
    turns = _turn_values_from_items(items)
    if not turns:
        return None
    return sum(turns) / _metric_denominator(denominator, len(turns))


def _retry_ratio_from_items(items: Any, *, denominator: int | None = None) -> float | None:
    attempts = _attempt_values_from_items(items)
    if not attempts:
        return None
    return sum(1 for value in attempts if value > 1) / _metric_denominator(denominator, len(attempts))


def _average_finished_task_turns(run_dir: Path | None, run_type: str, summary: dict[str, Any] | None) -> float | None:
    if run_type == "wide_search":
        eval_results = _load_eval_results_cached(str(run_dir)) if run_dir is not None else None
        items = eval_results.get("items") if isinstance(eval_results, dict) else None
        average = _average_finished_task_turns_from_items(items)
        if average is not None:
            return average
        live_results = _load_live_eval_results_cached(str(run_dir), run_type) if run_dir is not None else None
        live_items = live_results.get("items") if isinstance(live_results, dict) else None
        live_average = _average_finished_task_turns_from_items(live_items)
        if live_average is not None:
            return live_average
        eval_summary = summary.get("eval_results", {}) if isinstance(summary, dict) else {}
        if isinstance(eval_summary, dict):
            return _numeric_float(eval_summary.get("avg_num_turns"))
        return None

    eval_summary = summary.get("eval_results", {}) if isinstance(summary, dict) else {}
    if isinstance(eval_summary, dict):
        summary_average = _numeric_float(eval_summary.get("avg_num_turns"))
        if summary_average is not None:
            return summary_average
    if run_dir is None:
        return None
    eval_results = _load_eval_results_cached(str(run_dir))
    items = eval_results.get("items") if isinstance(eval_results, dict) else None
    average = _average_finished_task_turns_from_items(items)
    if average is not None:
        return average
    live_results = _load_live_eval_results_cached(str(run_dir), run_type)
    live_average = _numeric_float(live_results.get("avg_num_turns")) if isinstance(live_results, dict) else None
    if live_average is not None:
        return live_average
    live_items = live_results.get("items") if isinstance(live_results, dict) else None
    return _average_finished_task_turns_from_items(live_items)


def _retry_ratio(run_dir: Path | None, run_type: str, summary: dict[str, Any] | None) -> float | None:
    if run_type == "wide_search":
        eval_results = _load_eval_results_cached(str(run_dir)) if run_dir is not None else None
        items = eval_results.get("items") if isinstance(eval_results, dict) else None
        retry_ratio = _retry_ratio_from_items(items)
        if retry_ratio is not None:
            return retry_ratio
        live_results = _load_live_eval_results_cached(str(run_dir), run_type) if run_dir is not None else None
        live_items = live_results.get("items") if isinstance(live_results, dict) else None
        live_retry_ratio = _retry_ratio_from_items(live_items)
        if live_retry_ratio is not None:
            return live_retry_ratio
        eval_summary = summary.get("eval_results", {}) if isinstance(summary, dict) else {}
        if isinstance(eval_summary, dict):
            return _numeric_float(eval_summary.get("retry_ratio"))
        return None

    eval_summary = summary.get("eval_results", {}) if isinstance(summary, dict) else {}
    if isinstance(eval_summary, dict):
        summary_retry_ratio = _numeric_float(eval_summary.get("retry_ratio"))
        if summary_retry_ratio is not None:
            return summary_retry_ratio
    if run_dir is None:
        return None
    eval_results = _load_eval_results_cached(str(run_dir))
    if isinstance(eval_results, dict):
        payload_retry_ratio = _numeric_float(eval_results.get("retry_ratio"))
        if payload_retry_ratio is not None:
            return payload_retry_ratio
    items = eval_results.get("items") if isinstance(eval_results, dict) else None
    retry_ratio = _retry_ratio_from_items(items)
    if retry_ratio is not None:
        return retry_ratio
    live_results = _load_live_eval_results_cached(str(run_dir), run_type)
    live_retry_ratio = _numeric_float(live_results.get("retry_ratio")) if isinstance(live_results, dict) else None
    if live_retry_ratio is not None:
        return live_retry_ratio
    live_items = live_results.get("items") if isinstance(live_results, dict) else None
    return _retry_ratio_from_items(live_items)


def _job_status_emoji(progress_ratio: float | None, latest_log_update: datetime | None, *, now: datetime | None = None) -> str:
    if progress_ratio is not None and progress_ratio >= 1.0:
        return _JOB_DONE
    if latest_log_update is None:
        return _JOB_STOPPED

    current = now or datetime.now(tz=latest_log_update.tzinfo or UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    update = latest_log_update
    if update.tzinfo is None:
        update = update.replace(tzinfo=current.tzinfo)
    if current.astimezone(UTC).timestamp() - update.astimezone(UTC).timestamp() > _RUNNING_LOG_STALE_SECONDS:
        return _JOB_STOPPED
    return _JOB_RUNNING


def _experiment_suffix(exp_name: str) -> str:
    for prefix in _EXPERIMENT_PREFIXES:
        if exp_name.startswith(prefix):
            return exp_name.removeprefix(prefix)
    return exp_name


def _experiment_key(exp_name: str, run_type: str) -> str:
    del run_type
    return exp_name


def _experiment_name_sort_key(exp_name: object) -> tuple[str, str]:
    suffix = _experiment_suffix(str(exp_name))
    return (suffix.casefold(), suffix)


def _status_sort_key(status: object) -> int:
    return _EXPERIMENT_STATUS_SORT_ORDER.get(str(status), len(_EXPERIMENT_STATUS_SORT_ORDER))


def _unknown_experiment_statuses(table_rows: list[dict[str, Any]]) -> list[str]:
    statuses = {str(row.get("Status", "")) for row in table_rows}
    return sorted(status for status in statuses if status and status not in _EXPERIMENT_STATUS_SORT_ORDER)


def _type_sort_key(run_type: object) -> tuple[int, str]:
    type_name = str(run_type)
    return (_EXPERIMENT_TYPE_SORT_ORDER.get(type_name, len(_EXPERIMENT_TYPE_SORT_ORDER)), type_name)


def _merged_row_status_sort_key(row: dict[str, Any]) -> int:
    statuses = [
        row[status_key]
        for experiment_key, status_key in (("Left Experiment", "Left Status"), ("Right Experiment", "Right Status"))
        if row.get(experiment_key)
    ]
    return min((_status_sort_key(status) for status in statuses), default=len(_EXPERIMENT_STATUS_SORT_ORDER))


def _merged_row_sort_key(row: dict[str, Any]) -> tuple[int, str, str]:
    name_key = _experiment_name_sort_key(row["Experiment"])
    return (_merged_row_status_sort_key(row), *name_key)


def _table_row_sort_key(row: dict[str, Any]) -> tuple[int, str, str, int, str]:
    name_key = _experiment_name_sort_key(row["Experiment"])
    type_key = _type_sort_key(row["Type"])
    return (_status_sort_key(row["Status"]), *name_key, *type_key)


def _score_accuracy_ratio(
    scores: dict[str, dict[str, Any]],
    *,
    score_key: str,
    score_threshold: float,
) -> float | None:
    values: list[bool] = []
    for task_id, row in scores.items():
        if str(task_id).startswith("_"):
            continue
        score = row.get(score_key)
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            values.append(float(score) > score_threshold)
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _widesearch_resolved_expected_trials(run_dir: Path, num_trials: int) -> ExpectedTaskCount | None:
    expected_instances = _resolve_expected_task_count_cached(str(run_dir))
    if expected_instances.count is None:
        return None
    source = str(expected_instances.source)
    if source.startswith("eval_results.json"):
        return ExpectedTaskCount(
            count=expected_instances.count,
            source=source,
            dataset_path=expected_instances.dataset_path,
            dataset_max_tasks=expected_instances.dataset_max_tasks,
            warning=expected_instances.warning,
        )
    if source.startswith("dashboard_summary"):
        return None
    return ExpectedTaskCount(
        count=expected_instances.count * num_trials,
        source=f"{source} * widesearch_summary.json:num_trials",
        dataset_path=expected_instances.dataset_path,
        dataset_max_tasks=expected_instances.dataset_max_tasks,
        warning=expected_instances.warning,
    )


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _widesearch_progress(run_dir: Path) -> WideSearchProgress:
    """Read WideSearch progress from ``widesearch_summary.json`` + score sidecars.

    The default experiments view uses ``llm_judge`` artifacts (which WideSearch
    runs never produce), so completed WideSearch experiments would otherwise
    show 0% progress forever. ``judged`` is the count of usable per-trial
    sidecars in ``widesearch_scores/`` (smooth progress as trials finish,
    excluding retryable runner failures); ``expected`` prefers
    ``total_instances`` (the configured/scheduled count, written once at run
    start) over ``num_complete_instances + num_incomplete_instances``, because
    the latter only counts instances that already produced a sidecar — during a
    run, an instance whose first trial hasn't started yet would be silently
    excluded, which can flip an in-progress run to 100%/finished when only one
    instance is fully done.
    """
    summary = _load_widesearch_summary(str(run_dir)) or {}
    judged = len(_load_widesearch_trial_scores(str(run_dir)))
    if not isinstance(summary, dict):
        return WideSearchProgress(
            judged_trials=judged,
            expected=ExpectedTaskCount(count=None, source="missing widesearch_summary.json"),
            num_trials=None,
            total_instances=None,
        )

    num_trials = _positive_int(summary.get("num_trials"))
    total_instances = _positive_int(summary.get("total_instances"))
    if num_trials is None:
        expected_count: int | None = None
        expected_source = "missing widesearch_summary.json:num_trials"
    elif total_instances is not None:
        expected_count = total_instances * num_trials
        expected_source = "widesearch_summary.json:total_instances"
    else:
        resolved_expected = _widesearch_resolved_expected_trials(run_dir, num_trials)
        if resolved_expected is not None and resolved_expected.count is not None:
            expected_count = resolved_expected.count
            expected_source = resolved_expected.source
        else:
            # Fallback for older runs without ``total_instances`` (pre-fix).
            # Underestimates expected when some instances haven't started any
            # trial yet, but at least matches the run's eventual size once
            # every instance has produced at least one sidecar.
            complete = _nonnegative_int(summary.get("num_complete_instances")) or 0
            incomplete = _nonnegative_int(summary.get("num_incomplete_instances")) or 0
            seen = complete + incomplete
            expected_count = seen * num_trials if seen > 0 else None
            expected_source = "widesearch_summary.json:complete+incomplete"
    return WideSearchProgress(
        judged_trials=judged,
        expected=ExpectedTaskCount(count=expected_count, source=expected_source),
        num_trials=num_trials,
        total_instances=total_instances,
    )


def _widesearch_judged_and_expected(run_dir: Path) -> tuple[int, ExpectedTaskCount]:
    progress = _widesearch_progress(run_dir)
    return progress.judged_trials, progress.expected


def _widesearch_item_f1_pass_at_trials(run_dir: Path) -> float | None:
    """Use item F1 max@N/pass@N as the WideSearch homepage quality metric."""
    summary = _load_widesearch_summary(str(run_dir))
    if not isinstance(summary, dict):
        return None
    leaderboard = summary.get("leaderboard")
    value = leaderboard.get("item_f1_max@N") if isinstance(leaderboard, dict) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        summary_metrics = summary.get("summary")
        f1_item = summary_metrics.get("f1_by_item") if isinstance(summary_metrics, dict) else None
        value = f1_item.get("max_n") if isinstance(f1_item, dict) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _widesearch_leaderboard_accuracy(run_dir: Path) -> float | None:
    """Backward-compatible wrapper for the homepage WideSearch score."""
    return _widesearch_item_f1_pass_at_trials(run_dir)


def _widesearch_score_label(num_trials: int | None) -> str:
    return f"Item F1 pass@{num_trials}" if num_trials is not None else "Item F1 pass@N"


def _build_experiment_rows(source: ExperimentOverviewSource) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp_name in _discover_experiments(source.log_dir, source.run_type):
        exp_root = source.log_dir / exp_name
        run_name = _latest_run(exp_root)
        run_dir = exp_root / run_name if run_name else None
        summary = _load_dashboard_summary_cached(str(run_dir)) if run_dir else None
        llm_summary = summary.get("llm_judge", {}) if isinstance(summary, dict) else {}
        num_trials: int | None = None
        accuracy_label = "LLM Accuracy"
        if source.run_type == "wide_search" and run_dir is not None:
            progress = _widesearch_progress(run_dir)
            judged = progress.judged_trials
            expected = progress.expected
            num_trials = progress.num_trials
            accuracy = _widesearch_item_f1_pass_at_trials(run_dir)
            accuracy_label = _widesearch_score_label(num_trials)
        elif isinstance(llm_summary, dict) and isinstance(llm_summary.get("judged_task_count"), int):
            judged = int(llm_summary["judged_task_count"])
            accuracy_ratio = llm_summary.get("llm_judge_accuracy")
            accuracy = float(accuracy_ratio) if isinstance(accuracy_ratio, (int, float)) and not isinstance(accuracy_ratio, bool) else None
            expected = _resolve_expected_task_count_cached(str(run_dir)) if run_dir else ExpectedTaskCount(count=None, source="no run selected")
        else:
            llm_evals = _load_llm_judge_results(str(run_dir)) if run_dir else {}
            judged = _judged_task_count(llm_evals)
            accuracy = _score_accuracy_ratio(
                llm_evals,
                score_key="llm_judge_score",
                score_threshold=LLM_JUDGE_CORRECT_THRESHOLD,
            )
            expected = _resolve_expected_task_count_cached(str(run_dir)) if run_dir else ExpectedTaskCount(count=None, source="no run selected")
        progress_ratio = judged / expected.count if expected.count else None
        # LiveBrowseComp multi-run evals report avg@N over num_runs (paper Table 3).
        # When run_eval_config has written the cross-run rollup, show avg@N +/- std
        # instead of just the latest run's single-shot accuracy.
        accuracy_std: float | None = None
        if source.run_type == "web_search":
            aggregate = _load_aggregate_summary_cached(str(exp_root))
            avg_accuracy = _numeric_float(aggregate.get("avg_accuracy")) if isinstance(aggregate, dict) else None
            if avg_accuracy is not None:
                accuracy = avg_accuracy
                accuracy_std = _numeric_float(aggregate.get("std_accuracy"))
                accuracy_label = f"avg@{aggregate.get('num_runs')}"
        rows.append(
            {
                "_key": _experiment_key(exp_name, source.run_type),
                "Experiment": exp_name,
                "_progress_ratio": progress_ratio,
                "_accuracy_ratio": accuracy,
                "_accuracy_std": accuracy_std,
                "_accuracy_label": accuracy_label,
                "_avg_finished_task_turns": _average_finished_task_turns(run_dir, source.run_type, summary),
                "_retry_ratio": _retry_ratio(run_dir, source.run_type, summary),
                "_finished": bool(expected.count and judged >= expected.count),
                "_judged": judged,
                "_expected_tasks": expected.count,
                "_expected_tasks_source": expected.source,
                "_expected_tasks_warning": expected.warning,
                "_num_trials": num_trials,
                "_status": _job_status_emoji(progress_ratio, _latest_status_update_at(run_dir, source.run_type)),
            }
        )
    return rows


def _side_row_by_key(source: ExperimentOverviewSource) -> dict[str, dict[str, Any]]:
    return {str(row["_key"]): row for row in _build_experiment_rows(source)}


def _build_merged_experiment_rows(
    left: ExperimentOverviewSource,
    right: ExperimentOverviewSource,
) -> list[dict[str, Any]]:
    left_rows = _side_row_by_key(left)
    right_rows = _side_row_by_key(right)
    merged: list[dict[str, Any]] = []
    for key in sorted(set(left_rows) | set(right_rows)):
        left_row = left_rows.get(key)
        right_row = right_rows.get(key)
        left_finished = bool(left_row and left_row["_finished"])
        right_finished = bool(right_row and right_row["_finished"])
        existing_finished = [finished for row, finished in ((left_row, left_finished), (right_row, right_finished)) if row]
        merged.append(
            {
                "Experiment": key,
                "Left Experiment": str(left_row["Experiment"]) if left_row else "",
                "Left Type": left.run_type if left_row else "",
                "Left Status": str(left_row["_status"]) if left_row else "",
                "Left Progress %": _numeric_float(left_row["_progress_ratio"]) if left_row else None,
                "Left LLM Judge Accuracy %": left_row["_accuracy_ratio"] if left_row else None,
                "Left Accuracy Std": left_row.get("_accuracy_std") if left_row else None,
                "Left Accuracy Label": str(left_row.get("_accuracy_label", "LLM Accuracy")) if left_row else "",
                "Left Avg. Turns": left_row["_avg_finished_task_turns"] if left_row else None,
                "Left Retry %": left_row["_retry_ratio"] if left_row else None,
                "_left_judged": int(left_row["_judged"]) if left_row else 0,
                "_left_expected_tasks": left_row["_expected_tasks"] if left_row else None,
                "_left_expected_tasks_warning": left_row["_expected_tasks_warning"] if left_row else None,
                "_left_num_trials": left_row.get("_num_trials") if left_row else None,
                "Right Experiment": str(right_row["Experiment"]) if right_row else "",
                "Right Type": right.run_type if right_row else "",
                "Right Status": str(right_row["_status"]) if right_row else "",
                "Right Progress %": _numeric_float(right_row["_progress_ratio"]) if right_row else None,
                "Right LLM Judge Accuracy %": right_row["_accuracy_ratio"] if right_row else None,
                "Right Accuracy Std": right_row.get("_accuracy_std") if right_row else None,
                "Right Accuracy Label": str(right_row.get("_accuracy_label", "LLM Accuracy")) if right_row else "",
                "Right Avg. Turns": right_row["_avg_finished_task_turns"] if right_row else None,
                "Right Retry %": right_row["_retry_ratio"] if right_row else None,
                "_right_judged": int(right_row["_judged"]) if right_row else 0,
                "_right_expected_tasks": right_row["_expected_tasks"] if right_row else None,
                "_right_expected_tasks_warning": right_row["_expected_tasks_warning"] if right_row else None,
                "_right_num_trials": right_row.get("_num_trials") if right_row else None,
                "_all_finished": bool(existing_finished) and all(existing_finished),
                "_max_judged": max(int(row["_judged"]) for row in (left_row, right_row) if row),
            }
        )
    return sorted(merged, key=_merged_row_sort_key)


def _experiment_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table_rows: list[dict[str, Any]] = []
    for row in rows:
        left_type = str(row["Left Type"])
        right_type = str(row["Right Type"])
        if row["Left Experiment"] and row["Right Experiment"] and left_type == right_type:
            use_right = int(row["_right_judged"]) > int(row["_left_judged"])
            progress = row["Right Progress %"] if use_right else row["Left Progress %"]
            accuracy = row["Right LLM Judge Accuracy %"] if use_right else row["Left LLM Judge Accuracy %"]
            accuracy_label = row.get("Right Accuracy Label", "LLM Accuracy") if use_right else row.get("Left Accuracy Label", "LLM Accuracy")
            status = row["Right Status"] if use_right else row["Left Status"]
            finished_tasks = row["_right_judged"] if use_right else row["_left_judged"]
            avg_turns = row.get("Right Avg. Turns") if use_right else row.get("Left Avg. Turns")
            retry_ratio = row.get("Right Retry %") if use_right else row.get("Left Retry %")
            accuracy_std = row.get("Right Accuracy Std") if use_right else row.get("Left Accuracy Std")
            table_rows.append(
                {
                    "Experiment": row["Experiment"],
                    "Type": left_type,
                    "Status": status,
                    "Running Progress": None if progress is None else float(progress) * 100,
                    "Finished Tasks": int(finished_tasks),
                    "Score Metric": accuracy_label,
                    "LLM Accuracy": None if accuracy is None else float(accuracy) * 100,
                    "LLM Accuracy Value": _accuracy_value_text(accuracy, accuracy_std),
                    "Avg. Turns": None if avg_turns is None else float(avg_turns),
                    "Retry%": None if retry_ratio is None else float(retry_ratio) * 100,
                }
            )
            continue
        if row["Left Experiment"]:
            table_rows.append(
                {
                    "Experiment": row["Experiment"],
                    "Type": left_type,
                    "Status": row["Left Status"],
                    "Running Progress": None if row["Left Progress %"] is None else float(row["Left Progress %"]) * 100,
                    "Finished Tasks": int(row["_left_judged"]),
                    "Score Metric": row.get("Left Accuracy Label", "LLM Accuracy"),
                    "LLM Accuracy": None if row["Left LLM Judge Accuracy %"] is None else float(row["Left LLM Judge Accuracy %"]) * 100,
                    "LLM Accuracy Value": _accuracy_value_text(row["Left LLM Judge Accuracy %"], row.get("Left Accuracy Std")),
                    "Avg. Turns": None if row.get("Left Avg. Turns") is None else float(row["Left Avg. Turns"]),
                    "Retry%": None if row.get("Left Retry %") is None else float(row["Left Retry %"]) * 100,
                }
            )
        if row["Right Experiment"]:
            table_rows.append(
                {
                    "Experiment": row["Experiment"],
                    "Type": right_type,
                    "Status": row["Right Status"],
                    "Running Progress": None if row["Right Progress %"] is None else float(row["Right Progress %"]) * 100,
                    "Finished Tasks": int(row["_right_judged"]),
                    "Score Metric": row.get("Right Accuracy Label", "LLM Accuracy"),
                    "LLM Accuracy": None if row["Right LLM Judge Accuracy %"] is None else float(row["Right LLM Judge Accuracy %"]) * 100,
                    "LLM Accuracy Value": _accuracy_value_text(row["Right LLM Judge Accuracy %"], row.get("Right Accuracy Std")),
                    "Avg. Turns": None if row.get("Right Avg. Turns") is None else float(row["Right Avg. Turns"]),
                    "Retry%": None if row.get("Right Retry %") is None else float(row["Right Retry %"]) * 100,
                }
            )
    table_rows = sorted(table_rows, key=_table_row_sort_key)
    if table_rows and all(row.get("Score Metric") == "LLM Accuracy" for row in table_rows):
        for row in table_rows:
            row.pop("Score Metric", None)
    return table_rows


def _score_column_label(table_rows: list[dict[str, Any]]) -> str:
    labels = {str(row.get("Score Metric", "")).strip() for row in table_rows if row.get("Score Metric")}
    if len(labels) == 1:
        return next(iter(labels))
    if not labels or labels == {"LLM Accuracy"}:
        return "LLM Accuracy"
    return "Score"


def _render_experiment_table(rows: list[dict[str, Any]]) -> None:
    table_rows = _experiment_table_rows(rows)
    unknown_statuses = _unknown_experiment_statuses(table_rows)
    if unknown_statuses:
        st.warning("Statuses outside the default order: " + ", ".join(f"`{status}`" for status in unknown_statuses))
    score_label = _score_column_label(table_rows)
    display_rows = [dict(row) for row in table_rows]
    if score_label != "Score":
        for row in display_rows:
            row.pop("Score Metric", None)
    df = pd.DataFrame(display_rows)
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Running Progress": st.column_config.ProgressColumn(
                "Running Progress",
                format="%.0f%%",
                min_value=0.0,
                max_value=100.0,
                color="blue",
            ),
            "LLM Accuracy": st.column_config.ProgressColumn(
                score_label,
                format="%.0f%%",
                min_value=0.0,
                max_value=100.0,
                color="green",
            ),
            "LLM Accuracy Value": st.column_config.TextColumn("Score Value"),
            "Avg. Turns": st.column_config.NumberColumn(
                "Avg. Turns",
                format="%.1f",
            ),
            "Retry%": st.column_config.NumberColumn(
                "Retry%",
                format="%.1f%%",
            ),
        },
    )


def _render_expected_task_warnings(rows: list[dict[str, Any]]) -> None:
    warnings: list[str] = []
    for row in rows:
        for side_name, experiment_key, expected_key, warning_key in (
            ("Left", "Left Experiment", "_left_expected_tasks", "_left_expected_tasks_warning"),
            ("Right", "Right Experiment", "_right_expected_tasks", "_right_expected_tasks_warning"),
        ):
            experiment = row.get(experiment_key)
            warning = row.get(warning_key)
            if experiment and row.get(expected_key) is None and warning:
                warnings.append(f"{side_name} `{experiment}`: {warning}")
    for warning in sorted(set(warnings)):
        st.warning(warning)


def _render_experiments_tab(
    left: ExperimentOverviewSource,
    right: ExperimentOverviewSource,
) -> None:
    st.header("Experiments")
    st.caption(
        "Progress counts tasks with numeric `llm_judge_score`; WideSearch counts completed trial score sidecars "
        "against `total_instances * num_trials`. "
        "Expected counts are read from run artifacts and effective configs. "
        "Status is completed at 100%, running when incomplete logs or WideSearch artifacts updated within 30 minutes, and stopped otherwise. "
        "Running rows are shown first, then completed and stopped/paused rows."
    )
    st.caption(f"Left: `{left.log_dir}` | type: `{left.run_type}`  Right: `{right.log_dir}` | type: `{right.run_type}`")
    rows = _build_merged_experiment_rows(left, right)
    if not rows:
        st.info("No experiments found under the selected log directories.")
        return
    _render_expected_task_warnings(rows)
    _render_experiment_table(rows)
