# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Guards for accidental resume into already-final benchmark run dirs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class FinalizedRunResumeError(RuntimeError):
    """Raised when a resume would write into an already-final run directory."""


@dataclass(frozen=True)
class FinalizedRunStatus:
    run_dir: str
    expected_task_count: int | None
    benchmark_exists: bool
    benchmark_row_count: int
    benchmark_unique_task_count: int
    benchmark_missing_task_id_count: int
    eval_results_exists: bool
    eval_num_items: int | None
    is_finalized: bool
    reason: str

    def abort_message(self) -> str:
        expected = self.expected_task_count if self.expected_task_count is not None else "unknown"
        return (
            f"Refusing --resume into finalized run_dir={self.run_dir}: "
            f"benchmark_unique_task_count={self.benchmark_unique_task_count}, "
            f"benchmark_row_count={self.benchmark_row_count}, "
            f"eval_num_items={self.eval_num_items}, expected_task_count={expected}. "
            "Use a new output_dir, or pass --force_resume_finalized_run to override intentionally."
        )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _as_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None


def count_dataset_items(data_path: str | Path, *, max_tasks: int | None = None) -> int:
    """Count dataset rows for final-run completion checks."""
    if max_tasks is not None and max_tasks > 0:
        return max_tasks

    path = Path(data_path)
    if path.is_dir():
        for name in ("standardized_data.jsonl", "DSQA-full.csv"):
            candidate = path / name
            if candidate.exists():
                return count_dataset_items(candidate)
        msg = f"unsupported dataset directory: {path}"
        raise ValueError(msg)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return sum(1 for _ in csv.DictReader(handle))

    if suffix == ".jsonl":
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count

    if suffix == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            return len(loaded)
        if isinstance(loaded, dict):
            for key in ("items", "data", "examples", "tasks"):
                value = loaded.get(key)
                if isinstance(value, list):
                    return len(value)
        raise ValueError("JSON dataset is not a list and has no list-valued items/data/examples/tasks field")

    raise ValueError(f"unsupported dataset file suffix: {suffix or '<none>'}")


def inspect_finalized_run(run_dir: str | Path, *, expected_task_count: int | None) -> FinalizedRunStatus:
    """Inspect benchmark/eval artifacts without considering trace count."""
    run_path = Path(run_dir)
    expected = expected_task_count if expected_task_count is not None and expected_task_count > 0 else None
    benchmark_path = run_path / "benchmark_results.jsonl"
    benchmark_row_count = 0
    missing_task_id_count = 0
    task_ids: set[str] = set()
    if benchmark_path.exists():
        try:
            lines = benchmark_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            lines = []
        for line in lines:
            if not line.strip():
                continue
            benchmark_row_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                missing_task_id_count += 1
                continue
            task_id = row.get("task_id") if isinstance(row, dict) else None
            if task_id is None:
                missing_task_id_count += 1
            else:
                task_ids.add(str(task_id))

    eval_path = run_path / "eval_results.json"
    eval_data = _load_json(eval_path)
    eval_num_items = _as_positive_int(eval_data.get("num_items"))

    if not benchmark_path.exists():
        reason = "benchmark_results.jsonl is missing"
    elif expected is None:
        reason = "expected task count is unavailable"
    elif len(task_ids) < expected:
        reason = "benchmark_results.jsonl unique task count is below expected task count"
    elif eval_num_items is None:
        reason = "eval_results.json num_items is missing"
    elif eval_num_items < expected:
        reason = "eval_results.json num_items is below expected task count"
    else:
        reason = "finalized benchmark/eval artifacts are complete"

    is_finalized = (
        benchmark_path.exists() and expected is not None and len(task_ids) >= expected and eval_num_items is not None and eval_num_items >= expected
    )
    return FinalizedRunStatus(
        run_dir=str(run_path),
        expected_task_count=expected,
        benchmark_exists=benchmark_path.exists(),
        benchmark_row_count=benchmark_row_count,
        benchmark_unique_task_count=len(task_ids),
        benchmark_missing_task_id_count=missing_task_id_count,
        eval_results_exists=eval_path.exists(),
        eval_num_items=eval_num_items,
        is_finalized=is_finalized,
        reason=reason,
    )


def guard_resume_into_finalized_run(
    run_dir: str | Path,
    *,
    expected_task_count: int | None,
    force: bool = False,
) -> FinalizedRunStatus:
    """Raise unless ``--resume`` is safe for this run directory."""
    status = inspect_finalized_run(run_dir, expected_task_count=expected_task_count)
    if status.is_finalized and not force:
        raise FinalizedRunResumeError(status.abort_message())
    return status
