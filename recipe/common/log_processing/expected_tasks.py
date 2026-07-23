# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the expected task count for a recipe run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExpectedTaskCount:
    count: int | None
    source: str
    dataset_path: str | None = None
    dataset_max_tasks: int | None = None
    warning: str | None = None

    @property
    def is_known(self) -> bool:
        return self.count is not None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
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


def _run_candidate_paths(run_dir: Path, name: str) -> tuple[Path, ...]:
    return (run_dir / name, run_dir.parent / name)


def _dataset_from_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config.get("dataset")
    if isinstance(dataset, dict):
        return dataset
    benchmark = config.get("benchmark")
    if isinstance(benchmark, dict):
        return benchmark
    effective = config.get("effective_config")
    if isinstance(effective, dict):
        dataset = effective.get("dataset")
        if isinstance(dataset, dict):
            return dataset
        benchmark = effective.get("benchmark")
        if isinstance(benchmark, dict):
            return benchmark
    cli_args = config.get("cli_args")
    if isinstance(cli_args, dict):
        return cli_args
    return {}


def _config_dataset(run_dir: Path) -> tuple[dict[str, Any], str | None]:
    metadata = _load_json(run_dir / "run_metadata.json")
    dataset = _dataset_from_config(metadata)
    if dataset:
        return dataset, "run_metadata.json"

    for path in _run_candidate_paths(run_dir, "run_config.effective.yaml"):
        config = _load_yaml(path)
        dataset = _dataset_from_config(config)
        if dataset:
            return dataset, str(path.relative_to(run_dir) if path.is_relative_to(run_dir) else path)
    return {}, None


def _count_dataset_rows(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
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


def _expected_from_eval_results(run_dir: Path) -> ExpectedTaskCount | None:
    data = _load_json(run_dir / "eval_results.json")
    count = _as_positive_int(data.get("num_items"))
    if count is None:
        return None
    return ExpectedTaskCount(count=count, source="eval_results.json num_items")


def _expected_from_dashboard_summary(run_dir: Path) -> ExpectedTaskCount | None:
    data = _load_json(run_dir / "dashboard_summary.json")
    count = _as_positive_int(data.get("expected_tasks"))
    if count is None:
        return None
    return ExpectedTaskCount(
        count=count,
        source=str(data.get("expected_tasks_source") or "dashboard_summary.json expected_tasks"),
        dataset_path=str(data.get("dataset_path")) if data.get("dataset_path") else None,
        dataset_max_tasks=_as_positive_int(data.get("dataset_max_tasks")),
        warning=str(data.get("expected_tasks_warning")) if data.get("expected_tasks_warning") else None,
    )


def _expected_from_config(run_dir: Path) -> ExpectedTaskCount | None:
    dataset, source = _config_dataset(run_dir)
    if not dataset:
        return None

    max_tasks = _as_positive_int(dataset.get("max_tasks"))
    dataset_path = str(dataset.get("data_path") or "") or None
    if max_tasks is not None:
        return ExpectedTaskCount(
            count=max_tasks,
            source=f"{source} dataset.max_tasks" if source else "dataset.max_tasks",
            dataset_path=dataset_path,
            dataset_max_tasks=max_tasks,
        )

    if not dataset_path:
        return ExpectedTaskCount(
            count=None,
            source=f"{source} dataset.data_path" if source else "dataset.data_path",
            warning="dataset.data_path is not set",
        )

    path = Path(dataset_path)
    if not path.exists():
        return ExpectedTaskCount(
            count=None,
            source=f"{source} dataset.data_path" if source else "dataset.data_path",
            dataset_path=dataset_path,
            warning=f"Dataset path is not readable: {dataset_path}",
        )

    try:
        count = _count_dataset_rows(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return ExpectedTaskCount(
            count=None,
            source=f"{source} dataset.data_path" if source else "dataset.data_path",
            dataset_path=dataset_path,
            warning=f"Dataset path is not readable: {dataset_path} ({type(exc).__name__}: {exc})",
        )

    return ExpectedTaskCount(
        count=count,
        source=f"{source} dataset.data_path row count" if source else "dataset.data_path row count",
        dataset_path=dataset_path,
        dataset_max_tasks=None,
    )


def resolve_expected_task_count(run_dir: str | Path) -> ExpectedTaskCount:
    """Resolve expected task count for a run directory.

    Resolution order:
    1. effective run config ``dataset.max_tasks`` / ``benchmark.max_tasks``
    2. row count from effective run config ``dataset.data_path`` / ``benchmark.data_path``
    3. ``eval_results.json["num_items"]``
    4. ``dashboard_summary.json["expected_tasks"]``

    Configured dataset size is preferred over generated artifacts because live
    dashboard refreshes can create partial ``eval_results.json`` or stale
    ``dashboard_summary.json`` files while an evaluation is still running.
    """
    run_path = Path(run_dir)
    config_result = _expected_from_config(run_path)
    if config_result is not None and config_result.count is not None:
        return config_result
    for resolver in (_expected_from_eval_results, _expected_from_dashboard_summary):
        result = resolver(run_path)
        if result is not None:
            return result
    if config_result is not None:
        return config_result
    return ExpectedTaskCount(count=None, source="unavailable", warning="Expected task count is unavailable from run artifacts")
