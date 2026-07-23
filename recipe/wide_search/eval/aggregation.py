# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

METRIC_KEYS: tuple[str, ...] = (
    "score",
    "precision_by_row",
    "recall_by_row",
    "f1_by_row",
    "precision_by_item",
    "recall_by_item",
    "f1_by_item",
)


@dataclass
class TrialResult:
    """One trial's per-instance metrics. Field names mirror the official keys."""

    instance_id: str
    trial_index: int
    score: float = 0.0
    precision_by_row: float = 0.0
    recall_by_row: float = 0.0
    f1_by_row: float = 0.0
    precision_by_item: float = 0.0
    recall_by_item: float = 0.0
    f1_by_item: float = 0.0
    msg: str = ""

    def values(self) -> dict[str, float]:
        return {m: float(getattr(self, m)) for m in METRIC_KEYS}


@dataclass
class PerTaskAggregate:
    """avg_n / max_n / min_n per metric for a single instance over k trials."""

    instance_id: str
    num_trials: int
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)


def per_task_aggregate(instance_id: str, trials: Iterable[TrialResult], *, expected_trials: int | None = None) -> PerTaskAggregate:
    trials = list(trials)
    if expected_trials is not None and len(trials) != expected_trials:
        msg = f"per_task_aggregate: instance {instance_id} has {len(trials)} trials, expected {expected_trials}"
        raise ValueError(msg)
    metrics: dict[str, dict[str, float]] = {}
    for m in METRIC_KEYS:
        values = [float(getattr(t, m)) for t in trials]
        if not values:
            metrics[m] = {"avg_n": 0.0, "max_n": 0.0, "min_n": 0.0}
            continue
        metrics[m] = {
            "avg_n": float(np.mean(values)),
            "max_n": float(np.max(values)),
            "min_n": float(np.min(values)),
        }
    return PerTaskAggregate(instance_id=instance_id, num_trials=len(trials), metrics=metrics)


def per_task_aggregates(trials: Iterable[TrialResult], *, expected_trials: int | None = None) -> list[PerTaskAggregate]:
    by_iid: dict[str, list[TrialResult]] = defaultdict(list)
    for t in trials:
        by_iid[t.instance_id].append(t)
    out: list[PerTaskAggregate] = []
    for iid in sorted(by_iid.keys()):
        out.append(per_task_aggregate(iid, by_iid[iid], expected_trials=expected_trials))
    return out


def global_summary(per_task: list[PerTaskAggregate]) -> dict[str, dict[str, float]]:
    """Mean of per-task ``avg_n / max_n / min_n`` across instances.

    Output shape mirrors the official ``summary.json``:
    ``{metric: {avg_n, max_n, min_n}}`` for each of the 7 metrics.
    """
    summary: dict[str, dict[str, float]] = {}
    if not per_task:
        for m in METRIC_KEYS:
            summary[m] = {"avg_n": 0.0, "max_n": 0.0, "min_n": 0.0}
        return summary
    for m in METRIC_KEYS:
        avg = float(np.mean([pt.metrics[m]["avg_n"] for pt in per_task]))
        mx = float(np.mean([pt.metrics[m]["max_n"] for pt in per_task]))
        mn = float(np.mean([pt.metrics[m]["min_n"] for pt in per_task]))
        summary[m] = {"avg_n": avg, "max_n": mx, "min_n": mn}
    return summary


def leaderboard_view(summary: dict[str, dict[str, float]]) -> dict[str, float]:
    """Flatten the global summary into the six numbers shown on the leaderboard."""
    score = summary.get("score", {})
    f1_row = summary.get("f1_by_row", {})
    f1_item = summary.get("f1_by_item", {})
    return {
        "success_rate_avg@N": float(score.get("avg_n", 0.0)),
        "success_rate_pass@N": float(score.get("max_n", 0.0)),
        "row_f1_avg@N": float(f1_row.get("avg_n", 0.0)),
        "row_f1_max@N": float(f1_row.get("max_n", 0.0)),
        "item_f1_avg@N": float(f1_item.get("avg_n", 0.0)),
        "item_f1_max@N": float(f1_item.get("max_n", 0.0)),
    }


def serialize_per_task(per_task: list[PerTaskAggregate]) -> list[dict[str, Any]]:
    return [{"instance_id": pt.instance_id, "num_trials": pt.num_trials, "metrics": pt.metrics} for pt in per_task]
