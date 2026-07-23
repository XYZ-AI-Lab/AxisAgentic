# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy formatting helpers for the dashboard."""

from __future__ import annotations

from typing import Any


def _format_score_accuracy(
    scores: dict[str, dict[str, Any]],
    *,
    score_key: str = "score",
    bool_key: str | None = None,
    score_threshold: float = 0.0,
    task_ids: set[str] | None = None,
) -> str:
    values: list[bool] = []
    for tid, row in scores.items():
        if str(tid).startswith("_"):
            continue
        if task_ids is not None and str(tid) not in task_ids:
            continue
        if bool_key is not None and row.get(bool_key) is not None:
            values.append(bool(row.get(bool_key)))
            continue
        score = row.get(score_key)
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            values.append(float(score) > score_threshold)
    if not values:
        return "-"
    correct = sum(1 for value in values if value)
    return f"{correct}/{len(values)} ({correct / len(values) * 100:.0f}%)"
