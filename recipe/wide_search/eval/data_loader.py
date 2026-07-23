# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from recipe.wide_search.eval.preprocess import norm_column

_GOLD_CSV_ENCODING = "utf-8-sig"


@dataclass
class WideSearchQuery:
    instance_id: str
    query: str
    evaluation: dict[str, Any]
    answer: pd.DataFrame
    language: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _load_evaluation_field(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return dict(raw)
    msg = f"unsupported evaluation field type: {type(raw).__name__}"
    raise TypeError(msg)


def load_gold_dataframe(gold_csv_path: str | Path, required_columns: list[str]) -> pd.DataFrame | None:
    path = Path(gold_csv_path)
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding=_GOLD_CSV_ENCODING)
    df.columns = [norm_column(str(col).strip()) for col in df.columns]
    for col in required_columns:
        if col not in df.columns:
            return None
    return df[required_columns]


def load_widesearch_queries(
    data_path: str | Path,
    gold_dir: str | Path,
    *,
    instance_ids: list[str] | None = None,
) -> list[WideSearchQuery]:
    data_path = Path(data_path)
    gold_dir = Path(gold_dir)
    if not data_path.exists():
        msg = f"widesearch data_path not found: {data_path}"
        raise FileNotFoundError(msg)
    if not gold_dir.is_dir():
        msg = f"widesearch gold_dir not a directory: {gold_dir}"
        raise NotADirectoryError(msg)

    selected: set[str] | None = set(instance_ids) if instance_ids else None
    queries: list[WideSearchQuery] = []
    skipped: list[tuple[str, str]] = []

    with data_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            instance_id = row["instance_id"]
            if selected is not None and instance_id not in selected:
                continue
            evaluation = _load_evaluation_field(row["evaluation"])
            required = list(evaluation.get("required") or [])
            gold_csv_path = gold_dir / f"{instance_id}.csv"
            answer_df = load_gold_dataframe(gold_csv_path, required)
            if answer_df is None:
                skipped.append((instance_id, str(gold_csv_path)))
                continue
            queries.append(
                WideSearchQuery(
                    instance_id=instance_id,
                    query=row["query"],
                    evaluation=evaluation,
                    answer=answer_df,
                    language=row.get("language", ""),
                    metadata={k: v for k, v in row.items() if k not in {"instance_id", "query", "evaluation", "language"}},
                )
            )
    queries.sort(key=lambda q: q.instance_id)
    return queries
