# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Dataset loader shared by web-search benchmark recipes."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from agentic.datasets.base import BaseDataset, DatasetItem

logger = logging.getLogger(__name__)


class BenchmarkDataset(BaseDataset):
    """Dataset loader for the supported JSONL and CSV benchmark formats.

    Expects a JSONL file where each line contains::

        {"task_id": "...", "task_question": "...", "ground_truth": "...", "file_name": "..."}

    Args:
        data_path: Path to the JSONL data file.
        max_items: Maximum number of items to load. ``None`` loads all.
    """

    def __init__(self, *, data_path: str | Path, max_items: int | None = None) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.max_items = max_items

    @staticmethod
    def _dataset_file_for_dir(path: Path) -> Path:
        candidates = ("standardized_data.jsonl", "DSQA-full.csv")
        for name in candidates:
            candidate = path / name
            if candidate.exists():
                return candidate
        msg = f"Dataset directory has no supported data file ({', '.join(candidates)}): {path}"
        raise FileNotFoundError(msg)

    def load(self) -> None:
        data_path = self._dataset_file_for_dir(self.data_path) if self.data_path.is_dir() else self.data_path
        if not data_path.exists():
            msg = f"Data file not found: {data_path}"
            raise FileNotFoundError(msg)

        if data_path.suffix.lower() == ".csv":
            self._load_csv(data_path)
        else:
            self._load_jsonl(data_path)

        logger.info("Loaded %d benchmark items from %s", len(self.items), data_path)

    def _append_record(self, idx: int, record: dict[str, Any], *, data_path: Path) -> None:
        if "task_question" in record:
            task_id = record.get("task_id", str(idx))
            problem = record["task_question"]
            label = record.get("ground_truth", "")
            excluded = {"task_question", "ground_truth"}
        elif "problem" in record and "answer" in record:
            task_id = record.get("task_id", f"deepsearchqa__{idx}")
            problem = record["problem"]
            label = record.get("answer", "")
            excluded = {"problem", "answer"}
        else:
            msg = f"Unsupported dataset record fields at {data_path}:{idx + 1}: {sorted(record)}"
            raise ValueError(msg)

        self.items.append(
            DatasetItem(
                problem=str(problem),
                label=str(label),
                source=json.dumps({"task_id": str(task_id), "file_name": record.get("file_name")}),
                metadata={k: v for k, v in record.items() if k not in excluded},
            )
        )

    def _load_jsonl(self, data_path: Path) -> None:
        with data_path.open(encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if self.max_items is not None and idx >= self.max_items:
                    break
                strip_line = line.strip()
                if not strip_line:
                    continue
                record = json.loads(strip_line)
                self._append_record(idx, record, data_path=data_path)

    def _load_csv(self, data_path: Path) -> None:
        with data_path.open(newline="", encoding="utf-8-sig") as f:
            for idx, record in enumerate(csv.DictReader(f)):
                if self.max_items is not None and idx >= self.max_items:
                    break
                self._append_record(idx, record, data_path=data_path)
