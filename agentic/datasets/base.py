# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetItem:
    """A single dataset entry with a problem, ground-truth label, and metadata."""

    problem: str
    label: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDataset(ABC):
    """Minimal dataset contract for agentic/ inference evaluation."""

    def __init__(self) -> None:
        self.items: list[DatasetItem] = []

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> DatasetItem:
        return self.items[idx]

    @staticmethod
    def _read_datafile(path: Path, repo_id: str, filename: str) -> pd.DataFrame:
        import pandas as pd

        if path.suffix == ".parquet":
            data = pd.read_parquet(path)
        elif path.suffix in (".json", ".jsonl"):
            data = pd.read_json(path, lines=path.suffix == ".jsonl")
        elif path.suffix == ".csv":
            data = pd.read_csv(path)
        else:
            msg = f"Unsupported file format: {path.suffix}"
            raise ValueError(msg)
        logger.info("Loaded %d records from %s (%s/%s).", len(data), path, repo_id, filename)
        return data
