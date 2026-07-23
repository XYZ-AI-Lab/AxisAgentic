# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic.datasets.base import BaseDataset
    from agentic.evaluation.verifier import BaseVerifier
    from agentic.orchestration.task_orchestrator import TaskOrchestrator

logger = logging.getLogger(__name__)

_UTC8 = timezone(timedelta(hours=8))


@dataclass
class EvaluationResult:
    """Aggregate evaluation result."""

    total: int = 0
    correct: int = 0
    scores: list[float] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


class BatchEvaluator:
    """Run an orchestrator over a dataset, verify outputs, and report accuracy.

    Optionally persists per-task evaluation results to disk.  When *eval_dir*
    is provided, each scored task is written to
    ``<eval_dir>/<eval_name>/<task_id>.json``.  The JSON file includes the
    score, prediction, ground truth, and — when *trace_log_dir* is given — a
    ``source_trace`` pointer to the inference trace log file.

    Args:
        orchestrator: Task orchestrator for running inference.
        verifier: Verifier for scoring model outputs.
        max_concurrent: Maximum concurrent tasks during :meth:`evaluate`.
        eval_dir: Directory for writing per-task eval result files.
            Should typically be the same root as the ``TaskLogger`` log dir.
        trace_log_dir: Directory where inference trace logs (from
            ``TaskLogger``) are stored.  Used to resolve ``source_trace``
            paths in eval result files.  When ``None``, ``source_trace``
            is omitted.
    """

    def __init__(
        self,
        *,
        orchestrator: TaskOrchestrator | None = None,
        verifier: BaseVerifier | None = None,
        max_concurrent: int = 16,
        eval_dir: str | Path | None = None,
        trace_log_dir: str | Path | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.verifier = verifier
        self.max_concurrent = max_concurrent
        self._eval_dir = Path(eval_dir) if eval_dir is not None else None
        self._trace_log_dir = Path(trace_log_dir) if trace_log_dir is not None else None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Per-task result logging
    # ------------------------------------------------------------------

    def log_result(
        self,
        task_id: str,
        *,
        eval_name: str,
        ground_truth: Any = None,
        prediction: Any = None,
        score: float | None = None,
        error: str | None = None,
        source_trace: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        """Write a single evaluation result file.

        Output path: ``<eval_dir>/<eval_name>/<task_id>.json``.

        If *source_trace* is not explicitly given but *trace_log_dir* was
        provided at init time, the method attempts to locate the latest
        trace file matching ``<trace_log_dir>/<task_id>*.json``.

        Args:
            task_id: Identifier of the evaluated task (used as filename).
            eval_name: Evaluation metric / aspect name (used as subdirectory).
            ground_truth: Expected answer.
            prediction: Model's predicted answer.
            score: Evaluation score.
            error: Error description if the task failed before scoring.
            source_trace: Explicit path to the inference trace log file.
                When ``None``, auto-resolved from *trace_log_dir*.
            metadata: Arbitrary extra fields merged into the record.

        Returns:
            The path to the written file, or ``None`` if *eval_dir* is not set.
        """
        if self._eval_dir is None:
            return None

        # Auto-resolve source trace when not explicitly given
        if source_trace is None and self._trace_log_dir is not None:
            source_trace = self._resolve_trace_path(task_id)

        record: dict[str, Any] = {
            "task_id": task_id,
            "eval_name": eval_name,
            "timestamp": datetime.now(tz=_UTC8).isoformat(),
        }
        if source_trace is not None:
            record["source_trace"] = source_trace
        if ground_truth is not None:
            record["ground_truth"] = ground_truth
        if prediction is not None:
            record["prediction"] = prediction
        if score is not None:
            record["score"] = score
        if error is not None:
            record["error"] = error
        if metadata:
            record.update(metadata)

        out_dir = self._eval_dir / eval_name
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{task_id}.json"

        content = json.dumps(record, indent=2, ensure_ascii=False)
        with self._lock:
            output_path.write_text(content, encoding="utf-8")
        logger.info("eval %s task=%s score=%s -> %s", eval_name, task_id, score, output_path)
        return output_path

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    async def evaluate(self, dataset: BaseDataset, *, eval_name: str = "default") -> EvaluationResult:
        """Run inference + scoring over *dataset* and optionally log results.

        Args:
            dataset: Dataset to evaluate.
            eval_name: Name used for eval result subdirectory.
        """
        assert self.orchestrator is not None, "orchestrator is required for evaluate()"
        assert self.verifier is not None, "verifier is required for evaluate()"

        result = EvaluationResult()
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _run_one(idx: int) -> dict[str, Any]:
            item = dataset[idx]
            task_id = f"eval_{idx}"
            async with semaphore:
                orch_result = await self.orchestrator.run(item.problem, task_id=task_id)  # type: ignore[union-attr]
            output = str(orch_result.output)
            score = self.verifier.verify(item.label, output)  # type: ignore[union-attr]
            detail = {
                "idx": idx,
                "task_id": task_id,
                "score": score,
                "output": output,
                "label": item.label,
                "problem": item.problem[:100],
            }
            self.log_result(
                task_id,
                eval_name=eval_name,
                ground_truth=item.label,
                prediction=output,
                score=score,
                metadata={"num_turns": getattr(orch_result, "num_turns", None)},
            )
            return detail

        tasks = [_run_one(i) for i in range(len(dataset))]
        details = await asyncio.gather(*tasks)

        for detail in details:
            result.total += 1
            result.scores.append(detail["score"])
            if detail["score"] > 0.5:
                result.correct += 1
            result.details.append(detail)

        logger.info("Evaluation complete: %d/%d correct (%.2f%% accuracy).", result.correct, result.total, result.accuracy * 100)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_trace_path(self, task_id: str) -> str | None:
        """Find the latest trace file for *task_id* under *trace_log_dir*."""
        if self._trace_log_dir is None or not self._trace_log_dir.exists():
            return None
        candidates = sorted(self._trace_log_dir.glob(f"{task_id}*.json"))
        return str(candidates[-1]) if candidates else None
