# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Run exact-match and LLM-as-judge scoring over existing benchmark results.

This post-processor does not rerun inference.  It reads each run directory's
``benchmark_results.jsonl`` or partial per-task outputs, compares the saved
prediction against the dataset ground truth with exact match and
:class:`LLMVerifier`, and writes:

* ``llm_judge_results.jsonl``: one row per task
* ``llm_judge_summary.json``: aggregate accuracy and configuration
* ``llm_judge_accuracy.txt``: human-readable accuracy
* ``em_results.jsonl`` / ``em_summary.json`` / ``em_accuracy.txt``: exact-match sidecar outputs

When ``eval_results.json`` already exists, the script also annotates it with
``llm_judge_*`` fields so timing dashboards can display both EM and judge scores.

Pass ``--judge-tag <TAG>`` to re-judge finished runs with a different judge
model additively: outputs are written as ``llm_judge_*.<TAG>.{jsonl,json,txt}``
and ``llm_judge_<TAG>/`` while the run's existing EM artifacts,
canonical ``llm_judge_*`` files, ``eval_results.json`` and dashboard artifacts
are left untouched (EM is reused read-only).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import json
import logging
import os
import re
import shlex
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

from agentic.model_clients.request_logger import ModelRequestLogger
from recipe.common.boxed_verifier import normalize_boxed_answer
from recipe.common.log_processing import write_dashboard_artifacts
from recipe.web_search.eval.benchmark_dataset import BenchmarkDataset
from recipe.web_search.eval.llm_judge import create_web_search_llm_verifier, resolve_judge_max_tokens_for_benchmark
from recipe.web_search.eval.llm_verifier import LLMJudgeError, LLMVerifier

logger = logging.getLogger(__name__)

_ORI_TASK_RE = re.compile(r"^task_(?P<base>[^_]+)_attempt-(?P<attempt>\d+)(?:_format-retry-(?P<retry>\d+))?_(?P<ts>[\d\-]+)\.json$")


class SourceRows(NamedTuple):
    rows: list[dict[str, Any]]
    source_result: str
    source_kind: str
    source_complete: bool


def _write_json_with_timing(path: Path, payload: dict[str, Any], *, timing_key: str | None = None) -> float:
    if timing_key is not None:
        payload[timing_key] = None
    start = time.perf_counter()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - start
    if timing_key is not None:
        payload[timing_key] = elapsed
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return elapsed


def _write_jsonl_with_timing(path: Path, rows: list[dict[str, Any]]) -> float:
    start = time.perf_counter()
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    return time.perf_counter() - start


def _write_text_with_timing(path: Path, content: str) -> float:
    start = time.perf_counter()
    path.write_text(content, encoding="utf-8")
    return time.perf_counter() - start


def _load_env_file(path: Path | None) -> None:
    """Load simple KEY=VALUE lines into the process env without overriding env."""
    if path is None or not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        try:
            parsed = shlex.split(value, comments=False, posix=True)
            os.environ[key] = parsed[0] if parsed else ""
        except ValueError:
            os.environ[key] = value.strip().strip('"').strip("'")


def _dataset_by_task_id(data_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    dataset = BenchmarkDataset(data_path=data_path)
    dataset.load()
    for idx, item in enumerate(dataset.items):
        source = json.loads(item.source) if isinstance(item.source, str) else item.source
        task_id = str(source.get("task_id", idx) if isinstance(source, dict) else idx)
        out[task_id] = {
            "question": item.problem,
            "ground_truth": item.label,
            "metadata": item.metadata,
        }
    logger.info("Loaded %d benchmark items from %s", len(out), data_path)
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_existing_judge_sidecars(run_dir: Path, judge_eval_dir: Path | None = None) -> list[dict[str, Any]]:
    judge_eval_dir = judge_eval_dir if judge_eval_dir is not None else run_dir / "llm_judge"
    if not judge_eval_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(judge_eval_dir.glob("*.json"), key=lambda p: (len(p.stem), p.stem)):
        row = _read_json_file(path)
        if row:
            rows.append(row)
    return rows


def _has_numeric_llm_judge_score(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    try:
        float(row["llm_judge_score"])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _is_stable_file(path: Path, stable_seconds: float) -> bool:
    try:
        age_s = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return age_s >= stable_seconds


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Skipping unreadable/partial JSON file %s: %s", path, exc)
        return None


def _read_agentic_eval_folder(run_dir: Path, *, stable_seconds: float) -> list[dict[str, Any]]:
    """Read partial agentic rows from ``exact_match/*.json``."""
    eval_dir = run_dir / "exact_match"
    if not eval_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(eval_dir.glob("*.json"), key=lambda p: (len(p.stem), p.stem)):
        if not _is_stable_file(path, stable_seconds):
            continue
        row = _read_json_file(path)
        if not row:
            continue
        rows.append(
            {
                "task_id": str(row.get("task_id", path.stem)),
                "task_question": row.get("task_question") or row.get("question") or "",
                "ground_truth": row.get("ground_truth") or "",
                "output": row.get("prediction") or "",
                "score": row.get("score"),
                "reason": row.get("reason"),
                "num_turns": row.get("num_turns"),
                "source_eval_file": str(path),
            }
        )
    return rows


def _read_ori_trace_folder(run_dir: Path, *, stable_seconds: float) -> list[dict[str, Any]]:
    """Read partial upstream rows from completed ``task_*.json`` traces."""
    grouped: dict[str, list[tuple[tuple[int, int, str], Path]]] = {}
    for path in run_dir.glob("task_*.json"):
        match = _ORI_TASK_RE.match(path.name)
        if not match:
            continue
        base = match.group("base")
        attempt = int(match.group("attempt") or 1)
        retry = int(match.group("retry") or 0)
        ts = match.group("ts") or ""
        grouped.setdefault(base, []).append(((attempt, retry, ts), path))

    rows: list[dict[str, Any]] = []
    for base, entries in sorted(grouped.items(), key=lambda item: (len(item[0]), item[0])):
        entries.sort()
        _, path = entries[-1]
        if not _is_stable_file(path, stable_seconds):
            logger.debug("Skipping task %s: latest trace is still being written (%s)", base, path)
            continue
        trace = _read_json_file(path)
        if not trace:
            continue
        if trace.get("end_time") is None or trace.get("status") not in {None, "success", "failed"}:
            continue
        prediction = trace.get("final_boxed_answer") or ""
        rows.append(
            {
                "task_id": base,
                "task_question": (trace.get("input") or {}).get("task_description", ""),
                "ground_truth": trace.get("ground_truth") or "",
                "model_boxed_answer": prediction,
                "status": trace.get("status"),
                "log_file_path": str(path),
                "attempts": [
                    {
                        "attempt_number": 1,
                        "model_boxed_answer": prediction,
                        "status": trace.get("status"),
                        "log_file_path": str(path),
                    }
                ],
            }
        )
    return rows


def _load_source_rows(run_dir: Path, *, allow_partial: bool, stable_seconds: float, predictions_jsonl: Path | None = None) -> SourceRows:
    if predictions_jsonl is not None:
        if not predictions_jsonl.exists():
            raise FileNotFoundError(f"--predictions-jsonl not found: {predictions_jsonl}")
        return SourceRows(
            rows=_read_jsonl(predictions_jsonl),
            source_result=str(predictions_jsonl),
            source_kind="benchmark_results",
            source_complete=True,
        )
    results_path = run_dir / "benchmark_results.jsonl"
    if results_path.exists():
        return SourceRows(
            rows=_read_jsonl(results_path),
            source_result=str(results_path),
            source_kind="benchmark_results",
            source_complete=True,
        )
    if not allow_partial:
        raise FileNotFoundError(f"Missing benchmark_results.jsonl under {run_dir}; pass --allow-partial to judge in-progress runs")

    ori_rows = _read_ori_trace_folder(run_dir, stable_seconds=stable_seconds)
    if ori_rows:
        return SourceRows(
            rows=ori_rows,
            source_result=str(run_dir / "task_*.json"),
            source_kind="ori_task_traces",
            source_complete=False,
        )

    agentic_rows = _read_agentic_eval_folder(run_dir, stable_seconds=stable_seconds)
    if agentic_rows:
        return SourceRows(
            rows=agentic_rows,
            source_result=str(run_dir / "exact_match"),
            source_kind="agentic_em_sidecars",
            source_complete=False,
        )

    raise FileNotFoundError(f"No complete result rows found under {run_dir}")


def _source_matches_current(existing: dict[str, Any] | None, source_id: str) -> bool:
    """Return whether a cached sidecar still matches the current task source.

    Partial original runs can write newer ``format-retry`` files for the same
    task after an earlier judge pass.  In that case task_id alone is not enough
    to decide reuse.  Final ``benchmark_results.jsonl`` rows do not currently
    carry a per-task source id, so empty source ids keep the old reuse behavior.
    """
    if existing is None or not source_id:
        return existing is not None
    existing_source = str(existing.get("source_id") or existing.get("source_eval_file") or existing.get("log_file_path") or "")
    return existing_source == source_id


def _task_attempts(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return saved predictions to judge for one task row.

    Most rows have a single ``output``. Rows may alternatively include an
    ``attempts`` list; judging every attempt preserves pass@k semantics.
    """
    attempts = row.get("attempts")
    if isinstance(attempts, list) and attempts:
        out = []
        for idx, attempt in enumerate(attempts, start=1):
            if not isinstance(attempt, dict):
                continue
            prediction = attempt.get("model_boxed_answer") or attempt.get("output") or ""
            out.append(
                {
                    "attempt": attempt.get("attempt", idx),
                    "prediction": prediction,
                    "original_judge_result": attempt.get("final_judge_result"),
                    "original_is_correct": attempt.get("is_correct"),
                }
            )
        return out

    prediction = row.get("output")
    if prediction is None:
        prediction = row.get("model_boxed_answer")
    if prediction is None:
        prediction = row.get("prediction")
    return [{"attempt": 1, "prediction": prediction or ""}]


def _prediction_from_row(row: dict[str, Any]) -> str:
    attempts = _task_attempts(row)
    if attempts:
        return str(attempts[0].get("prediction") or "")
    return ""


def _row_source_id(row: dict[str, Any]) -> str:
    source = row.get("source_eval_file") or row.get("log_file_path")
    if source:
        return str(source)
    attempts = row.get("attempts")
    if isinstance(attempts, list):
        paths = [str(a.get("log_file_path")) for a in attempts if isinstance(a, dict) and a.get("log_file_path")]
        if paths:
            return "|".join(paths)
    return ""


def _original_score(row: dict[str, Any]) -> float | None:
    score = row.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return float(score)
    pass_at_k = row.get("pass_at_k_success")
    if isinstance(pass_at_k, bool):
        return 1.0 if pass_at_k else 0.0
    return None


def _exact_match_score(ground_truth: str, prediction: str | None) -> float:
    if not prediction:
        return 0.0
    return 1.0 if normalize_boxed_answer(prediction) == normalize_boxed_answer(ground_truth) else 0.0


def _read_existing_em_sidecars(run_dir: Path) -> list[dict[str, Any]]:
    em_dir = run_dir / "exact_match"
    if not em_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(em_dir.glob("*.json"), key=lambda p: (len(p.stem), p.stem)):
        row = _read_json_file(path)
        if row:
            rows.append(row)
    return rows


def _normalize_em_record(
    *,
    task_id: str,
    row: dict[str, Any],
    dataset: dict[str, dict[str, Any]],
    existing: dict[str, Any] | None,
    overwrite: bool,
) -> dict[str, Any]:
    data = dataset.get(task_id, {})
    source_id = _row_source_id(row)
    if existing is not None and not overwrite and existing.get("score") is not None and _source_matches_current(existing, source_id):
        reused = dict(existing)
        reused.setdefault("task_id", task_id)
        reused.setdefault("eval_name", "exact_match")
        reused.setdefault("source_id", source_id)
        reused.setdefault("question", row.get("task_question") or data.get("question", ""))
        reused.setdefault("ground_truth", row.get("ground_truth") or data.get("ground_truth", ""))
        reused.setdefault("prediction", _prediction_from_row(row))
        reused.setdefault("em_result", "CORRECT" if float(reused.get("score") or 0.0) > 0.5 else "INCORRECT")
        return reused

    question = row.get("task_question") or (existing.get("question") if existing else None) or data.get("question", "")
    ground_truth = row.get("ground_truth") or (existing.get("ground_truth") if existing else None)
    if not ground_truth:
        ground_truth = data.get("ground_truth", "")
    prediction = row.get("prediction")
    if prediction is None:
        prediction = row.get("output")
    if prediction is None:
        prediction = row.get("model_boxed_answer")
    if prediction is None:
        prediction = existing.get("prediction") if existing else None
    if prediction is None:
        prediction = _prediction_from_row(row)

    score = _exact_match_score(str(ground_truth), str(prediction or ""))
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    return {
        "task_id": task_id,
        "eval_name": "exact_match",
        "source_id": source_id or (existing.get("source_id") if existing else ""),
        "timestamp": timestamp,
        "question": question or "",
        "ground_truth": ground_truth or "",
        "prediction": prediction or "",
        "score": score,
        "em_result": "CORRECT" if score > 0.5 else "INCORRECT",
        "reason": row.get("reason") or (existing.get("reason") if existing else None),
        "num_turns": row.get("num_turns") or (existing.get("num_turns") if existing else None),
    }


def _write_em_outputs(
    *,
    run_dir: Path,
    rows: list[dict[str, Any]],
    dataset: dict[str, dict[str, str]],
    overwrite: bool,
    allow_partial: bool,
    stable_seconds: float,
    source_result: str,
    source_kind: str,
    source_complete: bool,
    source_num_items: int,
    dataset_num_items: int,
    persist: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    em_dir = run_dir / "exact_match"
    if persist:
        em_dir.mkdir(parents=True, exist_ok=True)
    # When not persisting (additive tag mode) always reuse existing EM sidecars
    # read-only; we never rewrite the run's canonical EM artifacts.
    existing_rows = [] if (overwrite and persist) else _read_existing_em_sidecars(run_dir)
    existing_by_task = {str(row.get("task_id")): row for row in existing_rows}

    final_rows: list[dict[str, Any]] = []
    current_ids: set[str] = set()
    for index, row in enumerate(rows):
        task_id = str(row.get("task_id", index))
        current_ids.add(task_id)
        record = _normalize_em_record(
            task_id=task_id,
            row=row,
            dataset=dataset,
            existing=existing_by_task.get(task_id),
            overwrite=overwrite and persist,
        )
        final_rows.append(record)
        if persist:
            _write_json_with_timing(em_dir / f"{task_id}.json", record)

    if source_complete:
        for task_id, existing in existing_by_task.items():
            if task_id not in current_ids:
                final_rows.append(existing)

    final_rows.sort(key=lambda row: (len(str(row.get("task_id", ""))), str(row.get("task_id", ""))))
    total = len(final_rows)
    correct = sum(1 for row in final_rows if float(row.get("score") or 0.0) > 0.5)
    accuracy = correct / total if total else 0.0

    results_write_s = _write_jsonl_with_timing(run_dir / "em_results.jsonl", final_rows) if persist else None
    accuracy_write_s = _write_text_with_timing(run_dir / "em_accuracy.txt", f"{accuracy * 100:.2f}%\n") if persist else None
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "source_result_file": source_result,
        "source_kind": source_kind,
        "source_complete": source_complete,
        "source_num_items": source_num_items,
        "dataset_num_items": dataset_num_items,
        "num_items": total,
        "eval_skipped": max(0, source_num_items - total),
        "em_correct": correct,
        "em_accuracy": accuracy,
        "allow_partial": allow_partial,
        "stable_seconds": stable_seconds,
        "sidecar_eval_dir": str(em_dir),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "em_results_write_elapsed_s": results_write_s,
        "em_accuracy_write_elapsed_s": accuracy_write_s,
    }
    if persist:
        summary_write_s = _write_json_with_timing(run_dir / "em_summary.json", summary, timing_key="em_summary_write_elapsed_s")
        summary["em_summary_write_elapsed_s"] = summary_write_s
    else:
        summary["em_summary_write_elapsed_s"] = None
    logger.info(
        "EM %s for %s: %.2f%% (%d/%d)",
        "computed (read-only)" if not persist else "complete",
        run_dir,
        accuracy * 100,
        correct,
        total,
    )
    return final_rows, summary


def _judge_metadata(verifier: LLMVerifier) -> dict[str, str | int | None]:
    return {
        "judge_model": verifier.judge_model,
        "judge_base_url": verifier.judge_base_url or os.environ.get("OPENAI_BASE_URL"),
        "judge_api_key_env": verifier.judge_api_key_env,
        "judge_times": verifier.judge_times,
        "judge_max_tokens": verifier.judge_max_tokens,
        "judge_empty_length_retry_max_tokens": verifier.judge_empty_length_retry_max_tokens,
        "judge_prompt_profile": getattr(verifier, "judge_prompt_profile", "browsecomp"),
    }


def _stamp_judge_metadata(row: dict[str, Any], verifier: LLMVerifier) -> dict[str, Any]:
    row.update(_judge_metadata(verifier))
    return row


def _stamp_em_fields(row: dict[str, Any], em_by_task: dict[str, dict[str, Any]]) -> dict[str, Any]:
    em = em_by_task.get(str(row.get("task_id", "")))
    if em:
        row["em_score"] = em.get("score")
        row["em_result"] = em.get("em_result")
    return row


def _existing_judge_times(row: dict[str, Any]) -> int | None:
    value = row.get("judge_times") or row.get("llm_judge_times")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _existing_judge_max_tokens(row: dict[str, Any]) -> int:
    value = row.get("judge_max_tokens")
    if value is None:
        return 2
    try:
        return int(value)
    except (TypeError, ValueError):
        return 2


def _existing_judge_empty_length_retry_max_tokens(row: dict[str, Any]) -> int:
    value = row.get("judge_empty_length_retry_max_tokens")
    if value is None:
        value = row.get("llm_judge_empty_length_retry_max_tokens")
    if value is None:
        return 1024
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1024


def _normalize_base_url(value: Any) -> str:
    if value is None:
        return ""
    return str(value).rstrip("/")


def _judge_metadata_matches_current(row: dict[str, Any], verifier: LLMVerifier) -> bool:
    expected_base_url = verifier.judge_base_url or os.environ.get("OPENAI_BASE_URL")
    expected_prompt_profile = getattr(verifier, "judge_prompt_profile", "browsecomp")
    return (
        row.get("judge_model") == verifier.judge_model
        and _normalize_base_url(row.get("judge_base_url")) == _normalize_base_url(expected_base_url)
        and _existing_judge_max_tokens(row) == verifier.judge_max_tokens
        and _existing_judge_empty_length_retry_max_tokens(row) == verifier.judge_empty_length_retry_max_tokens
        and (row.get("judge_prompt_profile") or "browsecomp") == expected_prompt_profile
    )


def _read_reusable_judge_rows(run_dir: Path, output_jsonl: Path, *, overwrite: bool, judge_eval_dir: Path | None = None) -> list[dict[str, Any]]:
    if overwrite:
        return []
    existing_rows = [row for row in _read_existing_judge_sidecars(run_dir, judge_eval_dir) if _has_numeric_llm_judge_score(row)]
    if not output_jsonl.exists():
        return existing_rows

    # Aggregate rows win over sidecars when both exist, but either source is
    # enough to skip a repeated judge call for the same task.
    by_task = {str(row.get("task_id")): row for row in existing_rows}
    by_task.update({str(row.get("task_id")): row for row in _read_jsonl(output_jsonl) if _has_numeric_llm_judge_score(row)})
    return list(by_task.values())


def _initialize_judged_rows(
    rows: list[dict[str, Any]],
    existing_by_task: dict[str, dict[str, Any]],
    em_by_task: dict[str, dict[str, Any]],
    verifier: LLMVerifier,
) -> list[dict[str, Any] | None]:
    judged_rows: list[dict[str, Any] | None] = []
    for index, row in enumerate(rows):
        existing = existing_by_task.get(str(row.get("task_id", index)))
        source_id = _row_source_id(row)
        if (
            existing
            and _has_numeric_llm_judge_score(existing)
            and _existing_judge_times(existing) == verifier.judge_times
            and _source_matches_current(existing, source_id)
            and _judge_metadata_matches_current(existing, verifier)
        ):
            existing = _stamp_judge_metadata(existing, verifier)
            existing = _stamp_em_fields(existing, em_by_task)
        else:
            existing = None
        judged_rows.append(existing)
    return judged_rows


_ATTEMPT_STRUCTURAL_KEYS = frozenset({"attempt", "attempt_id", "prediction", "original_judge_result", "original_is_correct"})


def _attempt_judge_payload(attempt: dict[str, Any]) -> dict[str, Any]:
    """Judge-only fields from a judged attempt, suitable for cache reuse/merge."""
    return {key: copy.deepcopy(value) for key, value in attempt.items() if key not in _ATTEMPT_STRUCTURAL_KEYS}


def _seed_judge_cache(judge_cache: dict[tuple[str, str], dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    """Populate the cross-dir judge cache from already-judged rows.

    Attempt-budget sweeps reuse the same rollout output across multiple budgets
    (and the max budget mirrors the parent run). Seeding lets later dirs reuse a
    verdict for an identical ``(task_id, prediction)`` instead of re-querying the
    judge model.
    """
    for row in rows:
        task_id = str(row.get("task_id", ""))
        for attempt in row.get("attempts") or []:
            if not isinstance(attempt, dict) or not _has_numeric_llm_judge_score(attempt):
                continue
            key = (task_id, str(attempt.get("prediction") or ""))
            judge_cache.setdefault(key, _attempt_judge_payload(attempt))


async def _judge_one_row(
    *,
    index: int,
    row: dict[str, Any],
    judged_rows: list[dict[str, Any] | None],
    dataset: dict[str, dict[str, str]],
    verifier: LLMVerifier,
    semaphore: asyncio.Semaphore,
    em_by_task: dict[str, dict[str, Any]],
    judge_eval_dir: Path,
    overwrite: bool,
    judge_cache: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> None:
    task_id = str(row.get("task_id", index))
    if judged_rows[index] is not None and not overwrite:
        return

    data = dataset.get(task_id, {})
    question = row.get("task_question") or data.get("question", "")
    ground_truth = row.get("ground_truth") or data.get("ground_truth", "")
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else None
    judged_attempts: list[dict[str, Any]] = []
    async with semaphore:
        for attempt in _task_attempts(row):
            prediction = str(attempt.get("prediction") or "")
            attempt_id = attempt.get("attempt_id") or len(judged_attempts)
            cache_key = (task_id, prediction)
            cached_payload = judge_cache.get(cache_key) if judge_cache is not None else None
            if cached_payload is not None:
                logger.debug("Reusing cached judge verdict for task %s attempt %s", task_id, attempt_id)
                judged_attempts.append({**attempt, **copy.deepcopy(cached_payload)})
                continue
            logger.debug("Judging task %s attempt %s", task_id, attempt_id)
            try:
                if ground_truth:
                    judge_details = await verifier.ajudge(str(ground_truth), prediction, question=str(question), metadata=metadata)
                    score = float(judge_details["llm_judge_score"])
                else:
                    judge_details = {
                        "llm_judge_score": 0.0,
                        "llm_judge_result": "INCORRECT",
                        "llm_judge_skipped_reason": "missing_ground_truth",
                        "llm_judge_times": verifier.judge_times,
                        "llm_judge_scores": [0.0] * verifier.judge_times,
                    }
                    score = 0.0
            except LLMJudgeError:
                logger.exception("Task %s attempt %s: LLM judge failed; skipping task sidecar", task_id, attempt_id)
                judged_rows[index] = None
                return
            judge_payload = {**judge_details, "llm_judge_score": score}
            judged_attempts.append({**attempt, **judge_payload})
            if judge_cache is not None:
                judge_cache[cache_key] = copy.deepcopy(judge_payload)

    best_score = max((float(a["llm_judge_score"]) for a in judged_attempts), default=0.0)
    best_attempt = next((a for a in judged_attempts if float(a["llm_judge_score"]) == best_score), None)
    judged_row = {
        "task_id": task_id,
        "source_id": _row_source_id(row),
        "question": question,
        "ground_truth": ground_truth,
        "prediction": best_attempt.get("prediction") if best_attempt else "",
        "llm_judge_score": best_score,
        "llm_judge_result": "CORRECT" if best_score > 0.5 else "INCORRECT",
        "llm_judge_request": best_attempt.get("llm_judge_request") if best_attempt else None,
        "llm_judge_response": best_attempt.get("llm_judge_response") if best_attempt else None,
        "llm_judge_skipped_reason": best_attempt.get("llm_judge_skipped_reason") if best_attempt else None,
        "em_score": (em_by_task.get(task_id) or {}).get("score"),
        "em_result": (em_by_task.get(task_id) or {}).get("em_result"),
        "original_score": _original_score(row),
        "attempts": judged_attempts,
    }
    _stamp_judge_metadata(judged_row, verifier)
    judged_rows[index] = judged_row
    _write_json_with_timing(judge_eval_dir / f"{task_id}.json", judged_row)
    logger.info("Judged task %s: %s (attempts=%d)", task_id, judged_row["llm_judge_result"], len(judged_attempts))


def _finalize_judge_rows(
    *,
    rows: list[dict[str, Any]],
    judged_rows: list[dict[str, Any] | None],
    existing_by_task: dict[str, dict[str, Any]],
    em_by_task: dict[str, dict[str, Any]],
    verifier: LLMVerifier,
    judge_eval_dir: Path,
    source_complete: bool,
) -> list[dict[str, Any]]:
    for row in judged_rows:
        if row is not None:
            _stamp_judge_metadata(row, verifier)
            _stamp_em_fields(row, em_by_task)
            _write_json_with_timing(judge_eval_dir / f"{row['task_id']}.json", row)

    current_ids = {str(row.get("task_id", index)) for index, row in enumerate(rows)}
    final_rows = [row for row in judged_rows if row is not None]
    if source_complete:
        for task_id, row in existing_by_task.items():
            if task_id not in current_ids:
                _stamp_judge_metadata(row, verifier)
                _stamp_em_fields(row, em_by_task)
                _write_json_with_timing(judge_eval_dir / f"{task_id}.json", row)
                final_rows.append(row)
    final_rows.sort(key=lambda row: (len(str(row.get("task_id", ""))), str(row.get("task_id", ""))))
    return final_rows


def _write_judge_outputs(
    *,
    run_dir: Path,
    output_jsonl: Path,
    summary_path: Path,
    accuracy_path: Path,
    final_rows: list[dict[str, Any]],
    source_item_count: int,
    source_result: str,
    source_kind: str,
    source_complete: bool,
    dataset_num_items: int,
    verifier: LLMVerifier,
    em_summary: dict[str, Any],
    allow_partial: bool,
    stable_seconds: float,
    judge_eval_dir: Path,
) -> dict[str, Any]:
    total = len(final_rows)
    skipped = max(0, source_item_count - total)
    correct = sum(1 for row in final_rows if float(row["llm_judge_score"]) > 0.5)
    accuracy = correct / total if total else 0.0
    now = datetime.now().astimezone().isoformat(timespec="seconds")

    jsonl_write_s = _write_jsonl_with_timing(output_jsonl, final_rows)
    accuracy_write_s = _write_text_with_timing(accuracy_path, f"{accuracy * 100:.2f}%\n")
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "source_result_file": source_result,
        "source_kind": source_kind,
        "source_complete": source_complete,
        **_judge_metadata(verifier),
        "em": em_summary,
        "source_num_items": source_item_count,
        "dataset_num_items": dataset_num_items,
        "num_items": total,
        "llm_judge_skipped": skipped,
        "llm_judge_correct": correct,
        "llm_judge_accuracy": accuracy,
        "allow_partial": allow_partial,
        "stable_seconds": stable_seconds,
        "sidecar_eval_dir": str(judge_eval_dir),
        "created_at": now,
        "llm_judge_results_write_elapsed_s": jsonl_write_s,
        "llm_judge_accuracy_write_elapsed_s": accuracy_write_s,
    }
    summary_write_s = _write_json_with_timing(summary_path, summary, timing_key="llm_judge_summary_write_elapsed_s")
    summary["llm_judge_summary_write_elapsed_s"] = summary_write_s
    logger.info(
        "Wrote judge outputs: results=%s summary=%s accuracy=%s evaluated=%d skipped=%d",
        output_jsonl,
        summary_path,
        accuracy_path,
        total,
        skipped,
    )
    return summary


async def _judge_run_dir(
    *,
    run_dir: Path,
    dataset: dict[str, dict[str, str]],
    verifier: LLMVerifier,
    request_logger: ModelRequestLogger | None,
    max_concurrent: int,
    overwrite: bool,
    update_eval_results: bool,
    allow_partial: bool,
    stable_seconds: float,
    judge_cache: dict[tuple[str, str], dict[str, Any]] | None = None,
    judge_tag: str | None = None,
    predictions_jsonl: Path | None = None,
) -> dict[str, Any]:
    additive = bool(judge_tag)
    tag_suffix = f".{judge_tag}" if additive else ""
    output_jsonl = run_dir / f"llm_judge_results{tag_suffix}.jsonl"
    summary_path = run_dir / f"llm_judge_summary{tag_suffix}.json"
    accuracy_path = run_dir / f"llm_judge_accuracy{tag_suffix}.txt"
    judge_eval_dir = run_dir / (f"llm_judge_{judge_tag}" if additive else "llm_judge")
    judge_eval_dir.mkdir(parents=True, exist_ok=True)
    verifier.set_request_logger(request_logger)

    source = _load_source_rows(run_dir, allow_partial=allow_partial, stable_seconds=stable_seconds, predictions_jsonl=predictions_jsonl)
    rows, source_result = source.rows, source.source_result
    logger.info(
        "Starting LLM judge for %s: source=%s rows=%d max_concurrent=%d overwrite=%s tag=%s",
        run_dir,
        source_result,
        len(rows),
        max_concurrent,
        overwrite,
        judge_tag or "-",
    )
    # In additive (tagged) mode the run's canonical EM artifacts are reused
    # read-only so no existing file is mutated.
    em_rows, em_summary = _write_em_outputs(
        run_dir=run_dir,
        rows=rows,
        dataset=dataset,
        overwrite=overwrite,
        allow_partial=allow_partial,
        stable_seconds=stable_seconds,
        source_result=source_result,
        source_kind=source.source_kind,
        source_complete=source.source_complete,
        source_num_items=len(rows),
        dataset_num_items=len(dataset),
        persist=not additive,
    )
    em_by_task = {str(row.get("task_id")): row for row in em_rows}
    existing_rows = _read_reusable_judge_rows(run_dir, output_jsonl, overwrite=overwrite, judge_eval_dir=judge_eval_dir)
    existing_by_task = {str(row.get("task_id")): row for row in existing_rows}
    semaphore = asyncio.Semaphore(max(1, max_concurrent))
    judged_rows = _initialize_judged_rows(rows, existing_by_task, em_by_task, verifier)
    await asyncio.gather(
        *(
            _judge_one_row(
                index=index,
                row=row,
                judged_rows=judged_rows,
                dataset=dataset,
                verifier=verifier,
                semaphore=semaphore,
                em_by_task=em_by_task,
                judge_eval_dir=judge_eval_dir,
                overwrite=overwrite,
                judge_cache=judge_cache,
            )
            for index, row in enumerate(rows)
        )
    )
    final_rows = _finalize_judge_rows(
        rows=rows,
        judged_rows=judged_rows,
        existing_by_task=existing_by_task,
        em_by_task=em_by_task,
        verifier=verifier,
        judge_eval_dir=judge_eval_dir,
        source_complete=source.source_complete,
    )
    if judge_cache is not None:
        _seed_judge_cache(judge_cache, final_rows)
    summary = _write_judge_outputs(
        run_dir=run_dir,
        output_jsonl=output_jsonl,
        summary_path=summary_path,
        accuracy_path=accuracy_path,
        final_rows=final_rows,
        source_item_count=len(rows),
        source_result=source_result,
        source_kind=source.source_kind,
        source_complete=source.source_complete,
        dataset_num_items=len(dataset),
        verifier=verifier,
        em_summary=em_summary,
        allow_partial=allow_partial,
        stable_seconds=stable_seconds,
        judge_eval_dir=judge_eval_dir,
    )

    # Additive tag mode must not touch the run's canonical eval_results.json or
    # dashboard artifacts; only the tagged judge files above are written.
    if update_eval_results and not additive:
        _annotate_eval_results(run_dir / "eval_results.json", summary, final_rows)

    if not additive:
        write_dashboard_artifacts(run_dir, run_type=_infer_dashboard_run_type(run_dir))

    logger.info(
        "LLM judge complete for %s: %.2f%% (%d/%d, skipped=%d)",
        run_dir,
        summary["llm_judge_accuracy"] * 100,
        summary["llm_judge_correct"],
        summary["num_items"],
        summary["llm_judge_skipped"],
    )
    return summary


def _annotate_eval_results(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_task_id = {str(row["task_id"]): row for row in rows}
    payload["llm_judge"] = summary
    payload["llm_judge_correct"] = summary["llm_judge_correct"]
    payload["llm_judge_accuracy"] = summary["llm_judge_accuracy"]
    for item in payload.get("items", []) or []:
        if not isinstance(item, dict):
            continue
        judged = by_task_id.get(str(item.get("task_id", "")))
        if judged:
            item["llm_judge_score"] = judged["llm_judge_score"]
            item["llm_judge_result"] = judged["llm_judge_result"]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Annotated %s with LLM judge fields", path)


def _infer_dashboard_run_type(run_dir: Path) -> str | None:
    eval_results = run_dir / "eval_results.json"
    if eval_results.exists():
        with contextlib.suppress(OSError, UnicodeDecodeError, json.JSONDecodeError):
            payload = json.loads(eval_results.read_text(encoding="utf-8"))
            source = payload.get("source") if isinstance(payload, dict) else None
            if source == "web_search":
                return "web_search"
            if source == "agentic":
                return "web_search"
    if (run_dir / "web-search-benchmark").exists():
        return "web_search"
    return None


def _has_judgeable_results(path: Path) -> bool:
    """Whether *path* carries result files a judge pass can read."""
    return (path / "benchmark_results.jsonl").exists() or (path / "exact_match").exists() or bool(list(path.glob("task_*.json")))


def _attempt_budget_sort_key(name: str) -> tuple[int, str]:
    suffix = name[len("attempt_budget_") :]
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _attempt_budget_subdirs(run_dir: Path) -> list[Path]:
    """Per-attempt-budget sweep subdirs (``attempt_budget_<n>``) with result files."""
    return sorted(
        (child for child in run_dir.glob("attempt_budget_*") if child.is_dir() and _has_judgeable_results(child)),
        key=lambda child: _attempt_budget_sort_key(child.name),
    )


def _discover_run_dirs(paths: list[Path], *, include_attempt_budget_subdirs: bool = True) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()

    def _add_run_dir(run_dir: Path) -> None:
        resolved = run_dir.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        out.append(run_dir)
        # Attempt-budget sweeps write a self-contained result dir per budget; judge
        # each one so every budget gets its own llm_judge_* artifacts.
        if not include_attempt_budget_subdirs:
            return
        for budget_dir in _attempt_budget_subdirs(run_dir):
            budget_resolved = budget_dir.resolve()
            if budget_resolved in seen:
                continue
            seen.add(budget_resolved)
            out.append(budget_dir)

    for path in paths:
        if _has_judgeable_results(path):
            _add_run_dir(path)
            continue
        for child in sorted(path.glob("run_*")):
            if _has_judgeable_results(child):
                _add_run_dir(child)
    return out


def _install_stop_event() -> asyncio.Event:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _stop() -> None:
        logger.info("Stop requested; exiting after current evaluator pass")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _stop)
    return stop_event


def _judge_summary_is_complete(summary: dict[str, Any]) -> bool:
    return bool(summary.get("source_complete")) and int(summary.get("llm_judge_skipped") or 0) == 0


def _request_logger_for_run(
    run_dir: Path, request_loggers: dict[tuple[Path, str | None], ModelRequestLogger], tag: str | None = None
) -> ModelRequestLogger:
    key = (run_dir.resolve(), tag)
    request_logger = request_loggers.get(key)
    if request_logger is None:
        request_logger = ModelRequestLogger(run_dir, name=f"judge_{tag}" if tag else "judge")
        request_loggers[key] = request_logger
    return request_logger


async def _run_once(
    args: argparse.Namespace,
    dataset: dict[str, dict[str, str]],
    verifier: LLMVerifier,
    request_loggers: dict[tuple[Path, str | None], ModelRequestLogger],
) -> tuple[int, bool]:
    run_dirs = _discover_run_dirs(args.run_dir, include_attempt_budget_subdirs=not args.no_attempt_budget_subdirs)
    if not run_dirs:
        if args.watch or args.allow_partial:
            logger.info("No run directories with result files found yet")
            return 0, False
        raise FileNotFoundError("No run directories with benchmark_results.jsonl were found")

    evaluated = 0
    complete = True
    # Shared across all dirs judged in this pass (parent run + its attempt-budget
    # sweep dirs, and across runs): an identical (task_id, prediction) is judged
    # once and reused, avoiding redundant judge-model calls.
    judge_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for run_dir in run_dirs:
        try:
            summary = await _judge_run_dir(
                run_dir=run_dir,
                dataset=dataset,
                verifier=verifier,
                request_logger=_request_logger_for_run(run_dir, request_loggers, tag=args.judge_tag) if args.judge_request_logging else None,
                max_concurrent=args.max_concurrent,
                overwrite=args.overwrite,
                update_eval_results=not args.no_update_eval_results,
                allow_partial=args.allow_partial,
                stable_seconds=args.stable_seconds,
                judge_cache=judge_cache,
                judge_tag=args.judge_tag,
                predictions_jsonl=args.predictions_jsonl,
            )
        except FileNotFoundError as exc:
            if not args.allow_partial:
                raise
            logger.warning("Skipping %s for now: %s", run_dir, exc)
            complete = False
            continue
        evaluated += 1
        complete = complete and _judge_summary_is_complete(summary)
    return evaluated, complete


async def _amain(args: argparse.Namespace) -> None:
    _load_env_file(args.env_file)
    data_path = args.data_path
    if not data_path:
        raise ValueError("--data_path is required")

    dataset = _dataset_by_task_id(Path(data_path))
    verifier = create_web_search_llm_verifier(
        benchmark_name=args.benchmark_name,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url or os.environ.get("OPENAI_BASE_URL"),
        judge_api_key_env=args.judge_api_key_env,
        max_retries=args.max_retries,
        judge_times=args.judge_times,
        judge_max_tokens=args.judge_max_tokens,
        judge_empty_length_retry_max_tokens=args.judge_empty_length_retry_max_tokens,
    )
    request_loggers: dict[tuple[Path, str | None], ModelRequestLogger] = {}
    try:
        if args.watch:
            stop_event = _install_stop_event()
            logger.info(
                "Watching %s for incremental EM/LLM judge evaluation every %.1fs",
                ", ".join(str(path) for path in args.run_dir),
                args.poll_seconds,
            )
            while not stop_event.is_set():
                evaluated, complete = await _run_once(args, dataset, verifier, request_loggers)
                if evaluated and complete:
                    logger.info("All watched run dirs are source-complete and fully judged; exiting watch mode")
                    break
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop_event.wait(), timeout=max(1.0, args.poll_seconds))
        else:
            await _run_once(args, dataset, verifier, request_loggers)
    finally:
        verifier.set_request_logger(None)
        await verifier.aclose()
        for request_logger in request_loggers.values():
            request_logger.close()


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Add exact-match and LLM-as-judge results to existing benchmark runs")
    parser.add_argument(
        "--run_dir",
        type=Path,
        action="append",
        required=True,
        help="Run dir containing benchmark_results.jsonl, or parent dir containing run_* subdirs. Pass multiple times.",
    )
    parser.add_argument("--data_path", type=Path, required=True, help="Benchmark data file or directory")
    parser.add_argument(
        "--benchmark_name",
        choices=["browsecomp", "browsecomp_zh", "gaia", "hle", "deepsearchqa", "livebrowsecomp"],
        default="browsecomp",
        help="Benchmark prompt profile for LLM judge",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".envs/.env"), help="Env file to load")
    parser.add_argument("--judge_model", default=None, help="Judge model name")
    parser.add_argument("--judge_base_url", default=None, help="OpenAI-compatible judge base URL (defaults to OPENAI_BASE_URL)")
    parser.add_argument("--judge_api_key_env", default="JUDGE_API_KEY", help="Env var holding judge API key")
    parser.add_argument("--max_concurrent", type=int, default=20, help="Maximum concurrent tasks judged per run")
    parser.add_argument("--max_retries", type=int, default=3, help="Judge retries for failed/unparseable responses")
    parser.add_argument("--judge_times", type=int, default=5, help="Judge calls per QA pair; averaged into llm_judge_score")
    parser.add_argument("--judge_max_tokens", type=int, default=None, help="Maximum tokens requested for each judge decision")
    parser.add_argument(
        "--judge_request_logging",
        action="store_true",
        help=(
            "Enable exact judge request/response JSONL logging. Default is disabled because "
            "serializing judge payloads can add client-side latency under high concurrency."
        ),
    )
    parser.add_argument(
        "--no_judge_request_logging",
        action="store_false",
        dest="judge_request_logging",
        help="Keep exact judge request/response JSONL logging disabled. This is the default.",
    )
    parser.add_argument(
        "--judge_empty_length_retry_max_tokens",
        type=int,
        default=1024,
        help="Maximum retry cap used for unparseable finish_reason=length judge responses",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if llm_judge_results.jsonl exists")
    parser.add_argument(
        "--judge-tag",
        "--judge_tag",
        dest="judge_tag",
        default=None,
        help=(
            "Additive re-judge mode: write a new judge model's results as tagged files "
            "(llm_judge_results.<TAG>.jsonl, llm_judge_summary.<TAG>.json, llm_judge_accuracy.<TAG>.txt, "
            "llm_judge_<TAG>/) without modifying existing judge/EM/eval_results/dashboard files."
        ),
    )
    parser.add_argument(
        "--allow-partial", action="store_true", help="Judge currently available per-task result files when benchmark_results.jsonl is not present"
    )
    parser.add_argument("--stable-seconds", type=float, default=30.0, help="Skip partial source files modified more recently than this")
    parser.add_argument("--no-update-eval-results", action="store_true", help="Do not annotate eval_results.json")
    parser.add_argument("--watch", action="store_true", help="Keep polling run dirs and incrementally evaluate newly completed task files")
    parser.add_argument("--poll-seconds", type=float, default=15.0, help="Polling interval for --watch")
    parser.add_argument(
        "--predictions-jsonl",
        "--predictions_jsonl",
        dest="predictions_jsonl",
        type=Path,
        default=None,
        help=(
            "Override the predictions source: a benchmark_results-style JSONL with per-task "
            "{task_id, output} used instead of <run_dir>/benchmark_results.jsonl. Ground truth "
            "still comes from --data_path. Intended for single --run_dir usage."
        ),
    )
    parser.add_argument(
        "--no-attempt-budget-subdirs",
        dest="no_attempt_budget_subdirs",
        action="store_true",
        help="Do not descend into attempt_budget_* sweep subdirs; judge only the given run dir(s).",
    )
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    if args.judge_tag is not None:
        args.judge_tag = args.judge_tag.strip()
        if not args.judge_tag or not re.fullmatch(r"[A-Za-z0-9._-]+", args.judge_tag):
            parser.error("--judge-tag must be a non-empty token of letters, digits, '.', '_' or '-'")
    args.judge_times = max(1, args.judge_times)
    args.judge_max_tokens = resolve_judge_max_tokens_for_benchmark(args.benchmark_name, args.judge_max_tokens)
    args.judge_empty_length_retry_max_tokens = max(1, args.judge_empty_length_retry_max_tokens)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(_amain(args))


if __name__ == "__main__":
    _main()
