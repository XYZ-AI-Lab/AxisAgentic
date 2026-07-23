# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Build ``eval_results.json`` and dashboard artifacts from existing run logs."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Literal

from recipe.common.log_processing.dashboard_artifacts import write_dashboard_artifacts
from recipe.common.log_processing.live_eval_results import scan_agentic_trace_index

RunType = Literal["web_search", "wide_search"]

logger = logging.getLogger(__name__)

_WIDESEARCH_RETRYABLE_SCORE_PREFIXES = (
    "orchestrator error:",
    "transport error:",
    "connection error:",
    "timeout error:",
    "request error:",
)


def _agentic_trace_dir(run_type: RunType) -> str:
    if run_type == "web_search":
        return "web-search-benchmark"
    return "wide-search"


def _write_eval_results(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def _normalize_em_result(row: dict[str, Any], task_id: str) -> dict[str, Any]:
    normalized = dict(row)
    normalized["task_id"] = str(normalized.get("task_id", task_id))
    score = normalized.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        normalized["score"] = float(score)
        normalized.setdefault("is_correct", float(score) > 0)
        normalized.setdefault("em_result", "CORRECT" if float(score) > 0 else "INCORRECT")
    elif normalized.get("is_correct") is not None:
        normalized["score"] = 1.0 if bool(normalized["is_correct"]) else 0.0
        normalized.setdefault("em_result", "CORRECT" if normalized["score"] > 0 else "INCORRECT")
    return normalized


def _is_retryable_widesearch_score_sidecar(row: dict[str, Any]) -> bool:
    msg = str(row.get("msg") or "").strip().lower()
    return msg.startswith(_WIDESEARCH_RETRYABLE_SCORE_PREFIXES)


def _load_em_results(run_dir: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    jsonl_path = run_dir / "em_results.jsonl"
    if jsonl_path.exists():
        try:
            lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            lines = []
        for line in lines:
            if not line.strip():
                continue
            with contextlib.suppress(json.JSONDecodeError):
                row = json.loads(line)
                tid = str(row.get("task_id", ""))
                if tid:
                    results[tid] = _normalize_em_result(row, tid)

    sidecar_dir = run_dir / "exact_match"
    if sidecar_dir.exists():
        for path in sorted(sidecar_dir.glob("*.json")):
            with contextlib.suppress(UnicodeDecodeError, json.JSONDecodeError, OSError):
                row = json.loads(path.read_text(encoding="utf-8"))
                tid = str(row.get("task_id", path.stem))
                if tid and tid not in results:
                    results[tid] = _normalize_em_result(row, tid)

    widesearch_scores_dir = run_dir / "widesearch_scores"
    if widesearch_scores_dir.exists():
        for path in sorted(widesearch_scores_dir.glob("*.json")):
            with contextlib.suppress(UnicodeDecodeError, json.JSONDecodeError, OSError):
                row = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(row, dict) or _is_retryable_widesearch_score_sidecar(row):
                    continue
                tid = str(row.get("task_id") or path.stem)
                if tid and tid not in results:
                    results[tid] = _normalize_em_result(row, tid)
    return results


def build_eval_results_from_run(run_dir: str | Path, *, run_type: RunType, force: bool = False) -> Path | None:
    """Build or refresh ``eval_results.json`` for one run directory."""
    run_path = Path(run_dir)
    eval_results_path = run_path / "eval_results.json"
    if eval_results_path.exists() and not force:
        logger.info("%s already exists; use --force to rebuild", eval_results_path)
        write_dashboard_artifacts(run_path, run_type=run_type)
        return eval_results_path

    from recipe.common.log_processing.live_eval_results import _build_live_agentic_eval_results

    index = scan_agentic_trace_index(run_path, (_agentic_trace_dir(run_type),))
    payload = _build_live_agentic_eval_results(str(run_path), index, _load_em_results(run_path))
    if payload is None:
        logger.warning("No completed agentic traces found under %s", run_path)
        return None
    path = _write_eval_results(eval_results_path, payload)
    write_dashboard_artifacts(run_path, run_type=run_type)
    return path


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build eval_results.json from existing run logs")
    parser.add_argument("--run-dir", type=Path, action="append", required=True, help="Run directory. Pass multiple times to process several runs.")
    parser.add_argument("--run-type", required=True, choices=["web_search", "wide_search"], help="Run type for trace discovery")
    parser.add_argument("--force", action="store_true", help="Rebuild eval_results.json even when it already exists")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_type = args.run_type
    for run_dir in args.run_dir:
        build_eval_results_from_run(run_dir, run_type=run_type, force=args.force)


if __name__ == "__main__":
    _main()
