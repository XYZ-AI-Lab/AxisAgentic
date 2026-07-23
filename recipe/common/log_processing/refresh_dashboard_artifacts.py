# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Refresh dashboard artifacts for every run under one or more log roots."""

from __future__ import annotations

import argparse
import fnmatch
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from recipe.common.log_processing.assistant_message_stats import ARTIFACT_NAME as ASSISTANT_MESSAGE_STATS_NAME
from recipe.common.log_processing.build_eval_results import RunType, build_eval_results_from_run
from recipe.common.log_processing.trace_distributions import ARTIFACT_NAME as TRACE_DISTRIBUTIONS_NAME

RefreshStatus = Literal["ok", "skipped", "failed", "dry-run"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunRefreshPlan:
    run_dir: Path
    run_type: RunType


@dataclass(frozen=True)
class RunRefreshResult:
    run_dir: Path
    run_type: RunType
    status: RefreshStatus
    elapsed_s: float
    message: str = ""


def infer_run_type(run_dir: Path) -> RunType | None:
    """Infer dashboard run type from directory layout and experiment name."""
    if (run_dir / "wide-search").exists() or any(part.startswith("wide_search") for part in run_dir.parts):
        return "wide_search"
    if (run_dir / "web-search-benchmark").exists() or any(part.startswith("web_search") for part in run_dir.parts):
        return "web_search"
    return None


def discover_run_dirs(
    log_roots: list[Path],
    *,
    pattern: str | None = None,
    run_type: RunType | None = None,
) -> tuple[list[RunRefreshPlan], list[Path]]:
    """Discover refreshable ``run_*`` directories.

    Returns ``(plans, unknown_runs)``. ``pattern`` matches the string path with
    shell-style wildcards; without wildcard characters it is treated as
    ``*pattern*``.
    """
    plans: list[RunRefreshPlan] = []
    unknown: list[Path] = []
    seen: set[Path] = set()
    normalized_pattern = _normalize_pattern(pattern)
    for root in log_roots:
        if not root.exists():
            logger.warning("Log root does not exist: %s", root)
            continue
        for candidate in sorted(path for path in root.glob("**/run_*") if path.is_dir()):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if normalized_pattern and not fnmatch.fnmatch(str(candidate), normalized_pattern):
                continue
            inferred = infer_run_type(candidate)
            if inferred is None:
                unknown.append(candidate)
                continue
            if run_type is not None and inferred != run_type:
                continue
            plans.append(RunRefreshPlan(run_dir=candidate, run_type=inferred))
    return plans, unknown


def refresh_run(plan: RunRefreshPlan, *, force: bool = False, dry_run: bool = False, skip_unchanged: bool = False) -> RunRefreshResult:
    start = time.perf_counter()
    if dry_run:
        return RunRefreshResult(plan.run_dir, plan.run_type, "dry-run", time.perf_counter() - start)
    if force and skip_unchanged and _artifacts_are_current(plan):
        return RunRefreshResult(plan.run_dir, plan.run_type, "skipped", time.perf_counter() - start, "artifacts are current")
    try:
        path = build_eval_results_from_run(plan.run_dir, run_type=plan.run_type, force=force)
    except Exception as exc:
        return RunRefreshResult(plan.run_dir, plan.run_type, "failed", time.perf_counter() - start, f"{type(exc).__name__}: {exc}")
    if path is None:
        return RunRefreshResult(plan.run_dir, plan.run_type, "skipped", time.perf_counter() - start, "no completed traces found")
    return RunRefreshResult(plan.run_dir, plan.run_type, "ok", time.perf_counter() - start, str(path))


def refresh_runs(
    plans: list[RunRefreshPlan],
    *,
    force: bool = False,
    dry_run: bool = False,
    jobs: int = 1,
    skip_unchanged: bool = False,
) -> list[RunRefreshResult]:
    """Refresh the provided plans, preserving result order for serial runs."""
    if jobs <= 1 or dry_run:
        return [refresh_run(plan, force=force, dry_run=dry_run, skip_unchanged=skip_unchanged) for plan in plans]

    results: list[RunRefreshResult] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(refresh_run, plan, force=force, dry_run=False, skip_unchanged=skip_unchanged) for plan in plans]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda result: str(result.run_dir))


def refresh_cycle(
    log_roots: list[Path],
    *,
    pattern: str | None = None,
    run_type: RunType | None = None,
    force: bool = False,
    dry_run: bool = False,
    jobs: int = 1,
    skip_unchanged: bool = False,
) -> tuple[list[RunRefreshPlan], list[Path], list[RunRefreshResult]]:
    """Discover and refresh runs once.

    Watch mode calls this repeatedly from a separate CLI process. Evaluation
    runners do not call this loop, so it cannot add work to the evaluation hot
    path unless the user starts it manually in another shell.
    """
    plans, unknown = discover_run_dirs(log_roots, pattern=pattern, run_type=run_type)
    results = refresh_runs(plans, force=force, dry_run=dry_run, jobs=jobs, skip_unchanged=skip_unchanged)
    return plans, unknown, results


def _iter_source_files(plan: RunRefreshPlan) -> list[Path]:
    run_dir = plan.run_dir
    files: list[Path] = []
    files.extend(path for path in (run_dir / "benchmark_results.jsonl", run_dir / "em_results.jsonl") if path.exists())
    files.extend(run_dir.glob("llm_judge*.json*"))
    files.extend((run_dir / "exact_match").glob("*.json") if (run_dir / "exact_match").exists() else [])
    if plan.run_type == "web_search":
        files.extend((run_dir / "web-search-benchmark").glob("*.json"))
    else:
        files.extend((run_dir / "wide-search").glob("*.json"))
        if (run_dir / "widesearch_scores").exists():
            files.extend((run_dir / "widesearch_scores").glob("*.json"))
    return [path for path in files if path.is_file()]


def _artifacts_are_current(plan: RunRefreshPlan) -> bool:
    artifacts = [
        plan.run_dir / "eval_results.json",
        plan.run_dir / "dashboard_summary.json",
        plan.run_dir / "dashboard_events.jsonl",
        plan.run_dir / TRACE_DISTRIBUTIONS_NAME,
    ]
    artifacts.append(plan.run_dir / ASSISTANT_MESSAGE_STATS_NAME)
    if not all(path.exists() for path in artifacts):
        return False
    artifact_mtime = min(path.stat().st_mtime for path in artifacts)
    source_files = _iter_source_files(plan)
    if not source_files:
        return True
    return max(path.stat().st_mtime for path in source_files) <= artifact_mtime


def _normalize_pattern(pattern: str | None) -> str | None:
    if not pattern:
        return None
    if any(char in pattern for char in "*?[]"):
        return pattern
    return f"*{pattern}*"


def _print_plan(plans: list[RunRefreshPlan], unknown: list[Path]) -> None:
    counts = _counts_by_type(plans)
    print(f"Discovered runs: web_search={counts['web_search']} wide_search={counts['wide_search']} unknown={len(unknown)}")
    for plan in plans:
        print(f"[plan] {plan.run_type:10s} {plan.run_dir}")
    for path in unknown:
        print(f"[unknown] {path}")


def _print_result(result: RunRefreshResult) -> None:
    suffix = f" - {result.message}" if result.message and result.status == "failed" else ""
    print(f"[{result.status}] {result.run_type:10s} {result.elapsed_s:6.2f}s {result.run_dir}{suffix}", flush=True)


def _print_summary(results: list[RunRefreshResult]) -> None:
    by_status: dict[str, int] = {"ok": 0, "skipped": 0, "failed": 0, "dry-run": 0}
    for result in results:
        by_status[result.status] += 1
    print("Summary:")
    for status in ("ok", "skipped", "failed", "dry-run"):
        print(f"  {status}: {by_status[status]}")


def _counts_by_type(plans: list[RunRefreshPlan]) -> dict[str, int]:
    counts = {"web_search": 0, "wide_search": 0}
    for plan in plans:
        counts[plan.run_type] += 1
    return counts


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Refresh dashboard artifacts for run_* directories under log roots")
    parser.add_argument("--log-root", type=Path, action="append", default=None, help="Log root to scan. Can be passed multiple times.")
    parser.add_argument("--run-type", choices=["web_search", "wide_search"], default=None, help="Only refresh runs of this inferred type.")
    parser.add_argument("--pattern", default=None, help="Only refresh paths matching this glob or substring.")
    parser.add_argument("--force", action="store_true", help="Rebuild eval_results.json even when it already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Only print discovered runs; do not write artifacts.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of runs to refresh concurrently. Default: 1.")
    parser.add_argument("--watch", action="store_true", help="Keep polling log roots and refreshing artifacts until interrupted.")
    parser.add_argument("--interval", type=float, default=30.0, help="Seconds between watch refresh cycles. Default: 30.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(name)s %(levelname)s %(message)s")
    log_roots = args.log_root or [Path("logs")]
    jobs = max(int(args.jobs), 1)
    interval = max(float(args.interval), 1.0)

    try:
        while True:
            if args.watch:
                print(f"Refresh cycle at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            plans, unknown, results = refresh_cycle(
                log_roots,
                pattern=args.pattern,
                run_type=args.run_type,
                force=args.force,
                dry_run=args.dry_run,
                jobs=jobs,
                skip_unchanged=args.watch,
            )
            _print_plan(plans, unknown)
            for result in results:
                _print_result(result)
            _print_summary(results)
            if not args.watch:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopped watch refresh.", flush=True)


if __name__ == "__main__":
    _main()
