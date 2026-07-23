# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate per-run accuracy across the runs of a multi-run web-search eval.

``run_eval_config`` with ``run.num_runs > 1`` produces independent ``run_1 ..
run_N`` directories, each with its own single-shot accuracy but no cross-run
summary. This runner reads each run's ``llm_judge_summary.json`` (or ``em``),
then reports avg@N (mean of the per-run accuracies) and the sample standard
deviation, writing ``aggregate_summary.json`` + ``aggregate_summary.txt`` into
the parent runs directory.

Example:
    python -m recipe.web_search.runners.aggregate_runs \
        --runs-dir logs/web_search_infer/livebrowsecomp_default
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

_METRIC_FIELDS = {
    "llm_judge": ("llm_judge_summary.json", "llm_judge_accuracy", "llm_judge_correct"),
    "em": ("em_summary.json", "em_accuracy", "em_correct"),
}
_RUN_DIR_PATTERN = re.compile(r"^run_(\d+)$")


@dataclass
class RunAccuracy:
    run: str
    accuracy: float
    correct: int | None
    num_items: int | None
    complete: bool | None


def _run_sort_key(path: Path) -> tuple[int, str]:
    match = _RUN_DIR_PATTERN.match(path.name)
    # Natural ordering by the integer suffix; non-matching dirs sort last.
    return (int(match.group(1)) if match else 1 << 30, path.name)


def discover_run_dirs(runs_dir: Path) -> list[Path]:
    """Return ``run_*`` subdirectories in natural (run_1, run_2, ...) order."""
    return sorted((p for p in runs_dir.iterdir() if p.is_dir() and _RUN_DIR_PATTERN.match(p.name)), key=_run_sort_key)


def load_run_accuracy(run_dir: Path, metric: str) -> RunAccuracy | None:
    """Read one run's accuracy from its judge summary; ``None`` if unavailable."""
    summary_name, accuracy_key, correct_key = _METRIC_FIELDS[metric]
    summary_path = run_dir / summary_name
    if not summary_path.is_file():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if metric == "em" and isinstance(payload.get("em"), dict):
        # llm_judge_summary.json nests the em block; em_summary.json is flat.
        payload = payload["em"]
    accuracy = payload.get(accuracy_key)
    if accuracy is None:
        return None
    return RunAccuracy(
        run=run_dir.name,
        accuracy=float(accuracy),
        correct=payload.get(correct_key),
        num_items=payload.get("num_items"),
        complete=payload.get("source_complete"),
    )


def aggregate_accuracies(runs: list[RunAccuracy], metric: str) -> dict:
    """Compute avg@N and the sample standard deviation over per-run accuracies."""
    accuracies = [r.accuracy for r in runs]
    num_runs = len(accuracies)
    item_counts = {r.num_items for r in runs if r.num_items is not None}
    return {
        "metric": metric,
        "num_runs": num_runs,
        "avg_accuracy": statistics.fmean(accuracies) if accuracies else None,
        # Sample standard deviation (ddof=1); undefined for fewer than 2 runs.
        "std_accuracy": statistics.stdev(accuracies) if num_runs >= 2 else None,
        "min_accuracy": min(accuracies) if accuracies else None,
        "max_accuracy": max(accuracies) if accuracies else None,
        # Only meaningful when every run scored the same number of items.
        "num_items": next(iter(item_counts)) if len(item_counts) == 1 else None,
        "incomplete_runs": [r.run for r in runs if r.complete is False],
        "runs": [asdict(r) for r in runs],
    }


def render_text_report(summary: dict) -> str:
    n = summary["num_runs"]
    avg = summary["avg_accuracy"]
    std = summary["std_accuracy"]
    lines = [f"metric: {summary['metric']}", f"num_runs: {n}"]
    if avg is not None and std is not None:
        lines.append(f"avg@{n}: {avg * 100:.2f}% ± {std * 100:.2f}% (sample std)")
    elif avg is not None:
        lines.append(f"avg@{n}: {avg * 100:.2f}% (std undefined for <2 runs)")
    if summary["num_items"] is not None:
        lines.append(f"num_items: {summary['num_items']}")
    if summary["incomplete_runs"]:
        lines.append(f"WARNING incomplete runs: {', '.join(summary['incomplete_runs'])}")
    lines.append("per-run: " + " ".join(f"{r['run']}={r['accuracy'] * 100:.2f}%" for r in summary["runs"]))
    return "\n".join(lines) + "\n"


def aggregate_runs(runs_dir: Path, metric: str) -> dict:
    run_dirs = discover_run_dirs(runs_dir)
    if not run_dirs:
        msg = f"no run_* subdirectories under {runs_dir}"
        raise FileNotFoundError(msg)
    runs = [acc for acc in (load_run_accuracy(d, metric) for d in run_dirs) if acc is not None]
    if not runs:
        msg = f"no run had a readable {_METRIC_FIELDS[metric][0]} under {runs_dir}"
        raise FileNotFoundError(msg)
    return {"runs_dir": str(runs_dir), **aggregate_accuracies(runs, metric)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate avg@N + sample std across multi-run web-search evals.")
    parser.add_argument("--runs-dir", type=Path, required=True, help="Parent directory holding run_1 .. run_N.")
    parser.add_argument("--metric", choices=sorted(_METRIC_FIELDS), default="llm_judge")
    parser.add_argument("--output", type=Path, default=None, help="JSON output path (default: <runs-dir>/aggregate_summary.json).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = aggregate_runs(args.runs_dir, args.metric)
    output = args.output or (args.runs_dir / "aggregate_summary.json")
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = render_text_report(summary)
    output.with_suffix(".txt").write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
