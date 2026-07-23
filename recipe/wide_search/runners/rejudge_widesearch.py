# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Additively re-judge a finished WideSearch run with a different judge model.

This post-processor does not re-run the agent. It reuses the persisted agent
traces under ``<run_dir>/wide-search/`` (read-only), re-scores each trial with a
new :class:`WideSearchJudgeClient`, and writes tagged sidecars/summaries:

* ``widesearch_scores_<TAG>/{instance}__trial-{k}.json``: per-trial metrics
* ``widesearch_summary.<TAG>.json``: leaderboard + global summary
* ``widesearch_per_task.<TAG>.json``: per-task aggregation

The run's canonical ``widesearch_scores/``, ``widesearch_summary.json``,
``widesearch_per_task.json``, ``wide-search/`` traces, ``eval_results.json`` and
dashboard artifacts are left untouched.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import yaml

from agentic.observability.task_logger import TaskTrace
from agentic.orchestration.task_orchestrator import OrchestrationResult
from recipe.web_search.config import RunConfig
from recipe.wide_search.config import WideSearchBenchmarkConfig, WideSearchEvalSettings
from recipe.wide_search.eval.aggregation import (
    TrialResult,
    global_summary,
    leaderboard_view,
    per_task_aggregates,
    serialize_per_task,
)
from recipe.wide_search.eval.data_loader import load_widesearch_queries
from recipe.wide_search.eval.evaluation import aevaluate_single_query
from recipe.wide_search.eval.judge_client import WideSearchJudgeClient, WideSearchJudgeConfig
from recipe.wide_search.runners.evaluate_widesearch import (
    ORCHESTRATOR_NAME,
    _load_scored_trials,
    _select_response_for_scoring,
    _trial_from_eval,
    _write_trial_sidecar,
    trial_task_id,
)
from recipe.wide_search.runners.run_eval_config import (
    _load_env_file,
    _resolve_env_file_path,
    _resolve_path,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from recipe.wide_search.eval.answer_extractor import ExtractorMode
    from recipe.wide_search.eval.data_loader import WideSearchQuery

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"[A-Za-z0-9._-]+")


@dataclass
class _RejudgeContext:
    """Minimal config slice re-judging needs, tolerant of agent-side drift."""

    benchmark: WideSearchBenchmarkConfig
    eval: WideSearchEvalSettings
    model: SimpleNamespace
    env_file: str | None


def _pick_known(model_cls: Any, data: dict[str, Any] | None) -> Any:
    """Validate ``data`` against ``model_cls`` after dropping unknown top-level keys."""
    known = set(model_cls.model_fields)
    return model_cls.model_validate({k: v for k, v in (data or {}).items() if k in known})


def _load_rejudge_context(config_path: Path) -> _RejudgeContext:
    """Load only the fields re-judging needs from a WideSearch eval YAML.

    Old run configs can carry agent-side sections (``tools``/``model``) that no
    longer validate under the current strict schema. Re-judging never runs the
    agent, so validate just ``benchmark`` and ``eval`` (dropping unknown keys)
    and read the few judge-fallback values from the raw ``model``/``run`` maps.
    """
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model_raw = raw.get("model") or {}
    run_raw = raw.get("run") or {}
    return _RejudgeContext(
        benchmark=_pick_known(WideSearchBenchmarkConfig, raw.get("benchmark")),
        eval=_pick_known(WideSearchEvalSettings, raw.get("eval")),
        model=SimpleNamespace(
            openai_model=model_raw.get("openai_model"),
            openai_base_url=model_raw.get("openai_base_url"),
            api_key_env=model_raw.get("api_key_env") or "OPENAI_API_KEY",
        ),
        env_file=run_raw.get("env_file", RunConfig().env_file),
    )


def _index_completed_traces(traces_dir: Path) -> dict[str, Path]:
    """Map ``{instance}__trial-{k}`` -> highest-attempt finalized trace file.

    Built purely from filenames in a single directory listing (no file reads),
    so it is O(number_of_traces). Finalized traces are ``{task_id}.json``;
    in-progress ``.partial.json`` files are ignored.
    """
    best: dict[str, tuple[int, Path]] = {}
    for path in traces_dir.glob("*.json"):
        name = path.name
        if name.endswith(".partial.json"):
            continue
        stem = name[: -len(".json")]
        base, sep, attempt_str = stem.rpartition("_attempt-")
        if not sep or "__trial-" not in base:
            continue
        try:
            attempt = int(attempt_str)
        except ValueError:
            continue
        current = best.get(base)
        if current is None or attempt > current[0]:
            best[base] = (attempt, path)
    return {base: path for base, (_, path) in best.items()}


def _errored_trial_ids(scores_dir: Path) -> set[str]:
    """Trial ids whose tagged sidecar recorded one or more judge errors.

    Used by ``--rejudge-errored`` to force those specific trials to be scored
    again on a resume, rather than being reused as completed.
    """
    errored: set[str] = set()
    if not scores_dir.exists():
        return errored
    for path in scores_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("judge_errors"):
            errored.add(path.stem)
    return errored


def _load_trace_result(path: Path) -> OrchestrationResult | None:
    """Load one trace into an OrchestrationResult without rebuilding conversation.

    Scoring only needs the final answer (``metadata.output``) plus, as a
    fallback, the last assistant message. Skipping conversation reconstruction
    (``reconstruct_conversation=False``) avoids materializing hundreds of
    message objects per trace, which is the dominant cost on large WideSearch
    traces.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("re-judge: failed to read trace %s", path)
        return None
    try:
        trace = TaskTrace.from_dict(data)
    except (KeyError, TypeError, ValueError):
        logger.warning("re-judge: malformed trace %s", path)
        return None
    return OrchestrationResult.from_trace(trace, reconstruct_conversation=False)


def _write_tagged_per_task_and_summary(
    output_dir: Path,
    tag: str,
    trials: list[TrialResult],
    expected_trials: int,
    *,
    total_instances: int,
    all_instance_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Tagged twin of evaluate_widesearch._write_per_task_and_summary.

    ``all_instance_ids`` is the full set of *scheduled* instances. Seeding it
    ensures instances whose traces were entirely missing (zero scored trials)
    still surface in ``incomplete_instance_ids`` / ``num_incomplete_instances``
    instead of silently vanishing from the counts.
    """
    by_iid: dict[str, list[TrialResult]] = {iid: [] for iid in (all_instance_ids or ())}
    for trial in trials:
        by_iid.setdefault(trial.instance_id, []).append(trial)
    completed_trials = [t for ts in by_iid.values() if len(ts) >= expected_trials for t in ts]
    incomplete = sorted(iid for iid, ts in by_iid.items() if len(ts) < expected_trials)
    completed_per_task = per_task_aggregates(completed_trials, expected_trials=expected_trials)

    summary = global_summary(completed_per_task)
    leaderboard = leaderboard_view(summary)

    per_task_payload = {
        "expected_trials": expected_trials,
        "total_instances": total_instances,
        "num_complete_instances": len(completed_per_task),
        "num_incomplete_instances": len(incomplete),
        "incomplete_instance_ids": incomplete,
        "per_task": serialize_per_task(completed_per_task),
    }
    (output_dir / f"widesearch_per_task.{tag}.json").write_text(json.dumps(per_task_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_payload = {
        "num_trials": expected_trials,
        "total_instances": total_instances,
        "num_complete_instances": len(completed_per_task),
        "num_incomplete_instances": len(incomplete),
        "judge_tag": tag,
        "summary": summary,
        "leaderboard": leaderboard,
    }
    (output_dir / f"widesearch_summary.{tag}.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_payload


async def rejudge_run_dir(
    *,
    run_dir: Path,
    queries: list[WideSearchQuery],
    expected_trials: int,
    extractor_mode: ExtractorMode,
    prompts_profile: str,
    judge_config: WideSearchJudgeConfig,
    judge_max_concurrent: int,
    tag: str,
    resume: bool = True,
    rejudge_errored: bool = False,
) -> dict[str, Any]:
    """Re-judge every completed trial trace in ``run_dir`` with ``judge_config``.

    Trials whose agent trace is missing are skipped (never re-run); their
    instances land in ``num_incomplete_instances``. When ``resume`` is true,
    trials that already have a valid tagged sidecar under
    ``widesearch_scores_<TAG>/`` are reused as-is and not re-judged, so an
    interrupted run can be continued (e.g. at a different concurrency) without
    redoing finished work. When ``rejudge_errored`` is also true, sidecars that
    recorded judge errors are dropped from the reuse set so those specific
    trials are scored again.
    """
    run_dir = Path(run_dir).expanduser()
    traces_dir = run_dir / ORCHESTRATOR_NAME
    if not traces_dir.exists():
        msg = f"no agent traces found at {traces_dir}; nothing to re-judge"
        raise FileNotFoundError(msg)

    scores_dir = run_dir / f"widesearch_scores_{tag}"
    scores_dir.mkdir(parents=True, exist_ok=True)

    # Index finalized traces once (filenames only); scoring reads each trial's
    # single file directly, avoiding the O(N^2) re-scan of a per-trial loader.
    trace_index = _index_completed_traces(traces_dir)
    logger.info("re-judge: %d completed trial traces indexed under %s", len(trace_index), run_dir)

    judge_client = WideSearchJudgeClient(judge_config)
    judge_client.attach_semaphore(asyncio.Semaphore(max(1, judge_max_concurrent)))

    # Resume: reuse valid tagged sidecars from a prior (possibly interrupted)
    # run so we only judge what's left. Sidecars with retryable errors are
    # ignored by the loader and get re-judged.
    scored_by_id = _load_scored_trials(scores_dir) if resume else {}
    if scored_by_id and rejudge_errored:
        errored = _errored_trial_ids(scores_dir) & set(scored_by_id)
        for task_id in errored:
            scored_by_id.pop(task_id, None)
        if errored:
            logger.info("re-judge: %d errored trials will be re-scored", len(errored))
    if scored_by_id:
        logger.info("re-judge: resuming, %d trials already scored (skipping)", len(scored_by_id))
    trials: list[TrialResult] = list(scored_by_id.values())
    trials_lock = asyncio.Lock()

    async def _rejudge_trial(query: WideSearchQuery, trial_index: int) -> None:
        task_id = trial_task_id(query.instance_id, trial_index)
        if task_id in scored_by_id:
            return
        trace_path = trace_index.get(task_id)
        if trace_path is None:
            return
        result = _load_trace_result(trace_path)
        if result is None:
            return
        raw_response, df = _select_response_for_scoring(result, mode=extractor_mode)
        eval_result = await aevaluate_single_query(query, df, client=judge_client, prompt_profile=prompts_profile)
        trial = _trial_from_eval(query.instance_id, trial_index, eval_result)
        async with trials_lock:
            trials.append(trial)
            _write_trial_sidecar(scores_dir, trial, eval_result, raw_response=raw_response)

    started_at = time.perf_counter()
    try:
        await asyncio.gather(*(_rejudge_trial(query, trial_index) for query in queries for trial_index in range(expected_trials)))
    finally:
        elapsed = time.perf_counter() - started_at
        logger.info(
            "re-judge finished in %.1fs (judge_calls=%d, trials_scored=%d)",
            elapsed,
            judge_client.total_calls,
            len(trials),
        )
        await judge_client.aclose()

    return _write_tagged_per_task_and_summary(
        run_dir,
        tag,
        trials,
        expected_trials,
        total_instances=len(queries),
        all_instance_ids=[query.instance_id for query in queries],
    )


def _resolve_judge_config(
    config: Any,
    *,
    judge_model: str | None,
    judge_base_url: str | None,
    judge_api_key_env: str | None,
) -> WideSearchJudgeConfig:
    # Mirror run_eval_config._resolve_config: fall back through the judge env
    # vars and finally the agent model/endpoint, so configs that leave
    # eval.judge_* unset (relying on model.openai_*) re-judge without extra flags.
    model = judge_model or config.eval.judge_model or os.environ.get("JUDGE_MODEL") or config.model.openai_model or os.environ.get("OPENAI_MODEL")
    if not model:
        msg = "judge model is required: pass --judge-model, set JUDGE_MODEL/OPENAI_MODEL, or set eval.judge_model / model.openai_model in the config"
        raise ValueError(msg)
    base_url = (
        judge_base_url
        or config.eval.judge_base_url
        or os.environ.get("JUDGE_BASE_URL")
        or config.model.openai_base_url
        or os.environ.get("OPENAI_BASE_URL")
    )
    api_key_env = judge_api_key_env or config.eval.judge_api_key_env
    if api_key_env == "JUDGE_API_KEY" and "JUDGE_API_KEY" not in os.environ:
        api_key_env = config.model.api_key_env
    return WideSearchJudgeConfig(
        judge_model=model,
        judge_base_url=base_url,
        judge_api_key_env=api_key_env,
        judge_max_tokens=config.eval.judge_max_tokens,
        judge_temperature=config.eval.judge_temperature,
        request_timeout=config.eval.request_timeout,
        max_retries=config.eval.max_retries,
        retryable_status_codes=list(config.eval.retryable_status_codes),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Additively re-judge a finished WideSearch run with a new judge model")
    parser.add_argument("--config", required=True, help="WideSearch eval YAML used for the original run (or a copy)")
    parser.add_argument(
        "--run-dir",
        "--run_dir",
        dest="run_dir",
        action="append",
        required=True,
        help="Run dir containing wide-search/ traces (e.g. .../run_1). Pass multiple times.",
    )
    parser.add_argument("--judge-tag", "--judge_tag", dest="judge_tag", required=True, help="Tag for the additive output files")
    parser.add_argument("--judge-model", "--judge_model", dest="judge_model", default=None, help="New judge model (overrides config/env)")
    parser.add_argument("--judge-base-url", "--judge_base_url", dest="judge_base_url", default=None)
    parser.add_argument("--judge-api-key-env", "--judge_api_key_env", dest="judge_api_key_env", default=None)
    parser.add_argument("--num-trials", type=int, default=None, help="Override benchmark.num_trials (must match the original run)")
    parser.add_argument("--judge-max-concurrent", type=int, default=None, help="Override eval.judge_max_concurrent")
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Re-judge every trial even if a tagged sidecar already exists (default: resume/skip existing).",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--rejudge-errored",
        dest="rejudge_errored",
        action="store_true",
        help="On resume, re-score trials whose existing tagged sidecar recorded judge errors.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    tag = args.judge_tag.strip()
    if not tag or not _TAG_RE.fullmatch(tag):
        parser.error("--judge-tag must be a non-empty token of letters, digits, '.', '_' or '-'")

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    config_path = Path(args.config)
    ctx = _load_rejudge_context(config_path)
    _load_env_file(_resolve_env_file_path(config_path, ctx.env_file))

    judge_config = _resolve_judge_config(
        ctx,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        judge_api_key_env=args.judge_api_key_env,
    )
    expected_trials = args.num_trials if args.num_trials is not None else ctx.benchmark.num_trials
    judge_max_concurrent = args.judge_max_concurrent if args.judge_max_concurrent is not None else ctx.eval.judge_max_concurrent

    data_path = _resolve_path(ctx.benchmark.data_path, label="benchmark.data_path")
    gold_dir = _resolve_path(ctx.benchmark.gold_dir, label="benchmark.gold_dir")
    queries = load_widesearch_queries(str(data_path), str(gold_dir), instance_ids=ctx.benchmark.instance_ids)
    if not queries:
        msg = f"no widesearch queries loaded from {data_path} / {gold_dir}"
        raise RuntimeError(msg)
    # Reproduce run_evaluation's selection so subset runs (shuffle_tasks /
    # max_tasks) re-judge exactly the scheduled instances, not the full dataset.
    if ctx.benchmark.shuffle_tasks:
        random.Random(ctx.benchmark.shuffle_seed).shuffle(queries)
    if ctx.benchmark.max_tasks is not None:
        queries = queries[: ctx.benchmark.max_tasks]

    print("=== WideSearch additive re-judge ===")
    print(f"Config:       {config_path}")
    print(f"Judge model:  {judge_config.judge_model}")
    print(f"Judge tag:    {tag}")
    print(f"Num trials:   {expected_trials}")
    print(f"Queries:      {len(queries)}")

    for raw_run_dir in args.run_dir:
        run_dir = _resolve_path(raw_run_dir, label="run-dir")
        print(f"--- Re-judging {run_dir} ---")
        summary = asyncio.run(
            rejudge_run_dir(
                run_dir=run_dir,
                queries=queries,
                expected_trials=expected_trials,
                extractor_mode=ctx.eval.extractor,
                prompts_profile=ctx.eval.prompts,
                judge_config=judge_config,
                judge_max_concurrent=judge_max_concurrent,
                tag=tag,
                resume=args.resume,
                rejudge_errored=args.rejudge_errored,
            )
        )
        for key, value in summary.get("leaderboard", {}).items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
