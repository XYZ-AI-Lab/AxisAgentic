# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from agentic.config import FormatErrorConfig, OrchestrationConfig, ToolArgumentRepairConfig, ToolManagerConfig
from agentic.contracts import ConversationMessage, FormatErrorStrategy, MessageRole
from agentic.observability.task_logger import TaskLogger
from agentic.orchestration.task_orchestrator import OrchestrationResult
from agentic.tools import ToolManager
from recipe.common.log_processing.build_eval_results import build_eval_results_from_run
from recipe.web_search.agent.prompts import FORMAT_ERROR_MESSAGE
from recipe.web_search.agent.runtime import WebSearchConversationConfig
from recipe.web_search.runners.evaluate_benchmark import (
    _build_context_token_estimator,
    build_model_client,
    build_web_search_tools,
)
from recipe.wide_search.agent.orchestrator import WideSearchTaskOrchestrator
from recipe.wide_search.agent.prompts import widesearch_system_prompt
from recipe.wide_search.agent.runtime import WideSearchConversationRuntime
from recipe.wide_search.eval.aggregation import (
    TrialResult,
    global_summary,
    leaderboard_view,
    per_task_aggregates,
    serialize_per_task,
)
from recipe.wide_search.eval.answer_extractor import ExtractorMode, extract_dataframe
from recipe.wide_search.eval.data_loader import WideSearchQuery, load_widesearch_queries
from recipe.wide_search.eval.evaluation import WideSearchEvalResult, aevaluate_single_query
from recipe.wide_search.eval.judge_client import WideSearchJudgeClient, WideSearchJudgeConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from recipe.web_search.agent.orchestrator import WebSearchTaskOrchestrator
    from recipe.wide_search.config import WideSearchEvalConfig

logger = logging.getLogger(__name__)

ORCHESTRATOR_NAME = "wide-search"


def _config_to_namespace(config: WideSearchEvalConfig) -> argparse.Namespace:
    """Adapt typed config to the argparse Namespace expected by shared builders."""
    return argparse.Namespace(
        # build_model_client
        model=config.model.openai_model,
        base_url=config.model.openai_base_url,
        api_key_env=config.model.api_key_env,
        max_context_length=config.model.context.max_context_length,
        max_output_tokens=config.model.max_output_tokens,
        temperature=config.model.temperature,
        top_p=config.model.top_p,
        timeout=config.model.timeout,
        endpoint_profile=config.model.endpoint_profile,
        preserve_reasoning_content=config.model.preserve_reasoning_content,
        response_reasoning_fields=config.model.response_reasoning_fields,
        max_tokens_field=config.model.max_tokens_field,
        parallel_tool_calls=config.model.parallel_tool_calls,
        parse_embedded_thinking=config.model.parse_embedded_thinking,
        endpoint_error_exit_status_codes=config.model.endpoint_error_exit_status_codes,
        endpoint_error_exit_after_seconds=config.model.endpoint_error_exit_after_seconds,
        endpoint_connection_error_retry_wait_seconds=config.model.endpoint_connection_error_retry_wait_seconds,
        endpoint_error_retry_backoff_multiplier=config.model.transport.retry.backoff_multiplier,
        endpoint_error_retry_backoff_max_seconds=config.model.transport.retry.backoff_max_seconds,
        endpoint_error_retry_jitter=config.model.transport.retry.jitter,
        endpoint_error_respect_retry_after=config.model.transport.retry.respect_retry_after,
        endpoint_retry_on_timeout=config.model.transport.retry.retry_on_timeout,
        endpoint_retry_on_connection_error=config.model.transport.retry.retry_on_connection_error,
        token_estimation_chars_per_token=config.model.context.token_estimation_chars_per_token,
        repetition_penalty=config.model.repetition_penalty,
        request_extra_body_json=json.dumps(config.model.request_extra_body) if config.model.request_extra_body else None,
        max_response_retries=config.model.max_response_retries,
        retry_wait_seconds=config.model.retry_wait_seconds,
        context_safety_margin=config.model.context.safety_margin,
        context_limit_detection=config.model.context.limit_detection,
        context_estimator=config.model.context.estimator,
        context_tokenizer_path=config.model.context.tokenizer_path,
        system_prompt_render_template=config.run.system_prompt_render_template,
        system_prompt_render_extract_start=config.run.system_prompt_render_extract_start,
        system_prompt_render_extract_end=config.run.system_prompt_render_extract_end,
        # build_web_search_tools
        prompt_profile="default",  # widesearch reuses the default web-search tool surface
        jina_base_url=config.tools.jina_base_url,
        max_content_length=config.tools.max_content_length,
        summary_llm_base_url=config.tools.summary_llm.base_url,
        summary_llm_model_name=config.tools.summary_llm.model_name,
        summary_llm_api_key_env=config.tools.summary_llm.api_key_env,
        summary_llm_cache_enabled=config.tools.summary_llm.cache_enabled,
        summary_llm_timeout=config.tools.summary_llm.timeout.to_runtime(),
        summary_llm_retry=config.tools.summary_llm.retry.to_runtime(),
        scrape_timeout=config.tools.scrape.timeout.to_runtime(),
        scrape_retry=config.tools.scrape.retry.to_runtime(),
        scrape_fallback_retry=config.tools.scrape.fallback_retry_runtime(),
        search_timeout=config.tools.search.timeout.to_runtime(),
        search_retry=config.tools.search.retry.to_runtime(),
        raw_scrape_cache_enabled=config.tools.raw_scrape_cache.enabled,
        raw_scrape_cache_scope=config.tools.raw_scrape_cache.scope,
        raw_scrape_cache_provider=config.tools.raw_scrape_cache.provider,
        raw_scrape_cache_normalize_url=config.tools.raw_scrape_cache.normalize_url,
        serper_base_url=config.tools.serper_base_url,
    )


def build_widesearch_orchestrator(
    *,
    model_client: Any,
    task_logger: TaskLogger | None,
    config: WideSearchEvalConfig,
    context_token_estimator: Callable[[list[ConversationMessage]], int] | None = None,
) -> WebSearchTaskOrchestrator:
    r"""Construct a WebSearchTaskOrchestrator parameterized for widesearch.

    Crucially uses ``name="wide-search"`` so traces land in ``<output_dir>/wide-search/``
    and the resume cache picks up the right files. The system prompt is
    widesearch-specific (project default = boxed Markdown table).

    Returns a :class:`WideSearchTaskOrchestrator`, which extends the base
    orchestrator to also accept unboxed Markdown tables as final answers — the
    official agent prompt does not wrap its table in ``\\boxed{}``.
    """
    args = _config_to_namespace(config)
    tools = build_web_search_tools(args)
    tool_manager = ToolManager(
        tools=tools,
        config=ToolManagerConfig(argument_repair=ToolArgumentRepairConfig(enabled=True)),
    )
    system_prompt = widesearch_system_prompt(
        profile=config.agent_prompt.profile,
        language=config.agent_prompt.language,
        date=config.agent_prompt.date,
    )
    format_error = FormatErrorConfig(strategy=FormatErrorStrategy.IGNORE, keywords=[])
    preflight_enabled = config.model.context.limit_detection == "estimated"
    return WideSearchTaskOrchestrator(
        config=OrchestrationConfig(
            name=ORCHESTRATOR_NAME,
            conversation=WebSearchConversationConfig(
                system_prompt=system_prompt,
                user_prompt_template="{task}",
                max_turns=config.agent.max_turns,
                context_window=config.model.context.max_context_length,
                context_safety_margin=config.model.context.safety_margin,
                min_tokens_for_generation=config.model.context.min_tokens_for_generation,
                context_warning_threshold=config.model.context.warning_threshold or config.model.max_output_tokens,
                format_error=format_error,
                early_stop_announcement_prompt=None,
                final_response_prompt=None,
                keep_tool_result=config.agent.keep_tool_result,
                tool_result_role=config.agent.tool_result_role,
            ),
        ),
        model_client=model_client,
        tool_manager=tool_manager,
        task_logger=task_logger,
        conversation_runtime_class=WideSearchConversationRuntime,
        max_task_retries=config.agent.retry.max_task_retries,
        include_failure_summary_in_retry=config.agent.retry.include_failure_summary,
        max_final_answer_attempts=config.agent.retry.max_final_answer_attempts,
        prompt_profile="default",
        generation_limit_recovery_non_final_attempt=config.agent.retry.generation_limit_recovery.non_final_attempt,
        generation_limit_recovery_final_attempt=config.agent.retry.generation_limit_recovery.final_attempt,
        context_token_estimator=context_token_estimator,
        context_limit_preflight_enabled=preflight_enabled,
        extractor_mode=config.eval.extractor,
        agent_prompt_profile=config.agent_prompt.profile,
    )


def _build_widesearch_context_token_estimator(config: WideSearchEvalConfig, model_client: Any) -> Callable[[list[ConversationMessage]], int] | None:
    """Build the context-token estimator iff preflight is enabled.

    Mirrors the web-search runner: when ``model.context.limit_detection`` is
    ``"estimated"``, build the configured estimator and wire it onto the model
    client (so the model client also estimates against it for max-tokens caps).
    Returns None when preflight is disabled or the estimator mode is
    ``"model_client"``.
    """
    if config.model.context.limit_detection != "estimated":
        return None
    args = _config_to_namespace(config)
    tools = build_web_search_tools(args)
    estimator = _build_context_token_estimator(args, [tool.to_tool_definition() for tool in tools])
    if estimator is not None and hasattr(model_client, "set_context_token_estimator"):
        cast("Any", model_client).set_context_token_estimator(estimator)
    return estimator


def trial_task_id(instance_id: str, trial_index: int) -> str:
    return f"{instance_id}__trial-{trial_index}"


def _select_response_for_scoring(result: OrchestrationResult, *, mode: ExtractorMode) -> tuple[str, Any]:
    """Pick the assistant text to score against and return ``(text, df)``.

    Tries the latest assistant message first — for normal completions that
    text equals ``result.output`` (or contains it as a boxed payload) and has
    the most diagnostic detail for ``raw_response_preview``. Falls back to
    ``result.output`` when the latest message doesn't yield a parseable
    DataFrame: this catches the retry-policy ``fallback_output_used`` path,
    where the orchestrator returns an earlier intermediate boxed answer in
    ``result.output`` while ``visible_conversation`` ends with the failed
    forced-final response.
    """
    last_assistant = ""
    for message in reversed(result.visible_conversation):
        if message.role == MessageRole.ASSISTANT and message.content:
            last_assistant = str(message.content)
            break

    df = extract_dataframe(last_assistant, mode=mode) if last_assistant else None
    if df is not None and not df.empty:
        return last_assistant, df

    output = str(result.output or "")
    if output and output != FORMAT_ERROR_MESSAGE:
        df_out = extract_dataframe(output, mode=mode)
        if df_out is not None and not df_out.empty:
            return output, df_out

    return last_assistant, df


def _scan_completed_trial_ids(task_logger: TaskLogger | None) -> set[str]:
    if task_logger is None:
        return set()
    completed = set()
    for attempt_id in task_logger.scan_completed_task_ids(tool_path=[ORCHESTRATOR_NAME]):
        base = attempt_id.rsplit("_attempt-", maxsplit=1)[0]
        if "__trial-" in base:
            completed.add(base)
    return completed


def _load_completed_result(task_logger: TaskLogger, base_id: str) -> OrchestrationResult | None:
    """Pick the highest-numbered attempt for ``base_id`` and load it."""
    attempt_ids = [
        aid for aid in task_logger.scan_completed_task_ids(tool_path=[ORCHESTRATOR_NAME]) if aid.rsplit("_attempt-", maxsplit=1)[0] == base_id
    ]
    if not attempt_ids:
        return None
    attempt_ids.sort(key=lambda aid: int(aid.rsplit("-", maxsplit=1)[-1]))
    trace = task_logger.load_trace(attempt_ids[-1], tool_path=[ORCHESTRATOR_NAME])
    if trace is None:
        return None
    return OrchestrationResult.from_trace(trace)


def _write_run_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    payload = {"started_at": datetime.now().astimezone().isoformat(timespec="seconds"), **metadata}
    (output_dir / "run_metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_trial_sidecar(scores_dir: Path, trial: TrialResult, eval_result: WideSearchEvalResult, raw_response: str | None) -> None:
    payload = {
        "instance_id": trial.instance_id,
        "trial_index": trial.trial_index,
        **eval_result.to_dict(),
        "raw_response_preview": (raw_response or "")[:2048],
    }
    path = scores_dir / f"{trial_task_id(trial.instance_id, trial.trial_index)}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_trial_id_from_score_path(path: Path) -> tuple[str, int] | None:
    if "__trial-" not in path.stem:
        return None
    instance_id, trial_suffix = path.stem.rsplit("__trial-", maxsplit=1)
    with contextlib.suppress(ValueError):
        return instance_id, int(trial_suffix)
    return None


def _score_sidecar_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    with contextlib.suppress(TypeError, ValueError):
        return float(value)
    return 0.0


def _is_retryable_score_sidecar(payload: dict[str, Any]) -> bool:
    msg = str(payload.get("msg") or "").strip().lower()
    retryable_prefixes = (
        "orchestrator error:",
        "transport error:",
        "connection error:",
        "timeout error:",
        "request error:",
    )
    return msg.startswith(retryable_prefixes)


def _quarantine_score_sidecar(path: Path, reason: str) -> None:
    if not path.exists():
        return
    quarantine_dir = path.parent / "_ignored" / reason
    try:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        target = quarantine_dir / path.name
        suffix = 1
        while target.exists():
            target = quarantine_dir / f"{path.stem}.{suffix}{path.suffix}"
            suffix += 1
        path.replace(target)
        logger.info("resume: moved ignored widesearch score sidecar %s to %s", path, target)
    except OSError:
        logger.warning("resume: failed to quarantine ignored widesearch score sidecar %s", path, exc_info=True)


def _quarantine_score_sidecars(scores_dir: Path, trial_ids: set[str], reason: str) -> None:
    for task_id in sorted(trial_ids):
        _quarantine_score_sidecar(scores_dir / f"{task_id}.json", reason)


def _trial_from_score_sidecar(path: Path, *, quarantine_ignored: bool = False) -> TrialResult | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        logger.warning("resume: failed to load widesearch score sidecar %s", path)
        return None
    if not isinstance(payload, dict):
        return None
    if _is_retryable_score_sidecar(payload):
        logger.info("resume: ignoring retryable widesearch score sidecar %s", path)
        if quarantine_ignored:
            _quarantine_score_sidecar(path, "retryable")
        return None

    parsed = _parse_trial_id_from_score_path(path)
    instance_id = payload.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id:
        instance_id = parsed[0] if parsed is not None else ""
    trial_index = payload.get("trial_index")
    if not isinstance(trial_index, int) or isinstance(trial_index, bool):
        trial_index = parsed[1] if parsed is not None else None
    if not instance_id or trial_index is None:
        logger.warning("resume: skipping malformed widesearch score sidecar %s", path)
        return None

    return TrialResult(
        instance_id=instance_id,
        trial_index=trial_index,
        score=_score_sidecar_float(payload, "score"),
        precision_by_row=_score_sidecar_float(payload, "precision_by_row"),
        recall_by_row=_score_sidecar_float(payload, "recall_by_row"),
        f1_by_row=_score_sidecar_float(payload, "f1_by_row"),
        precision_by_item=_score_sidecar_float(payload, "precision_by_item"),
        recall_by_item=_score_sidecar_float(payload, "recall_by_item"),
        f1_by_item=_score_sidecar_float(payload, "f1_by_item"),
        msg=str(payload.get("msg") or ""),
    )


def _load_scored_trials(scores_dir: Path, *, quarantine_ignored: bool = False) -> dict[str, TrialResult]:
    scored: dict[str, TrialResult] = {}
    if not scores_dir.exists():
        return scored
    for path in sorted(scores_dir.glob("*.json")):
        trial = _trial_from_score_sidecar(path, quarantine_ignored=quarantine_ignored)
        if trial is None:
            continue
        scored[trial_task_id(trial.instance_id, trial.trial_index)] = trial
    return scored


def _filter_scored_trials_to_schedule(
    scored_trials_by_id: dict[str, TrialResult],
    scheduled_trial_ids: set[str],
) -> dict[str, TrialResult]:
    return {task_id: trial for task_id, trial in scored_trials_by_id.items() if task_id in scheduled_trial_ids}


def _filter_scored_trials_to_completed_traces(
    scored_trials_by_id: dict[str, TrialResult],
    completed_trial_ids: set[str],
) -> dict[str, TrialResult]:
    return {task_id: trial for task_id, trial in scored_trials_by_id.items() if task_id in completed_trial_ids}


def _write_per_task_and_summary(
    output_dir: Path,
    trials: list[TrialResult],
    expected_trials: int,
    *,
    total_instances: int,
) -> dict[str, Any]:
    by_iid: dict[str, list[TrialResult]] = {}
    for t in trials:
        by_iid.setdefault(t.instance_id, []).append(t)
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
    (output_dir / "widesearch_per_task.json").write_text(json.dumps(per_task_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_payload = {
        "num_trials": expected_trials,
        # ``total_instances`` is the authoritative count of scheduled instances
        # (after ``max_tasks`` slicing), written once at run start. The
        # ``num_complete_instances + num_incomplete_instances`` fields below
        # only count instances that have already produced at least one trial
        # sidecar, so they cannot be used as the run's expected size for
        # in-progress dashboard progress.
        "total_instances": total_instances,
        "num_complete_instances": len(completed_per_task),
        "num_incomplete_instances": len(incomplete),
        "summary": summary,
        "leaderboard": leaderboard,
    }
    (output_dir / "widesearch_summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_payload


def _trial_from_eval(instance_id: str, trial_index: int, ev: WideSearchEvalResult) -> TrialResult:
    return TrialResult(
        instance_id=instance_id,
        trial_index=trial_index,
        score=ev.score,
        precision_by_row=ev.precision_by_row,
        recall_by_row=ev.recall_by_row,
        f1_by_row=ev.f1_by_row,
        precision_by_item=ev.precision_by_item,
        recall_by_item=ev.recall_by_item,
        f1_by_item=ev.f1_by_item,
        msg=ev.msg,
    )


async def run_widesearch_eval_loop(  # noqa: PLR0915
    *,
    queries: list[WideSearchQuery],
    output_dir: Path,
    resume: bool,
    expected_trials: int,
    max_concurrent: int,
    extractor_mode: ExtractorMode,
    prompts_profile: str,
    judge_config: WideSearchJudgeConfig,
    judge_max_concurrent: int,
    model_client: Any,
    orchestrator_factory: Callable[[Any, TaskLogger | None, Any], Any],
    context_token_estimator: Any = None,
    run_metadata: dict[str, Any] | None = None,
    run_type: str = "wide_search",
) -> dict[str, Any]:
    """Shared k-trial widesearch scheduling / scoring / aggregation loop.

    Decoupled from any specific agent stack: the caller supplies a prebuilt
    ``model_client`` and an ``orchestrator_factory`` invoked fresh per trial as
    ``orchestrator_factory(model_client, task_logger, context_token_estimator)``.
    The caller supplies the runtime-specific model client and orchestrator.
    Traces land under the ``ORCHESTRATOR_NAME`` subdir and dashboard artifacts
    are built with ``run_type`` (default ``wide_search``).
    """
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        # On a fresh run, drop traces and per-trial sidecars from any previous
        # run sharing this directory. The final dashboard rebuild scans these
        # files via build_eval_results_from_run, so leftovers from a longer
        # earlier run would otherwise pollute the new metrics.
        shutil.rmtree(output_dir / ORCHESTRATOR_NAME, ignore_errors=True)
        shutil.rmtree(output_dir / "widesearch_scores", ignore_errors=True)
    scores_dir = output_dir / "widesearch_scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    if not queries:
        msg = "no widesearch queries loaded — check benchmark.data_path / gold_dir"
        raise RuntimeError(msg)

    _write_run_metadata(output_dir, run_metadata or {})

    task_logger = TaskLogger(log_dir=str(output_dir), persist_json=True, background_persist_json=True)

    judge_client = WideSearchJudgeClient(judge_config)
    judge_semaphore = asyncio.Semaphore(max(1, judge_max_concurrent))
    judge_client.attach_semaphore(judge_semaphore)

    agent_semaphore = asyncio.Semaphore(max(1, max_concurrent))
    total_instances = len(queries)
    scheduled_trial_ids = {trial_task_id(query.instance_id, trial_index) for query in queries for trial_index in range(expected_trials)}

    completed_trial_ids: set[str] = set()
    if resume:
        completed_trial_ids = _scan_completed_trial_ids(task_logger)
        logger.info("resume: %d completed trials cached", len(completed_trial_ids))

    scored_trials_by_id = _load_scored_trials(scores_dir, quarantine_ignored=True) if resume else {}
    stale_score_ids = set(scored_trials_by_id) - scheduled_trial_ids
    if stale_score_ids:
        logger.info("resume: ignoring %d scored trials outside current schedule", len(stale_score_ids))
        _quarantine_score_sidecars(scores_dir, stale_score_ids, "outside-schedule")
        scored_trials_by_id = _filter_scored_trials_to_schedule(scored_trials_by_id, scheduled_trial_ids)
    scored_without_trace_ids = set(scored_trials_by_id) - completed_trial_ids
    if scored_without_trace_ids:
        logger.info(
            "resume: rerunning %d scored trials without completed traces",
            len(scored_without_trace_ids),
        )
        _quarantine_score_sidecars(scores_dir, scored_without_trace_ids, "missing-trace")
        scored_trials_by_id = _filter_scored_trials_to_completed_traces(scored_trials_by_id, completed_trial_ids)
    if scored_trials_by_id:
        logger.info("resume: %d trace-backed scored trials loaded from sidecars", len(scored_trials_by_id))

    trials: list[TrialResult] = list(scored_trials_by_id.values())
    trials_lock = asyncio.Lock()
    # Seed widesearch_summary.json so the dashboard sees the correct expected
    # task count from t=0, before any trial finishes. Without this seed, the
    # file only exists after the first trial completes, which makes early
    # progress reports look like "0 / 0".
    _write_per_task_and_summary(output_dir, trials, expected_trials, total_instances=total_instances)

    async def _run_trial(query: WideSearchQuery, trial_index: int) -> None:
        task_id = trial_task_id(query.instance_id, trial_index)
        if task_id in scored_trials_by_id:
            return
        result: OrchestrationResult | None = None
        if task_id in completed_trial_ids:
            result = _load_completed_result(task_logger, task_id)
        if result is None:
            orchestrator = orchestrator_factory(model_client, task_logger, context_token_estimator)
            async with agent_semaphore:
                try:
                    result = await orchestrator.run(task=query.query, task_id=task_id)
                except Exception as exc:
                    logger.exception("orchestrator run failed task_id=%s", task_id)
                    eval_result = WideSearchEvalResult(instance_id=query.instance_id, msg=f"orchestrator error: {exc!r}")
                    trial = _trial_from_eval(query.instance_id, trial_index, eval_result)
                    async with trials_lock:
                        trials.append(trial)
                        scored_trials_by_id[task_id] = trial
                        _write_trial_sidecar(scores_dir, trial, eval_result, raw_response=None)
                        _write_per_task_and_summary(output_dir, trials, expected_trials, total_instances=total_instances)
                    return

        raw_response, df = _select_response_for_scoring(result, mode=extractor_mode)
        eval_result = await aevaluate_single_query(query, df, client=judge_client, prompt_profile=prompts_profile)
        trial = _trial_from_eval(query.instance_id, trial_index, eval_result)
        async with trials_lock:
            trials.append(trial)
            scored_trials_by_id[task_id] = trial
            _write_trial_sidecar(scores_dir, trial, eval_result, raw_response=raw_response)
            _write_per_task_and_summary(output_dir, trials, expected_trials, total_instances=total_instances)

    pending = [asyncio.create_task(_run_trial(query, trial_index)) for query in queries for trial_index in range(expected_trials)]
    started_at = time.perf_counter()
    try:
        await asyncio.gather(*pending)
    finally:
        elapsed = time.perf_counter() - started_at
        logger.info("widesearch eval finished in %.1fs (judge_calls=%d)", elapsed, judge_client.total_calls)
        await judge_client.aclose()
        task_logger.close()

    summary_payload = _write_per_task_and_summary(output_dir, trials, expected_trials, total_instances=total_instances)
    try:
        build_eval_results_from_run(output_dir, run_type=run_type, force=True)
    except Exception:
        logger.exception("failed to build eval_results.json / dashboard artifacts for %s", output_dir)
    return summary_payload


def _default_run_metadata(config: WideSearchEvalConfig) -> dict[str, Any]:
    return {
        "schema_version": config.schema_version,
        "benchmark": "widesearch",
        "num_trials": config.benchmark.num_trials,
        "max_concurrent": config.benchmark.max_concurrent,
        "agent_prompt_profile": config.agent_prompt.profile,
        "judge_model": config.eval.judge_model,
        "extractor": config.eval.extractor,
        "prompts": config.eval.prompts,
    }


async def run_evaluation(config: WideSearchEvalConfig) -> dict[str, Any]:
    """Run the full k-trial widesearch evaluation with the native web-search agent."""
    queries = load_widesearch_queries(
        config.benchmark.data_path,
        config.benchmark.gold_dir,
        instance_ids=config.benchmark.instance_ids,
    )
    if config.benchmark.shuffle_tasks:
        random.Random(config.benchmark.shuffle_seed).shuffle(queries)
    if config.benchmark.max_tasks is not None:
        queries = queries[: config.benchmark.max_tasks]

    model_client = build_model_client(_config_to_namespace(config))
    context_token_estimator = _build_widesearch_context_token_estimator(config, model_client)

    judge_config = WideSearchJudgeConfig(
        judge_model=config.eval.judge_model,
        judge_base_url=config.eval.judge_base_url,
        judge_api_key_env=config.eval.judge_api_key_env,
        judge_max_tokens=config.eval.judge_max_tokens,
        judge_temperature=config.eval.judge_temperature,
        request_timeout=config.eval.request_timeout,
        max_retries=config.eval.max_retries,
        retryable_status_codes=config.eval.retryable_status_codes,
    )

    def _orchestrator_factory(mc: Any, tl: TaskLogger | None, est: Any) -> Any:
        return build_widesearch_orchestrator(
            model_client=mc,
            task_logger=tl,
            config=config,
            context_token_estimator=est,
        )

    return await run_widesearch_eval_loop(
        queries=queries,
        output_dir=Path(config.run.output_dir).expanduser(),
        resume=config.run.resume,
        expected_trials=config.benchmark.num_trials,
        max_concurrent=config.benchmark.max_concurrent,
        extractor_mode=config.eval.extractor,
        prompts_profile=config.eval.prompts,
        judge_config=judge_config,
        judge_max_concurrent=config.eval.judge_max_concurrent,
        model_client=model_client,
        orchestrator_factory=_orchestrator_factory,
        context_token_estimator=context_token_estimator,
        run_metadata=_default_run_metadata(config),
        run_type="wide_search",
    )
