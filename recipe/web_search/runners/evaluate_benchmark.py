# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from agentic.config import FormatErrorConfig, OrchestrationConfig, ToolArgumentRepairConfig, ToolManagerConfig
from agentic.contracts import ConversationMessage, FormatErrorStrategy
from agentic.evaluation.evaluator import BatchEvaluator
from agentic.model_assets import get_chat_template_render_config, normalize_openai_tool_call_arguments_for_chat_template, resolve_asset_uri
from agentic.model_clients import ModelRequestLogger, OpenAICompatibleModelClient, OpenAICompatibleModelClientConfig, RetryingModelClient
from agentic.model_clients.openai_client import DEFAULT_TRANSIENT_ENDPOINT_ERROR_STATUS_CODES
from agentic.model_clients.sglang_client import _chat_template_messages
from agentic.observability.task_logger import TaskLogger
from agentic.orchestration.task_orchestrator import OrchestrationResult
from agentic.tools import ToolManager
from agentic.tools.code_sandbox import (
    DEFAULT_E2B_TEMPLATE_ID,
    DEFAULT_SANDBOX_TIMEOUT,
    E2BSandboxOptions,
    create_code_sandbox_tools,
    validate_code_sandbox_environment,
)
from agentic.tools.schema_order import validate_rendered_tool_argument_order
from agentic.tools.web_search import (
    SIMPLE_SCRAPE_AND_EXTRACT_PARAMETERS,
    SIMPLE_WEB_SEARCH_PARAMETERS,
    LLMExtractionOptions,
    RawScrapeCacheOptions,
    RetryConfig,
    ScrapeBackendOptions,
    TimeoutConfig,
    create_scrape_and_extract_tool,
    create_web_search_tool,
)
from recipe.common.boxed_verifier import BoxedAnswerVerifier
from recipe.common.eval_results import TaskTimingRow, build_eval_results_payload, compute_non_model_overhead_s
from recipe.common.io_timing import write_json_with_timing, write_jsonl_with_timing, write_text_with_timing
from recipe.common.log_processing import write_dashboard_artifacts
from recipe.common.log_processing.finalized_run_guard import FinalizedRunResumeError, guard_resume_into_finalized_run
from recipe.common.log_processing.live_eval_results import build_agentic_timing_extras_from_index, scan_agentic_trace_index
from recipe.web_search.agent.context_compression_manager import ContextCompressionConfig, ContextCompressionManager
from recipe.web_search.agent.discard_all_manager import DiscardAllConfig, DiscardAllManager
from recipe.web_search.agent.orchestrator import WebSearchTaskOrchestrator
from recipe.web_search.agent.prompts import FORMAT_ERROR_MESSAGE, extract_boxed_content, generate_system_prompt, generate_user_prompt_template
from recipe.web_search.agent.runtime import WebSearchConversationConfig, WebSearchConversationRuntime
from recipe.web_search.config import (
    DEFAULT_SYSTEM_PROMPT_RENDER_EXTRACT_END,
    DEFAULT_SYSTEM_PROMPT_RENDER_EXTRACT_START,
    DEFAULT_SYSTEM_PROMPT_RENDER_TEMPLATE,
)
from recipe.web_search.eval.benchmark_dataset import BenchmarkDataset

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentic.model_clients.base import ModelClient

DEFAULT_MODEL = "native-tool-call-model"
DEFAULT_MAX_TURNS = 300
DEFAULT_KEEP_TOOL_RESULT = 5
DEFAULT_MAX_TASK_RETRIES = 5
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_MAX_CONTEXT_LENGTH = 262_144
DEFAULT_MAX_OUTPUT_TOKENS = 16_384
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_CONCURRENT = 10
_CONTEXT_TOKENIZER_EXACT_CACHE_SIZE = 128
_CONTEXT_TOKENIZER_NEAR_LIMIT_BAND = 65_536


def _git_value(args: list[str]) -> str | None:
    git_executable = shutil.which("git")
    if git_executable is None:
        return None
    try:
        result = subprocess.run(
            [git_executable, *args],
            cwd=Path(__file__).resolve().parents[3],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def _jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in sorted(vars(args).items())}


def _masked_env_flag(name: str) -> dict[str, object]:
    value = os.environ.get(name)
    return {"name": name, "present": bool(value), "length": len(value or "")}


def _parse_bool(value: str | bool | None) -> bool | None:  # noqa: FBT001
    if value is None or isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def _parse_optional_float(value: str | float | None) -> float | None:
    if value is None or isinstance(value, float):
        return value
    lowered = value.strip().lower()
    if lowered in {"none", "null"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid optional float value: {value!r}") from exc


def build_model_client(args: argparse.Namespace) -> ModelClient:
    extra_body: dict[str, Any] = {}
    if args.repetition_penalty is not None and args.repetition_penalty != 1.0:
        extra_body["repetition_penalty"] = args.repetition_penalty
    if args.request_extra_body_json:
        extra_body.update(json.loads(args.request_extra_body_json))

    inner = OpenAICompatibleModelClient(
        OpenAICompatibleModelClientConfig(
            model=args.model,
            api_base=args.base_url,
            api_key_env=args.api_key_env,
            context_window=args.max_context_length,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout_seconds=args.timeout,
            endpoint_profile=args.endpoint_profile,
            preserve_reasoning_content=args.preserve_reasoning_content,
            response_reasoning_fields=args.response_reasoning_fields,
            max_tokens_field=args.max_tokens_field,
            parallel_tool_calls=args.parallel_tool_calls,
            parse_embedded_thinking=args.parse_embedded_thinking,
            endpoint_error_exit_status_codes=args.endpoint_error_exit_status_codes or [],
            endpoint_error_exit_after_seconds=args.endpoint_error_exit_after_seconds,
            endpoint_connection_error_retry_wait_seconds=args.endpoint_connection_error_retry_wait_seconds,
            endpoint_error_retry_backoff_multiplier=args.endpoint_error_retry_backoff_multiplier,
            endpoint_error_retry_backoff_max_seconds=args.endpoint_error_retry_backoff_max_seconds,
            endpoint_error_retry_jitter=args.endpoint_error_retry_jitter,
            endpoint_error_respect_retry_after=args.endpoint_error_respect_retry_after,
            endpoint_retry_on_timeout=args.endpoint_retry_on_timeout,
            endpoint_retry_on_connection_error=args.endpoint_retry_on_connection_error,
            token_estimation_chars_per_token=args.token_estimation_chars_per_token,
            extra_body=extra_body,
            enable_tool_calls=True,
        )
    )
    retrying = RetryingModelClient(
        inner,
        max_retries=args.max_response_retries,
        retry_wait_seconds=args.retry_wait_seconds,
        context_safety_margin=args.context_safety_margin,
        cap_max_tokens_for_context=getattr(args, "context_limit_detection", "provider_error") == "estimated",
    )
    return retrying


def build_web_search_tools(
    args: argparse.Namespace,
    *,
    summary_llm_request_logger: ModelRequestLogger | None = None,
) -> list[Any]:
    if getattr(args, "disable_tools", False):
        return []
    scrape_backend_options = ScrapeBackendOptions(
        jina_base_url=args.jina_base_url,
        max_content_length=args.max_content_length,
        timeout=getattr(args, "scrape_timeout", None),
        retry=getattr(args, "scrape_retry", None),
        fallback_retry=getattr(args, "scrape_fallback_retry", None),
    )
    llm_extraction_options = LLMExtractionOptions(
        base_url=args.summary_llm_base_url,
        model_name=args.summary_llm_model_name,
        api_key=os.environ.get(args.summary_llm_api_key_env, ""),
        request_logger=summary_llm_request_logger,
        cache_enabled=getattr(args, "summary_llm_cache_enabled", False),
        timeout=getattr(args, "summary_llm_timeout", None),
        retry=getattr(args, "summary_llm_retry", None),
        max_input_chars=getattr(args, "summary_llm_max_input_chars", None),
        chunk_overlap_chars=getattr(args, "summary_llm_chunk_overlap_chars", 1_200),
        max_chunks=getattr(args, "summary_llm_max_chunks", 12),
        chunk_max_concurrent=getattr(args, "summary_llm_chunk_max_concurrent", 4),
        chunked_extraction=getattr(args, "summary_llm_chunked_extraction", True),
        chunk_strategy=getattr(args, "summary_llm_chunk_strategy", "single"),
        max_recursion_depth=getattr(args, "summary_llm_max_recursion_depth", 4),
        global_anchor_enabled=getattr(args, "summary_llm_global_anchor_enabled", False),
        chunk_envelope_mode=getattr(args, "summary_llm_chunk_envelope_mode", "strict_caveat"),
        csv_layer_b_enabled=getattr(args, "summary_llm_csv_layer_b_enabled", False),
        request_extra_body=(
            json.loads(args.summary_llm_request_extra_body_json) if getattr(args, "summary_llm_request_extra_body_json", None) else None
        ),
    )
    search_timeout = getattr(args, "search_timeout", None)
    search_retry = getattr(args, "search_retry", None)
    raw_scrape_cache_options = RawScrapeCacheOptions(
        enabled=getattr(args, "raw_scrape_cache_enabled", False),
        scope=getattr(args, "raw_scrape_cache_scope", "task"),
        provider=getattr(args, "raw_scrape_cache_provider", "web_search"),
        normalize_url=getattr(args, "raw_scrape_cache_normalize_url", False),
    )
    code_execs = _build_code_execs(args)
    return [
        create_web_search_tool(
            parameters=SIMPLE_WEB_SEARCH_PARAMETERS,
            serper_base_url=args.serper_base_url,
            timeout=search_timeout,
            retry=search_retry,
        ),
        create_scrape_and_extract_tool(
            parameters=SIMPLE_SCRAPE_AND_EXTRACT_PARAMETERS,
            scrape_backend_options=scrape_backend_options,
            llm_extraction_options=llm_extraction_options,
            raw_scrape_cache_options=raw_scrape_cache_options,
        ),
        *code_execs,
    ]


def _build_code_execs(args: argparse.Namespace) -> list[Any]:
    """Build the optional session-scoped code sandbox tools.

    Returns an empty list only when ``--code_exec_enabled false`` is set. When enabled,
    the E2B environment is validated up front so a missing dependency/API key
    fails fast instead of failing every tool call mid-run. The tools lazily
    create one shared sandbox on the first ``python_exec``/``shell_exec`` call in
    each task attempt, reuse it afterwards, and are registered without a
    ``server_name`` so they stay flat native function tools, matching the other
    web-search tools.
    """
    if not getattr(args, "code_exec_enabled", False):
        return []
    options = E2BSandboxOptions(
        template_id=getattr(args, "code_exec_template_id", None) or DEFAULT_E2B_TEMPLATE_ID,
        sandbox_timeout=getattr(args, "code_exec_sandbox_timeout", None) or DEFAULT_SANDBOX_TIMEOUT,
    )
    problems = validate_code_sandbox_environment(options=options)
    if problems:
        details = "\n".join(f"- {problem}" for problem in problems)
        raise RuntimeError(
            "Code exec is enabled (tools.code_exec.enabled) but unavailable in the agentic runtime:\n"
            f"{details}\n"
            "Install e2b-code-interpreter and set E2B_API_KEY, or disable tools.code_exec.enabled."
        )
    return [
        *create_code_sandbox_tools(
            options=options,
            retry=getattr(args, "code_exec_retry", None),
            max_calls_per_task=getattr(args, "code_exec_max_calls_per_task", None),
        )
    ]


def build_orchestrator(
    model_client: ModelClient,
    task_logger: TaskLogger | None,
    args: argparse.Namespace,
    *,
    summary_llm_request_logger: ModelRequestLogger | None = None,
) -> WebSearchTaskOrchestrator:
    prompt_profile = getattr(args, "prompt_profile", "default")
    tools = build_web_search_tools(args, summary_llm_request_logger=summary_llm_request_logger)
    # Discard-all's max-tool budget is handled by the orchestrator as the
    # boundary for entering the final no-discard attempt. It is not a hard
    # ToolManager cap: the batch that crosses the threshold is allowed to finish.
    tool_manager = ToolManager(
        tools=tools,
        config=ToolManagerConfig(
            argument_repair=ToolArgumentRepairConfig(enabled=True),
        ),
    )
    context_token_estimator = None
    if getattr(args, "context_limit_detection", "provider_error") == "estimated":
        context_token_estimator = _build_context_token_estimator(args, [tool.to_tool_definition() for tool in tools])
    if context_token_estimator is not None and hasattr(model_client, "set_context_token_estimator"):
        cast("Any", model_client).set_context_token_estimator(context_token_estimator)
    system_prompt = generate_system_prompt(
        getattr(args, "system_prompt_date", None),
        prompt_profile=prompt_profile,
        code_exec_enabled=getattr(args, "code_exec_enabled", False),
    )
    format_error = FormatErrorConfig(strategy=FormatErrorStrategy.IGNORE, keywords=[])
    max_output_tokens = getattr(args, "max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
    context_warning_threshold = getattr(args, "context_warning_threshold", None) or max_output_tokens
    context_compression_manager = _build_context_compression_manager(args)
    discard_all_manager = _build_discard_all_manager(args)
    # Mirror the config-model validator on the direct CLI path: discard-all and
    # Summary-style context_compression are alternative context-management
    # strategies and must not run together in the same orchestrator.
    if context_compression_manager is not None and discard_all_manager is not None:
        msg = "context_compression and discard_all are mutually exclusive; enable at most one context-management strategy."
        raise ValueError(msg)
    if discard_all_manager is not None and getattr(args, "context_limit_detection", "provider_error") == "estimated":
        msg = (
            "discard_all is incompatible with context_limit_detection=estimated: the estimated "
            "local context-limit preflight force-finalizes (or terminates) near the limit before "
            "discard-all can reset the trajectory. Use context_limit_detection=provider_error with discard_all."
        )
        raise ValueError(msg)
    # Discard-all supersedes the turn budget with its global tool-call cap so the
    # run can keep resetting context and exploring well past max_turns.
    effective_max_turns = args.max_turns
    if discard_all_manager is not None:
        effective_max_turns = discard_all_manager.max_tool_calls
    return WebSearchTaskOrchestrator(
        config=OrchestrationConfig(
            name="web-search-benchmark",
            conversation=WebSearchConversationConfig(
                system_prompt=system_prompt,
                user_prompt_template=generate_user_prompt_template(prompt_profile=prompt_profile),
                max_turns=effective_max_turns,
                context_window=args.max_context_length,
                context_safety_margin=args.context_safety_margin,
                min_tokens_for_generation=getattr(args, "min_tokens_for_generation", 2048),
                context_warning_threshold=context_warning_threshold,
                format_error=format_error,
                early_stop_announcement_prompt=None,
                final_response_prompt=None,
                keep_tool_result=args.keep_tool_result,
                tool_result_role=args.tool_result_role,
            ),
        ),
        model_client=model_client,
        tool_manager=tool_manager,
        task_logger=task_logger,
        conversation_runtime_class=WebSearchConversationRuntime,
        max_task_retries=args.max_task_retries,
        include_failure_summary_in_retry=args.include_failure_summary_in_retry,
        max_final_answer_attempts=args.max_final_answer_attempts,
        prompt_profile=prompt_profile,
        context_token_estimator=context_token_estimator,
        context_limit_preflight_enabled=getattr(args, "context_limit_detection", "provider_error") == "estimated",
        semantic_query_budget_enabled=getattr(args, "semantic_query_budget_enabled", False),
        semantic_query_budget_max_unique=getattr(args, "semantic_query_budget_max_unique", None),
        retry_attempt_provenance_enabled=getattr(args, "retry_attempt_provenance_enabled", False),
        retry_no_box_turn_limit_cap_enabled=getattr(args, "retry_no_box_turn_limit_cap_enabled", False),
        retry_no_box_turn_limit_cap=getattr(args, "retry_no_box_turn_limit_cap", 3),
        generation_limit_recovery_non_final_attempt=getattr(args, "generation_limit_recovery_non_final_attempt", "retry"),
        generation_limit_recovery_final_attempt=getattr(args, "generation_limit_recovery_final_attempt", "rollback"),
        self_verification_enabled=getattr(args, "self_verification_enabled", False),
        self_verification_max_reanswer_attempts=getattr(args, "self_verification_max_reanswer_attempts", 1),
        self_verification_max_turns=getattr(args, "self_verification_max_turns", None),
        self_verification_verdict_resample_max_attempts=getattr(args, "self_verification_verdict_resample_max_attempts", 3),
        rollback_storm_shadow_enabled=getattr(args, "rollback_storm_shadow_enabled", False),
        rollback_storm_duplicate_threshold=getattr(args, "rollback_storm_duplicate_threshold", 20),
        rollback_storm_tool_error_threshold=getattr(args, "rollback_storm_tool_error_threshold", 10),
        rollback_storm_late_turn_threshold=getattr(args, "rollback_storm_late_turn_threshold", 250),
        rollback_storm_preview_max_items=getattr(args, "rollback_storm_preview_max_items", 5),
        context_compression_manager=context_compression_manager,
        discard_all_manager=discard_all_manager,
        discard_all_last_attempt_max_turns=args.max_turns if discard_all_manager is not None else None,
    )


def _build_context_compression_manager(args: argparse.Namespace) -> ContextCompressionManager | None:
    if not getattr(args, "context_compression_enabled", False):
        return None
    base_url = getattr(args, "context_compression_llm_base_url", None)
    model_name = getattr(args, "context_compression_llm_model_name", None)
    if not base_url or not model_name:
        logger.warning(
            "Context compression enabled but llm.base_url/model_name missing; skipping",
        )
        return None
    api_key_env = getattr(args, "context_compression_llm_api_key_env", "sk-dummy")
    api_key = os.environ.get(api_key_env, "")
    cfg = ContextCompressionConfig(
        enabled=True,
        interval=getattr(args, "context_compression_interval", 10),
        recent_window=getattr(args, "context_compression_recent_window", 10),
    )
    return ContextCompressionManager(cfg, base_url=base_url, model_name=model_name, api_key=api_key)


def _build_discard_all_manager(args: argparse.Namespace) -> DiscardAllManager | None:
    if not getattr(args, "discard_all_enabled", False):
        return None
    cfg = DiscardAllConfig(
        enabled=True,
        trigger_ratio=getattr(args, "discard_all_trigger_ratio", 0.80),
        min_turns_between=getattr(args, "discard_all_min_turns_between", 3),
        max_tool_calls=getattr(args, "discard_all_max_tool_calls", 1800),
    )
    return DiscardAllManager(cfg)


def _estimate_context_tokens_by_chars(messages: list[ConversationMessage], *, chars_per_token: float) -> int:
    serialized = json.dumps(
        [message.to_model_message() for message in messages],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return max(1, int(len(serialized) / max(chars_per_token, 0.1)))


def _context_tokenizer_exact_threshold(args: argparse.Namespace) -> int:
    context_window = getattr(args, "max_context_length", None)
    if not isinstance(context_window, int) or context_window <= 0:
        return 0
    max_output_tokens = int(getattr(args, "max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS) or DEFAULT_MAX_OUTPUT_TOKENS)
    warning_threshold = int(getattr(args, "context_warning_threshold", None) or max_output_tokens)
    safety_margin = int(getattr(args, "context_safety_margin", 0) or 0)
    near_limit_band = max(_CONTEXT_TOKENIZER_NEAR_LIMIT_BAND, max_output_tokens + warning_threshold + safety_margin)
    return max(0, context_window - near_limit_band)


def _build_cheap_context_token_estimator(chars_per_token: float) -> Callable[[list[ConversationMessage]], int]:
    def cheap_estimate(messages: list[ConversationMessage]) -> int:
        return _estimate_context_tokens_by_chars(messages, chars_per_token=chars_per_token)

    cast("Any", cheap_estimate).token_estimator_includes_tools = False
    cast("Any", cheap_estimate).token_estimator_is_additive = True
    return cheap_estimate


def _build_context_token_estimator(args: argparse.Namespace, tools: list[dict[str, Any]]) -> Callable[[list[ConversationMessage]], int] | None:
    estimator_mode = str(getattr(args, "context_estimator", "model_client") or "model_client")
    tokenizer_path = getattr(args, "context_tokenizer_path", None)
    chars_per_token = float(getattr(args, "token_estimation_chars_per_token", 3.0) or 3.0)
    if estimator_mode == "model_client":
        return None
    if estimator_mode == "cheap":
        logger.info(
            "Using cheap additive context token estimator (chars_per_token=%.3f)",
            chars_per_token,
        )
        return _build_cheap_context_token_estimator(chars_per_token)
    if estimator_mode != "chat_template":
        msg = f"unsupported context estimator mode: {estimator_mode!r}"
        raise RuntimeError(msg)
    if not tokenizer_path:
        msg = "--context_estimator chat_template requires --context_tokenizer_path"
        raise RuntimeError(msg)

    resolved_path = _resolve_system_prompt_render_template(tokenizer_path)
    template_path = getattr(args, "system_prompt_render_template", None) or DEFAULT_SYSTEM_PROMPT_RENDER_TEMPLATE
    if str(template_path).lower() == "auto":
        render_config = get_chat_template_render_config(model=getattr(args, "model", None), base_url=getattr(args, "base_url", None))
        if render_config is None:
            msg = f"no chat-template asset profile found for model={getattr(args, 'model', None)!r}, base_url={getattr(args, 'base_url', None)!r}"
            raise RuntimeError(msg)
        template_path = render_config.asset_uri
    resolved_template_path = _resolve_system_prompt_render_template(str(template_path))
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        msg = f"--context_tokenizer_path requires transformers: {exc}"
        raise RuntimeError(msg) from exc
    tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=True)
    exact_threshold_tokens = _context_tokenizer_exact_threshold(args)
    exact_cache: OrderedDict[tuple[int, ...], int] = OrderedDict()

    def exact_estimate(messages: list[ConversationMessage]) -> int:
        cache_key = tuple(id(message) for message in messages)
        cached = exact_cache.get(cache_key)
        if cached is not None:
            exact_cache.move_to_end(cache_key)
            return cached
        rendered_messages = _chat_template_messages(messages)
        rendered = _render_chat_template_string(resolved_template_path, messages=rendered_messages, tools=tools)
        tokens = tokenizer(str(rendered), add_special_tokens=False)["input_ids"]
        token_count = len(tokens)
        exact_cache[cache_key] = token_count
        if len(exact_cache) > _CONTEXT_TOKENIZER_EXACT_CACHE_SIZE:
            exact_cache.popitem(last=False)
        return token_count

    def estimate(messages: list[ConversationMessage]) -> int:
        # Keep normal rollout checks cheap. Full chat-template tokenization scales
        # with the whole conversation and can otherwise block the asyncio loop.
        cheap_tokens = _estimate_context_tokens_by_chars(messages, chars_per_token=chars_per_token)
        if cheap_tokens < exact_threshold_tokens:
            return cheap_tokens
        return exact_estimate(messages)

    cast("Any", estimate).token_estimator_includes_tools = True
    cast("Any", estimate).token_estimator_is_additive = False
    logger.info(
        "Using hybrid context token estimator: tokenizer=%s template=%s exact_threshold_tokens=%d chars_per_token=%.3f",
        resolved_path,
        resolved_template_path,
        exact_threshold_tokens,
        chars_per_token,
    )
    return estimate


def _self_verification_settings_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "enabled": bool(getattr(args, "self_verification_enabled", False)),
        "max_reanswer_attempts": int(getattr(args, "self_verification_max_reanswer_attempts", 1) or 0),
        "verification_max_turns": getattr(args, "self_verification_max_turns", None),
        "verdict_resample_max_attempts": int(getattr(args, "self_verification_verdict_resample_max_attempts", 3) or 1),
    }


def _write_run_metadata(output_dir: Path, args: argparse.Namespace, system_prompt: str) -> None:
    git_status = _git_value(["status", "--short"])
    tool_names = ["web_search", "scrape_and_extract_info"]
    if getattr(args, "code_exec_enabled", False):
        tool_names.extend(["python_exec", "shell_exec"])
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "entrypoint": "recipe.web_search.runners.evaluate_benchmark",
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version},
        "git": {
            "commit": _git_value(["rev-parse", "HEAD"]),
            "branch": _git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "is_dirty": bool(git_status),
            "status_short": git_status,
        },
        "cli_args": _jsonable_args(args),
        "env": {
            name: _masked_env_flag(name) for name in ("OPENAI_API_KEY", "SERPER_API_KEY", "JINA_API_KEY", "SUMMARY_LLM_API_KEY", "JUDGE_API_KEY")
        },
        "artifacts": {"output_dir": str(output_dir), "system_prompt_chars": len(system_prompt)},
        "benchmark": {"name": args.benchmark_name, "data_path": str(args.data_path), "max_tasks": args.max_tasks},
        "tools": tool_names,
        "native_tool_calls": True,
        "self_verification": _self_verification_settings_from_args(args),
    }
    hydra_dir = output_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (hydra_dir / "config.yaml").write_text(json.dumps(metadata["cli_args"], indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _clear_non_resume_artifacts(output_dir: Path) -> None:
    """Remove generated run artifacts so non-resume reruns cannot show stale traces."""
    for directory_name in ("web-search-benchmark", "model_requests", "exact_match", "llm_judge"):
        shutil.rmtree(output_dir / directory_name, ignore_errors=True)
    for path in output_dir.glob("attempt_budget_*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            # Includes attempt_budget_sweep_ledger.jsonl: a non-resume rerun must start from a
            # clean ledger, else new entries append onto stale cross-run ones and a later resume
            # reconstructs from traces this rerun just deleted.
            path.unlink()
    for file_name in (
        "accuracy.txt",
        "benchmark_results.jsonl",
        "em_accuracy.txt",
        "em_results.jsonl",
        "em_summary.json",
        "eval_results.json",
        "eval_results_io_timing.json",
        "llm_judge_accuracy.txt",
        "llm_judge_results.jsonl",
        "llm_judge_summary.json",
        "run_metadata.json",
    ):
        path = output_dir / file_name
        if path.exists():
            path.unlink()


def _attempt_budget_dir(output_dir: Path, budget: int) -> Path:
    return output_dir / f"attempt_budget_{budget}"


def _trace_ref_for_result(output_dir: Path, *, task_id: str, budget: int, result: OrchestrationResult) -> dict[str, object] | None:
    trace_task_id = result.metadata.get("task_id") if isinstance(result.metadata, dict) else None
    if not isinstance(trace_task_id, str) or not trace_task_id:
        return None
    trace_path = output_dir / "web-search-benchmark" / f"{trace_task_id}.json"
    return {
        "task_id": task_id,
        "attempt": int(result.metadata.get("attempt_budget_actual_attempts") or budget),
        "trace_task_id": trace_task_id,
        "trace_path": os.path.relpath(trace_path, _attempt_budget_dir(output_dir, budget) / "web-search-benchmark"),
        "attempt_budget": budget,
        "actual_attempts": result.metadata.get("attempt_budget_actual_attempts"),
        "reused_from_budget": result.metadata.get("attempt_budget_reused_from_budget"),
    }


def _write_attempt_budget_trace_refs(output_dir: Path, budget: int, refs: list[dict[str, object]]) -> None:
    trace_dir = _attempt_budget_dir(output_dir, budget) / "web-search-benchmark"
    trace_dir.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "attempt_budget": budget, "refs": refs}
    (trace_dir / "trace_refs.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sweep_resume_ledger_path(output_dir: Path) -> Path:
    """Append-only ledger recording each task whose attempt-budget sweep fully completed.

    Each JSONL line maps one task_id -> the per-budget trace_task_ids produced by that
    task's single sweep rollout. Written incrementally as tasks finish so a crashed run
    can be resumed at task granularity (a sweep produces ALL budgets from one rollout, so
    the natural resume unit is the task_id, not the (task_id, budget) cell).
    """
    return output_dir / "attempt_budget_sweep_ledger.jsonl"


def _append_sweep_resume_ledger(output_dir: Path, task_id: str, budget_trace_ids: dict[int, str | None]) -> None:
    path = _sweep_resume_ledger_path(output_dir)
    record = {
        "task_id": task_id,
        "budget_trace_ids": {str(budget): trace_id for budget, trace_id in budget_trace_ids.items()},
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    # If a prior crash left a torn (newline-less) final line, prefix a newline so the
    # new record cannot fuse onto the partial one and corrupt both.
    prefix = ""
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                prefix = "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(prefix + line)


def _load_sweep_resume_ledger(output_dir: Path) -> dict[str, dict[int, str | None]]:
    """Reload the sweep resume ledger -> {task_id: {budget: trace_task_id}}.

    Last record for a task_id wins (idempotent re-writes are harmless).
    """
    path = _sweep_resume_ledger_path(output_dir)
    completed: dict[str, dict[int, str | None]] = {}
    if not path.exists():
        return completed
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a torn final line from a crash mid-write
        task_id = record.get("task_id")
        raw = record.get("budget_trace_ids")
        if not isinstance(task_id, str) or not isinstance(raw, dict):
            continue
        completed[task_id] = {int(budget): trace_id for budget, trace_id in raw.items()}
    return completed


def _build_completed_sweep_cache(
    task_logger: TaskLogger | None,
    output_dir: Path,
    orchestrator_name: str,
    sweep_max_budget: int,
    *,
    self_verification_settings: dict[str, object] | None = None,
) -> dict[str, dict[int, OrchestrationResult]]:
    """Reconstruct completed sweeps from the resume ledger + persisted traces.

    Returns {task_id: {budget: OrchestrationResult}} for every task whose sweep fully
    completed (all budgets present and loadable). Partially-written tasks are dropped so
    the caller re-runs them from scratch.
    """
    if task_logger is None:
        return {}
    ledger = _load_sweep_resume_ledger(output_dir)
    cache: dict[str, dict[int, OrchestrationResult]] = {}
    for task_id, budget_trace_ids in ledger.items():
        budget_map: dict[int, OrchestrationResult] = {}
        complete = True
        for budget in range(1, sweep_max_budget + 1):
            trace_id = budget_trace_ids.get(budget)
            if not trace_id:
                complete = False
                break
            trace = task_logger.load_trace(trace_id, tool_path=[orchestrator_name])
            if trace is None:
                complete = False
                break
            if not _trace_matches_self_verification_settings(trace.metadata, self_verification_settings):
                complete = False
                break
            result = OrchestrationResult.from_trace(trace)
            output = str(result.output or "")
            boxed = extract_boxed_content(output)
            if boxed:
                result = result.model_copy(update={"output": boxed})
            budget_map[budget] = result
        if complete:
            cache[task_id] = budget_map
    logger.info("Resume(sweep): %d/%d completed tasks loaded from ledger", len(cache), len(ledger))
    return cache


def _extract_rendered_system_block(rendered_prompt: str, *, start_token: str | None, end_token: str | None) -> str:
    if not start_token or not end_token:
        return ""
    start = rendered_prompt.find(start_token)
    if start < 0:
        return ""
    content_start = start + len(start_token)
    if content_start < len(rendered_prompt) and rendered_prompt[content_start] == "\n":
        content_start += 1
    end = rendered_prompt.find(end_token, content_start)
    if end < 0:
        return ""
    return rendered_prompt[content_start:end].rstrip()


def _render_chat_template_system_prompt(
    *,
    template_name_or_path: str | None,
    extract_start: str | None,
    extract_end: str | None,
    model: str | None,
    base_url: str | None,
    system_prompt: str,
    user_prompt: str,
    tools: list[dict[str, Any]],
) -> tuple[dict[str, Any], str | None]:
    """Render the initial request through a tokenizer chat template and extract the system block."""
    if not template_name_or_path:
        return {"status": "skipped", "reason": "system prompt render template was not provided"}, "system prompt render template was not provided"
    endpoint_profile = None
    configured_template = template_name_or_path
    if template_name_or_path.lower() == "auto":
        render_config = get_chat_template_render_config(model=model, base_url=base_url)
        if render_config is None:
            reason = f"no chat-template asset profile found for model={model!r}, base_url={base_url!r}"
            return {"status": "skipped", "reason": reason, "system_prompt_render_template": configured_template}, reason
        endpoint_profile = render_config.endpoint_profile
        template_name_or_path = render_config.asset_uri
        extract_start = extract_start or render_config.extract_start
        extract_end = extract_end or render_config.extract_end
    try:
        template_name_or_path = _resolve_system_prompt_render_template(template_name_or_path)
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "system_prompt_render_template": configured_template,
        }, f"{type(exc).__name__}: {exc}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        rendered = _render_chat_template_string(template_name_or_path, messages=messages, tools=tools)
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "system_prompt_render_template": configured_template,
            "resolved_system_prompt_render_template": template_name_or_path,
            "endpoint_profile": endpoint_profile,
        }, f"{type(exc).__name__}: {exc}"

    if not isinstance(rendered, str):
        reason = f"chat template returned {type(rendered).__name__}, expected str"
        return {"status": "error", "reason": reason}, reason

    system_block = _extract_rendered_system_block(rendered, start_token=extract_start, end_token=extract_end)
    if not system_block:
        reason = f"rendered prompt did not contain the configured system block delimiters: {extract_start!r} ... {extract_end!r}"
        return {
            "artifact": "chat_template_render.json",
            "status": "error",
            "reason": reason,
            "source": "chat_template_render",
            "endpoint_profile": endpoint_profile,
            "system_prompt_render_template": configured_template,
            "resolved_system_prompt_render_template": template_name_or_path,
            "extract_start": extract_start,
            "extract_end": extract_end,
            "rendered_text": rendered,
        }, reason
    order_error = validate_rendered_tool_argument_order(rendered, tools)
    if order_error is not None:
        return {
            "artifact": "chat_template_render.json",
            "status": "error",
            "reason": order_error,
            "source": "chat_template_render",
            "endpoint_profile": endpoint_profile,
            "system_prompt_render_template": configured_template,
            "resolved_system_prompt_render_template": template_name_or_path,
            "extract_start": extract_start,
            "extract_end": extract_end,
            "rendered_text": rendered,
            "extracted_system_block": system_block,
        }, order_error
    return {
        "artifact": "chat_template_render.json",
        "status": "success",
        "source": "chat_template_render",
        "endpoint_profile": endpoint_profile,
        "system_prompt_render_template": configured_template,
        "resolved_system_prompt_render_template": template_name_or_path,
        "extract_start": extract_start,
        "extract_end": extract_end,
        "rendered_text": rendered,
        "extracted_system_block": system_block,
    }, None


def _render_chat_template_string(template_name_or_path: str, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    path = Path(template_name_or_path)
    if path.is_file() and path.suffix in {".jinja", ".json"}:
        return _render_asset_chat_template(path, messages=messages, tools=tools)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        msg = f"transformers is not installed: {exc}"
        raise RuntimeError(msg) from exc
    tokenizer = AutoTokenizer.from_pretrained(template_name_or_path, trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=True,
    )
    return str(rendered)


def _render_asset_chat_template(path: Path, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    try:
        from jinja2 import Environment
    except ImportError as exc:
        msg = f"jinja2 is not installed: {exc}"
        raise RuntimeError(msg) from exc
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        template = data.get("chat_template")
        if not isinstance(template, str) or not template:
            msg = f"{path} does not contain a non-empty chat_template"
            raise ValueError(msg)
    else:
        template = path.read_text(encoding="utf-8")

    def _tojson(value: Any, **kwargs: Any) -> str:
        kwargs.setdefault("ensure_ascii", False)
        return json.dumps(value, **kwargs)

    env = Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])  # noqa: S701
    env.filters["tojson"] = _tojson
    return env.from_string(template).render(
        messages=normalize_openai_tool_call_arguments_for_chat_template(messages),
        tools=tools or None,
        add_generation_prompt=True,
        enable_thinking=True,
        clear_thinking=False,
        preserve_thinking=True,
    )


def _resolve_system_prompt_render_template(value: str) -> str:
    if value.startswith("asset://"):
        return str(resolve_asset_uri(value))
    if value.startswith("axis_model://"):
        root = os.environ.get("AXIS_MODEL_DIR")
        if not root:
            raise RuntimeError(f"{value} requires AXIS_MODEL_DIR to be set")
        return str(Path(root).expanduser() / value[len("axis_model://") :].lstrip("/"))
    if value.startswith("repo://"):
        return str(Path(__file__).resolve().parents[3] / value[len("repo://") :].lstrip("/"))
    return value


def _build_completed_task_cache(
    task_logger: TaskLogger | None,
    orchestrator_name: str,
    max_task_retries: int,
    *,
    retry_no_box_turn_limit_cap_enabled: bool = False,
    retry_no_box_turn_limit_cap: int | None = None,
    self_verification_settings: dict[str, object] | None = None,
) -> dict[str, OrchestrationResult]:
    if task_logger is None:
        return {}
    completed_ids = task_logger.scan_completed_task_ids(tool_path=[orchestrator_name])
    max_attempts = max_task_retries + 1
    base_to_attempts: dict[str, list[str]] = {}
    for attempt_id in completed_ids:
        parts = attempt_id.rsplit("_attempt-", maxsplit=1)
        if len(parts) == 2:
            base_to_attempts.setdefault(parts[0], []).append(attempt_id)

    cache: dict[str, OrchestrationResult] = {}
    for base_id, attempt_ids in base_to_attempts.items():
        attempt_ids.sort(key=lambda aid: int(aid.rsplit("-", maxsplit=1)[-1]))
        trace = task_logger.load_trace(attempt_ids[-1], tool_path=[orchestrator_name])
        if trace is None:
            continue
        if not _trace_matches_self_verification_settings(trace.metadata, self_verification_settings):
            continue
        result = OrchestrationResult.from_trace(trace)
        output = str(result.output or "")
        boxed = extract_boxed_content(output)
        if boxed:
            result = result.model_copy(update={"output": boxed})
            output = boxed
        cap_completed = retry_no_box_turn_limit_cap_enabled and bool(result.metadata.get("retry_blocked_by_no_box_turn_limit_cap"))
        if cap_completed and retry_no_box_turn_limit_cap is not None:
            cap_completed = result.metadata.get("retry_no_box_turn_limit_cap") == retry_no_box_turn_limit_cap
        if (output and output != FORMAT_ERROR_MESSAGE) or len(attempt_ids) >= max_attempts or cap_completed:
            cache[base_id] = result
    logger.info("Resume: %d/%d tasks loaded from existing traces", len(cache), len(base_to_attempts))
    return cache


def _trace_matches_self_verification_settings(
    metadata: dict[str, Any],
    expected_settings: dict[str, object] | None,
) -> bool:
    expected = expected_settings or {"enabled": False}
    if not bool(expected.get("enabled")):
        self_verification = metadata.get("self_verification")
        if isinstance(self_verification, dict):
            settings = self_verification.get("settings")
            if isinstance(settings, dict):
                return not bool(settings.get("enabled"))
        return True
    self_verification = metadata.get("self_verification")
    if not isinstance(self_verification, dict):
        return False
    settings = self_verification.get("settings")
    return isinstance(settings, dict) and settings == expected


def _extract_timing_from_result(result: Any) -> tuple[float | None, float | None, dict[str, float], dict[str, int], dict[str, int], dict[str, int]]:
    info = getattr(result, "info", None) or {}
    metadata = getattr(result, "metadata", None) or {}
    inference_timing = info.get("inference_timing") if isinstance(info, dict) else None
    model_client_s = None
    if isinstance(inference_timing, dict):
        ms = inference_timing.get("model_client_elapsed_ms_sum")
        if isinstance(ms, (int, float)) and ms > 0:
            model_client_s = float(ms) / 1000.0
    total_elapsed_s = None
    total_ms = metadata.get("total_elapsed_ms") if isinstance(metadata, dict) else None
    if isinstance(total_ms, (int, float)) and total_ms > 0:
        total_elapsed_s = float(total_ms) / 1000.0

    latency_by_tool: dict[str, float] = {}
    count_by_tool: dict[str, int] = {}
    success_by_tool: dict[str, int] = {}
    failed_by_tool: dict[str, int] = {}
    tool_metrics = info.get("tool_metrics") if isinstance(info, dict) else None
    if isinstance(tool_metrics, dict):
        for tool_name, metrics in tool_metrics.items():
            if tool_name == "overall_metrics" or not isinstance(metrics, dict):
                continue
            latencies = metrics.get("latency_ms", []) or []
            if isinstance(latencies, list):
                numeric = [float(value) for value in latencies if isinstance(value, (int, float)) and not isinstance(value, bool)]
                if numeric:
                    latency_by_tool[tool_name] = sum(numeric)
                    count_by_tool[tool_name] = len(numeric)
            if isinstance(metrics.get("num_success"), int):
                success_by_tool[tool_name] = metrics["num_success"]
            if isinstance(metrics.get("num_failed"), int):
                failed_by_tool[tool_name] = metrics["num_failed"]
    return model_client_s, total_elapsed_s, latency_by_tool, count_by_tool, success_by_tool, failed_by_tool


def _build_timing_row(
    *,
    task_id: str,
    idx: int,
    result: Any,
    cached: bool,
    task_elapsed_s: float | None,
    orchestrator_build_elapsed_s: float | None,
    group_completion_elapsed_s: float | None,
) -> TaskTimingRow:
    model_s, cached_total_s, latency_by_tool, count_by_tool, success_by_tool, failed_by_tool = _extract_timing_from_result(result)
    effective_task_s = task_elapsed_s if task_elapsed_s is not None else cached_total_s
    tool_sum_ms = sum(latency_by_tool.values()) if latency_by_tool else 0.0
    tool_count = sum(count_by_tool.values()) if count_by_tool else 0
    return TaskTimingRow(
        task_id=task_id,
        idx=idx,
        num_turns=result.num_turns,
        finish_reason=result.reason,
        cached=cached,
        task_elapsed_s=effective_task_s,
        orchestrator_build_elapsed_s=orchestrator_build_elapsed_s,
        model_client_elapsed_s=model_s,
        group_completion_elapsed_s=group_completion_elapsed_s,
        non_model_overhead_s=compute_non_model_overhead_s(effective_task_s, model_s, tool_sum_ms),
        tool_latency_ms_sum=tool_sum_ms if latency_by_tool else None,
        tool_latency_ms_mean=(tool_sum_ms / tool_count) if tool_count else None,
        tool_latency_ms_max=max(latency_by_tool.values()) if latency_by_tool else None,
        tool_count=tool_count if latency_by_tool else None,
        tool_latency_ms_by_tool=latency_by_tool,
        tool_count_by_tool=count_by_tool,
        tool_success_by_tool=success_by_tool,
        tool_failed_by_tool=failed_by_tool,
    )


async def _score_and_record(
    *,
    task_id: str,
    result: OrchestrationResult,
    item: Any,
    verifier: BoxedAnswerVerifier,
    evaluator: BatchEvaluator | None,
    results: list[dict[str, object]],
    timing: TaskTimingRow,
    state: dict[str, int],
    score_cache: dict[tuple[object, ...], float] | None = None,
) -> None:
    extracted = str(result.output or "") or None
    score_cache_key = _score_cache_key(item=item, extracted=extracted)
    cached_score = score_cache.get(score_cache_key) if score_cache is not None else None
    if cached_score is not None:
        score = cached_score
    elif item.label:
        score = verifier.score(item.label, extracted)
    else:
        score = 0.0
    if score_cache is not None and cached_score is None:
        score_cache[score_cache_key] = float(score)

    timing.score = float(score)
    record: dict[str, object] = {
        "task_id": task_id,
        "output": extracted,
        "ground_truth": item.label,
        "score": score,
        "reason": result.reason,
        "num_turns": result.num_turns,
        "task_elapsed_s": timing.task_elapsed_s,
        "model_client_elapsed_s": timing.model_client_elapsed_s,
        "non_model_overhead_s": timing.non_model_overhead_s,
        "tool_latency_ms_sum": timing.tool_latency_ms_sum,
        "tool_count": timing.tool_count,
        "cached": timing.cached,
    }
    results.append(record)
    if evaluator is not None:
        evaluator.log_result(task_id, eval_name="exact_match", ground_truth=item.label, prediction=extracted, score=score)
    if score > 0:
        state["correct"] += 1
    state["total"] += 1
    logger.info(
        "Task %s: extracted=%s, score=%.1f, running accuracy=%.2f%% (%d/%d)",
        task_id,
        (extracted or "")[:80],
        score,
        state["correct"] / state["total"] * 100,
        state["correct"],
        state["total"],
    )


def _score_cache_key(*, item: Any, extracted: str | None) -> tuple[object, ...]:
    metadata = getattr(item, "metadata", None)
    try:
        metadata_key = json.dumps(metadata, sort_keys=True, default=str, ensure_ascii=False)
    except TypeError:
        metadata_key = repr(metadata)
    return (
        getattr(item, "label", None),
        extracted,
        getattr(item, "problem", None),
        metadata_key,
    )


async def run_evaluation(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915
    script_start = time.perf_counter()
    script_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    dataset = BenchmarkDataset(data_path=args.data_path, max_items=args.max_tasks)
    dataset.load()
    if args.shuffle_tasks:
        random.Random(args.shuffle_seed).shuffle(dataset.items)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        try:
            guard_resume_into_finalized_run(
                output_dir,
                expected_task_count=len(dataset.items),
                force=args.force_resume_finalized_run,
            )
        except FinalizedRunResumeError as exc:
            raise SystemExit(str(exc)) from None
    if not args.resume:
        _clear_non_resume_artifacts(output_dir)

    model_build_start = time.perf_counter()
    model_client = build_model_client(args)
    model_client_build_elapsed_s = time.perf_counter() - model_build_start
    if args.disable_file_writes:
        args.no_logging = True

    inference_request_logger = None
    summary_llm_request_logger = None
    if not args.no_logging and not args.no_model_request_logging:
        inference_request_logger = ModelRequestLogger(output_dir, name="inference")
        summary_llm_request_logger = ModelRequestLogger(output_dir, name="summary_llm")
        model_client.set_request_logger(inference_request_logger)

    task_logger = TaskLogger(log_dir=str(output_dir), persist_json=True, background_persist_json=True) if not args.no_logging else None
    evaluator = BatchEvaluator(eval_dir=output_dir, trace_log_dir=output_dir / "web-search-benchmark") if not args.no_logging else None
    prototype_orchestrator = build_orchestrator(model_client, task_logger, args, summary_llm_request_logger=summary_llm_request_logger)
    system_prompt = prototype_orchestrator.config.conversation.system_prompt or ""
    _write_run_metadata(output_dir, args, system_prompt)
    if task_logger is not None:
        task_logger.save_config({"tool_names": prototype_orchestrator.tool_manager.list_tool_names()}, "env_info.json")
        if dataset.items:
            initial_user_prompt = prototype_orchestrator.config.conversation.user_prompt_template.format(task=dataset.items[0].problem)
            chat_template_render, chat_template_error = _render_chat_template_system_prompt(
                template_name_or_path=args.system_prompt_render_template,
                extract_start=args.system_prompt_render_extract_start,
                extract_end=args.system_prompt_render_extract_end,
                model=args.model,
                base_url=args.base_url,
                system_prompt=system_prompt,
                user_prompt=initial_user_prompt,
                tools=prototype_orchestrator.tool_manager.list_tool_definitions(),
            )
            task_logger.save_config(chat_template_render, "chat_template_render.json")
            if chat_template_render.get("status") == "success":
                compatibility_artifact = dict(chat_template_render)
                compatibility_artifact["artifact"] = "chat_template_system_prompt.json"
                compatibility_artifact["rendered_text"] = chat_template_render.get("extracted_system_block", "")
                task_logger.save_config(compatibility_artifact, "chat_template_system_prompt.json")
            if chat_template_error:
                logger.warning("Could not render chat-template system prompt: %s", chat_template_error)

    verifier = BoxedAnswerVerifier()

    completed_cache = None
    completed_sweep_cache: dict[str, dict[int, OrchestrationResult]] = {}
    _sweep_max_budget_for_resume = max(1, int(args.max_task_retries or 0) + 1)
    self_verification_settings = _self_verification_settings_from_args(args)
    if args.resume:
        if args.disable_file_writes:
            raise ValueError("--resume requires persisted traces; do not combine it with --disable_file_writes")
        if args.attempt_budget_sweep_enabled:
            completed_sweep_cache = _build_completed_sweep_cache(
                task_logger,
                output_dir,
                prototype_orchestrator.config.name,
                _sweep_max_budget_for_resume,
                self_verification_settings=self_verification_settings,
            )
        else:
            completed_cache = _build_completed_task_cache(
                task_logger,
                prototype_orchestrator.config.name,
                args.max_task_retries,
                retry_no_box_turn_limit_cap_enabled=getattr(args, "retry_no_box_turn_limit_cap_enabled", False),
                retry_no_box_turn_limit_cap=getattr(args, "retry_no_box_turn_limit_cap", None),
                self_verification_settings=self_verification_settings,
            )

    semaphore = asyncio.Semaphore(max(1, args.max_concurrent))
    results_lock = asyncio.Lock()
    results: list[dict[str, object]] = []
    timing_rows: list[TaskTimingRow] = []
    state = {"correct": 0, "total": 0}
    sweep_enabled = bool(args.attempt_budget_sweep_enabled)
    # Consistent with the non-sweep path: max_task_retries counts retries, so the
    # largest attempt budget is retries + 1 and equals a full non-sweep run.
    sweep_max_budget = int(args.max_task_retries) + 1
    sweep_results: dict[int, list[dict[str, object]]] = {}
    sweep_timing_rows: dict[int, list[TaskTimingRow]] = {}
    sweep_states: dict[int, dict[str, int]] = {}
    sweep_trace_refs: dict[int, list[dict[str, object]]] = {}
    if sweep_enabled:
        for budget in range(1, sweep_max_budget + 1):
            sweep_results[budget] = results if budget == sweep_max_budget else []
            sweep_timing_rows[budget] = timing_rows if budget == sweep_max_budget else []
            sweep_states[budget] = state if budget == sweep_max_budget else {"correct": 0, "total": 0}
            sweep_trace_refs[budget] = []
    eval_start = time.perf_counter()

    async def _run_item(idx: int, item: Any) -> None:  # noqa: PLR0915
        source = json.loads(item.source) if isinstance(item.source, str) else item.source
        task_id = str(source.get("task_id", idx) if isinstance(source, dict) else idx)
        cached = False
        task_elapsed_s = None
        orchestrator_build_elapsed_s = None
        if completed_cache and task_id in completed_cache:
            result = completed_cache[task_id]
            cached = True
        elif sweep_enabled:
            # budget -> OrchestrationResult for this task, from cache (resume) or a fresh rollout.
            budget_results_map: dict[int, OrchestrationResult] = {}
            if task_id in completed_sweep_cache:
                budget_results_map = dict(completed_sweep_cache[task_id])
                cached = True
            else:
                build_start = time.perf_counter()
                orchestrator = build_orchestrator(model_client, task_logger, args, summary_llm_request_logger=summary_llm_request_logger)
                orchestrator_build_elapsed_s = time.perf_counter() - build_start
                async with semaphore:
                    try:
                        task_start = time.perf_counter()
                        budget_map = await orchestrator.run_attempt_budget_sweep(task=item.problem, task_id=task_id)
                        task_elapsed_s = time.perf_counter() - task_start
                    except Exception as exc:
                        logger.exception("Task %s failed", task_id)
                        async with results_lock:
                            for budget in range(1, sweep_max_budget + 1):
                                sweep_results[budget].append({"task_id": task_id, "output": None, "score": 0.0, "error": str(exc)})
                                sweep_states[budget]["total"] += 1
                        return
                budget_results_map = {
                    budget: budget_map[budget].result for budget in range(1, sweep_max_budget + 1) if budget_map.get(budget) is not None
                }
                # Persist the per-budget trace ids for this task BEFORE scoring, so a crash
                # mid-scoring still leaves this fully-rolled-out task resumable.
                if task_logger is not None and len(budget_results_map) == sweep_max_budget:
                    budget_trace_ids = {
                        budget: (res.metadata.get("task_id") if isinstance(res.metadata, dict) else None)
                        for budget, res in budget_results_map.items()
                    }
                    if all(budget_trace_ids.get(b) for b in range(1, sweep_max_budget + 1)):
                        _append_sweep_resume_ledger(output_dir, task_id, budget_trace_ids)
            async with results_lock:
                score_cache: dict[tuple[object, ...], float] = {}
                for budget in range(1, sweep_max_budget + 1):
                    budget_orchestration_result = budget_results_map.get(budget)
                    if budget_orchestration_result is None:
                        sweep_results[budget].append({"task_id": task_id, "output": None, "score": 0.0, "error": "missing attempt-budget result"})
                        sweep_states[budget]["total"] += 1
                        continue
                    ref = _trace_ref_for_result(output_dir, task_id=task_id, budget=budget, result=budget_orchestration_result)
                    if ref is not None:
                        sweep_trace_refs[budget].append(ref)
                    timing_row = _build_timing_row(
                        task_id=task_id,
                        idx=idx,
                        result=budget_orchestration_result,
                        cached=cached,
                        task_elapsed_s=task_elapsed_s if budget == sweep_max_budget else None,
                        orchestrator_build_elapsed_s=orchestrator_build_elapsed_s if budget == sweep_max_budget else None,
                        group_completion_elapsed_s=time.perf_counter() - eval_start,
                    )
                    timing_row.num_attempts = int(budget_orchestration_result.metadata.get("attempt_budget_actual_attempts") or budget)
                    sweep_timing_rows[budget].append(timing_row)
                    await _score_and_record(
                        task_id=task_id,
                        result=budget_orchestration_result,
                        item=item,
                        verifier=verifier,
                        evaluator=evaluator if budget == sweep_max_budget else None,
                        results=sweep_results[budget],
                        timing=timing_row,
                        state=sweep_states[budget],
                        score_cache=score_cache,
                    )
            return
        else:
            build_start = time.perf_counter()
            orchestrator = build_orchestrator(model_client, task_logger, args, summary_llm_request_logger=summary_llm_request_logger)
            orchestrator_build_elapsed_s = time.perf_counter() - build_start
            async with semaphore:
                try:
                    task_start = time.perf_counter()
                    result = await orchestrator.run(task=item.problem, task_id=task_id)
                    task_elapsed_s = time.perf_counter() - task_start
                except Exception as exc:
                    logger.exception("Task %s failed", task_id)
                    async with results_lock:
                        results.append({"task_id": task_id, "output": None, "score": 0.0, "error": str(exc)})
                        state["total"] += 1
                    return
        timing_row = _build_timing_row(
            task_id=task_id,
            idx=idx,
            result=result,
            cached=cached,
            task_elapsed_s=task_elapsed_s,
            orchestrator_build_elapsed_s=orchestrator_build_elapsed_s,
            group_completion_elapsed_s=time.perf_counter() - eval_start,
        )
        async with results_lock:
            timing_rows.append(timing_row)
            await _score_and_record(
                task_id=task_id,
                result=result,
                item=item,
                verifier=verifier,
                evaluator=evaluator,
                results=results,
                timing=timing_row,
                state=state,
            )

    pending = [asyncio.create_task(_run_item(idx, item)) for idx, item in enumerate(dataset.items)]
    try:
        await asyncio.gather(*pending)
    except (KeyboardInterrupt, asyncio.CancelledError):
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise

    total_eval_elapsed_s = time.perf_counter() - eval_start
    task_logger_close_drain_elapsed_s = 0.0
    if task_logger is not None:
        close_start = time.perf_counter()
        task_logger.close()
        task_logger_close_drain_elapsed_s = time.perf_counter() - close_start
    if inference_request_logger is not None:
        inference_request_logger.close()
    if summary_llm_request_logger is not None:
        summary_llm_request_logger.close()

    accuracy = state["correct"] / state["total"] if state["total"] else 0.0
    results_write_s = write_jsonl_with_timing(output_dir / "benchmark_results.jsonl", results)
    accuracy_write_s = write_text_with_timing(output_dir / "accuracy.txt", f"{accuracy * 100:.2f}%\n")
    timing_rows.sort(key=lambda row: row.idx if row.idx is not None else 0)
    start_to_end_elapsed_s = time.perf_counter() - script_start
    script_ended_at = datetime.now().astimezone().isoformat(timespec="seconds")
    dashboard_timing_extras_elapsed_s = 0.0
    dashboard_timing_extras: dict[str, Any] = {}
    if not args.disable_file_writes:
        dashboard_timing_extras_start = time.perf_counter()
        trace_index = scan_agentic_trace_index(output_dir, ("web-search-benchmark",))
        dashboard_timing_extras = build_agentic_timing_extras_from_index(trace_index)
        dashboard_timing_extras_elapsed_s = time.perf_counter() - dashboard_timing_extras_start
    eval_results_payload = build_eval_results_payload(
        timing_rows,
        source="web_search",
        model=args.model,
        num_items=state["total"],
        script_started_at=script_started_at,
        script_ended_at=script_ended_at,
        start_to_end_elapsed_s=start_to_end_elapsed_s,
        total_eval_elapsed_s=total_eval_elapsed_s,
        model_client_build_elapsed_s=model_client_build_elapsed_s,
        task_logger_close_drain_elapsed_s=task_logger_close_drain_elapsed_s,
        trace_file_writes_disabled=args.disable_file_writes,
        extra={
            "correct": state["correct"],
            "accuracy": accuracy,
            "max_concurrent": args.max_concurrent,
            "benchmark_results_write_elapsed_s": results_write_s,
            "accuracy_write_elapsed_s": accuracy_write_s,
            "dashboard_timing_extras_elapsed_s": dashboard_timing_extras_elapsed_s,
            "run_metadata_path": str(output_dir / "run_metadata.json"),
            # When sweeping, the run dir itself is the largest attempt budget
            # (== a full non-sweep run); no separate attempt_budget_<max> dir is
            # written, so mark the run dir so dashboards can identify the budget.
            **({"attempt_budget_sweep": True, "attempt_budget": sweep_max_budget, "max_attempt_budget": sweep_max_budget} if sweep_enabled else {}),
            **dashboard_timing_extras,
        },
    )
    eval_results_write_s = write_json_with_timing(output_dir / "eval_results.json", eval_results_payload, timing_key="eval_results_write_elapsed_s")
    write_json_with_timing(
        output_dir / "eval_results_io_timing.json",
        {
            "benchmark_results_write_elapsed_s": results_write_s,
            "accuracy_write_elapsed_s": accuracy_write_s,
            "eval_results_write_elapsed_s": eval_results_write_s,
            "dashboard_timing_extras_elapsed_s": dashboard_timing_extras_elapsed_s,
            "task_logger_close_drain_elapsed_s": task_logger_close_drain_elapsed_s,
            "trace_file_writes_disabled": args.disable_file_writes,
        },
        timing_key="io_timing_write_elapsed_s",
    )
    write_dashboard_artifacts(output_dir, run_type="web_search")
    if sweep_enabled:
        run_metadata_path = output_dir / "run_metadata.json"
        # The largest budget equals a full non-sweep run and is already written
        # to the run dir itself, so only emit subdirs for the reduced budgets
        # 1..max-1. This avoids duplicating the max-budget artifacts on disk.
        for budget in range(1, sweep_max_budget):
            budget_dir = _attempt_budget_dir(output_dir, budget)
            budget_dir.mkdir(parents=True, exist_ok=True)
            if run_metadata_path.exists():
                shutil.copy2(run_metadata_path, budget_dir / "run_metadata.json")
            _write_attempt_budget_trace_refs(output_dir, budget, sweep_trace_refs[budget])
            budget_state = sweep_states[budget]
            budget_accuracy = budget_state["correct"] / budget_state["total"] if budget_state["total"] else 0.0
            budget_results_write_s = write_jsonl_with_timing(budget_dir / "benchmark_results.jsonl", sweep_results[budget])
            budget_accuracy_write_s = write_text_with_timing(budget_dir / "accuracy.txt", f"{budget_accuracy * 100:.2f}%\n")
            budget_rows = sweep_timing_rows[budget]
            budget_rows.sort(key=lambda row: row.idx if row.idx is not None else 0)
            budget_dashboard_timing_extras_elapsed_s = 0.0
            budget_dashboard_timing_extras: dict[str, Any] = {}
            if not args.disable_file_writes:
                budget_dashboard_start = time.perf_counter()
                budget_trace_index = scan_agentic_trace_index(budget_dir, ("web-search-benchmark",))
                budget_dashboard_timing_extras = build_agentic_timing_extras_from_index(budget_trace_index)
                budget_dashboard_timing_extras_elapsed_s = time.perf_counter() - budget_dashboard_start
            budget_payload = build_eval_results_payload(
                budget_rows,
                source="web_search_attempt_budget_sweep",
                model=args.model,
                num_items=budget_state["total"],
                script_started_at=script_started_at,
                script_ended_at=script_ended_at,
                start_to_end_elapsed_s=start_to_end_elapsed_s,
                total_eval_elapsed_s=total_eval_elapsed_s,
                model_client_build_elapsed_s=model_client_build_elapsed_s,
                task_logger_close_drain_elapsed_s=task_logger_close_drain_elapsed_s,
                trace_file_writes_disabled=args.disable_file_writes,
                extra={
                    "correct": budget_state["correct"],
                    "accuracy": budget_accuracy,
                    "max_concurrent": args.max_concurrent,
                    "attempt_budget_sweep": True,
                    "attempt_budget": budget,
                    "max_attempt_budget": sweep_max_budget,
                    "benchmark_results_write_elapsed_s": budget_results_write_s,
                    "accuracy_write_elapsed_s": budget_accuracy_write_s,
                    "dashboard_timing_extras_elapsed_s": budget_dashboard_timing_extras_elapsed_s,
                    "run_metadata_path": str(budget_dir / "run_metadata.json"),
                    **budget_dashboard_timing_extras,
                },
            )
            budget_eval_write_s = write_json_with_timing(
                budget_dir / "eval_results.json",
                budget_payload,
                timing_key="eval_results_write_elapsed_s",
            )
            write_json_with_timing(
                budget_dir / "eval_results_io_timing.json",
                {
                    "benchmark_results_write_elapsed_s": budget_results_write_s,
                    "accuracy_write_elapsed_s": budget_accuracy_write_s,
                    "eval_results_write_elapsed_s": budget_eval_write_s,
                    "dashboard_timing_extras_elapsed_s": budget_dashboard_timing_extras_elapsed_s,
                    "task_logger_close_drain_elapsed_s": task_logger_close_drain_elapsed_s,
                    "trace_file_writes_disabled": args.disable_file_writes,
                },
                timing_key="io_timing_write_elapsed_s",
            )
            write_dashboard_artifacts(budget_dir, run_type="web_search")
    logger.info("Evaluation complete: accuracy=%.2f%% (%d/%d)", accuracy * 100, state["correct"], state["total"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description="Native structured-tool web-search benchmark evaluation")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--system_prompt_render_template",
        default=os.environ.get("SYSTEM_PROMPT_RENDER_TEMPLATE", DEFAULT_SYSTEM_PROMPT_RENDER_TEMPLATE),
        help=(
            "Optional local/HuggingFace tokenizer/template path used only to render the logged dashboard "
            "system-prompt artifact. Use 'auto' to select the repo-managed asset for the base model. It does not affect inference."
        ),
    )
    parser.add_argument(
        "--system_prompt_render_extract_start",
        default=os.environ.get("SYSTEM_PROMPT_RENDER_EXTRACT_START", DEFAULT_SYSTEM_PROMPT_RENDER_EXTRACT_START),
        help="Start delimiter for extracting the dashboard system-prompt block from the rendered chat-template string.",
    )
    parser.add_argument(
        "--system_prompt_render_extract_end",
        default=os.environ.get("SYSTEM_PROMPT_RENDER_EXTRACT_END", DEFAULT_SYSTEM_PROMPT_RENDER_EXTRACT_END),
        help="End delimiter for extracting the dashboard system-prompt block from the rendered chat-template string.",
    )
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--endpoint_profile", default=None)
    parser.add_argument(
        "--prompt_profile",
        choices=["default", "deepsearchqa", "livebrowsecomp", "livebrowsecomp_notools"],
        default="default",
        help="Prompt and web-search tool schema profile.",
    )
    parser.add_argument("--system_prompt_date", default=None, help="Override the date rendered into prompt-profile system prompts.")
    parser.add_argument("--preserve_reasoning_content", type=_parse_bool, default=True)
    parser.add_argument("--response_reasoning_fields", nargs="*", default=None)
    parser.add_argument("--max_tokens_field", default=None)
    parser.add_argument("--parallel_tool_calls", type=_parse_bool, default=None)
    parser.add_argument("--parse_embedded_thinking", type=_parse_bool, default=None)
    parser.add_argument(
        "--endpoint_error_exit_status_codes",
        nargs="*",
        type=int,
        default=list(DEFAULT_TRANSIENT_ENDPOINT_ERROR_STATUS_CODES),
    )
    parser.add_argument("--endpoint_error_exit_after_seconds", type=_parse_optional_float, default=300.0)
    parser.add_argument("--endpoint_connection_error_retry_wait_seconds", type=float, default=60.0)
    parser.add_argument("--endpoint_error_retry_backoff_multiplier", type=float, default=2.0)
    parser.add_argument("--endpoint_error_retry_backoff_max_seconds", type=float, default=300.0)
    parser.add_argument("--endpoint_error_retry_jitter", type=_parse_bool, default=True)
    parser.add_argument("--endpoint_error_respect_retry_after", type=_parse_bool, default=True)
    parser.add_argument("--endpoint_retry_on_timeout", type=_parse_bool, default=True)
    parser.add_argument("--endpoint_retry_on_connection_error", type=_parse_bool, default=True)
    parser.add_argument("--request_extra_body_json", default=None)
    parser.add_argument("--summary_llm_request_extra_body_json", default=None)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--repetition_penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    parser.add_argument("--max_output_tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--max_context_length", type=int, default=DEFAULT_MAX_CONTEXT_LENGTH)
    parser.add_argument("--context_safety_margin", type=int, default=4096)
    parser.add_argument("--min_tokens_for_generation", type=int, default=2048)
    parser.add_argument("--context_warning_threshold", type=int, default=None)
    parser.add_argument(
        "--context_limit_detection",
        choices=["estimated", "provider_error"],
        default="provider_error",
        help=(
            "provider_error skips local context-budget preflight and relies on provider context-length errors or "
            "finish_reason=length. estimated runs local token-estimate preflight before model calls; use it only "
            "when provider errors are unavailable or unreliable."
        ),
    )
    parser.add_argument(
        "--context_estimator",
        choices=["model_client", "cheap", "chat_template"],
        default="model_client",
        help=(
            "Context preflight estimator. cheap enables additive incremental tracking; "
            "chat_template uses full chat-template tokenization near the limit and can add client-side latency."
        ),
    )
    parser.add_argument(
        "--context_tokenizer_path",
        default=None,
        help=("Optional local/HuggingFace tokenizer path. Required only when --context_estimator=chat_template."),
    )
    parser.add_argument("--token_estimation_chars_per_token", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max_response_retries", type=int, default=10)
    parser.add_argument("--retry_wait_seconds", type=float, default=30.0)
    parser.add_argument("--max_turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--keep_tool_result", type=int, default=DEFAULT_KEEP_TOOL_RESULT)
    parser.add_argument(
        "--tool_result_role",
        choices=["tool", "user"],
        default="tool",
        help="Role used for model-visible tool-result messages; internal traces still use role=tool.",
    )
    parser.add_argument("--max_task_retries", type=int, default=DEFAULT_MAX_TASK_RETRIES)
    parser.add_argument("--include_failure_summary_in_retry", action="store_true")
    parser.add_argument(
        "--max_final_answer_attempts",
        type=int,
        default=1,
        help="Maximum total final-answer attempts.",
    )
    parser.add_argument(
        "--attempt_budget_sweep_enabled",
        type=_parse_bool,
        default=False,
        help="Run one task rollout that emits per-attempt-budget results up to --max_task_retries.",
    )
    parser.add_argument(
        "--semantic_query_budget_enabled",
        type=_parse_bool,
        default=False,
        help="Enable opt-in search-query budget gating for web-search attempts.",
    )
    parser.add_argument(
        "--semantic_query_budget_max_unique",
        type=int,
        default=None,
        help="Maximum normalized unique google_search/web_search query keys per attempt when semantic query budget gating is enabled.",
    )
    parser.add_argument(
        "--retry_attempt_provenance_enabled",
        type=_parse_bool,
        default=False,
        help="Enable default-off retry attempt provenance metadata; does not change retry behavior or add model calls.",
    )
    parser.add_argument(
        "--retry_no_box_turn_limit_cap_enabled",
        type=_parse_bool,
        default=False,
        help="Enable V2+ default-off retry cap after repeated no-box turn-limit attempts.",
    )
    parser.add_argument(
        "--retry_no_box_turn_limit_cap",
        type=int,
        default=3,
        help="Consecutive no-box turn-limit attempts allowed before the V2+ retry cap blocks another attempt.",
    )
    parser.add_argument(
        "--generation_limit_recovery_non_final_attempt",
        choices=["retry", "rollback"],
        default="retry",
        help=(
            "Behavior on non-final attempts for provider context-limit errors and finish_reason=length: "
            "retry ends the attempt and uses the outer retry quota; rollback rolls back the latest assistant/tool exchange in-place."
        ),
    )
    parser.add_argument(
        "--generation_limit_recovery_final_attempt",
        choices=["rollback", "terminate"],
        default="rollback",
        help=(
            "Behavior on the final attempt for provider context-limit errors and finish_reason=length: "
            "rollback repeatedly rolls back older assistant/tool exchanges and re-generates; terminate stops immediately."
        ),
    )
    parser.add_argument(
        "--self_verification_enabled",
        type=_parse_bool,
        default=False,
        help="After a boxed answer is produced, run a tool-using self-verification loop before accepting it.",
    )
    parser.add_argument(
        "--self_verification_max_reanswer_attempts",
        type=int,
        default=1,
        help="Separate post-verification re-answer attempts available after incorrect/unparseable verdicts.",
    )
    parser.add_argument(
        "--self_verification_max_turns",
        type=int,
        default=None,
        help="Maximum tool-use turns for each self-verification attempt. Defaults to --max_turns when omitted.",
    )
    parser.add_argument(
        "--self_verification_verdict_resample_max_attempts",
        type=int,
        default=3,
        help="Maximum final-verdict resamples after an unparseable self-verification verdict.",
    )
    parser.add_argument(
        "--rollback_storm_shadow_enabled",
        type=_parse_bool,
        default=False,
        help="Enable default-off rollback-storm shadow telemetry; does not change retry behavior or add model calls.",
    )
    parser.add_argument(
        "--rollback_storm_duplicate_threshold",
        type=int,
        default=20,
        help="Duplicate-query rollback count threshold used only by rollback-storm shadow telemetry.",
    )
    parser.add_argument(
        "--rollback_storm_tool_error_threshold",
        type=int,
        default=10,
        help="Tool-error rollback count threshold used only by rollback-storm shadow telemetry.",
    )
    parser.add_argument(
        "--rollback_storm_late_turn_threshold",
        type=int,
        default=250,
        help="Turn threshold used only by rollback-storm shadow telemetry late-storm flags.",
    )
    parser.add_argument(
        "--rollback_storm_preview_max_items",
        type=int,
        default=5,
        help="Maximum bounded preview items stored by rollback-storm shadow telemetry.",
    )
    parser.add_argument(
        "--context_compression_enabled",
        type=_parse_bool,
        default=False,
        help="Enable periodic context compression that summarizes earlier turns into a single [context_summary] block.",
    )
    parser.add_argument(
        "--context_compression_interval",
        type=int,
        default=10,
        help="Fire context compression every N assistant turns.",
    )
    parser.add_argument(
        "--context_compression_recent_window",
        type=int,
        default=10,
        help="Number of recent turns kept intact after compression (1 turn ~ 2 messages).",
    )
    parser.add_argument("--context_compression_llm_base_url", default=None)
    parser.add_argument("--context_compression_llm_model_name", default=None)
    parser.add_argument("--context_compression_llm_api_key_env", default="sk-dummy")
    parser.add_argument(
        "--discard_all_enabled",
        type=_parse_bool,
        default=False,
        help=(
            "Enable discard-all context management: when observed prompt tokens exceed "
            "trigger_ratio of the context window, discard all prior tool-call history and "
            "reopen the task from a clean context. Mutually exclusive with context_compression."
        ),
    )
    parser.add_argument(
        "--discard_all_trigger_ratio",
        type=float,
        default=0.80,
        help="Discard-all fires when observed prompt tokens exceed this fraction of the context window.",
    )
    parser.add_argument(
        "--discard_all_min_turns_between",
        type=int,
        default=3,
        help="Minimum assistant turns between two discard-all resets (anti-thrash guard).",
    )
    parser.add_argument(
        "--discard_all_max_tool_calls",
        type=int,
        default=1800,
        help=(
            "Global tool-call cap for the attempt; does not reset on a discard and supersedes "
            "max_turns when discard-all is enabled. On hit, forces a graceful final answer."
        ),
    )
    parser.add_argument("--serper_base_url", default=None)
    parser.add_argument("--disable_tools", action="store_true", help="Register zero tools (pure-model baseline: no search/scrape/python).")
    parser.add_argument("--jina_base_url", default=None)
    parser.add_argument(
        "--code_exec_enabled",
        type=_parse_bool,
        default=False,
        help=(
            "Register session-scoped python_exec and shell_exec tools "
            "(lazy one shared E2B sandbox per task attempt). Default: false. "
            "Requires e2b-code-interpreter + E2B_API_KEY."
        ),
    )
    parser.add_argument(
        "--code_exec_sandbox_timeout",
        type=int,
        default=600,
        help="E2B sandbox keep-alive timeout in seconds for python_exec/shell_exec.",
    )
    parser.add_argument(
        "--code_exec_template_id",
        default=None,
        help="Optional E2B sandbox template ID for python_exec/shell_exec. Defaults to the built-in template.",
    )
    parser.add_argument(
        "--code_exec_max_calls_per_task",
        type=int,
        default=None,
        help="Optional per-tool cap on python_exec and shell_exec executions per task.",
    )
    parser.add_argument(
        "--code_exec_retry_json",
        default=None,
        help="JSON RetryConfig for E2B sandbox calls (429/transient retry). Absent => the tool's default 429-mitigation policy.",
    )
    parser.add_argument("--max_content_length", type=int, default=409_600)
    parser.add_argument(
        "--raw_scrape_cache_enabled",
        type=_parse_bool,
        default=False,
        help="Enable default-off successful non-empty raw page-content cache for scrape_and_extract_info.",
    )
    parser.add_argument(
        "--raw_scrape_cache_scope",
        choices=["task"],
        default="task",
        help="Scope for raw page-content cache. Currently only task-local cache is supported.",
    )
    parser.add_argument(
        "--raw_scrape_cache_provider",
        choices=["web_search"],
        default="web_search",
        help="Provider provenance recorded with raw page-content cache entries.",
    )
    parser.add_argument(
        "--raw_scrape_cache_normalize_url",
        type=_parse_bool,
        default=False,
        help="Explicitly normalize URL case and query ordering before hashing raw scrape cache keys.",
    )
    parser.add_argument("--summary_llm_base_url", default=None)
    parser.add_argument("--summary_llm_model_name", default=None)
    parser.add_argument("--summary_llm_api_key_env", default="SUMMARY_LLM_API_KEY")
    parser.add_argument(
        "--summary_llm_max_input_chars",
        type=int,
        default=None,
        help="Approximate prompt character budget before scrape summary switches to chunk-map-reduce. Use 0 to disable.",
    )
    parser.add_argument(
        "--summary_llm_chunk_overlap_chars",
        type=int,
        default=1_200,
        help="Character overlap between adjacent long-content summary chunks.",
    )
    parser.add_argument(
        "--summary_llm_max_chunks",
        type=int,
        default=12,
        help="Soft target chunk count for long-content summary logging. The per-chunk input budget is preserved.",
    )
    parser.add_argument(
        "--summary_llm_chunk_max_concurrent",
        type=int,
        default=4,
        help="Maximum concurrent chunk-map summary LLM calls for one long scraped document.",
    )
    parser.add_argument(
        "--summary_llm_chunked_extraction",
        type=_parse_bool,
        default=True,
        help=(
            "Master switch for chunked map-reduce extraction of long scraped pages. "
            "When false, long content uses the single-shot path (main behavior) regardless of max_input_chars."
        ),
    )
    parser.add_argument(
        "--summary_llm_chunk_strategy",
        choices=["single", "recursive"],
        default="single",
        help=(
            "Chunk reduce strategy when chunked extraction is on: 'single' (map every chunk then one reduce call) "
            "or 'recursive' (recursively reduce concatenated findings until they fit the budget; no silent tail loss)."
        ),
    )
    parser.add_argument(
        "--summary_llm_max_recursion_depth",
        type=int,
        default=4,
        help="Recursive strategy only: max reduce-recursion depth before falling back to a lossless concatenation.",
    )
    parser.add_argument(
        "--summary_llm_global_anchor_enabled",
        type=_parse_bool,
        default=False,
        help="Enable global anchor extraction in summary LLM structured scraping.",
    )
    parser.add_argument(
        "--summary_llm_chunk_envelope_mode",
        choices=["strict", "soft", "strict_caveat"],
        default="strict_caveat",
        help="Chunk envelope mode for summary LLM extraction prompts.",
    )
    parser.add_argument(
        "--summary_llm_csv_layer_b_enabled",
        type=_parse_bool,
        default=False,
        help="Enable Layer-B CSV extraction fallbacks in summary LLM structured scraping.",
    )
    parser.add_argument(
        "--summary_llm_cache_enabled",
        type=_parse_bool,
        default=False,
        help=(
            "Enable the shared LLM extraction cache for summary extraction calls. "
            "Default is disabled because persistent cache writes can add client-side latency under high concurrency."
        ),
    )
    # Timeout/retry policies are passed as JSON blobs (see _retry.TimeoutConfig /
    # RetryConfig). Absent => the tool layer's behavior-preserving defaults.
    parser.add_argument("--summary_llm_timeout_json", default=None)
    parser.add_argument("--summary_llm_retry_json", default=None)
    parser.add_argument("--scrape_timeout_json", default=None)
    parser.add_argument("--scrape_retry_json", default=None)
    parser.add_argument("--scrape_fallback_retry_json", default=None)
    parser.add_argument("--search_timeout_json", default=None)
    parser.add_argument("--search_retry_json", default=None)
    parser.add_argument("--data_path", required=True, help="Benchmark data file or directory")
    parser.add_argument(
        "--benchmark_name",
        choices=["browsecomp", "browsecomp_zh", "gaia", "hle", "deepsearchqa", "livebrowsecomp"],
        default="browsecomp",
    )
    parser.add_argument("--output_dir", default="logs/web_search_infer/agentic")
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--shuffle_tasks", action="store_true")
    parser.add_argument("--shuffle_seed", type=int, default=None)
    parser.add_argument("--no_logging", action="store_true")
    parser.add_argument(
        "--model_request_logging",
        action="store_false",
        dest="no_model_request_logging",
        default=argparse.SUPPRESS,
        help=(
            "Enable exact model request/response JSONL logging. Default is disabled because "
            "serializing full payloads can add client-side latency under high concurrency."
        ),
    )
    parser.add_argument(
        "--no_model_request_logging",
        action="store_true",
        dest="no_model_request_logging",
        default=argparse.SUPPRESS,
        help="Keep exact model request/response JSONL logging disabled. This is the default.",
    )
    parser.set_defaults(no_model_request_logging=True)
    parser.add_argument("--disable_file_writes", "--no_file_writes", action="store_true")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--max_concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force_resume_finalized_run",
        "--force-resume-finalized-run",
        action="store_true",
        help="Allow --resume to write into a run_dir that already has complete benchmark/eval final artifacts.",
    )
    args = parser.parse_args(argv)
    _attach_runtime_retry_configs(args)
    return args


def _attach_runtime_retry_configs(args: argparse.Namespace) -> None:
    """Decode the JSON timeout/retry blobs into runtime objects.

    ``build_web_search_tools`` consumes. ``None`` leaves the attribute as a
    runtime ``None`` so the tool layer applies its behavior-preserving defaults.
    """

    def _timeout(raw: str | None) -> TimeoutConfig | None:
        return TimeoutConfig.from_json(raw) if raw else None

    def _retry(raw: str | None) -> RetryConfig | None:
        return RetryConfig.from_json(raw) if raw else None

    args.summary_llm_timeout = _timeout(args.summary_llm_timeout_json)
    args.summary_llm_retry = _retry(args.summary_llm_retry_json)
    args.scrape_timeout = _timeout(args.scrape_timeout_json)
    args.scrape_retry = _retry(args.scrape_retry_json)
    args.scrape_fallback_retry = _retry(args.scrape_fallback_retry_json)
    args.search_timeout = _timeout(args.search_timeout_json)
    args.search_retry = _retry(args.search_retry_json)
    args.code_exec_retry = _retry(getattr(args, "code_exec_retry_json", None))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
