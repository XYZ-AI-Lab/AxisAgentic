# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from recipe.common.log_processing.finalized_run_guard import (
    FinalizedRunResumeError,
    count_dataset_items,
    guard_resume_into_finalized_run,
)
from recipe.web_search.config import WebSearchEvalConfig, dump_web_search_eval_config, load_web_search_eval_config
from recipe.web_search.eval.dsqa_utils import run_dsqa_f1_verify
from recipe.web_search.eval.llm_judge import resolve_judge_max_tokens_for_benchmark

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PATH_SCHEME_ENV = {
    "axis_data": "AXIS_DATA_DIR",
    "axis_log": "AXIS_LOG_DIR",
    "axis_model": "AXIS_MODEL_DIR",
}


def _load_env_file(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    print(f"Loading env from: {path}")
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


def _resolve_env_file_path(config_path: Path, env_file: str | None) -> Path | None:
    if not env_file:
        return None
    path = Path(env_file).expanduser()
    if path.is_absolute():
        return path
    repo_candidate = PROJECT_ROOT / path
    if repo_candidate.exists() or path.as_posix() == ".envs/.env":
        return repo_candidate
    config_candidate = config_path.expanduser().resolve().parent / path
    return config_candidate if config_candidate.exists() else repo_candidate


def _env_path_root(env_name: str, label: str) -> Path:
    value = os.environ.get(env_name)
    if not value:
        raise ValueError(f"{label} uses {env_name}, but {env_name} is not set.")
    return Path(value).expanduser()


def _resolve_path(value: str | Path, *, label: str) -> Path:
    raw = str(value)
    for scheme, env_name in _PATH_SCHEME_ENV.items():
        prefix = f"{scheme}://"
        if raw.startswith(prefix):
            return _env_path_root(env_name, label) / raw[len(prefix) :].lstrip("/")
    if raw.startswith("repo://"):
        return PROJECT_ROOT / raw[len("repo://") :].lstrip("/")
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _provider_value(value: str | None, env_names: tuple[str, ...], label: str) -> str:
    resolved = value or next((os.environ[name] for name in env_names if os.environ.get(name)), None)
    if not resolved:
        raise ValueError(f"{label} is required: set it in YAML or provider env {', '.join(env_names)}")
    return resolved


def _resolve_config(config: WebSearchEvalConfig) -> WebSearchEvalConfig:
    resolved = config.model_copy(deep=True)
    judge_max_tokens_is_default = "judge_max_tokens" not in config.judge.model_fields_set
    resolved.model.openai_model = _provider_value(resolved.model.openai_model, ("OPENAI_MODEL",), "model.openai_model")
    resolved.model.openai_base_url = _provider_value(resolved.model.openai_base_url, ("OPENAI_BASE_URL",), "model.openai_base_url")
    resolved.run.system_prompt_render_template = resolved.run.system_prompt_render_template or os.environ.get("SYSTEM_PROMPT_RENDER_TEMPLATE")
    resolved.run.system_prompt_render_extract_start = resolved.run.system_prompt_render_extract_start or os.environ.get(
        "SYSTEM_PROMPT_RENDER_EXTRACT_START"
    )
    resolved.run.system_prompt_render_extract_end = resolved.run.system_prompt_render_extract_end or os.environ.get(
        "SYSTEM_PROMPT_RENDER_EXTRACT_END"
    )
    if resolved.run.system_prompt_render_template and resolved.run.system_prompt_render_template.startswith(("axis_model://", "repo://")):
        resolved.run.system_prompt_render_template = str(
            _resolve_path(resolved.run.system_prompt_render_template, label="run.system_prompt_render_template")
        )
    if not resolved.benchmark.data_path:
        raise ValueError("benchmark.data_path is required")
    resolved.benchmark.data_path = str(_resolve_path(resolved.benchmark.data_path, label="benchmark.data_path"))
    resolved.run.output_dir = str(_resolve_path(resolved.run.output_dir, label="run.output_dir"))
    resolved.judge.judge_model = resolved.judge.judge_model or os.environ.get("JUDGE_MODEL") or resolved.model.openai_model
    resolved.judge.judge_base_url = resolved.judge.judge_base_url or os.environ.get("JUDGE_BASE_URL") or resolved.model.openai_base_url
    if resolved.judge.api_key_env == "JUDGE_API_KEY" and "JUDGE_API_KEY" not in os.environ:
        resolved.judge.api_key_env = resolved.model.api_key_env
    resolved.tools.serper_base_url = resolved.tools.serper_base_url or os.environ.get("SERPER_BASE_URL")
    resolved.tools.jina_base_url = resolved.tools.jina_base_url or os.environ.get("JINA_BASE_URL")
    resolved.tools.summary_llm.base_url = resolved.tools.summary_llm.base_url or os.environ.get("SUMMARY_LLM_BASE_URL")
    resolved.tools.summary_llm.model_name = resolved.tools.summary_llm.model_name or os.environ.get("SUMMARY_LLM_MODEL_NAME")
    resolved.agent.context_compression.llm.base_url = resolved.agent.context_compression.llm.base_url or os.environ.get("COMPRESSION_LLM_BASE_URL")
    resolved.agent.context_compression.llm.model_name = resolved.agent.context_compression.llm.model_name or os.environ.get(
        "COMPRESSION_LLM_MODEL_NAME"
    )
    resolved.judge.judge_times = max(1, resolved.judge.judge_times)
    resolved.judge.judge_max_tokens = resolve_judge_max_tokens_for_benchmark(
        resolved.benchmark.name,
        None if judge_max_tokens_is_default else resolved.judge.judge_max_tokens,
    )
    resolved.judge.judge_empty_length_retry_max_tokens = max(1, resolved.judge.judge_empty_length_retry_max_tokens)
    # The DSQA macro-F1 pass hard-codes the DeepSearchQA prompt schema and the
    # ``deepsearchqa__<idx>`` -> CSV row mapping, so force it off for any other
    # benchmark. Forcing it into the resolved config (not just at pass time)
    # keeps the effective YAML dump honest — non-DSQA runs never show
    # ``dsqa_f1_verify.enabled: true`` in ``run_config.effective.yaml``.
    if resolved.benchmark.name != "deepsearchqa":
        resolved.judge.dsqa_f1_verify.enabled = False
    return resolved


def _optional_arg(args: list[str], flag: str, value: Any) -> None:
    if value is not None:
        args.extend([flag, str(value)])


def _append_retry_no_box_turn_limit_cap_args(args: list[str], config: WebSearchEvalConfig) -> None:
    if not config.agent.retry.no_box_turn_limit_cap_enabled:
        return
    args.extend(["--retry_no_box_turn_limit_cap_enabled", "true"])
    args.extend(["--retry_no_box_turn_limit_cap", str(config.agent.retry.no_box_turn_limit_cap)])


def _append_semantic_query_budget_args(args: list[str], config: WebSearchEvalConfig) -> None:
    if not config.agent.semantic_query_budget.enabled:
        return
    args.extend(["--semantic_query_budget_enabled", "true"])
    _optional_arg(args, "--semantic_query_budget_max_unique", config.agent.semantic_query_budget.max_unique)


def _append_rollback_storm_shadow_args(args: list[str], config: WebSearchEvalConfig) -> None:
    if not config.agent.rollback_storm_shadow.enabled:
        return
    args.extend(["--rollback_storm_shadow_enabled", "true"])
    args.extend(["--rollback_storm_duplicate_threshold", str(config.agent.rollback_storm_shadow.duplicate_threshold)])
    args.extend(["--rollback_storm_tool_error_threshold", str(config.agent.rollback_storm_shadow.tool_error_threshold)])
    args.extend(["--rollback_storm_late_turn_threshold", str(config.agent.rollback_storm_shadow.late_turn_threshold)])
    args.extend(["--rollback_storm_preview_max_items", str(config.agent.rollback_storm_shadow.preview_max_items)])


def _append_context_compression_args(args: list[str], config: WebSearchEvalConfig) -> None:
    if not config.agent.context_compression.enabled:
        return
    cc = config.agent.context_compression
    args.extend(["--context_compression_enabled", "true"])
    args.extend(["--context_compression_interval", str(cc.interval)])
    args.extend(["--context_compression_recent_window", str(cc.recent_window)])
    _optional_arg(args, "--context_compression_llm_base_url", cc.llm.base_url)
    _optional_arg(args, "--context_compression_llm_model_name", cc.llm.model_name)
    args.extend(["--context_compression_llm_api_key_env", cc.llm.api_key_env])


def _append_self_verification_args(args: list[str], config: WebSearchEvalConfig) -> None:
    sv = config.agent.self_verification
    if not sv.enabled:
        return
    args.extend(["--self_verification_enabled", "true"])
    args.extend(["--self_verification_max_reanswer_attempts", str(sv.max_reanswer_attempts)])
    _optional_arg(args, "--self_verification_max_turns", sv.verification_max_turns)
    args.extend(["--self_verification_verdict_resample_max_attempts", str(sv.verdict_resample_max_attempts)])


def _append_discard_all_args(args: list[str], config: WebSearchEvalConfig) -> None:
    if not config.agent.discard_all.enabled:
        return
    da = config.agent.discard_all
    args.extend(["--discard_all_enabled", "true"])
    args.extend(["--discard_all_trigger_ratio", str(da.trigger_ratio)])
    args.extend(["--discard_all_min_turns_between", str(da.min_turns_between)])
    args.extend(["--discard_all_max_tool_calls", str(da.max_tool_calls)])


def _append_raw_scrape_cache_args(args: list[str], config: WebSearchEvalConfig) -> None:
    if not config.tools.raw_scrape_cache.enabled:
        return
    args.extend(["--raw_scrape_cache_enabled", "true"])
    args.extend(["--raw_scrape_cache_scope", config.tools.raw_scrape_cache.scope])
    args.extend(["--raw_scrape_cache_provider", config.tools.raw_scrape_cache.provider])
    args.extend(["--raw_scrape_cache_normalize_url", str(config.tools.raw_scrape_cache.normalize_url).lower()])


def _append_code_exec_args(args: list[str], config: WebSearchEvalConfig) -> None:
    if not config.tools.code_exec.enabled:
        return
    args.extend(["--code_exec_enabled", "true"])
    args.extend(["--code_exec_sandbox_timeout", str(config.tools.code_exec.sandbox_timeout)])
    _optional_arg(args, "--code_exec_template_id", config.tools.code_exec.template_id)
    _optional_arg(args, "--code_exec_max_calls_per_task", config.tools.code_exec.max_calls_per_task)
    args.extend(["--code_exec_retry_json", config.tools.code_exec.retry.to_runtime().to_json()])


def _runner_args(config: WebSearchEvalConfig, run_dir: Path) -> list[str]:  # noqa: PLR0915
    args = [
        sys.executable,
        "-m",
        "recipe.web_search.runners.evaluate_benchmark",
        "--model",
        str(config.model.openai_model),
        "--base_url",
        str(config.model.openai_base_url),
        "--api_key_env",
        config.model.api_key_env,
        "--data_path",
        str(config.benchmark.data_path),
        "--benchmark_name",
        config.benchmark.name,
        "--output_dir",
        str(run_dir),
        "--max_turns",
        str(config.agent.max_turns),
        "--keep_tool_result",
        str(config.agent.keep_tool_result),
        "--tool_result_role",
        config.agent.tool_result_role,
        "--max_task_retries",
        str(config.agent.retry.max_task_retries),
        "--generation_limit_recovery_non_final_attempt",
        config.agent.retry.generation_limit_recovery.non_final_attempt,
        "--generation_limit_recovery_final_attempt",
        config.agent.retry.generation_limit_recovery.final_attempt,
        "--prompt_profile",
        config.agent.prompt_profile,
        "--temperature",
        str(config.model.temperature),
        "--max_output_tokens",
        str(config.model.max_output_tokens),
        "--max_context_length",
        str(config.model.context.max_context_length),
        "--context_safety_margin",
        str(config.model.context.safety_margin),
        "--min_tokens_for_generation",
        str(config.model.context.min_tokens_for_generation),
        "--context_estimator",
        config.model.context.estimator,
        "--context_limit_detection",
        config.model.context.limit_detection,
        "--token_estimation_chars_per_token",
        str(config.model.context.token_estimation_chars_per_token),
        "--timeout",
        str(config.model.timeout),
        "--max_response_retries",
        str(config.model.max_response_retries),
        "--retry_wait_seconds",
        str(config.model.retry_wait_seconds),
        "--log_level",
        config.run.log_level,
        "--max_concurrent",
        str(config.benchmark.max_concurrent),
        "--max_content_length",
        str(config.tools.max_content_length),
        "--summary_llm_api_key_env",
        config.tools.summary_llm.api_key_env,
    ]
    _optional_arg(args, "--top_p", config.model.top_p)
    _optional_arg(args, "--repetition_penalty", config.model.repetition_penalty)
    _optional_arg(args, "--context_warning_threshold", config.model.context.warning_threshold)
    _optional_arg(args, "--endpoint_profile", config.model.endpoint_profile)
    _optional_arg(args, "--system_prompt_date", config.agent.system_prompt_date)
    _optional_arg(args, "--max_tokens_field", config.model.max_tokens_field)
    _optional_arg(args, "--parallel_tool_calls", config.model.parallel_tool_calls)
    _optional_arg(args, "--parse_embedded_thinking", config.model.parse_embedded_thinking)
    if config.model.endpoint_error_exit_status_codes is not None:
        args.append("--endpoint_error_exit_status_codes")
        args.extend(str(code) for code in config.model.endpoint_error_exit_status_codes)
    if config.model.endpoint_error_exit_after_seconds is None:
        args.extend(["--endpoint_error_exit_after_seconds", "null"])
    else:
        _optional_arg(args, "--endpoint_error_exit_after_seconds", config.model.endpoint_error_exit_after_seconds)
    _optional_arg(args, "--endpoint_connection_error_retry_wait_seconds", config.model.endpoint_connection_error_retry_wait_seconds)
    _optional_arg(args, "--endpoint_error_retry_backoff_multiplier", config.model.transport.retry.backoff_multiplier)
    _optional_arg(args, "--endpoint_error_retry_backoff_max_seconds", config.model.transport.retry.backoff_max_seconds)
    _optional_arg(args, "--endpoint_error_retry_jitter", str(config.model.transport.retry.jitter).lower())
    _optional_arg(args, "--endpoint_error_respect_retry_after", str(config.model.transport.retry.respect_retry_after).lower())
    _optional_arg(args, "--endpoint_retry_on_timeout", str(config.model.transport.retry.retry_on_timeout).lower())
    _optional_arg(args, "--endpoint_retry_on_connection_error", str(config.model.transport.retry.retry_on_connection_error).lower())
    _optional_arg(args, "--context_tokenizer_path", config.model.context.tokenizer_path)
    args.extend(["--preserve_reasoning_content", str(config.model.preserve_reasoning_content).lower()])
    if config.model.response_reasoning_fields:
        args.append("--response_reasoning_fields")
        args.extend(config.model.response_reasoning_fields)
    if config.model.request_extra_body:
        args.extend(["--request_extra_body_json", json.dumps(config.model.request_extra_body, ensure_ascii=False)])
    if config.tools.summary_llm.request_extra_body:
        args.extend(
            [
                "--summary_llm_request_extra_body_json",
                json.dumps(config.tools.summary_llm.request_extra_body, ensure_ascii=False),
            ]
        )
    _optional_arg(args, "--system_prompt_render_template", config.run.system_prompt_render_template)
    _optional_arg(args, "--system_prompt_render_extract_start", config.run.system_prompt_render_extract_start)
    _optional_arg(args, "--system_prompt_render_extract_end", config.run.system_prompt_render_extract_end)
    _optional_arg(args, "--max_final_answer_attempts", config.agent.retry.max_final_answer_attempts)
    if config.agent.retry.attempt_budget_sweep_enabled:
        args.extend(["--attempt_budget_sweep_enabled", "true"])
    _append_semantic_query_budget_args(args, config)
    if config.agent.retry.attempt_provenance_enabled:
        args.extend(["--retry_attempt_provenance_enabled", "true"])
    _append_retry_no_box_turn_limit_cap_args(args, config)
    _append_rollback_storm_shadow_args(args, config)
    _append_context_compression_args(args, config)
    _append_self_verification_args(args, config)
    _append_discard_all_args(args, config)
    _optional_arg(args, "--max_tasks", config.benchmark.max_tasks)
    _optional_arg(args, "--shuffle_seed", config.benchmark.shuffle_seed)
    _optional_arg(args, "--serper_base_url", config.tools.serper_base_url)
    _optional_arg(args, "--jina_base_url", config.tools.jina_base_url)
    _append_raw_scrape_cache_args(args, config)
    _append_code_exec_args(args, config)
    if config.tools.disable_all:
        args.append("--disable_tools")
    _optional_arg(args, "--summary_llm_base_url", config.tools.summary_llm.base_url)
    _optional_arg(args, "--summary_llm_model_name", config.tools.summary_llm.model_name)
    _optional_arg(args, "--summary_llm_max_input_chars", config.tools.summary_llm.max_input_chars)
    _optional_arg(args, "--summary_llm_chunk_overlap_chars", config.tools.summary_llm.chunk_overlap_chars)
    _optional_arg(args, "--summary_llm_max_chunks", config.tools.summary_llm.max_chunks)
    _optional_arg(args, "--summary_llm_chunk_max_concurrent", config.tools.summary_llm.chunk_max_concurrent)
    # Always pass the master switch explicitly (default is on, so absence is ambiguous).
    args.extend(["--summary_llm_chunked_extraction", "true" if config.tools.summary_llm.chunked_extraction else "false"])
    _optional_arg(args, "--summary_llm_chunk_strategy", config.tools.summary_llm.chunk_strategy)
    _optional_arg(args, "--summary_llm_max_recursion_depth", config.tools.summary_llm.max_recursion_depth)
    args.extend(["--summary_llm_global_anchor_enabled", str(config.tools.summary_llm.global_anchor_enabled).lower()])
    _optional_arg(args, "--summary_llm_chunk_envelope_mode", config.tools.summary_llm.chunk_envelope_mode)
    args.extend(["--summary_llm_csv_layer_b_enabled", str(config.tools.summary_llm.csv_layer_b_enabled).lower()])
    if config.tools.summary_llm.cache_enabled:
        args.extend(["--summary_llm_cache_enabled", "true"])
    args.extend(["--summary_llm_timeout_json", config.tools.summary_llm.timeout.to_runtime().to_json()])
    args.extend(["--summary_llm_retry_json", config.tools.summary_llm.retry.to_runtime().to_json()])
    args.extend(["--scrape_timeout_json", config.tools.scrape.timeout.to_runtime().to_json()])
    args.extend(["--scrape_retry_json", config.tools.scrape.retry.to_runtime().to_json()])
    args.extend(["--scrape_fallback_retry_json", config.tools.scrape.fallback_retry_runtime().to_json()])
    args.extend(["--search_timeout_json", config.tools.search.timeout.to_runtime().to_json()])
    args.extend(["--search_retry_json", config.tools.search.retry.to_runtime().to_json()])
    if config.run.resume:
        args.append("--resume")
        if config.run.force_resume_finalized_run:
            args.append("--force_resume_finalized_run")
    if config.run.model_request_logging:
        args.append("--model_request_logging")
    if config.benchmark.shuffle_tasks:
        args.append("--shuffle_tasks")
    if config.agent.retry.include_failure_summary:
        args.append("--include_failure_summary_in_retry")
    return args


def _write_run_config_files(input_config_path: Path, resolved: WebSearchEvalConfig, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.input.yaml").write_text(input_config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "run_config.effective.yaml").write_text(dump_web_search_eval_config(resolved), encoding="utf-8")


def _preflight_resume_targets(config: WebSearchEvalConfig, output_dir: Path) -> None:
    if not config.run.resume:
        return
    expected_count = count_dataset_items(str(config.benchmark.data_path), max_tasks=config.benchmark.max_tasks)
    for idx in range(1, config.run.num_runs + 1):
        guard_resume_into_finalized_run(
            output_dir / f"run_{idx}",
            expected_task_count=expected_count,
            force=config.run.force_resume_finalized_run,
        )


def _start_online_judge(config: WebSearchEvalConfig, output_dir: Path, env_file: Path | None) -> subprocess.Popen[str] | None:
    if not config.judge.online:
        print("Online LLM judge evaluation: disabled")
        return None
    judge_max_tokens = resolve_judge_max_tokens_for_benchmark(
        config.benchmark.name,
        config.judge.judge_max_tokens if "judge_max_tokens" in config.judge.model_fields_set else None,
    )
    print(
        "Online LLM judge evaluation: enabled "
        f"(poll={config.judge.poll_seconds}s stable={config.judge.stable_seconds}s "
        f"max_concurrent={config.judge.max_concurrent} judge_times={config.judge.judge_times} "
        f"judge_max_tokens={judge_max_tokens} "
        f"judge_empty_length_retry_max_tokens={config.judge.judge_empty_length_retry_max_tokens})"
    )
    args = [
        sys.executable,
        "-m",
        "recipe.web_search.runners.judge_existing",
        "--run_dir",
        str(output_dir),
        "--data_path",
        str(config.benchmark.data_path),
        "--benchmark_name",
        config.benchmark.name,
        "--max_concurrent",
        str(config.judge.max_concurrent),
        "--judge_times",
        str(config.judge.judge_times),
        "--judge_max_tokens",
        str(judge_max_tokens),
        "--judge_empty_length_retry_max_tokens",
        str(config.judge.judge_empty_length_retry_max_tokens),
        "--allow-partial",
        "--stable-seconds",
        str(config.judge.stable_seconds),
        "--watch",
        "--poll-seconds",
        str(config.judge.poll_seconds),
        "--log_level",
        config.run.log_level,
    ]
    if env_file is not None:
        args.extend(["--env-file", str(env_file)])
    if config.judge.judge_model:
        args.extend(["--judge_model", config.judge.judge_model])
    if config.judge.judge_base_url:
        args.extend(["--judge_base_url", config.judge.judge_base_url])
    if config.judge.api_key_env:
        args.extend(["--judge_api_key_env", config.judge.api_key_env])
    if config.judge.request_logging:
        args.append("--judge_request_logging")
    return subprocess.Popen(args, text=True, start_new_session=True)


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError:
            process.kill()
        process.wait()


def _final_judge_existing_args(config: WebSearchEvalConfig, run_dir: Path, env_file: Path | None, judge_max_tokens: int) -> list[str]:
    args = [
        sys.executable,
        "-m",
        "recipe.web_search.runners.judge_existing",
        "--run_dir",
        str(run_dir),
        "--data_path",
        str(config.benchmark.data_path),
        "--benchmark_name",
        config.benchmark.name,
        "--max_concurrent",
        str(config.judge.max_concurrent),
        "--judge_times",
        str(config.judge.judge_times),
        "--judge_max_tokens",
        str(judge_max_tokens),
        "--judge_empty_length_retry_max_tokens",
        str(config.judge.judge_empty_length_retry_max_tokens),
        "--allow-partial",
        "--stable-seconds",
        "0",
        "--log_level",
        config.run.log_level,
    ]
    if env_file is not None:
        args.extend(["--env-file", str(env_file)])
    if config.judge.judge_model:
        args.extend(["--judge_model", config.judge.judge_model])
    if config.judge.judge_base_url:
        args.extend(["--judge_base_url", config.judge.judge_base_url])
    if config.judge.api_key_env:
        args.extend(["--judge_api_key_env", config.judge.api_key_env])
    if config.judge.request_logging:
        args.append("--judge_request_logging")
    return args


def _final_online_judge(config: WebSearchEvalConfig, output_dir: Path, env_file: Path | None) -> None:
    if not config.judge.online:
        return
    print("--- Final EM/LLM judge catch-up pass ---")
    judge_max_tokens = resolve_judge_max_tokens_for_benchmark(
        config.benchmark.name,
        config.judge.judge_max_tokens if "judge_max_tokens" in config.judge.model_fields_set else None,
    )
    subprocess.run(_final_judge_existing_args(config, output_dir, env_file, judge_max_tokens), check=False)
    # Attempt-budget sweep: each attempt_budget_N/ is a full result dir (benchmark_results.jsonl
    # + traces). The top-level judge above only covers the max budget; judge each budget dir too
    # so the per-attempt judge curve (attempt_budget_N/llm_judge_accuracy.txt) is produced, and is
    # complete after a resume. Non-sweep runs have no such dirs, so this loop is a no-op.
    # output_dir is the parent; per-run results (and their attempt_budget_N/ dirs) live under run_*/.
    budget_dirs = sorted(
        (p for p in output_dir.glob("run_*/attempt_budget_*") if p.is_dir() and (p / "benchmark_results.jsonl").exists()),
        key=lambda p: (p.parent.name, p.name),
    )
    for budget_dir in budget_dirs:
        print(f"--- Per-budget judge: {budget_dir.parent.name}/{budget_dir.name} ---")
        subprocess.run(_final_judge_existing_args(config, budget_dir, env_file, judge_max_tokens), check=False)


def _dsqa_f1_verify_run_dirs(output_dir: Path) -> list[Path]:
    """Return each ``run_*/`` root that has judgeable final predictions.

    Only the run root is scored — the file there is the final answer for each
    task after any attempt-budget sweep. ``attempt_budget_*/`` subdirs are
    deliberately skipped so F1 reflects the final answer, not per-attempt
    intermediate predictions.
    """

    def _has_predictions(path: Path) -> bool:
        return (path / "llm_judge_results.jsonl").exists() or (path / "benchmark_results.jsonl").exists()

    dirs: list[Path] = []
    for run_dir in sorted(output_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        if _has_predictions(run_dir):
            dirs.append(run_dir)
    return dirs


def _run_dsqa_f1_verify_pass(config: WebSearchEvalConfig, output_dir: Path) -> None:
    cfg = config.judge.dsqa_f1_verify
    if not cfg.enabled:
        return
    if config.benchmark.name != "deepsearchqa":
        print(
            f"dsqa_f1_verify: skipped — benchmark.name is {config.benchmark.name!r} (only 'deepsearchqa' produces the required task-id/CSV mapping)"
        )
        return
    judge_model = cfg.judge_model or config.judge.judge_model
    judge_base_url = cfg.judge_base_url or config.judge.judge_base_url
    api_key_env = cfg.api_key_env or config.judge.api_key_env
    if not judge_model or not judge_base_url:
        print(
            "dsqa_f1_verify: skipped — resolved judge_model/judge_base_url is empty. Set them under judge.dsqa_f1_verify.* or inherit from judge.*."
        )
        return
    dataset_csv = Path(config.benchmark.data_path)
    if not dataset_csv.exists():
        print(f"dsqa_f1_verify: skipped — benchmark.data_path does not exist: {dataset_csv}")
        return
    run_dirs = _dsqa_f1_verify_run_dirs(output_dir)
    if not run_dirs:
        print(f"dsqa_f1_verify: no run dirs with predictions under {output_dir}")
        return
    print(f"--- DSQA F1 verify pass over {len(run_dirs)} run dir(s) (judge={judge_model}@{judge_base_url}) ---")
    for run_dir in run_dirs:
        print(f"--- dsqa_f1_verify: {run_dir} ---")
        try:
            run_dsqa_f1_verify(
                run_dir=run_dir,
                dataset_csv=dataset_csv,
                judge_model=judge_model,
                judge_base_url=judge_base_url,
                api_key_env=api_key_env,
                experiment=run_dir.name,
                workers=cfg.workers,
                retries=cfg.retries,
                request_timeout=cfg.request_timeout,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                retry_sleep_cap=cfg.retry_sleep_cap,
                progress_every=cfg.progress_every,
                resume=cfg.resume,
            )
        except Exception as exc:
            print(f"dsqa_f1_verify failed for {run_dir}: {exc!r}")


def _aggregate_runs_summary(config: WebSearchEvalConfig, output_dir: Path) -> None:
    """Write the parent-dir avg@N + sample-std summary for multi-run evals.

    Gated to LiveBrowseComp, whose paper protocol reports avg@N over num_runs
    samples (Table 3). Other benchmarks report single-run accuracy, so this
    cross-run rollup is skipped for them.
    """
    if config.benchmark.name != "livebrowsecomp" or config.run.num_runs <= 1:
        return
    from recipe.web_search.runners.aggregate_runs import aggregate_runs, render_text_report

    print(f"--- Aggregating avg@{config.run.num_runs} across runs ---")
    try:
        summary = aggregate_runs(output_dir, "llm_judge")
    except FileNotFoundError as exc:
        print(f"skipped run aggregation: {exc}")
        return
    summary_path = output_dir / "aggregate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = render_text_report(summary)
    summary_path.with_suffix(".txt").write_text(report, encoding="utf-8")
    print(report, end="")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run native web-search benchmark evaluation from YAML")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force-resume-finalized-run",
        "--force_resume_finalized_run",
        action="store_true",
        help="Allow --resume to write into run_* dirs that already have complete benchmark/eval final artifacts.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_web_search_eval_config(config_path)
    if args.resume:
        config.run.resume = True
    if args.force_resume_finalized_run:
        config.run.force_resume_finalized_run = True
    env_file = _resolve_env_file_path(config_path, config.run.env_file)
    _load_env_file(env_file)
    resolved = _resolve_config(config)

    print("=== Native Web Search YAML Evaluation ===")
    print(f"Config:         {config_path}")
    print(f"Model:          {resolved.model.openai_model}")
    print(f"Base URL:       {resolved.model.openai_base_url}")
    print(f"Benchmark:      {resolved.benchmark.name}")
    print(f"Data path:      {resolved.benchmark.data_path}")
    print(f"Output dir:     {resolved.run.output_dir}")
    print(f"Num runs:       {resolved.run.num_runs}")
    print(f"Max concurrent: {resolved.benchmark.max_concurrent}")
    if args.dry_run:
        print(dump_web_search_eval_config(resolved))
        return

    output_dir = _resolve_path(resolved.run.output_dir, label="run.output_dir").resolve()
    try:
        _preflight_resume_targets(resolved, output_dir)
    except FinalizedRunResumeError as exc:
        raise SystemExit(str(exc)) from None
    _write_run_config_files(config_path, resolved, output_dir)
    judge_process = _start_online_judge(resolved, output_dir, env_file)
    processes: list[subprocess.Popen[str]] = []
    try:
        for idx in range(1, resolved.run.num_runs + 1):
            run_dir = output_dir / f"run_{idx}"
            _write_run_config_files(config_path, resolved, run_dir)
            print(f"--- Starting run {idx}/{resolved.run.num_runs} (output: {run_dir}) ---")
            processes.append(subprocess.Popen(_runner_args(resolved, run_dir), text=True, start_new_session=True))
        status = 0
        for process in processes:
            status = process.wait() or status
    finally:
        _stop_process(judge_process)
    _final_online_judge(resolved, output_dir, env_file)
    _run_dsqa_f1_verify_pass(resolved, output_dir)
    _aggregate_runs_summary(resolved, output_dir)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
