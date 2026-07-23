# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic.model_clients.openai_client import DEFAULT_REASONING_RESPONSE_FIELDS
from recipe.common.retry_config import (
    ModelResponseRetryConfig,
    ModelTransportConfig,
    RetryConfig,
    ScrapeToolConfig,
    SearchToolConfig,
    TimeoutConfig,
    default_code_exec_retry,
    default_summary_retry,
    default_summary_timeout,
    legacy_model_retryable_status_codes,
)

DEFAULT_SYSTEM_PROMPT_RENDER_TEMPLATE = "auto"
DEFAULT_SYSTEM_PROMPT_RENDER_EXTRACT_START = None
DEFAULT_SYSTEM_PROMPT_RENDER_EXTRACT_END = None


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(StrictConfigModel):
    output_dir: str = "logs/web_search_infer"
    num_runs: int = 1
    resume: bool = False
    force_resume_finalized_run: bool = False
    # Default-off: exact payload logging serializes large request/response
    # bodies and can add client-side latency in high-concurrency runs.
    model_request_logging: bool = False
    system_prompt_render_template: str | None = DEFAULT_SYSTEM_PROMPT_RENDER_TEMPLATE
    system_prompt_render_extract_start: str | None = DEFAULT_SYSTEM_PROMPT_RENDER_EXTRACT_START
    system_prompt_render_extract_end: str | None = DEFAULT_SYSTEM_PROMPT_RENDER_EXTRACT_END
    env_file: str | None = ".envs/.env"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class ModelContextConfig(StrictConfigModel):
    max_context_length: int = 262_144
    safety_margin: int = 4_096
    min_tokens_for_generation: int = 2_048
    warning_threshold: int | None = None
    # model_client: use the model client estimate_tokens implementation.
    # cheap: use additive character estimates, enabling incremental context tracking.
    # chat_template: render/tokenize the full chat template near the context limit;
    # this is more exact but can add client-side latency under high concurrency.
    estimator: Literal["model_client", "cheap", "chat_template"] = "model_client"
    # provider_error: skip local context-limit checks; rely on provider context
    # errors and finish_reason=length recovery. estimated: run local preflight
    # checks before model calls; useful only when provider errors are unreliable.
    limit_detection: Literal["provider_error", "estimated"] = "provider_error"
    tokenizer_path: str | None = None
    token_estimation_chars_per_token: float = 3.0


class ModelRuntimeConfig(StrictConfigModel):
    openai_model: str | None = None
    openai_base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    endpoint_profile: str | None = None
    preserve_reasoning_content: bool = True
    response_reasoning_fields: list[str] = Field(default_factory=lambda: list(DEFAULT_REASONING_RESPONSE_FIELDS))
    max_tokens_field: str | None = None
    parallel_tool_calls: bool | None = None
    parse_embedded_thinking: bool = True
    transport: ModelTransportConfig = Field(default_factory=ModelTransportConfig)
    response_retry: ModelResponseRetryConfig = Field(default_factory=ModelResponseRetryConfig)
    request_extra_body: dict[str, Any] = Field(default_factory=dict)
    temperature: float = 1.0
    top_p: float | None = 0.95
    repetition_penalty: float | None = 1.05
    max_output_tokens: int = 16_384
    context: ModelContextConfig = Field(default_factory=ModelContextConfig)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_timeout_retry_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        migrated = dict(data)
        transport = dict(migrated.get("transport") or {})
        timeout = dict(transport.get("timeout") or {})
        retry = dict(transport.get("retry") or {})
        response_retry = dict(migrated.get("response_retry") or {})

        if "timeout" in migrated:
            timeout.setdefault("total_seconds", migrated.pop("timeout"))
        if "endpoint_error_exit_status_codes" in migrated:
            legacy_status_codes = migrated.pop("endpoint_error_exit_status_codes")
            retry.setdefault("retryable_status_codes", legacy_model_retryable_status_codes(legacy_status_codes))
        if "endpoint_connection_error_retry_wait_seconds" in migrated:
            retry.setdefault("backoff_base_seconds", migrated.pop("endpoint_connection_error_retry_wait_seconds"))
        if "endpoint_error_exit_after_seconds" in migrated:
            retry.setdefault("exit_after_seconds", migrated.pop("endpoint_error_exit_after_seconds"))
        if "retry_wait_seconds" in migrated:
            response_retry.setdefault("wait_seconds", migrated.pop("retry_wait_seconds"))
        if "max_response_retries" in migrated:
            response_retry.setdefault("max_attempts", migrated.pop("max_response_retries"))

        if timeout:
            transport["timeout"] = timeout
        if retry:
            transport["retry"] = retry
        if transport:
            migrated["transport"] = transport
        if response_retry:
            migrated["response_retry"] = response_retry
        return migrated

    @property
    def timeout(self) -> float:
        return self.transport.timeout_seconds()

    @timeout.setter
    def timeout(self, value: float) -> None:
        self.transport.timeout.total_seconds = value

    @property
    def endpoint_error_exit_status_codes(self) -> list[int]:
        return self.transport.retry.retryable_status_codes

    @endpoint_error_exit_status_codes.setter
    def endpoint_error_exit_status_codes(self, value: list[int]) -> None:
        self.transport.retry.retryable_status_codes = value

    @property
    def endpoint_error_exit_after_seconds(self) -> float | None:
        return self.transport.retry.exit_after_seconds

    @endpoint_error_exit_after_seconds.setter
    def endpoint_error_exit_after_seconds(self, value: float | None) -> None:
        self.transport.retry.exit_after_seconds = value

    @property
    def endpoint_connection_error_retry_wait_seconds(self) -> float:
        return self.transport.retry.backoff_base_seconds

    @endpoint_connection_error_retry_wait_seconds.setter
    def endpoint_connection_error_retry_wait_seconds(self, value: float) -> None:
        self.transport.retry.backoff_base_seconds = value

    @property
    def retry_wait_seconds(self) -> float:
        return self.response_retry.wait_seconds

    @retry_wait_seconds.setter
    def retry_wait_seconds(self, value: float) -> None:
        self.response_retry.wait_seconds = value

    @property
    def max_response_retries(self) -> int:
        return self.response_retry.max_attempts

    @max_response_retries.setter
    def max_response_retries(self, value: int) -> None:
        self.response_retry.max_attempts = value


class BenchmarkConfig(StrictConfigModel):
    name: Literal["browsecomp", "browsecomp_zh", "gaia", "hle", "deepsearchqa", "livebrowsecomp"] = "browsecomp"
    data_path: str | None = "axis_data://browsecomp/standardized_data.jsonl"
    max_tasks: int | None = None
    shuffle_tasks: bool = True
    shuffle_seed: int | None = None
    max_concurrent: int = 16


class SemanticQueryBudgetConfig(StrictConfigModel):
    enabled: bool = False
    max_unique: int | None = Field(default=None, ge=0)


class GenerationLimitRecoveryConfig(StrictConfigModel):
    # Applies to provider context-length errors and model responses with
    # finish_reason=length.
    non_final_attempt: Literal["retry", "rollback"] = "retry"
    final_attempt: Literal["rollback", "terminate"] = "rollback"


class AgentRetryConfig(StrictConfigModel):
    max_task_retries: int = 5
    include_failure_summary: bool = False
    max_final_answer_attempts: int = 1
    attempt_budget_sweep_enabled: bool = False
    attempt_provenance_enabled: bool = False
    no_box_turn_limit_cap_enabled: bool = False
    no_box_turn_limit_cap: int = Field(default=3, ge=1)
    generation_limit_recovery: GenerationLimitRecoveryConfig = Field(default_factory=GenerationLimitRecoveryConfig)


class RollbackStormShadowConfig(StrictConfigModel):
    enabled: bool = False
    duplicate_threshold: int = Field(default=20, ge=1)
    tool_error_threshold: int = Field(default=10, ge=1)
    late_turn_threshold: int = Field(default=250, ge=0)
    preview_max_items: int = Field(default=5, ge=0)


class ContextCompressionLLMConfig(StrictConfigModel):
    base_url: str | None = None
    model_name: str | None = None
    api_key_env: str = "sk-dummy"


class ContextCompressionConfig(StrictConfigModel):
    enabled: bool = False
    interval: int = Field(default=10, ge=1)
    recent_window: int = Field(default=10, ge=1)
    llm: ContextCompressionLLMConfig = Field(default_factory=ContextCompressionLLMConfig)


class SelfVerificationConfig(StrictConfigModel):
    enabled: bool = False
    max_reanswer_attempts: int = Field(default=1, ge=0)
    verification_max_turns: int | None = Field(default=None, ge=1)
    verdict_resample_max_attempts: int = Field(default=3, ge=1)


class DiscardAllConfig(StrictConfigModel):
    # Discard-all context management: when observed prompt tokens exceed
    # ``trigger_ratio`` of the context window, discard all prior tool-call
    # history and reopen the task from a clean context. Mutually exclusive with
    # ``context_compression`` (enforced fail-fast in AgentConfig).
    enabled: bool = False
    trigger_ratio: float = Field(default=0.80, gt=0.0, le=1.0)
    min_turns_between: int = Field(default=3, ge=0)
    # Global tool-call cap for the attempt; does not reset on discard and
    # supersedes ``max_turns`` when discard-all is enabled.
    max_tool_calls: int = Field(default=1800, ge=1)


class AgentConfig(StrictConfigModel):
    prompt_profile: Literal["default", "deepsearchqa", "livebrowsecomp", "livebrowsecomp_notools"] = "default"
    system_prompt_date: str | None = None
    max_turns: int = 300
    keep_tool_result: int = 5
    tool_result_role: Literal["tool", "user"] = "tool"
    retry: AgentRetryConfig = Field(default_factory=AgentRetryConfig)
    semantic_query_budget: SemanticQueryBudgetConfig = Field(default_factory=SemanticQueryBudgetConfig)
    rollback_storm_shadow: RollbackStormShadowConfig = Field(default_factory=RollbackStormShadowConfig)
    context_compression: ContextCompressionConfig = Field(default_factory=ContextCompressionConfig)
    self_verification: SelfVerificationConfig = Field(default_factory=SelfVerificationConfig)
    discard_all: DiscardAllConfig = Field(default_factory=DiscardAllConfig)

    @model_validator(mode="after")
    def _validate_context_management_mutual_exclusion(self) -> AgentConfig:
        # Fail fast before an experiment starts: discard-all and Summary-style
        # context_compression are alternative context-management strategies and
        # must not run together (they would fight over the same visible prefix).
        if self.discard_all.enabled and self.context_compression.enabled:
            msg = (
                "agent.discard_all.enabled and agent.context_compression.enabled are mutually "
                "exclusive; enable at most one context-management strategy."
            )
            raise ValueError(msg)
        return self


class RawScrapeCacheConfig(StrictConfigModel):
    enabled: bool = False
    scope: Literal["task"] = "task"
    provider: Literal["web_search"] = "web_search"
    normalize_url: bool = False


class SummaryLLMConfig(StrictConfigModel):
    base_url: str | None = None
    model_name: str | None = None
    api_key_env: str = "SUMMARY_LLM_API_KEY"
    # Extra request-body fields merged into every extraction LLM call. For a
    # DeepSeek-V4-Pro extraction LLM enable reasoning with e.g.
    # {"reasoning_effort": "max", "thinking": {"type": "enabled"}}.
    request_extra_body: dict[str, Any] = Field(default_factory=dict)
    max_input_chars: int = Field(default=120_000, ge=0)
    chunk_overlap_chars: int = Field(default=1_200, ge=0)
    max_chunks: int = Field(default=12, ge=1)
    chunk_max_concurrent: int = Field(default=4, ge=1)
    # Master switch for chunked map-reduce extraction of long scraped pages. When
    # False, long content uses the single-shot path (silent halving) regardless of
    # ``max_input_chars``. When True, ``chunk_strategy`` selects the method.
    chunked_extraction: bool = True
    # "single" — map every chunk then ONE reduce call; "recursive" — recursively
    # reduce the concatenated findings until they fit the budget (no tail loss).
    chunk_strategy: Literal["single", "recursive"] = "single"
    # Recursive strategy only: max reduce-recursion depth before a lossless concat.
    max_recursion_depth: int = Field(default=4, ge=0)
    global_anchor_enabled: bool = False
    chunk_envelope_mode: Literal["strict", "soft", "strict_caveat"] = "strict_caveat"
    csv_layer_b_enabled: bool = False
    # Default-off: the persistent extraction cache uses file locks/fsyncs and
    # can add client-side latency in high-concurrency scrape/extract workloads.
    cache_enabled: bool = False
    timeout: TimeoutConfig = Field(default_factory=default_summary_timeout)
    retry: RetryConfig = Field(default_factory=default_summary_retry)


class CodeExecConfig(StrictConfigModel):
    # Default-off: enabling registers session-scoped python_exec and shell_exec
    # tools. Both share one E2B sandbox per task attempt, so shell-installed
    # packages/files are visible to the stateful Python interpreter.
    enabled: bool = False
    # E2B sandbox keep-alive timeout in seconds per task attempt.
    sandbox_timeout: int = Field(default=600, ge=1)
    # Optional E2B sandbox template ID. Leave null to use the built-in template.
    template_id: str | None = None
    # Optional per-tool cap on python_exec and shell_exec executions per task.
    max_calls_per_task: int | None = Field(default=None, ge=1)
    # Transient-error retry for E2B sandbox calls. A 429 rate-limit is waited out
    # and retried (summary-LLM-style profile) rather than returned as the tool
    # response.
    retry: RetryConfig = Field(default_factory=default_code_exec_retry)


class ToolsConfig(StrictConfigModel):
    disable_all: bool = False
    serper_base_url: str | None = None
    jina_base_url: str | None = None
    max_content_length: int = 409_600
    raw_scrape_cache: RawScrapeCacheConfig = Field(default_factory=RawScrapeCacheConfig)
    summary_llm: SummaryLLMConfig = Field(default_factory=SummaryLLMConfig)
    scrape: ScrapeToolConfig = Field(default_factory=ScrapeToolConfig)
    search: SearchToolConfig = Field(default_factory=SearchToolConfig)
    code_exec: CodeExecConfig = Field(default_factory=CodeExecConfig)


class DsqaF1VerifyConfig(StrictConfigModel):
    # DeepSearchQA macro-F1 verifier. When enabled, runs after the standard
    # A/B judge over the same source predictions and writes a separate
    # ``dsqa_f1_*`` artifact family into each run dir. Defaults to True so
    # DeepSearchQA runs pick it up automatically; ``_resolve_config`` forces
    # it to False for any other benchmark so non-DSQA effective configs
    # cannot claim this pass is enabled.
    enabled: bool = True
    workers: int = Field(default=5, ge=1)
    retries: int = Field(default=5, ge=1)
    request_timeout: int = Field(default=180, ge=1)
    max_tokens: int = Field(default=8192, ge=1)
    progress_every: int = Field(default=50, ge=1)
    resume: bool = True
    retry_sleep_cap: float = Field(default=30.0, gt=0.0)
    temperature: float | None = None
    # When null, each field inherits from the parent ``judge.*`` section.
    judge_model: str | None = None
    judge_base_url: str | None = None
    api_key_env: str | None = None


class JudgeConfig(StrictConfigModel):
    online: bool = True
    # Default-off: judge request logging serializes judge prompts/responses and
    # can add client-side latency when many tasks are judged concurrently.
    request_logging: bool = False
    max_concurrent: int = 4
    judge_times: int = 5
    judge_max_tokens: int = Field(default=2, ge=1)
    judge_empty_length_retry_max_tokens: int = Field(default=1024, ge=1)
    stable_seconds: float = 10.0
    poll_seconds: float = 15.0
    judge_model: str | None = None
    judge_base_url: str | None = None
    api_key_env: str = "JUDGE_API_KEY"
    dsqa_f1_verify: DsqaF1VerifyConfig = Field(default_factory=DsqaF1VerifyConfig)


class WebSearchEvalConfig(StrictConfigModel):
    schema_version: int = Field(default=1, ge=1)
    run: RunConfig = Field(default_factory=RunConfig)
    model: ModelRuntimeConfig = Field(default_factory=ModelRuntimeConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)

    @model_validator(mode="after")
    def _validate_discard_all_context_detection(self) -> WebSearchEvalConfig:
        # Fail fast: the estimated local context-limit preflight force-finalizes
        # (or terminates) near the limit, which would pre-empt discard-all before
        # it can reset the trajectory. discard-all is built around provider-
        # observed prompt tokens and must run with provider_error detection.
        if self.agent.discard_all.enabled and self.model.context.limit_detection == "estimated":
            msg = (
                "agent.discard_all.enabled requires model.context.limit_detection='provider_error'; "
                "the 'estimated' preflight would force-finalize near the context limit before "
                "discard-all can reset the trajectory."
            )
            raise ValueError(msg)
        return self


def load_web_search_eval_config(path: str | Path) -> WebSearchEvalConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return WebSearchEvalConfig.model_validate(raw, strict=True)


def dump_web_search_eval_config(config: WebSearchEvalConfig) -> str:
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, indent=2)
