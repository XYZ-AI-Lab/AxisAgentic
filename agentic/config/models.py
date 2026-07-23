# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic.contracts.conversation import FormatErrorStrategy


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoggerConfig(StrictConfigModel):
    name: str = "agentic"
    level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    log_dir: str = "logs/agentic"
    persist_json: bool = True
    timezone_offset_hours: float = Field(default=8, description="Timezone offset from UTC in hours for log timestamps.")
    incremental_flush_steps: int = Field(
        default=0,
        ge=0,
        description=(
            "Write a partial trace JSON every N steps so progress is visible during execution. "
            "0 disables incremental flushing (trace is only written on finish)."
        ),
    )


class ModelClientConfig(StrictConfigModel):
    """Base configuration shared by all model client backends."""

    context_window: int = Field(default=128000, gt=0)
    max_output_tokens: int = Field(default=4096, gt=0)
    temperature: float = 0.0
    enable_tool_calls: bool = True


class ToolConfig(StrictConfigModel):
    name: str
    description: str
    strict_mode: bool = True
    timeout_seconds: float = 120.0
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExternalServerConfig(StrictConfigModel):
    name: str
    transport: Literal["stdio", "sse", "inprocess"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class ToolArgumentRepairConfig(StrictConfigModel):
    """Opt-in pre-tool argument repair policy.

    Recipes that emulate lenient MCP servers can enable this so tool calls are
    repaired against registered tool schemas before duplicate detection,
    validation, and execution. The default is disabled for backward
    compatibility.
    """

    enabled: bool = False
    repair_strict_tools: bool = False
    apply_schema_aliases: bool = True
    apply_schema_defaults: bool = True
    coerce_scalar_types: bool = True
    normalize_key_names: bool = True
    fuzzy_match_single_missing_property: bool = True
    fuzzy_match_threshold: float = Field(default=0.72, ge=0.0, le=1.0)


class ToolManagerConfig(StrictConfigModel):
    """Configuration for ToolManager execution policies."""

    max_tool_calls_per_step: int | None = Field(
        default=None, ge=1, description="Maximum number of tool calls allowed per assistant turn. None means no limit."
    )
    max_tool_calls_per_task: int | None = Field(
        default=None, ge=1, description="Maximum total tool calls allowed per task across all tools. None means no limit."
    )
    default_max_calls_per_tool_per_task: int | None = Field(
        default=None, ge=1, description="Default per-tool call budget per task. Overridden by Tool.max_calls_per_task when set."
    )
    default_max_consecutive_calls_with_same_args: int | None = Field(
        default=None,
        ge=1,
        description="Default max consecutive calls with identical arguments. Overridden by Tool.max_consecutive_calls_with_same_args when set.",
    )
    tool_call_limit_exceeded_message: str = "Tool call not invoked: exceeded the maximum number of allowed tool calls per step."
    task_budget_exceeded_message: str = "Tool call not invoked: total call budget for this task has been reached."
    default_call_budget_exceeded_message: str = "Tool call not invoked: call budget for this tool in this task has been reached."
    default_consecutive_call_exceeded_message: str = "Tool call not invoked: repeated consecutive call with the same arguments."
    argument_repair: ToolArgumentRepairConfig = Field(default_factory=ToolArgumentRepairConfig)


class FormatErrorConfig(StrictConfigModel):
    """Configuration for detecting and handling tool-call format errors.

    When the model produces text containing ``keywords`` but no actual structured tool calls,
    the ``strategy`` determines the recovery action.
    """

    keywords: list[str] = Field(
        default_factory=lambda: ["<tool_call>", "</tool_call>"],
        description="Substrings that indicate the model attempted a tool call in plain text.",
    )
    strategy: FormatErrorStrategy = Field(
        default=FormatErrorStrategy.REPROMPT,
        description=(
            "How to handle format errors: "
            "'reprompt' sends the reprompt text to let the model correct itself; "
            "'rollback' silently hides the message and retries (counts toward max_assistant_rollbacks); "
            "'ignore' accepts the malformed message as-is."
        ),
    )
    reprompt: str = Field(
        default=(
            "Your previous response looked like a tool call, but it did not produce a valid structured tool call. "
            "Please either emit a valid tool call or continue without tool-call tags."
        ),
        description="Recovery prompt sent to the model when strategy is 'reprompt'.",
    )


class ContextLimitConfig(StrictConfigModel):
    """Configuration for context-window and generation-budget checks."""

    window: int | None = Field(
        default=None,
        ge=1,
        description="Context window size for limit checking. None disables context-limit enforcement.",
    )
    safety_margin: int = Field(default=0, ge=0)
    min_tokens_for_generation: int = Field(
        default=0,
        ge=0,
        description="Minimum remaining context tokens required before another assistant generation; below this, terminate instead of finalizing.",
    )
    warning_threshold: int | None = Field(
        default=None,
        ge=1,
        description="Remaining-token threshold below which the runtime injects the force-final prompt before another assistant generation.",
    )


class AssistantRollbackConfig(StrictConfigModel):
    """Configuration for assistant-message rollback (refusal, format error, orchestrator-initiated).

    The runtime maintains a single rollback counter shared across all rollback types.
    The counter resets after a successful turn or via explicit ``reset_rollback_counter()`` calls.
    """

    refusal_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Keywords indicating the assistant is refusing the task. "
            "When detected in a text-only response (no tool calls), the message is rolled back. "
            "Empty list (default) disables refusal detection."
        ),
    )
    max_rollbacks: int = Field(
        default=0,
        ge=0,
        description=(
            "Maximum consecutive assistant-message rollbacks before giving up. "
            "Shared across refusal keywords, format error rollback, "
            "and orchestrator-initiated rollbacks (duplicate queries, tool errors). "
            "0 disables all assistant-message rollback."
        ),
    )


class ConversationConfig(StrictConfigModel):
    """Configuration for ConversationRuntime initialization."""

    @model_validator(mode="before")
    @classmethod
    def _gather_legacy_context_limit_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        values = dict(data)
        legacy_mapping = {
            "context_window": "window",
            "context_safety_margin": "safety_margin",
            "min_tokens_for_generation": "min_tokens_for_generation",
            "context_warning_threshold": "warning_threshold",
        }
        legacy_context = {new_key: values.pop(old_key) for old_key, new_key in legacy_mapping.items() if old_key in values}
        if legacy_context:
            existing = values.get("context_limit")
            if existing is None:
                values["context_limit"] = legacy_context
            elif isinstance(existing, ContextLimitConfig):
                values["context_limit"] = {**legacy_context, **existing.model_dump()}
            elif isinstance(existing, dict):
                values["context_limit"] = {**legacy_context, **existing}
        return values

    # -- Core prompts --
    system_prompt: str | None = "You are a helpful agent."
    user_prompt_template: str = "{task}"
    early_stop_announcement_prompt: str | None = (
        "You cannot call any tool now. Generate your final answer to the given task based on the conversation so far."
    )
    final_response_prompt: str | None = None

    # -- Budget limits --
    max_turns: int | None = Field(default=None, ge=1)
    context_limit: ContextLimitConfig = Field(default_factory=ContextLimitConfig)

    # -- Tool-call format error handling --
    format_error: FormatErrorConfig = Field(default_factory=FormatErrorConfig)

    # -- Assistant-message rollback --
    assistant_rollback: AssistantRollbackConfig = Field(default_factory=AssistantRollbackConfig)

    # -- Tool message handling --
    tool_result_role: Literal["tool", "user"] = Field(
        default="tool",
        description=(
            "Role used when serializing tool-result messages for model calls. "
            "'tool' preserves native tool-call chat semantics; 'user' presents tool results as ordinary user messages "
            "while keeping the runtime trace and tool accounting unchanged."
        ),
    )
    agg_reject_tool_message: bool = Field(
        default=True,
        description="Aggregate rejected tool messages with identical content into a single message, joining their tool_call_ids with commas.",
    )

    @property
    def context_window(self) -> int | None:
        return self.context_limit.window

    @property
    def context_safety_margin(self) -> int:
        return self.context_limit.safety_margin

    @property
    def min_tokens_for_generation(self) -> int:
        return self.context_limit.min_tokens_for_generation

    @property
    def context_warning_threshold(self) -> int | None:
        return self.context_limit.warning_threshold


class RewardConfig(StrictConfigModel):
    reward_per_success_tool_call: float = 1.0
    reward_per_failed_tool_call: float = -0.1
    reward_per_rejected_tool_call: float = -0.2
    reward_for_wrong_tool_call_format: float = 0.0
    max_success_reward: float = Field(default=1.0, description="Upper clamp for cumulative success reward per step.")
    min_failure_penalty: float = Field(default=-0.5, description="Lower clamp for cumulative failure penalty per step.")
    min_rejection_penalty: float = Field(default=-0.5, description="Lower clamp for cumulative rejection penalty per step.")


class OrchestrationConfig(StrictConfigModel):
    name: str = "default-orchestration"
    description: str = ""
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    tool_manager: ToolManagerConfig = Field(default_factory=ToolManagerConfig)
    allow_structured_task_input: bool = True


class RunConfig(StrictConfigModel):
    """Top-level configuration aggregating all sub-configs needed for a complete run."""

    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    model_client: ModelClientConfig = Field(default_factory=ModelClientConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    tools: list[ToolConfig] = Field(default_factory=list)
    external_servers: list[ExternalServerConfig] = Field(default_factory=list)
