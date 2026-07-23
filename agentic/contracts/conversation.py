# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypedDict, cast

from pydantic import BaseModel, Field

from agentic.contracts.messages import ConversationMessage as _ConversationMessage

if TYPE_CHECKING:
    from agentic.contracts.messages import ConversationMessage


class StepAction(StrEnum):
    """High-level action the orchestrator should take after a conversation step.

    The conversation runtime maps its detailed ``ConversationStage`` to one of
    these actions so that the orchestrator dispatches on a small, stable enum
    rather than pattern-matching on individual stages.
    """

    CALL_MODEL = "call_model"
    """Conversation is waiting for an assistant response — call the model."""

    EXECUTE_TOOLS = "execute_tools"
    """Pending tool calls — execute them via the tool manager."""

    DONE = "done"
    """Terminal — stop orchestration."""


class FormatErrorStrategy(StrEnum):
    """Strategy for handling tool-call format errors in assistant responses."""

    REPROMPT = "reprompt"
    """Send a recovery prompt so the model can correct itself."""

    ROLLBACK = "rollback"
    """Silently hide the message and let the model retry (counts toward max_assistant_rollbacks)."""

    IGNORE = "ignore"
    """Accept the malformed message as-is without any recovery action."""


class RollbackReason(StrEnum):
    """Reason for a rollback marker in the conversation transcript.

    Rollback markers are internal ``CONVERSATION_RUNTIME`` messages that hide
    preceding messages from the visible conversation.  The reason field makes
    the marker self-describing for observability and debugging.
    """

    CONTEXT_LIMIT = "context_limit"
    """Rollback triggered by context-window pressure."""

    ASSISTANT_REFUSAL = "assistant_refusal"
    """Rollback triggered by refusal-keyword detection in the assistant response."""

    FORMAT_ERROR = "format_error"
    """Rollback triggered by malformed tool-call format in the assistant response."""

    DUPLICATE_QUERY = "duplicate_query"
    """Rollback triggered because all tool calls are duplicates of previous queries."""

    TOOL_ERROR = "tool_error"
    """Rollback triggered by a tool execution error deemed retry-worthy."""

    TOKEN_EXHAUSTION = "token_exhaustion"  # noqa: S105
    """Rollback triggered after a length-truncated assistant generation."""


class FinalizationTrigger(StrEnum):
    """Reason why the runtime is force-finalizing the conversation.

    Stored in ``ConversationStepInfo["finalization_trigger"]`` when the
    stage is ``FORCE_FINAL_AWAIT_ASSISTANT`` or ``ASSISTANT_FORCE_COMPLETED``.
    """

    CONTEXT_LIMIT = "context_limit"
    TURN_LIMIT = "turn_limit"
    TOOLS_EXHAUSTED = "tools_exhausted"


class ConversationStage(StrEnum):
    """Structured stage emitted by the conversation runtime.

    Stages are grouped into five categories:

    1. **Setup** — initialization before the first model call.
    2. **Normal flow** — the standard user → assistant → tool loop.
    3. **Error recovery** — re-prompts after malformed tool calls.
    4. **Force finalization** — budget-triggered wrap-up (context or turn limit).
    5. **Terminal** — conversation is finished; no further steps.
    """

    # ── Setup ────────────────────────────────────────────────────────────
    NOT_INITIALIZED = "not_initialized"
    """No message yet; runtime has not appended anything."""
    SYSTEM_MESSAGE_SET = "system_message_set"
    """System message appended during initialization."""
    INITIAL_USER_MESSAGE_AWAIT_ASSISTANT = "initial_user_message_await_assistant"
    """Initial user message appended; waiting for assistant."""

    # ── Normal flow ──────────────────────────────────────────────────────
    TOOL_MESSAGES_AWAIT_ASSISTANT = "tool_messages_await_assistant"
    """Tool result messages appended; waiting for assistant."""
    ASSISTANT_MESSAGE_RECEIVED = "assistant_message_received"
    """Assistant message appended to transcript."""
    ASSISTANT_TOOL_CALLS_AWAIT_TOOL = "assistant_tool_calls_await_tool"
    """Assistant message included tool calls; waiting for tool outputs."""
    FINAL_RESPONSE_PROMPT_AWAIT_ASSISTANT = "final_response_prompt_await_assistant"
    """Final-response prompt appended; waiting for assistant."""

    # ── Error recovery ───────────────────────────────────────────────────
    TOOL_CALL_FORMAT_ERROR_PROMPT_AWAIT_ASSISTANT = "tool_call_format_error_prompt_await_assistant"
    """Recovery prompt appended after tool-call format error; waiting for assistant."""
    ASSISTANT_MESSAGE_ROLLBACK = "assistant_message_rollback"
    """Rollback marker appended to hide a bad assistant message (refusal, format error, duplicate query, tool error)."""

    # ── Force finalization ───────────────────────────────────────────────
    CONTEXT_LIMIT_ROLLBACK = "context_limit_rollback"
    """Rollback marker appended due to context pressure."""
    FORCE_FINAL_AWAIT_ASSISTANT = "force_final_await_assistant"
    """Force-finalization prompt (or silent) appended; waiting for assistant.

    The specific trigger (context, turns, tools) is in ``ConversationStepInfo["finalization_trigger"]``.
    Whether a prompt was injected can be determined from the appended messages.
    """

    # ── Terminal ─────────────────────────────────────────────────────────
    ASSISTANT_COMPLETED = "assistant_completed"
    """Assistant answer accepted as final (natural completion or silent force-final)."""
    ASSISTANT_FORCE_COMPLETED = "assistant_force_completed"
    """Forced final answer after an explicit force-finalization prompt.

    The specific trigger is in ``ConversationStepInfo["finalization_trigger"]``.
    """
    ASSISTANT_TOOL_CALLS_REJECTED_FORCE_FINAL = "assistant_tool_calls_rejected_force_final"
    """Assistant tool calls rejected during force-finalization."""
    ASSISTANT_TERMINATED_TOKEN_EXCEED = "assistant_terminated_token_exceed"  # noqa: S105
    """Assistant generation exhausted its token budget (finish_reason='length') without an accepted final answer."""


FORCE_FINALIZATION_STAGES: frozenset[ConversationStage] = frozenset(
    {
        ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT,
    }
)

CONTEXT_LIMIT_CHECKPOINT_STAGES: frozenset[ConversationStage] = frozenset(
    {
        ConversationStage.TOOL_MESSAGES_AWAIT_ASSISTANT,
        ConversationStage.ASSISTANT_MESSAGE_RECEIVED,
        ConversationStage.ASSISTANT_TOOL_CALLS_AWAIT_TOOL,
        ConversationStage.FINAL_RESPONSE_PROMPT_AWAIT_ASSISTANT,
        ConversationStage.TOOL_CALL_FORMAT_ERROR_PROMPT_AWAIT_ASSISTANT,
    }
)

RUNTIME_INTERNAL_STAGES: frozenset[ConversationStage] = frozenset(
    {
        ConversationStage.NOT_INITIALIZED,
        ConversationStage.SYSTEM_MESSAGE_SET,
        ConversationStage.ASSISTANT_MESSAGE_RECEIVED,
        ConversationStage.CONTEXT_LIMIT_ROLLBACK,
        ConversationStage.ASSISTANT_MESSAGE_ROLLBACK,
    }
)

STAGE_TO_ACTION: dict[ConversationStage, StepAction] = {
    # CALL_MODEL — orchestrator should call the model next.
    ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT: StepAction.CALL_MODEL,
    ConversationStage.TOOL_MESSAGES_AWAIT_ASSISTANT: StepAction.CALL_MODEL,
    ConversationStage.TOOL_CALL_FORMAT_ERROR_PROMPT_AWAIT_ASSISTANT: StepAction.CALL_MODEL,
    ConversationStage.FINAL_RESPONSE_PROMPT_AWAIT_ASSISTANT: StepAction.CALL_MODEL,
    ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT: StepAction.CALL_MODEL,
    # EXECUTE_TOOLS — orchestrator should run tool calls.
    ConversationStage.ASSISTANT_TOOL_CALLS_AWAIT_TOOL: StepAction.EXECUTE_TOOLS,
    # DONE — orchestration is complete.
    ConversationStage.ASSISTANT_COMPLETED: StepAction.DONE,
    ConversationStage.ASSISTANT_FORCE_COMPLETED: StepAction.DONE,
    ConversationStage.ASSISTANT_TOOL_CALLS_REJECTED_FORCE_FINAL: StepAction.DONE,
    ConversationStage.ASSISTANT_TERMINATED_TOKEN_EXCEED: StepAction.DONE,
}

STAGE_EMOJI: dict[ConversationStage, str] = {
    # Initialization
    ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT: "▶️",
    # Tool execution
    ConversationStage.ASSISTANT_TOOL_CALLS_AWAIT_TOOL: "〰️",
    ConversationStage.TOOL_MESSAGES_AWAIT_ASSISTANT: "〰️",
    # Tool-call format error recovery
    ConversationStage.TOOL_CALL_FORMAT_ERROR_PROMPT_AWAIT_ASSISTANT: "💢",
    # Refusal rollback
    ConversationStage.ASSISTANT_MESSAGE_ROLLBACK: "💢",
    # Final-response prompt
    ConversationStage.FINAL_RESPONSE_PROMPT_AWAIT_ASSISTANT: "〰️",
    # Force finalization
    ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT: "🔚",
    # Terminal
    ConversationStage.ASSISTANT_COMPLETED: "⏹️",
    ConversationStage.ASSISTANT_FORCE_COMPLETED: "⏹️",
    ConversationStage.ASSISTANT_TOOL_CALLS_REJECTED_FORCE_FINAL: "🚫",
    ConversationStage.ASSISTANT_TERMINATED_TOKEN_EXCEED: "🚫",
}

VALID_STAGE_TRANSITION: dict[ConversationStage, frozenset[ConversationStage]] = {
    ConversationStage.NOT_INITIALIZED: frozenset(
        {
            ConversationStage.SYSTEM_MESSAGE_SET,
            ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT,
        }
    ),
    ConversationStage.SYSTEM_MESSAGE_SET: frozenset({ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT}),
    ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT: frozenset(
        {
            ConversationStage.ASSISTANT_MESSAGE_RECEIVED,
            ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT,
        }
    ),
    ConversationStage.TOOL_MESSAGES_AWAIT_ASSISTANT: frozenset(
        {
            ConversationStage.ASSISTANT_MESSAGE_RECEIVED,
            ConversationStage.CONTEXT_LIMIT_ROLLBACK,
            ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT,
            ConversationStage.ASSISTANT_TERMINATED_TOKEN_EXCEED,
        }
    ),
    ConversationStage.ASSISTANT_MESSAGE_RECEIVED: frozenset(
        {
            ConversationStage.ASSISTANT_TOOL_CALLS_AWAIT_TOOL,
            ConversationStage.TOOL_CALL_FORMAT_ERROR_PROMPT_AWAIT_ASSISTANT,
            ConversationStage.FINAL_RESPONSE_PROMPT_AWAIT_ASSISTANT,
            ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT,
            ConversationStage.ASSISTANT_COMPLETED,
            ConversationStage.ASSISTANT_FORCE_COMPLETED,
            ConversationStage.ASSISTANT_TOOL_CALLS_REJECTED_FORCE_FINAL,
            ConversationStage.ASSISTANT_TERMINATED_TOKEN_EXCEED,
            ConversationStage.ASSISTANT_MESSAGE_ROLLBACK,
        }
    ),
    ConversationStage.ASSISTANT_TOOL_CALLS_AWAIT_TOOL: frozenset(
        {
            ConversationStage.TOOL_MESSAGES_AWAIT_ASSISTANT,
            ConversationStage.ASSISTANT_MESSAGE_ROLLBACK,
        }
    ),
    ConversationStage.TOOL_CALL_FORMAT_ERROR_PROMPT_AWAIT_ASSISTANT: frozenset(
        {
            ConversationStage.ASSISTANT_MESSAGE_RECEIVED,
            ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT,
        }
    ),
    ConversationStage.FINAL_RESPONSE_PROMPT_AWAIT_ASSISTANT: frozenset(
        {
            ConversationStage.ASSISTANT_MESSAGE_RECEIVED,
            ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT,
        }
    ),
    ConversationStage.CONTEXT_LIMIT_ROLLBACK: frozenset(
        {
            ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT,
            ConversationStage.ASSISTANT_TERMINATED_TOKEN_EXCEED,
        }
    ),
    ConversationStage.ASSISTANT_MESSAGE_ROLLBACK: frozenset(
        {
            ConversationStage.TOOL_MESSAGES_AWAIT_ASSISTANT,
            ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT,
        }
    ),
    ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT: frozenset(
        {
            ConversationStage.ASSISTANT_MESSAGE_RECEIVED,
            ConversationStage.CONTEXT_LIMIT_ROLLBACK,
        }
    ),
    ConversationStage.ASSISTANT_COMPLETED: frozenset(),
    ConversationStage.ASSISTANT_FORCE_COMPLETED: frozenset(),
    ConversationStage.ASSISTANT_TOOL_CALLS_REJECTED_FORCE_FINAL: frozenset(),
    ConversationStage.ASSISTANT_TERMINATED_TOKEN_EXCEED: frozenset(),
}


class ConversationStepInfo(TypedDict, total=False):
    """Diagnostic metadata emitted alongside a conversation transition."""

    stage: str
    emoji: str
    stage_path: list[str]
    visible_message_count: int
    raw_message_count: int
    appended_message_count: int
    assistant_turn_idx: int
    force_finalize_stage: str
    finalization_trigger: str
    has_early_stop_prompt: bool
    rollback_message_count: int
    tool_request_count: int
    wrong_tool_call_format_keywords: list[str]
    refusal_keywords: list[str]
    assistant_rollback_count: int
    rollback_reason: str
    context_limit_estimate: dict[str, Any]
    loop_accounting_step: dict[str, Any]
    context_budget_stop: bool
    skipped_budget_limit_final_response: bool
    skipped_turn_limit_final_response: bool
    skipped_context_limit_final_response: bool
    skipped_tools_exhausted_final_response: bool
    force_finalization_recovery: bool
    context_limit_error_recovery: bool
    discarded_context_limit_error_for_recovery: bool
    token_exhaustion_recovery: bool
    discarded_length_response_for_recovery: bool
    discarded_length_response_without_recovery: bool
    skipped_length_recovery_for_retry: bool
    # Format-error telemetry:
    # ``format_error_strategy`` records which strategy actually fired for this event
    # (``reprompt`` / ``rollback``) — inferrable from stage_path, but explicit here for the viz tool;
    # ``offending_content_preview`` is a single-line, length-capped snippet of the assistant content
    # that triggered the keyword hit, so dashboards can show what the model emitted
    # without pulling the full conversation;
    # ``offending_content_length`` is the original length (so a truncated preview can be flagged).
    format_error_strategy: str
    offending_content_preview: str
    offending_content_length: int
    direct_final_answer: str
    direct_final_answer_preview: str
    finish_reason: str
    token_exhaustion_had_content: bool


class ConversationStepResult(BaseModel):
    """Conversation-state snapshot returned by the conversation runtime."""

    stage: ConversationStage = Field(
        default=ConversationStage.NOT_INITIALIZED,
        description="Typed transition stage for the current runtime step.",
    )
    action: StepAction = Field(
        default=StepAction.CALL_MODEL,
        description="High-level action the orchestrator should take after this step.",
    )
    visible_conversation: list[ConversationMessage] = Field(
        description="Conversation view used for the next model turn. It may be masked or extended with forced-finalization prompts.",
    )
    full_conversation: list[ConversationMessage] = Field(
        default_factory=list,
        description="Unmasked transcript retained for tracing, reward evaluation, and final output extraction.",
    )
    appended_messages: list[ConversationMessage] = Field(
        default_factory=list,
        description="Messages appended during the current transition only.",
    )
    info: ConversationStepInfo = Field(
        default_factory=lambda: cast("ConversationStepInfo", {}),
        description="Diagnostic payload for counts and stage-specific metadata such as tool-request counts or forced-finalization causes.",
    )


ConversationStepResult.model_rebuild(_types_namespace={"ConversationMessage": _ConversationMessage})
