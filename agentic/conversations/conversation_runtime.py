# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Self, cast

from agentic.config.models import ConversationConfig  # noqa: TC001 — used at runtime in __init__
from agentic.contracts import (
    ConversationMessage,
    ConversationStage,
    ConversationStepInfo,
    ConversationStepResult,
    FinalizationTrigger,
    FormatErrorStrategy,
    MessageRole,
    RollbackReason,
    StepAction,
    ToolCall,
)
from agentic.contracts.conversation import (
    CONTEXT_LIMIT_CHECKPOINT_STAGES,
    FORCE_FINALIZATION_STAGES,
    STAGE_EMOJI,
    STAGE_TO_ACTION,
    VALID_STAGE_TRANSITION,
)
from agentic.contracts.markers import (
    CompactionMarker,
    DiscardAllMarker,
    parse_compaction_marker,
    parse_discard_all_marker,
    splice_compaction,
    splice_discard_all,
)
from agentic.conversations.context_budget import ContextBudgetEstimator
from agentic.conversations.context_length_tracker import ContextLengthTracker

if TYPE_CHECKING:
    from collections.abc import Callable


_OFFENDING_CONTENT_PREVIEW_CHARS = 240


class _RuntimeStepTimer:
    """Attribute public runtime-step wall-clock, excluding nested token-estimator time."""

    __slots__ = ("_runtime", "_start", "_start_tokenizer_ms")

    def __init__(self, runtime: ConversationRuntime) -> None:
        self._runtime = runtime
        self._start = 0.0
        self._start_tokenizer_ms = 0.0

    def __enter__(self) -> Self:
        self._start = time.perf_counter()
        self._start_tokenizer_ms = self._runtime._tokenizer_elapsed_ms
        return self

    def __exit__(self, *_: object) -> None:
        wall_ms = (time.perf_counter() - self._start) * 1000.0
        nested_tokenizer_ms = self._runtime._tokenizer_elapsed_ms - self._start_tokenizer_ms
        self._runtime._runtime_step_elapsed_ms += max(0.0, wall_ms - nested_tokenizer_ms)


def _preview_content(content: str, *, max_chars: int = _OFFENDING_CONTENT_PREVIEW_CHARS) -> str:
    """Render ``content`` as a single-line, length-capped snippet for trace metadata.

    Used by the format-error paths so the step log captures what the model emitted
    without pasting the full assistant message — the full message is still present in ``full_conversation``.
    """
    collapsed = " ".join(content.split())
    if len(collapsed) > max_chars:
        return collapsed[:max_chars] + "…"
    return collapsed


class ConversationRuntime:
    def __init__(
        self,
        *,
        config: ConversationConfig,
        tools: list[dict[str, Any]] | None,
        max_output_tokens: int,
        token_estimator: Callable[[list[ConversationMessage]], int],
        token_estimator_includes_tools: bool = False,
        token_estimator_is_additive: bool = False,
        context_limit_preflight_enabled: bool = True,
    ) -> None:
        self.config = config
        self.system_prompt = config.system_prompt
        self.user_prompt_template = config.user_prompt_template
        self.early_stop_announcement_prompt = config.early_stop_announcement_prompt
        self.final_response_prompt = config.final_response_prompt

        self.format_error = config.format_error
        self.tool_call_keywords = list(config.format_error.keywords)
        self.format_error_strategy = config.format_error.strategy
        self.format_error_reprompt = config.format_error.reprompt

        self.refusal_keywords = list(config.assistant_rollback.refusal_keywords)
        self.max_assistant_rollbacks = config.assistant_rollback.max_rollbacks

        self.tools = list(tools) if tools is not None else []
        self.max_turns = config.max_turns
        self.context_window = config.context_window
        self.max_output_tokens = max_output_tokens
        self.token_estimator = token_estimator
        self.token_estimator_includes_tools = token_estimator_includes_tools
        self.token_estimator_is_additive = token_estimator_is_additive
        self.context_limit_preflight_enabled = context_limit_preflight_enabled
        self.context_safety_margin = config.context_safety_margin
        self.min_tokens_for_generation = config.min_tokens_for_generation
        self.context_warning_threshold = config.context_warning_threshold or max_output_tokens
        self._context_budget = self._build_context_budget_estimator()
        self.agg_reject_tool_message = config.agg_reject_tool_message
        self._full_conversation: list[ConversationMessage] = []
        self._assistant_turn_count = 0
        self._pending_tool_calls: list[ToolCall] = []
        self._assistant_rollback_count = 0
        self._force_final_trigger: FinalizationTrigger | None = None
        self._force_final_has_prompt: bool = False
        self.stage: ConversationStage = ConversationStage.NOT_INITIALIZED
        self._tokenizer_elapsed_ms: float = 0.0
        self._runtime_step_elapsed_ms: float = 0.0
        visible_estimator = self._measured_estimate_tokens if self._can_incrementally_estimate_context() else None
        self._visible_context = ContextLengthTracker(visible_estimator)

    @property
    def tokenizer_elapsed_ms(self) -> float:
        """Cumulative time in milliseconds spent inside token-estimator calls."""
        return self._tokenizer_elapsed_ms

    @property
    def runtime_step_elapsed_ms(self) -> float:
        """Cumulative time in milliseconds spent inside runtime step methods, excluding token-estimator time."""
        return self._runtime_step_elapsed_ms

    def _measured_estimate_tokens(self, messages: list[ConversationMessage]) -> int:
        t0 = time.perf_counter()
        try:
            return self.token_estimator(messages)
        finally:
            self._tokenizer_elapsed_ms += (time.perf_counter() - t0) * 1000.0

    def _can_incrementally_estimate_context(self) -> bool:
        return (
            self.context_limit_preflight_enabled
            and self.context_window is not None
            and self.token_estimator_is_additive
            and not self.token_estimator_includes_tools
        )

    def _estimate_visible_tokens_with_additional(
        self,
        visible_conversation: list[ConversationMessage],
        additional_messages: list[ConversationMessage],
    ) -> int:
        if self._can_incrementally_estimate_context() and self._visible_context.matches(visible_conversation):
            return self._visible_context.estimate_with(additional_messages)
        # Defensive fallback for subclasses/tests that may pass a reconstructed
        # visible list. Non-additive estimators, including chat-template
        # renderers, must see the whole prompt because template overhead is not
        # a sum of independently rendered message diffs.
        return self._measured_estimate_tokens([*visible_conversation, *additional_messages])

    def record_observed_prompt_tokens(
        self,
        visible_conversation: list[ConversationMessage],
        *,
        input_tokens: int | None,
    ) -> None:
        """Calibrate incremental context estimates with provider-reported prompt tokens.

        Provider usage can correct the cheap incremental estimator in either
        direction, while exact tokenizer/rendering remains an opt-in concern for
        callers that need it.
        """
        if input_tokens is None or input_tokens <= 0:
            return
        if not self._can_incrementally_estimate_context():
            return
        if not self._visible_context.matches(visible_conversation):
            return
        raw_estimate = self._visible_context.raw_estimate_with(self._context_probe_messages())
        self._visible_context.anchor_observed_tokens(observed_tokens=input_tokens, raw_estimated_tokens=raw_estimate)

    def _context_probe_messages(self) -> list[ConversationMessage]:
        if self.tools and not self.token_estimator_includes_tools:
            return [ConversationMessage.user(str(self.tools))]
        return []

    def _build_context_budget_estimator(self) -> ContextBudgetEstimator:
        return ContextBudgetEstimator(
            context_window=self.context_window,
            safety_margin=self.context_safety_margin,
            min_tokens_for_generation=self.min_tokens_for_generation,
            warning_threshold=self.context_warning_threshold,
            max_output_tokens=self.max_output_tokens,
            tools=self.tools,
            token_estimator_includes_tools=self.token_estimator_includes_tools,
            estimate_tokens=self._measured_estimate_tokens,
            estimate_tokens_with_additional=self._estimate_visible_tokens_with_additional,
        )

    def _measure_runtime_step_elapsed(self) -> _RuntimeStepTimer:
        return _RuntimeStepTimer(self)

    def reset(self) -> None:
        self._full_conversation = []
        self._assistant_turn_count = 0
        self._pending_tool_calls = []
        self._assistant_rollback_count = 0
        self._force_final_trigger = None
        self._force_final_has_prompt = False
        self.stage = ConversationStage.NOT_INITIALIZED
        self._tokenizer_elapsed_ms = 0.0
        self._runtime_step_elapsed_ms = 0.0
        self._visible_context.reset()

    def clone(self) -> ConversationRuntime:
        cloned = self.__class__(
            config=self.config,
            tools=list(self.tools),
            max_output_tokens=self.max_output_tokens,
            token_estimator=self.token_estimator,
            token_estimator_includes_tools=self.token_estimator_includes_tools,
            token_estimator_is_additive=self.token_estimator_is_additive,
            context_limit_preflight_enabled=self.context_limit_preflight_enabled,
        )
        cloned._full_conversation = list(self._full_conversation)
        cloned._assistant_turn_count = self._assistant_turn_count
        cloned._pending_tool_calls = list(self._pending_tool_calls)
        cloned._assistant_rollback_count = self._assistant_rollback_count
        cloned._force_final_trigger = self._force_final_trigger
        cloned._force_final_has_prompt = self._force_final_has_prompt
        cloned.max_turns = self.max_turns
        cloned.stage = self.stage
        cloned._tokenizer_elapsed_ms = self._tokenizer_elapsed_ms
        cloned._runtime_step_elapsed_ms = self._runtime_step_elapsed_ms
        estimator = cloned._measured_estimate_tokens if cloned._can_incrementally_estimate_context() else None
        cloned._visible_context = self._visible_context.clone(estimate_tokens=estimator)
        self._clone_extra_state_to(cloned)
        return cloned

    def reopen_after_context_reset(self) -> None:
        """Resume normal generation after an external context reset.

        Discard-all resets truncate the visible prompt to the task prefix without
        resetting cumulative turn counters. If the reset preempts a force-final
        stage, the force-final prompt remains in the append-only trace but is
        hidden by the discard marker; the runtime should call the model as a
        fresh task continuation, not as a force-final response.
        """
        self._pending_tool_calls = []
        self._assistant_rollback_count = 0
        self._force_final_trigger = None
        self._force_final_has_prompt = False
        self.stage = ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT

    def _clone_extra_state_to(self, _cloned: ConversationRuntime) -> None:
        """Hook for subclasses with additional mutable runtime state."""

    def initialize_conversation(self, task_description: str) -> ConversationStepResult:
        with self._measure_runtime_step_elapsed():
            if self._full_conversation:
                msg = "Task initialization is only allowed on the first step."
                raise RuntimeError(msg)

            stage_path = [self.stage]
            appended_messages: list[ConversationMessage] = []

            if self.system_prompt:
                system_message = ConversationMessage.system(self.system_prompt)
                self._append_messages([system_message])
                appended_messages.append(system_message)
                self._transit_stage(ConversationStage.SYSTEM_MESSAGE_SET, stage_path=stage_path)

            initial_user_message = ConversationMessage.user(self.user_prompt_template.format(task=task_description))
            self._append_messages([initial_user_message])
            appended_messages.append(initial_user_message)
            self._transit_stage(
                ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT,
                stage_path=stage_path,
            )

            info = self._handle_wait_for_assistant_stage(stage_path=stage_path, appended_messages=appended_messages)
            return self._build_step_result(appended_messages=appended_messages, stage_path=stage_path, info=info)

    def apply_assistant_message(self, message: ConversationMessage, *, finish_reason: str | None = None) -> ConversationStepResult:
        """Process an assistant response and determine the next action.

        Dispatch order (highest priority first):

        1. **Token-budget exhaustion** — ``finish_reason="length"`` without a direct final answer → terminate.
        2. **Force finalization** — during force-final rounds, accept the answer (reject tool calls) → done.
        3. **Tool calls** — structured tool calls present → execute tools.
        4. **Content-quality checks** (only for text-only responses, skipped after final-response prompt):
           a. Refusal keyword detection → rollback (hide message, retry).
           b. Wrong tool-call format detection → rollback, re-prompt, or ignore per ``format_error_strategy``.
        5. **Direct final answer** — recipe-specific hook may accept an already-final text answer.
        6. **Final-response prompt** — inject summary/wrap-up prompt if configured → call model again.
        7. **Normal completion** → done.

        Content-quality checks (4a, 4b) are skipped in two situations:
        - During **force finalization** (#2): we must accept whatever the model produces; rolling back would loop.
        - After the **final-response prompt** (#5 already fired): the model's second attempt is accepted as-is.
        """
        with self._measure_runtime_step_elapsed():
            if message.role != MessageRole.ASSISTANT:
                msg = "apply_assistant_message only accepts assistant messages."
                raise ValueError(msg)

            self._ensure_initialized()
            self._ensure_stage_action(StepAction.CALL_MODEL, "assistant messages")

            prior_stage = self.stage
            stage_path = [self.stage]
            appended_messages = [message]

            self._append_messages([message])
            self._assistant_turn_count += 1
            self._transit_stage(ConversationStage.ASSISTANT_MESSAGE_RECEIVED, stage_path=stage_path)

            # 1. Token-budget exhaustion: the model produced nothing because the context window is full.
            if finish_reason == "length" and not message.content and not message.tool_calls:
                self._transit_stage(ConversationStage.ASSISTANT_TERMINATED_TOKEN_EXCEED, stage_path=stage_path)
                return self._build_step_result(appended_messages=appended_messages, stage_path=stage_path)

            # 2. Force finalization: accept the answer as-is (only reject tool calls).
            if prior_stage in FORCE_FINALIZATION_STAGES:
                return self._handle_force_finalization_response(message, stage_path=stage_path, appended_messages=appended_messages)

            # A length-truncated text response is terminal unless it already contains a direct final answer.
            # Accept boxed answers first, then stop on a length-limited response.
            if finish_reason == "length":
                direct_final_answer = self._extract_direct_final_answer(message)
                if direct_final_answer is not None:
                    self._assistant_rollback_count = 0
                    self._transit_stage(ConversationStage.ASSISTANT_COMPLETED, stage_path=stage_path)
                    return self._build_step_result(
                        appended_messages=appended_messages,
                        stage_path=stage_path,
                        info={
                            "direct_final_answer": direct_final_answer,
                            "direct_final_answer_preview": _preview_content(direct_final_answer),
                            "finish_reason": finish_reason,
                        },
                    )
                self._transit_stage(ConversationStage.ASSISTANT_TERMINATED_TOKEN_EXCEED, stage_path=stage_path)
                return self._build_step_result(
                    appended_messages=appended_messages,
                    stage_path=stage_path,
                    info={"finish_reason": finish_reason, "token_exhaustion_had_content": bool(message.content)},
                )

            budget_result = self._maybe_stop_or_warn_after_assistant(stage_path=stage_path, appended_messages=appended_messages)
            if budget_result is not None:
                return budget_result

            # 3. Tool calls: dispatch to tool execution.
            if message.tool_calls:
                return self._handle_tool_calls(message, stage_path=stage_path, appended_messages=appended_messages)

            # 4. Content-quality checks on text-only responses.
            #    Skipped after the final-response prompt round — the model's second attempt is accepted.
            if prior_stage != ConversationStage.FINAL_RESPONSE_PROMPT_AWAIT_ASSISTANT:
                # 4a. Refusal keyword detection.
                refusal_result = self._maybe_handle_refusal(message, prior_stage, stage_path=stage_path, appended_messages=appended_messages)
                if refusal_result is not None:
                    return refusal_result

                # 4b. Wrong tool-call format detection.
                format_result = self._maybe_handle_format_error(message, prior_stage, stage_path=stage_path, appended_messages=appended_messages)
                if format_result is not None:
                    return format_result

                # 5. Recipe-specific direct final answer.
                direct_final_answer = self._extract_direct_final_answer(message)
                if direct_final_answer is not None:
                    self._assistant_rollback_count = 0
                    self._transit_stage(ConversationStage.ASSISTANT_COMPLETED, stage_path=stage_path)
                    return self._build_step_result(
                        appended_messages=appended_messages,
                        stage_path=stage_path,
                        info={
                            "direct_final_answer": direct_final_answer,
                            "direct_final_answer_preview": _preview_content(direct_final_answer),
                        },
                    )

                # 6. Final-response prompt.
                if self.final_response_prompt is not None and self._should_prompt_for_final_response(message, prior_stage):
                    self._assistant_rollback_count = 0
                    return self._handle_final_response_prompt(stage_path=stage_path, appended_messages=appended_messages)

            # 7. Normal completion.
            self._assistant_rollback_count = 0
            self._transit_stage(ConversationStage.ASSISTANT_COMPLETED, stage_path=stage_path)
            return self._build_step_result(appended_messages=appended_messages, stage_path=stage_path)

    def apply_batch_tool_messages(
        self,
        tool_result_messages: list[ConversationMessage],
        *,
        tools_exhausted: bool = False,
        rejected_call_ids: frozenset[str] | None = None,
    ) -> ConversationStepResult:
        with self._measure_runtime_step_elapsed():
            all_messages = self._resolve_pending_tool_calls(tool_result_messages, rejected_call_ids=rejected_call_ids)

            if not all_messages:
                msg = "apply_batch_tool_messages requires at least one tool result message."
                raise ValueError(msg)

            if any(message.role != MessageRole.TOOL for message in all_messages):
                msg = "apply_batch_tool_messages requires a batch of tool messages."
                raise ValueError(msg)

            self._ensure_initialized()
            self._ensure_stage_in({ConversationStage.ASSISTANT_TOOL_CALLS_AWAIT_TOOL}, "tool messages")

            stage_path = [self.stage]
            appended_messages = list(all_messages)
            self._append_messages(all_messages)
            self._transit_stage(ConversationStage.TOOL_MESSAGES_AWAIT_ASSISTANT, stage_path=stage_path)

            if tools_exhausted:
                info = self._force_finalize_due_to_tools_exhausted(stage_path=stage_path, appended_messages=appended_messages)
                return self._build_step_result(appended_messages=appended_messages, stage_path=stage_path, info=info)

            budget_result = self._maybe_stop_or_warn_before_generation(stage_path=stage_path, appended_messages=appended_messages)
            if budget_result is not None:
                return budget_result

            info = self._handle_wait_for_assistant_stage(stage_path=stage_path, appended_messages=appended_messages)
            return self._build_step_result(appended_messages=appended_messages, stage_path=stage_path, info=info)

    # ------------------------------------------------------------------
    # Stage transition guards
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if not self._full_conversation:
            msg = "Conversation must be initialized before applying messages."
            raise RuntimeError(msg)

    def _ensure_stage_in(self, allowed_stages: set[ConversationStage] | frozenset[ConversationStage], label: str) -> None:
        if self.stage not in allowed_stages:
            msg = f"Cannot apply {label} while runtime stage is {self.stage.value}."
            raise RuntimeError(msg)

    def _ensure_stage_action(self, expected_action: StepAction, label: str) -> None:
        if STAGE_TO_ACTION.get(self.stage) != expected_action:
            msg = f"Cannot apply {label} while runtime stage is {self.stage.value}."
            raise RuntimeError(msg)

    def _transit_stage(self, next_stage: ConversationStage, *, stage_path: list[ConversationStage] | None = None) -> None:
        allowed_transitions = VALID_STAGE_TRANSITION.get(self.stage, frozenset())
        if next_stage not in allowed_transitions:
            msg = f"Invalid stage transition from {self.stage.value} to {next_stage.value}."
            raise RuntimeError(msg)
        self.stage = next_stage
        if stage_path is not None:
            stage_path.append(next_stage)

    # ------------------------------------------------------------------
    # Assistant message dispatch
    # ------------------------------------------------------------------

    def _handle_force_finalization_response(
        self,
        message: ConversationMessage,
        *,
        stage_path: list[ConversationStage],
        appended_messages: list[ConversationMessage],
    ) -> ConversationStepResult:
        if message.tool_calls:
            self._transit_stage(
                ConversationStage.ASSISTANT_TOOL_CALLS_REJECTED_FORCE_FINAL,
                stage_path=stage_path,
            )
            return self._build_step_result(
                appended_messages=appended_messages,
                stage_path=stage_path,
                info={"tool_request_count": len(message.tool_calls)},
            )
        self._transit_stage(self._resolve_force_completed_stage(), stage_path=stage_path)
        return self._build_step_result(appended_messages=appended_messages, stage_path=stage_path)

    def _handle_tool_calls(
        self,
        message: ConversationMessage,
        *,
        stage_path: list[ConversationStage],
        appended_messages: list[ConversationMessage],
    ) -> ConversationStepResult:
        assert message.tool_calls
        self._pending_tool_calls = list(message.tool_calls)
        self._transit_stage(ConversationStage.ASSISTANT_TOOL_CALLS_AWAIT_TOOL, stage_path=stage_path)
        info: dict[str, object] = {"tool_request_count": len(message.tool_calls)}
        return self._build_step_result(
            appended_messages=appended_messages,
            stage_path=stage_path,
            info=info,
        )

    def _handle_wrong_tool_call_format(
        self,
        detected_keywords: list[str],
        *,
        stage_path: list[ConversationStage],
        appended_messages: list[ConversationMessage],
        offending_content: str | None = None,
    ) -> ConversationStepResult:
        recovery_prompt = ConversationMessage.user(self.format_error_reprompt)
        self._append_messages([recovery_prompt])
        appended_messages.append(recovery_prompt)
        self._transit_stage(
            ConversationStage.TOOL_CALL_FORMAT_ERROR_PROMPT_AWAIT_ASSISTANT,
            stage_path=stage_path,
        )
        info = self._handle_wait_for_assistant_stage(
            stage_path=stage_path,
            appended_messages=appended_messages,
            check_context_limit=False,
        )
        info["wrong_tool_call_format_keywords"] = detected_keywords
        info["format_error_strategy"] = FormatErrorStrategy.REPROMPT.value
        if offending_content is not None:
            info["offending_content_preview"] = _preview_content(offending_content)
            info["offending_content_length"] = len(offending_content)
        return self._build_step_result(appended_messages=appended_messages, stage_path=stage_path, info=info)

    def _handle_final_response_prompt(
        self,
        *,
        stage_path: list[ConversationStage],
        appended_messages: list[ConversationMessage],
    ) -> ConversationStepResult:
        assert self.final_response_prompt
        final_prompt = ConversationMessage.user(self.final_response_prompt)
        self._append_messages([final_prompt])
        appended_messages.append(final_prompt)
        self._transit_stage(
            ConversationStage.FINAL_RESPONSE_PROMPT_AWAIT_ASSISTANT,
            stage_path=stage_path,
        )
        info = self._handle_wait_for_assistant_stage(
            stage_path=stage_path,
            appended_messages=appended_messages,
            check_context_limit=False,
        )
        return self._build_step_result(appended_messages=appended_messages, stage_path=stage_path, info=info)

    def _extract_direct_final_answer(self, _message: ConversationMessage) -> str | None:
        """Return a final answer to accept before injecting ``final_response_prompt``.

        Subclasses can override this for recipe-specific final-answer formats. The
        base runtime deliberately stays format-agnostic and preserves the existing
        summary-prompt behavior.
        """
        return None

    def _should_prompt_for_final_response(self, _message: ConversationMessage, _prior_stage: ConversationStage) -> bool:
        """Return whether a normal text response should get a final-answer reprompt."""
        return True

    # ------------------------------------------------------------------
    # Limit enforcement (context window and turn budget)
    # ------------------------------------------------------------------

    def _handle_wait_for_assistant_stage(
        self,
        *,
        stage_path: list[ConversationStage],
        appended_messages: list[ConversationMessage],
        check_context_limit: bool = True,
    ) -> dict[str, object]:
        if check_context_limit:
            info = self._maybe_force_due_to_context_limit(stage_path=stage_path, appended_messages=appended_messages)
            if info is not None:
                return info

        if self._should_force_finalize_due_to_turn_limit():
            return self._inject_force_finalization(FinalizationTrigger.TURN_LIMIT, stage_path=stage_path, appended_messages=appended_messages)

        return {}

    def _maybe_force_due_to_context_limit(
        self,
        *,
        stage_path: list[ConversationStage],
        appended_messages: list[ConversationMessage],
    ) -> dict[str, object] | None:
        if not self.context_limit_preflight_enabled:
            return None
        if self.stage not in CONTEXT_LIMIT_CHECKPOINT_STAGES:
            return None

        visible_conversation = self._build_visible_conversation()
        context_info = self._estimate_context_limit_budget(visible_conversation)
        if context_info["hard_stop"]:
            self._transit_stage(ConversationStage.ASSISTANT_TERMINATED_TOKEN_EXCEED, stage_path=stage_path)
            return {
                "context_limit_estimate": context_info,
                "finalization_trigger": FinalizationTrigger.CONTEXT_LIMIT.value,
                "context_budget_stop": True,
            }
        if not context_info["needs_final_warning"] and context_info["warning_prompt_fits"]:
            return None

        return self._force_finalize_due_to_context_limit(
            stage_path=stage_path,
            appended_messages=appended_messages,
            context_info=context_info,
        )

    def _maybe_stop_or_warn_after_assistant(
        self,
        *,
        stage_path: list[ConversationStage],
        appended_messages: list[ConversationMessage],
    ) -> ConversationStepResult | None:
        if not self.context_limit_preflight_enabled:
            return None
        if self.min_tokens_for_generation <= 0:
            return None
        if self.stage != ConversationStage.ASSISTANT_MESSAGE_RECEIVED:
            return None
        context_info = self._estimate_generation_budget(self._build_visible_conversation())
        if context_info["hard_stop"]:
            self._transit_stage(ConversationStage.ASSISTANT_TERMINATED_TOKEN_EXCEED, stage_path=stage_path)
            return self._build_step_result(
                appended_messages=appended_messages,
                stage_path=stage_path,
                info={
                    "context_limit_estimate": context_info,
                    "finalization_trigger": FinalizationTrigger.CONTEXT_LIMIT.value,
                    "context_budget_stop": True,
                },
            )
        if context_info["needs_final_warning"]:
            info = self._force_finalize_due_to_context_limit(
                stage_path=stage_path,
                appended_messages=appended_messages,
                context_info=context_info,
                rollback=False,
            )
            return self._build_step_result(appended_messages=appended_messages, stage_path=stage_path, info=info)
        return None

    def _maybe_stop_or_warn_before_generation(
        self,
        *,
        stage_path: list[ConversationStage],
        appended_messages: list[ConversationMessage],
    ) -> ConversationStepResult | None:
        if not self.context_limit_preflight_enabled:
            return None
        if self.min_tokens_for_generation <= 0:
            return None
        if self.stage not in CONTEXT_LIMIT_CHECKPOINT_STAGES:
            return None
        context_info = self._estimate_generation_budget(self._build_visible_conversation())
        if context_info["hard_stop"]:
            self._transit_stage(ConversationStage.ASSISTANT_TERMINATED_TOKEN_EXCEED, stage_path=stage_path)
            return self._build_step_result(
                appended_messages=appended_messages,
                stage_path=stage_path,
                info={
                    "context_limit_estimate": context_info,
                    "finalization_trigger": FinalizationTrigger.CONTEXT_LIMIT.value,
                    "context_budget_stop": True,
                },
            )
        if context_info["needs_final_warning"]:
            info = self._force_finalize_due_to_context_limit(
                stage_path=stage_path,
                appended_messages=appended_messages,
                context_info=context_info,
            )
            return self._build_step_result(appended_messages=appended_messages, stage_path=stage_path, info=info)
        return None

    def force_finalize_due_to_context_limit_after_error(
        self,
        *,
        error_info: dict[str, object] | None = None,
    ) -> ConversationStepResult:
        """Force-finalize after the provider rejects a request for context length.

        This is the recovery path for endpoint-side context checks that are more
        accurate than the local estimator.  It uses the same rollback/finalization
        mechanics as the proactive preflight path so trace format stays unchanged.
        """
        with self._measure_runtime_step_elapsed():
            self._ensure_initialized()
            stage_path = [self.stage]
            appended_messages: list[ConversationMessage] = []
            visible_conversation = self._build_visible_conversation()
            if self.context_limit_preflight_enabled:
                context_info = self._estimate_context_limit_budget(visible_conversation)
            else:
                context_info = self._provider_context_limit_info()
            if error_info:
                context_info["provider_error"] = dict(error_info)

            if self.stage == ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT:
                info = {
                    "finalization_trigger": FinalizationTrigger.CONTEXT_LIMIT.value,
                    "context_limit_estimate": context_info,
                    "context_limit_error_during_force_final": True,
                }
                return self._build_step_result(appended_messages=appended_messages, stage_path=stage_path, info=info)

            info = self._force_finalize_due_to_context_limit(
                stage_path=stage_path,
                appended_messages=appended_messages,
                context_info=context_info,
            )
            return self._build_step_result(appended_messages=appended_messages, stage_path=stage_path, info=info)

    def rollback_latest_tool_exchange_and_force_finalize(self, *, reason: str = RollbackReason.TOKEN_EXHAUSTION) -> ConversationStepResult | None:
        """Roll back the latest tool exchange and re-enter force-final generation.

        This recovery is intentionally explicit rather than part of the normal
        stage flow. Orchestrators can use it after provider-side generation
        limits, keeping the generic runtime independent of attempt policy.
        """
        with self._measure_runtime_step_elapsed():
            self._ensure_initialized()
            self._ensure_stage_action(StepAction.CALL_MODEL, "token-exhaustion recovery")
            visible_conversation = self._build_visible_conversation()
            rollback_message_count = self._compute_latest_tool_exchange_rollback_count(visible_conversation)
            if rollback_message_count <= 0:
                return None
            if ConversationStage.CONTEXT_LIMIT_ROLLBACK not in VALID_STAGE_TRANSITION.get(self.stage, frozenset()):
                return None

            stage_path = [self.stage]
            appended_messages: list[ConversationMessage] = []
            rollback_marker = self._build_rollback_marker(rollback_message_count, reason=reason)
            self._append_messages([rollback_marker])
            appended_messages.append(rollback_marker)
            self._transit_stage(ConversationStage.CONTEXT_LIMIT_ROLLBACK, stage_path=stage_path)
            info = self._inject_force_finalization(FinalizationTrigger.CONTEXT_LIMIT, stage_path=stage_path, appended_messages=appended_messages)
            info["rollback_message_count"] = rollback_message_count
            info["rollback_reason"] = reason
            info["force_finalization_recovery"] = True
            if reason == RollbackReason.TOKEN_EXHAUSTION:
                info["token_exhaustion_recovery"] = True
            elif reason == RollbackReason.CONTEXT_LIMIT:
                info["context_limit_error_recovery"] = True
            return self._build_step_result(appended_messages=appended_messages, stage_path=stage_path, info=info)

    def _force_finalize_due_to_context_limit(
        self,
        *,
        stage_path: list[ConversationStage],
        appended_messages: list[ConversationMessage],
        context_info: dict[str, object],
        rollback: bool = True,
    ) -> dict[str, object]:
        visible_conversation = self._build_visible_conversation()
        can_rollback = rollback and ConversationStage.CONTEXT_LIMIT_ROLLBACK in VALID_STAGE_TRANSITION.get(self.stage, frozenset())
        rollback_message_count = self._compute_rollback_message_count(visible_conversation) if can_rollback else 0
        if can_rollback and rollback_message_count > 0:
            rollback_marker = self._build_rollback_marker(rollback_message_count, reason=RollbackReason.CONTEXT_LIMIT)
            self._append_messages([rollback_marker])
            appended_messages.append(rollback_marker)
            self._transit_stage(ConversationStage.CONTEXT_LIMIT_ROLLBACK, stage_path=stage_path)

        if self.context_limit_preflight_enabled:
            warning_fit_info = self._estimate_final_warning_fit(self._build_visible_conversation())
        else:
            warning_fit_info = {
                "estimated_prompt_with_warning_tokens": 0,
                "warning_remaining_tokens": None,
                "warning_prompt_fits": True,
            }
        context_info.update(warning_fit_info)
        if self.min_tokens_for_generation > 0 and not warning_fit_info["warning_prompt_fits"]:
            self._transit_stage(ConversationStage.ASSISTANT_TERMINATED_TOKEN_EXCEED, stage_path=stage_path)
            return {
                "finalization_trigger": FinalizationTrigger.CONTEXT_LIMIT.value,
                "rollback_message_count": rollback_message_count,
                "context_limit_estimate": context_info,
                "context_budget_stop": True,
            }

        info = self._inject_force_finalization(FinalizationTrigger.CONTEXT_LIMIT, stage_path=stage_path, appended_messages=appended_messages)
        info["rollback_message_count"] = rollback_message_count
        info["context_limit_estimate"] = context_info
        return info

    def _should_force_finalize_due_to_context_limit(self, visible_conversation: list[ConversationMessage]) -> bool:
        if not self.context_limit_preflight_enabled:
            return False
        context_info = self._estimate_context_limit_budget(visible_conversation)
        return bool(context_info["hard_stop"] or context_info["needs_final_warning"])

    def _estimate_context_limit_budget(self, visible_conversation: list[ConversationMessage]) -> dict[str, object]:
        return self._context_budget.estimate_force_finalization_budget(
            visible_conversation,
            force_final_prompt=self._build_early_stop_prompt(),
        )

    def _provider_context_limit_info(self) -> dict[str, object]:
        return {
            "estimated_prompt_tokens": 0,
            "reserved_output_tokens": self.max_output_tokens,
            "context_safety_margin": self.context_safety_margin,
            "estimated_total_tokens": 0,
            "remaining_tokens": None,
            "min_tokens_for_generation": self.min_tokens_for_generation,
            "context_warning_threshold": self.context_warning_threshold,
            "context_window": self.context_window,
            "hard_stop": False,
            "needs_final_warning": False,
            "exceeds_context": True,
            "warning_prompt_fits": True,
            "estimated_prompt_with_warning_tokens": 0,
            "warning_remaining_tokens": None,
            "context_limit_detection": "provider_error",
        }

    def _estimate_final_warning_fit(self, visible_conversation: list[ConversationMessage]) -> dict[str, object]:
        return self._context_budget.estimate_force_final_prompt_fit(
            visible_conversation,
            force_final_prompt=self._build_early_stop_prompt(),
        )

    def _estimate_generation_budget(self, visible_conversation: list[ConversationMessage]) -> dict[str, object]:
        return self._context_budget.estimate_generation_budget(visible_conversation)

    def _should_force_finalize_due_to_turn_limit(self) -> bool:
        if self.max_turns is None:
            return False
        return self._assistant_turn_count + 1 == self.max_turns

    def _inject_force_finalization(
        self,
        trigger: FinalizationTrigger,
        *,
        stage_path: list[ConversationStage],
        appended_messages: list[ConversationMessage],
    ) -> dict[str, object]:
        """Inject a force-finalization prompt (or silent transition) and record the trigger."""
        early_stop_prompt = self._build_early_stop_prompt()
        has_prompt = early_stop_prompt is not None
        if has_prompt:
            force_prompt = ConversationMessage.user(early_stop_prompt)
            self._append_messages([force_prompt])
            appended_messages.append(force_prompt)
        self._force_final_trigger = trigger
        self._force_final_has_prompt = has_prompt
        self._transit_stage(ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT, stage_path=stage_path)
        return {
            "force_finalize_stage": self.stage.value,
            "finalization_trigger": trigger.value,
            "has_early_stop_prompt": has_prompt,
        }

    def _resolve_force_completed_stage(self) -> ConversationStage:
        """Determine the terminal stage after a force-finalized response."""
        if getattr(self, "_force_final_has_prompt", True):
            return ConversationStage.ASSISTANT_FORCE_COMPLETED
        return ConversationStage.ASSISTANT_COMPLETED

    def _force_finalize_due_to_tools_exhausted(
        self,
        *,
        stage_path: list[ConversationStage],
        appended_messages: list[ConversationMessage],
    ) -> dict[str, object]:
        """Inject a force-finalization prompt because all tool budgets are exhausted."""
        info = self._inject_force_finalization(FinalizationTrigger.TOOLS_EXHAUSTED, stage_path=stage_path, appended_messages=appended_messages)
        info["tools_exhausted"] = True
        return info

    def _build_early_stop_prompt(self) -> str | None:
        if self.early_stop_announcement_prompt is None and self.final_response_prompt is None:
            return None
        if self.final_response_prompt is None:
            return self.early_stop_announcement_prompt
        if self.early_stop_announcement_prompt is None:
            return self.final_response_prompt
        return f"{self.early_stop_announcement_prompt}\n\n{self.final_response_prompt}"

    # ------------------------------------------------------------------
    # Conversation visibility (rollback and masking)
    # ------------------------------------------------------------------

    def _build_visible_conversation(self) -> list[ConversationMessage]:
        return self._visible_context.visible_messages()

    def _build_rollback_marker(self, rollback_message_count: int, reason: str) -> ConversationMessage:
        """Build a rollback marker that hides *rollback_message_count* preceding messages from the visible conversation.

        Args:
            rollback_message_count: Number of messages to hide (working backward from the marker's position).
            reason: Why the rollback is happening (e.g. ``RollbackReason.CONTEXT_LIMIT``).
        """
        return ConversationMessage.runtime(
            json.dumps({"rollback_message_count": rollback_message_count, "reason": reason}),
            name="rollback",
        )

    def _compute_rollback_message_count(self, visible_conversation: list[ConversationMessage]) -> int:
        # Rollback markers operate on the currently visible conversation.
        rollback_count = 0
        for message in reversed(visible_conversation):
            if message.role == MessageRole.SYSTEM:
                break
            rollback_count += 1
            if message.role == MessageRole.ASSISTANT:
                return rollback_count
        return rollback_count

    def _compute_latest_tool_exchange_rollback_count(self, visible_conversation: list[ConversationMessage]) -> int:
        rollback_count = 0
        idx = len(visible_conversation) - 1
        final_prompt = self._build_early_stop_prompt()
        if idx >= 0 and final_prompt and visible_conversation[idx].role == MessageRole.USER and visible_conversation[idx].content == final_prompt:
            rollback_count += 1
            idx -= 1

        tool_result_count = 0
        while idx >= 0 and self._is_tool_result_message_for_rollback(visible_conversation[idx]):
            rollback_count += 1
            tool_result_count += 1
            idx -= 1

        if tool_result_count == 0:
            return 0
        if idx < 0:
            return 0
        if visible_conversation[idx].role == MessageRole.ASSISTANT and visible_conversation[idx].tool_calls:
            return rollback_count + 1
        return 0

    @staticmethod
    def _is_tool_result_message_for_rollback(message: ConversationMessage) -> bool:
        return message.role == MessageRole.TOOL or (message.role == MessageRole.USER and message.tool_call_id is not None)

    def _parse_rollback_marker(self, message: ConversationMessage) -> tuple[int, str | None]:
        """Parse rollback count and reason from a rollback marker message.

        Returns ``(0, None)`` for non-rollback messages.
        """
        if message.role != MessageRole.CONVERSATION_RUNTIME or message.name != "rollback" or message.content is None:
            return 0, None
        try:
            payload = json.loads(message.content)
        except (TypeError, json.JSONDecodeError):
            return 0, None
        if not isinstance(payload, dict):
            return 0, None
        rollback_count = payload.get("rollback_message_count", 0)
        reason = payload.get("reason")
        count = rollback_count if isinstance(rollback_count, int) and rollback_count > 0 else 0
        return count, reason

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def untruncated_visible_conversation(self) -> list[ConversationMessage]:
        """Visible conversation with markers applied but WITHOUT per-request rendering.

        Subclasses may override ``_build_visible_conversation`` with lossy
        per-request rendering (e.g. tool-result truncation). Compaction must
        operate on the underlying tracked view: its splice indices address this
        list, and the summary should be generated from real message content.
        """
        return self._visible_context.visible_messages()

    def _append_messages(self, messages: list[ConversationMessage]) -> None:
        self._validate_append_messages(messages)
        self._full_conversation.extend(messages)
        self._apply_to_visible_context(messages)

    def _apply_to_visible_context(self, messages: list[ConversationMessage]) -> None:
        """Replay *messages* into ``_visible_context``, interpreting runtime markers.

        ``_full_conversation`` is append-only; the model-visible conversation is
        always derived from it by replaying CONVERSATION_RUNTIME markers in order:

        - ``rollback`` marker: hide the last N visible messages.
        - ``compaction`` marker: replace a visible range with its summary message.
        - ``discard_all`` marker: truncate the visible conversation to its
          leading ``prefix_len`` messages (discard every later tool-call turn).

        Shared by incremental append (``_append_messages``) and full rebuild
        (``_rebuild_visible_context``) so both derivations always agree.
        """
        visible_messages: list[ConversationMessage] = []
        for message in messages:
            if message.role == MessageRole.CONVERSATION_RUNTIME:
                if visible_messages:
                    self._visible_context.append_many(visible_messages)
                    visible_messages = []
                rollback_message_count, _reason = self._parse_rollback_marker(message)
                if rollback_message_count:
                    self._visible_context.rollback_last(rollback_message_count)
                    continue
                compaction = parse_compaction_marker(message)
                if compaction is not None:
                    self._apply_compaction_to_visible_context(compaction)
                    continue
                discard = parse_discard_all_marker(message)
                if discard is not None:
                    self._apply_discard_all_to_visible_context(discard)
                continue
            visible_messages.append(message)
        if visible_messages:
            self._visible_context.append_many(visible_messages)

    def _apply_compaction_to_visible_context(self, compaction: CompactionMarker) -> None:
        """Splice the tracked visible conversation per a parsed compaction marker.

        No-op when the recorded range no longer fits the current visible
        conversation (e.g. a malformed or stale marker in a loaded trace).
        Token calibration learned from provider usage survives the splice
        (``replace_visible`` keeps the ratio and drops only the anchor).
        """
        rebuilt = splice_compaction(self._visible_context.visible_messages(), compaction)
        if rebuilt is None:
            return
        self._visible_context.replace_visible(rebuilt)

    def _apply_discard_all_to_visible_context(self, discard: DiscardAllMarker) -> None:
        """Truncate the tracked visible conversation per a parsed discard-all marker.

        No-op when the recorded prefix no longer fits the current visible
        conversation (e.g. a malformed or stale marker in a loaded trace).
        Token calibration learned from provider usage survives the reset
        (``replace_visible`` keeps the ratio and drops only the anchor).
        """
        rebuilt = splice_discard_all(self._visible_context.visible_messages(), discard)
        if rebuilt is None:
            return
        self._visible_context.replace_visible(rebuilt)

    def _rebuild_visible_context(self) -> None:
        """Rebuild the visible-context tracker by replaying ``_full_conversation``.

        ``_full_conversation`` is the append-only source of truth; the visible
        prompt is derived from it by replaying runtime markers (rollback and
        compaction) via ``_apply_to_visible_context``.
        """
        self._visible_context.reset()
        self._apply_to_visible_context(self._full_conversation)

    def _validate_append_messages(self, messages: list[ConversationMessage]) -> None:
        has_messages = bool(self._full_conversation)
        for message in messages:
            if not has_messages and message.role not in (MessageRole.SYSTEM, MessageRole.USER):
                msg = "Conversation must start with a system or user message."
                raise RuntimeError(msg)
            if message.role == MessageRole.SYSTEM and has_messages:
                msg = "System messages can only be set during conversation initialization."
                raise ValueError(msg)
            has_messages = True

    def _resolve_pending_tool_calls(
        self,
        tool_result_messages: list[ConversationMessage],
        *,
        rejected_call_ids: frozenset[str] | None = None,
    ) -> list[ConversationMessage]:
        """Validate tool results against pending state and clear pending state."""
        pending = self._pending_tool_calls
        self._pending_tool_calls = []

        if len(tool_result_messages) != len(pending):
            msg = (
                f"Expected {len(pending)} tool result messages, but got {len(tool_result_messages)}.\nExpected: {pending}Got: {tool_result_messages}"
            )
            raise ValueError(msg)

        expected_ids = {tc.id for tc in pending}
        actual_ids = {m.tool_call_id for m in tool_result_messages if m.tool_call_id}
        if expected_ids != actual_ids:
            msg = f"Tool result IDs {actual_ids} do not match expected call IDs {expected_ids}."
            raise ValueError(msg)

        if not self.agg_reject_tool_message:
            return list(tool_result_messages)

        return self._aggregate_rejected_tool_messages(list(tool_result_messages), rejected_call_ids=rejected_call_ids or frozenset())

    def _aggregate_rejected_tool_messages(
        self,
        messages: list[ConversationMessage],
        *,
        rejected_call_ids: frozenset[str],
    ) -> list[ConversationMessage]:
        """Merge rejected tool messages that share identical content into single messages with comma-joined tool_call_ids.

        Only messages whose ``tool_call_id`` appears in *rejected_call_ids* are
        candidates for merging.  Non-rejected messages always pass through as-is,
        even if they happen to share content with another message.
        """
        result: list[ConversationMessage] = []
        # Group rejected messages by content; non-rejected messages pass through immediately.
        rejected_content_groups: OrderedDict[str | None, list[ConversationMessage]] = OrderedDict()
        for msg in messages:
            is_rejected = msg.tool_call_id is not None and msg.tool_call_id in rejected_call_ids
            if not is_rejected:
                result.append(msg)
                continue
            key = msg.content
            if key not in rejected_content_groups:
                rejected_content_groups[key] = []
            rejected_content_groups[key].append(msg)

        for group in rejected_content_groups.values():
            if len(group) == 1:
                result.append(group[0])
            else:
                merged_id = ",".join(m.tool_call_id for m in group if m.tool_call_id)
                merged_name = ",".join(sorted({m.name for m in group if m.name})) or None
                merged = ConversationMessage(
                    role=group[0].role,
                    content=group[0].content,
                    name=merged_name,
                    tool_call_id=merged_id,
                )
                result.append(merged)
        return result

    def _detect_wrong_tool_call_format_keywords(self, message: ConversationMessage) -> list[str]:
        if message.tool_calls:
            return []
        content = message.content or ""
        return [keyword for keyword in self.tool_call_keywords if keyword and keyword in content]

    # ------------------------------------------------------------------
    # Content-quality checks (refusal, format error)
    # ------------------------------------------------------------------

    def _detect_refusal_keywords(self, message: ConversationMessage) -> list[str]:
        """Detect refusal keywords in an assistant text-only response."""
        if not self.refusal_keywords or message.tool_calls:
            return []
        content = message.content or ""
        return [kw for kw in self.refusal_keywords if kw and kw in content]

    def _maybe_handle_refusal(
        self,
        message: ConversationMessage,
        prior_stage: ConversationStage,
        *,
        stage_path: list[ConversationStage],
        appended_messages: list[ConversationMessage],
    ) -> ConversationStepResult | None:
        """Check for refusal keywords and roll back the assistant message if detected.

        Returns ``None`` when no refusal is detected or the rollback limit has been reached.
        """
        if self.max_assistant_rollbacks <= 0:
            return None
        if self._assistant_rollback_count >= self.max_assistant_rollbacks:
            return None

        detected = self._detect_refusal_keywords(message)
        if not detected:
            return None

        return self._do_assistant_message_rollback(
            prior_stage,
            reason=RollbackReason.ASSISTANT_REFUSAL,
            stage_path=stage_path,
            appended_messages=appended_messages,
            extra_info={"refusal_keywords": detected},
        )

    def _maybe_handle_format_error(
        self,
        message: ConversationMessage,
        prior_stage: ConversationStage,
        *,
        stage_path: list[ConversationStage],
        appended_messages: list[ConversationMessage],
    ) -> ConversationStepResult | None:
        """Handle wrong tool-call format in an assistant text-only response.

        Strategy is determined by ``format_error_strategy``:

        * ``"rollback"``: silently hide the message and let the model retry.
          Returns ``None`` if the rollback limit has been reached (falls through to completion).
        * ``"reprompt"``: append a recovery prompt so the model can correct itself.
        * ``"ignore"``: do nothing — the message is accepted as-is.
        """
        if self.format_error_strategy == FormatErrorStrategy.IGNORE:
            return None

        detected_keywords = self._detect_wrong_tool_call_format_keywords(message)
        if not detected_keywords:
            return None

        if self.format_error_strategy == FormatErrorStrategy.ROLLBACK:
            if self._assistant_rollback_count >= self.max_assistant_rollbacks:
                return None  # Limit hit — fall through to normal completion
            extra: dict[str, object] = {
                "wrong_tool_call_format_keywords": detected_keywords,
                "format_error_strategy": FormatErrorStrategy.ROLLBACK.value,
            }
            content = message.content or ""
            if content:
                extra["offending_content_preview"] = _preview_content(content)
                extra["offending_content_length"] = len(content)
            return self._do_assistant_message_rollback(
                prior_stage,
                reason=RollbackReason.FORMAT_ERROR,
                stage_path=stage_path,
                appended_messages=appended_messages,
                extra_info=extra,
            )

        # "reprompt" strategy (default)
        return self._handle_wrong_tool_call_format(
            detected_keywords,
            stage_path=stage_path,
            appended_messages=appended_messages,
            offending_content=message.content,
        )

    def _do_assistant_message_rollback(
        self,
        prior_stage: ConversationStage,
        *,
        reason: str,
        stage_path: list[ConversationStage],
        appended_messages: list[ConversationMessage],
        extra_info: dict[str, object] | None = None,
    ) -> ConversationStepResult:
        """Shared rollback logic: hide assistant message, transition back to await-assistant stage."""
        self._assistant_rollback_count += 1
        # Undo the turn increment from apply_assistant_message() so that rolled-back
        # turns don't count toward the turn limit.
        if self._assistant_turn_count > 0:
            self._assistant_turn_count -= 1

        rollback_marker = self._build_rollback_marker(1, reason=reason)
        self._append_messages([rollback_marker])
        appended_messages.append(rollback_marker)

        self._transit_stage(ConversationStage.ASSISTANT_MESSAGE_ROLLBACK, stage_path=stage_path)

        target = self._resolve_rollback_target(prior_stage)
        self._transit_stage(target, stage_path=stage_path)

        info: dict[str, object] = {
            "assistant_rollback_count": self._assistant_rollback_count,
            "rollback_reason": reason,
        }
        if extra_info:
            info.update(extra_info)
        return self._build_step_result(appended_messages=appended_messages, stage_path=stage_path, info=info)

    def _resolve_rollback_target(self, prior_stage: ConversationStage) -> ConversationStage:
        """Determine the target stage after an assistant-message rollback."""
        if prior_stage == ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT:
            return ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT
        return ConversationStage.TOOL_MESSAGES_AWAIT_ASSISTANT

    # ------------------------------------------------------------------
    # Orchestrator-initiated rollback (duplicate query, tool error)
    # ------------------------------------------------------------------

    def rollback_assistant_tool_calls(self, reason: str) -> ConversationStepResult | None:
        """Roll back the last assistant message that contained tool calls.

        Called by the orchestrator when pending tool calls should be discarded
        (e.g. all queries are duplicates, or tool execution returned errors).

        Returns ``None`` if the rollback limit has been reached (caller should proceed normally).
        """
        if self.max_assistant_rollbacks <= 0:
            return None
        if self._assistant_rollback_count >= self.max_assistant_rollbacks:
            return None

        self._assistant_rollback_count += 1
        self._pending_tool_calls = []
        # Undo the turn increment so rolled-back turns don't count toward the turn limit.
        if self._assistant_turn_count > 0:
            self._assistant_turn_count -= 1

        rollback_marker = self._build_rollback_marker(1, reason=reason)
        self._append_messages([rollback_marker])

        stage_path = [self.stage]
        self._transit_stage(ConversationStage.ASSISTANT_MESSAGE_ROLLBACK, stage_path=stage_path)

        # Transition back to the appropriate await-assistant stage.
        # We're coming from ASSISTANT_TOOL_CALLS_AWAIT_TOOL, which means there were
        # prior tool messages or an initial user message before the assistant's response.
        target = ConversationStage.TOOL_MESSAGES_AWAIT_ASSISTANT
        self._transit_stage(target, stage_path=stage_path)

        info: dict[str, object] = {
            "assistant_rollback_count": self._assistant_rollback_count,
            "rollback_reason": reason,
        }
        return self._build_step_result(appended_messages=[rollback_marker], stage_path=stage_path, info=info)

    def reset_rollback_counter(self) -> None:
        """Reset the assistant-message rollback counter.

        Called by orchestrators after successful tool execution.
        """
        self._assistant_rollback_count = 0

    # ------------------------------------------------------------------
    # Result building
    # ------------------------------------------------------------------

    def _build_step_result(
        self,
        *,
        appended_messages: list[ConversationMessage],
        stage_path: list[ConversationStage],
        info: dict[str, object] | None = None,
    ) -> ConversationStepResult:
        action = STAGE_TO_ACTION.get(self.stage)
        if action is None:
            msg = f"Stage {self.stage.value} is runtime-internal and must not appear in a ConversationStepResult."
            raise RuntimeError(msg)

        visible_conversation = self._build_visible_conversation()
        step_info: dict[str, object] = {
            "stage": self.stage.value,
            "emoji": STAGE_EMOJI.get(self.stage, ""),
            "stage_path": [stage.value for stage in stage_path],
            "visible_message_count": len(visible_conversation),
            "raw_message_count": len(self._full_conversation),
            "appended_message_count": len(appended_messages),
            "assistant_turn_idx": self._assistant_turn_count,
        }
        if self.stage in FORCE_FINALIZATION_STAGES:
            step_info["force_finalize_stage"] = self.stage.value
        # Propagate finalization trigger to terminal stages so the orchestrator
        # can derive the done_reason without needing to track it separately.
        if self._force_final_trigger is not None and self.stage in {
            ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT,
            ConversationStage.ASSISTANT_FORCE_COMPLETED,
            ConversationStage.ASSISTANT_TOOL_CALLS_REJECTED_FORCE_FINAL,
            ConversationStage.ASSISTANT_COMPLETED,
        }:
            step_info["finalization_trigger"] = self._force_final_trigger.value
        if info:
            step_info.update(info)

        return ConversationStepResult(
            stage=self.stage,
            action=action,
            visible_conversation=visible_conversation,
            full_conversation=list(self._full_conversation),
            appended_messages=list(appended_messages),
            info=cast("ConversationStepInfo", step_info),
        )
