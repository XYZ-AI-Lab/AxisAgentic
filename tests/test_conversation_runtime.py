from agentic.config import AssistantRollbackConfig, ConversationConfig, FormatErrorConfig
from agentic.contracts import (
    ConversationMessage,
    ConversationStage,
    FormatErrorStrategy,
    MessageRole,
    RollbackReason,
    StepAction,
    ToolCall,
    ToolCallSpec,
)
from agentic.contracts.conversation import VALID_STAGE_TRANSITION
from agentic.conversations import ConversationRuntime


def _estimate_tokens(messages: list[ConversationMessage]) -> int:
    return sum(len(message.content or "") for message in messages)


WRONG_TOOL_CALL_FORMAT_PROMPT = "Please emit a valid tool call."
TOOL_CALL_KEYWORDS = ["<tool_call>", "</tool_call>"]
EARLY_STOP_ANNOUNCEMENT_PROMPT = "You cannot call any tool now. Generate your final answer to the given task based on the conversation so far."
FINAL_RESPONSE_PROMPT = "Please provide the final answer directly."

DEFAULT_FORMAT_ERROR = FormatErrorConfig(keywords=TOOL_CALL_KEYWORDS, reprompt=WRONG_TOOL_CALL_FORMAT_PROMPT)


def _runtime(**overrides: object) -> ConversationRuntime:
    config_fields = {
        "system_prompt": "system",
        "user_prompt_template": "Task: {task}",
        "early_stop_announcement_prompt": EARLY_STOP_ANNOUNCEMENT_PROMPT,
        "format_error": DEFAULT_FORMAT_ERROR,
        "max_turns": 12,
        "context_window": 200,
        "context_safety_margin": 0,
    }
    runtime_fields = {
        "tools": None,
        "max_output_tokens": 20,
        "token_estimator": _estimate_tokens,
        "token_estimator_includes_tools": False,
        "token_estimator_is_additive": False,
        "context_limit_preflight_enabled": True,
    }
    # Apply overrides to the right dict
    for k, v in overrides.items():
        if k in runtime_fields:
            runtime_fields[k] = v
        else:
            config_fields[k] = v
    return ConversationRuntime(config=ConversationConfig(**config_fields), **runtime_fields)


class _DirectAnswerRuntime(ConversationRuntime):
    def _extract_direct_final_answer(self, message: ConversationMessage) -> str | None:
        if message.content == "direct final":
            return "accepted answer"
        return None


def test_initialize_conversation_records_system_and_user_stages() -> None:
    runtime = _runtime()

    result = runtime.initialize_conversation("say hello")

    assert result.stage == ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT
    assert runtime.stage == ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT
    assert [message.role.value for message in result.appended_messages] == ["system", "user"]
    assert result.info["stage_path"] == [
        ConversationStage.NOT_INITIALIZED.value,
        ConversationStage.SYSTEM_MESSAGE_SET.value,
        ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT.value,
    ]


def test_runtime_clone_preserves_visible_state_independently() -> None:
    runtime = _runtime(token_estimator_is_additive=True)
    result = runtime.initialize_conversation("say hello")
    runtime.record_observed_prompt_tokens(result.visible_conversation, input_tokens=19)

    cloned = runtime.clone()

    assert cloned is not runtime
    assert cloned.stage == runtime.stage
    assert cloned.untruncated_visible_conversation() == runtime.untruncated_visible_conversation()
    assert cloned._visible_context.calibration_ratio == runtime._visible_context.calibration_ratio

    source_tokenizer_elapsed = runtime.tokenizer_elapsed_ms
    clone_tokenizer_elapsed = cloned.tokenizer_elapsed_ms
    cloned._append_messages([ConversationMessage.user("branch only")])

    assert [message.content for message in runtime.untruncated_visible_conversation()] == ["system", "Task: say hello"]
    assert [message.content for message in cloned.untruncated_visible_conversation()] == [
        "system",
        "Task: say hello",
        "branch only",
    ]
    assert runtime.tokenizer_elapsed_ms == source_tokenizer_elapsed
    assert cloned.tokenizer_elapsed_ms > clone_tokenizer_elapsed


def test_assistant_tool_call_path_records_append_stage_before_waiting_for_tools() -> None:
    runtime = _runtime()
    runtime.initialize_conversation("say hello")

    result = runtime.apply_assistant_message(
        ConversationMessage.assistant(tool_calls=[ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "hello"}))])
    )

    assert result.stage == ConversationStage.ASSISTANT_TOOL_CALLS_AWAIT_TOOL
    assert result.info["tool_request_count"] == 1
    assert result.info["stage_path"] == [
        ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT.value,
        ConversationStage.ASSISTANT_MESSAGE_RECEIVED.value,
        ConversationStage.ASSISTANT_TOOL_CALLS_AWAIT_TOOL.value,
    ]


def test_tool_message_batch_returns_to_waiting_for_assistant() -> None:
    runtime = _runtime()
    runtime.initialize_conversation("say hello")

    tc_1 = ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "hello"}))
    tc_2 = ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "world"}))
    runtime.apply_assistant_message(ConversationMessage.assistant(tool_calls=[tc_1, tc_2]))

    result = runtime.apply_batch_tool_messages(
        [
            ConversationMessage.tool("hello", tool_call_id=tc_1.id, name="echo"),
            ConversationMessage.tool("world", tool_call_id=tc_2.id, name="echo"),
        ]
    )

    assert result.stage == ConversationStage.TOOL_MESSAGES_AWAIT_ASSISTANT
    assert result.info["stage_path"] == [
        ConversationStage.ASSISTANT_TOOL_CALLS_AWAIT_TOOL.value,
        ConversationStage.TOOL_MESSAGES_AWAIT_ASSISTANT.value,
    ]
    assert [message.content for message in result.appended_messages] == ["hello", "world"]


def test_wrong_tool_call_format_appends_recovery_prompt() -> None:
    runtime = _runtime()
    runtime.initialize_conversation("say hello")

    result = runtime.apply_assistant_message(ConversationMessage.assistant("<tool_call>bad format</tool_call>"))

    assert result.stage == ConversationStage.TOOL_CALL_FORMAT_ERROR_PROMPT_AWAIT_ASSISTANT
    assert result.info["wrong_tool_call_format_keywords"] == TOOL_CALL_KEYWORDS
    assert result.info["format_error_strategy"] == "reprompt"
    # The offending content is captured as a single-line, length-capped preview,
    # and the full length is recorded so downstream tools can flag truncation.
    assert result.info["offending_content_preview"] == "<tool_call>bad format</tool_call>"
    assert result.info["offending_content_length"] == len("<tool_call>bad format</tool_call>")
    assert [message.role.value for message in result.appended_messages] == ["assistant", "user"]
    assert result.info["stage_path"] == [
        ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT.value,
        ConversationStage.ASSISTANT_MESSAGE_RECEIVED.value,
        ConversationStage.TOOL_CALL_FORMAT_ERROR_PROMPT_AWAIT_ASSISTANT.value,
    ]


def test_final_response_prompt_is_inserted_before_completion() -> None:
    runtime = _runtime(final_response_prompt=FINAL_RESPONSE_PROMPT)
    runtime.initialize_conversation("say hello")

    prompted = runtime.apply_assistant_message(ConversationMessage.assistant("draft answer"))
    completed = runtime.apply_assistant_message(ConversationMessage.assistant("final answer"))

    assert prompted.stage == ConversationStage.FINAL_RESPONSE_PROMPT_AWAIT_ASSISTANT
    assert prompted.info["stage_path"] == [
        ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT.value,
        ConversationStage.ASSISTANT_MESSAGE_RECEIVED.value,
        ConversationStage.FINAL_RESPONSE_PROMPT_AWAIT_ASSISTANT.value,
    ]
    assert prompted.appended_messages[-1].content == FINAL_RESPONSE_PROMPT
    assert completed.stage == ConversationStage.ASSISTANT_COMPLETED
    assert completed.info["stage_path"] == [
        ConversationStage.FINAL_RESPONSE_PROMPT_AWAIT_ASSISTANT.value,
        ConversationStage.ASSISTANT_MESSAGE_RECEIVED.value,
        ConversationStage.ASSISTANT_COMPLETED.value,
    ]


def test_direct_final_answer_hook_completes_before_final_response_prompt() -> None:
    runtime = _DirectAnswerRuntime(
        config=ConversationConfig(
            system_prompt="system",
            user_prompt_template="Task: {task}",
            final_response_prompt=FINAL_RESPONSE_PROMPT,
            format_error=DEFAULT_FORMAT_ERROR,
            max_turns=12,
            context_window=200,
            context_safety_margin=0,
        ),
        tools=None,
        max_output_tokens=20,
        token_estimator=_estimate_tokens,
    )
    runtime.initialize_conversation("say hello")

    result = runtime.apply_assistant_message(ConversationMessage.assistant("direct final"))

    assert result.stage == ConversationStage.ASSISTANT_COMPLETED
    assert result.action == StepAction.DONE
    assert result.info["direct_final_answer"] == "accepted answer"
    assert result.info["stage_path"] == [
        ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT.value,
        ConversationStage.ASSISTANT_MESSAGE_RECEIVED.value,
        ConversationStage.ASSISTANT_COMPLETED.value,
    ]
    assert [message.role for message in result.appended_messages] == [MessageRole.ASSISTANT]
    assert FINAL_RESPONSE_PROMPT not in [message.content for message in result.visible_conversation]


def test_length_response_with_content_terminates_without_final_response_prompt() -> None:
    runtime = _runtime(final_response_prompt=FINAL_RESPONSE_PROMPT)
    runtime.initialize_conversation("say hello")

    result = runtime.apply_assistant_message(ConversationMessage.assistant("partial answer"), finish_reason="length")

    assert result.stage == ConversationStage.ASSISTANT_TERMINATED_TOKEN_EXCEED
    assert result.action == StepAction.DONE
    assert result.info["finish_reason"] == "length"
    assert result.info["token_exhaustion_had_content"] is True
    assert result.info["stage_path"] == [
        ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT.value,
        ConversationStage.ASSISTANT_MESSAGE_RECEIVED.value,
        ConversationStage.ASSISTANT_TERMINATED_TOKEN_EXCEED.value,
    ]
    assert [message.content for message in result.appended_messages] == ["partial answer"]
    assert FINAL_RESPONSE_PROMPT not in [message.content for message in result.visible_conversation]


def test_length_response_with_direct_final_answer_is_accepted() -> None:
    runtime = _DirectAnswerRuntime(
        config=ConversationConfig(
            system_prompt="system",
            user_prompt_template="Task: {task}",
            final_response_prompt=FINAL_RESPONSE_PROMPT,
            format_error=DEFAULT_FORMAT_ERROR,
            max_turns=12,
            context_window=200,
            context_safety_margin=0,
        ),
        tools=None,
        max_output_tokens=20,
        token_estimator=_estimate_tokens,
    )
    runtime.initialize_conversation("say hello")

    result = runtime.apply_assistant_message(ConversationMessage.assistant("direct final"), finish_reason="length")

    assert result.stage == ConversationStage.ASSISTANT_COMPLETED
    assert result.action == StepAction.DONE
    assert result.info["direct_final_answer"] == "accepted answer"
    assert result.info["finish_reason"] == "length"
    assert FINAL_RESPONSE_PROMPT not in [message.content for message in result.visible_conversation]


def test_initialization_does_not_apply_context_limit_checkpoint() -> None:
    runtime = _runtime(context_window=20, max_output_tokens=8)

    result = runtime.initialize_conversation("this task is intentionally long")

    assert result.stage == ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT
    assert result.info["stage_path"] == [
        ConversationStage.NOT_INITIALIZED.value,
        ConversationStage.SYSTEM_MESSAGE_SET.value,
        ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT.value,
    ]
    assert [message.role for message in result.full_conversation] == [MessageRole.SYSTEM, MessageRole.USER]
    assert [message.role for message in result.visible_conversation] == [MessageRole.SYSTEM, MessageRole.USER]
    assert result.visible_conversation[-1].content == "Task: this task is intentionally long"


def test_context_limit_after_tool_results_rolls_back_last_assistant_turn() -> None:
    runtime = _runtime(
        system_prompt="s",
        user_prompt_template="T:{task}",
        early_stop_announcement_prompt="stop",
        context_window=25,
        max_output_tokens=1,
    )
    runtime.initialize_conversation("x")

    tc_1 = ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "hello"}))
    tc_2 = ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "world"}))
    runtime.apply_assistant_message(ConversationMessage.assistant(tool_calls=[tc_1, tc_2]))

    result = runtime.apply_batch_tool_messages(
        [
            ConversationMessage.tool("0123456789", tool_call_id=tc_1.id, name="echo"),
            ConversationMessage.tool("abcdefghij", tool_call_id=tc_2.id, name="echo"),
        ]
    )

    assert result.stage == ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT
    assert result.info["rollback_message_count"] == 3
    assert result.info["finalization_trigger"] == "context_limit"
    assert result.info["stage_path"] == [
        ConversationStage.ASSISTANT_TOOL_CALLS_AWAIT_TOOL.value,
        ConversationStage.TOOL_MESSAGES_AWAIT_ASSISTANT.value,
        ConversationStage.CONTEXT_LIMIT_ROLLBACK.value,
        ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT.value,
    ]
    assert [message.role for message in result.visible_conversation] == [MessageRole.SYSTEM, MessageRole.USER, MessageRole.USER]
    assert result.visible_conversation[-1].content == "stop"


def test_context_limit_reserves_output_tokens() -> None:
    runtime = _runtime(
        system_prompt="s",
        user_prompt_template="T:{task}",
        early_stop_announcement_prompt="stop",
        context_window=27,
        max_output_tokens=10,
    )
    runtime.initialize_conversation("x")

    tool_call = ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "hello"}))
    runtime.apply_assistant_message(ConversationMessage.assistant(tool_calls=[tool_call]))
    result = runtime.apply_batch_tool_messages([ConversationMessage.tool("0123456789", tool_call_id=tool_call.id, name="echo")])

    assert result.stage == ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT
    estimate = result.info["context_limit_estimate"]
    assert estimate["estimated_prompt_tokens"] <= 27
    assert estimate["reserved_output_tokens"] == 10
    assert estimate["estimated_total_tokens"] > 27


def test_context_budget_estimator_uses_incremental_visible_diffs() -> None:
    estimator_call_sizes: list[int] = []

    def estimate_tokens(messages: list[ConversationMessage]) -> int:
        estimator_call_sizes.append(len(messages))
        return sum(len(message.content or "") for message in messages)

    runtime = _runtime(
        system_prompt="s",
        user_prompt_template="T:{task}",
        early_stop_announcement_prompt="stop",
        context_window=10_000,
        max_output_tokens=10,
        token_estimator=estimate_tokens,
        token_estimator_is_additive=True,
    )
    runtime.initialize_conversation("x")
    estimator_call_sizes.clear()

    tool_call_1 = ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "hello"}))
    tool_call_2 = ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "world"}))
    runtime.apply_assistant_message(ConversationMessage.assistant(tool_calls=[tool_call_1, tool_call_2]))
    runtime.apply_batch_tool_messages(
        [
            ConversationMessage.tool("tool result 1", tool_call_id=tool_call_1.id, name="echo"),
            ConversationMessage.tool("tool result 2", tool_call_id=tool_call_2.id, name="echo"),
        ]
    )

    assert estimator_call_sizes
    assert max(estimator_call_sizes) == 2


def test_context_budget_uses_full_estimator_for_non_additive_estimators() -> None:
    estimator_call_sizes: list[int] = []

    def estimate_tokens(messages: list[ConversationMessage]) -> int:
        estimator_call_sizes.append(len(messages))
        return sum(len(message.content or "") for message in messages)

    runtime = _runtime(
        system_prompt="s",
        user_prompt_template="T:{task}",
        early_stop_announcement_prompt="stop",
        context_window=10_000,
        max_output_tokens=10,
        token_estimator=estimate_tokens,
    )
    runtime.initialize_conversation("x")
    estimator_call_sizes.clear()

    tool_call = ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "hello"}))
    runtime.apply_assistant_message(ConversationMessage.assistant(tool_calls=[tool_call]))
    runtime.apply_batch_tool_messages([ConversationMessage.tool("tool result", tool_call_id=tool_call.id, name="echo")])

    assert estimator_call_sizes
    assert max(estimator_call_sizes) >= 4


def test_context_preflight_disabled_skips_local_context_estimator() -> None:
    def estimate_tokens(_messages: list[ConversationMessage]) -> int:
        msg = "local context estimator should not run when preflight is disabled"
        raise AssertionError(msg)

    runtime = _runtime(
        system_prompt="s",
        user_prompt_template="T:{task}",
        early_stop_announcement_prompt="stop",
        context_window=20,
        max_output_tokens=10,
        token_estimator=estimate_tokens,
        token_estimator_is_additive=True,
        context_limit_preflight_enabled=False,
    )
    runtime.initialize_conversation("x")
    tool_call = ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "hello"}))
    runtime.apply_assistant_message(ConversationMessage.assistant(tool_calls=[tool_call]))

    result = runtime.apply_batch_tool_messages([ConversationMessage.tool("x" * 100, tool_call_id=tool_call.id, name="echo")])

    assert result.stage == ConversationStage.TOOL_MESSAGES_AWAIT_ASSISTANT
    assert "context_limit_estimate" not in result.info


def test_provider_context_error_recovery_skips_local_context_estimator_when_preflight_disabled() -> None:
    def estimate_tokens(_messages: list[ConversationMessage]) -> int:
        msg = "local context estimator should not run for provider-error recovery"
        raise AssertionError(msg)

    runtime = _runtime(
        system_prompt="s",
        user_prompt_template="T:{task}",
        early_stop_announcement_prompt="stop",
        context_window=20,
        max_output_tokens=10,
        token_estimator=estimate_tokens,
        token_estimator_is_additive=True,
        context_limit_preflight_enabled=False,
    )
    runtime.initialize_conversation("x")
    tool_call = ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "hello"}))
    runtime.apply_assistant_message(ConversationMessage.assistant(tool_calls=[tool_call]))
    runtime.apply_batch_tool_messages([ConversationMessage.tool("x" * 100, tool_call_id=tool_call.id, name="echo")])

    result = runtime.force_finalize_due_to_context_limit_after_error(error_info={"input_tokens": 25, "context_window": 20})

    assert result.stage == ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT
    assert result.info["context_limit_estimate"]["context_limit_detection"] == "provider_error"
    assert result.info["context_limit_estimate"]["provider_error"]["input_tokens"] == 25


def test_context_disabled_does_not_call_token_estimator_on_append() -> None:
    def estimate_tokens(_messages: list[ConversationMessage]) -> int:
        msg = "token estimator should not run when context limit enforcement is disabled"
        raise AssertionError(msg)

    runtime = _runtime(context_window=None, token_estimator=estimate_tokens)

    runtime.initialize_conversation("x")
    result = runtime.apply_assistant_message(ConversationMessage.assistant("final"))

    assert result.stage == ConversationStage.ASSISTANT_COMPLETED


def test_context_estimator_calibrates_from_observed_prompt_tokens() -> None:
    runtime = _runtime(
        system_prompt="s",
        user_prompt_template="T:{task}",
        context_window=10_000,
        max_output_tokens=10,
        token_estimator_is_additive=True,
    )
    result = runtime.initialize_conversation("x")
    before = runtime._estimate_generation_budget(result.visible_conversation)["estimated_prompt_tokens"]

    runtime.record_observed_prompt_tokens(result.visible_conversation, input_tokens=int(before) * 3)
    after = runtime._estimate_generation_budget(result.visible_conversation)["estimated_prompt_tokens"]

    assert after > before


def test_token_exhaustion_recovery_rolls_back_latest_tool_exchange_and_force_finalizes() -> None:
    runtime = _runtime(
        system_prompt="s",
        user_prompt_template="T:{task}",
        early_stop_announcement_prompt=None,
        final_response_prompt=FINAL_RESPONSE_PROMPT,
        context_window=10_000,
        max_output_tokens=10,
    )
    runtime.initialize_conversation("x")
    tool_call = ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "hello"}))
    runtime.apply_assistant_message(ConversationMessage.assistant(tool_calls=[tool_call]))
    runtime.apply_batch_tool_messages([ConversationMessage.tool("tool result", tool_call_id=tool_call.id, name="echo")])

    result = runtime.rollback_latest_tool_exchange_and_force_finalize()

    assert result is not None
    assert result.stage == ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT
    assert result.info["token_exhaustion_recovery"] is True
    assert result.info["rollback_message_count"] == 2
    assert [message.role for message in result.visible_conversation] == [MessageRole.SYSTEM, MessageRole.USER, MessageRole.USER]
    assert result.visible_conversation[-1].content == FINAL_RESPONSE_PROMPT


def test_token_exhaustion_recovery_reinserts_existing_force_final_prompt() -> None:
    runtime = _runtime(
        system_prompt="s",
        user_prompt_template="T:{task}",
        early_stop_announcement_prompt=None,
        final_response_prompt=FINAL_RESPONSE_PROMPT,
        context_window=10_000,
        max_output_tokens=10,
    )
    runtime.initialize_conversation("x")
    tool_call = ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "hello"}))
    runtime.apply_assistant_message(ConversationMessage.assistant(tool_calls=[tool_call]))
    runtime.apply_batch_tool_messages([ConversationMessage.tool("tool result", tool_call_id=tool_call.id, name="echo")], tools_exhausted=True)

    result = runtime.rollback_latest_tool_exchange_and_force_finalize()

    assert result is not None
    assert result.stage == ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT
    assert result.info["rollback_message_count"] == 3
    assert [message.content for message in result.visible_conversation] == ["s", "T:x", FINAL_RESPONSE_PROMPT]


def test_budget_warning_after_assistant_replaces_tool_execution() -> None:
    runtime = _runtime(
        system_prompt="s",
        user_prompt_template="T:{task}",
        early_stop_announcement_prompt="stop",
        context_window=12,
        max_output_tokens=5,
        min_tokens_for_generation=2,
        context_warning_threshold=10,
    )
    runtime.initialize_conversation("x")

    tool_call = ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "hello"}))
    result = runtime.apply_assistant_message(ConversationMessage.assistant(tool_calls=[tool_call]))

    assert result.stage == ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT
    assert result.action == StepAction.CALL_MODEL
    assert result.info["finalization_trigger"] == "context_limit"
    assert result.info["rollback_message_count"] == 0
    assert [message.role for message in result.appended_messages] == [MessageRole.ASSISTANT, MessageRole.USER]
    assert result.appended_messages[-1].content == "stop"


def test_budget_hard_stops_when_tool_result_leaves_too_few_tokens() -> None:
    runtime = _runtime(
        system_prompt="s",
        user_prompt_template="T:{task}",
        early_stop_announcement_prompt="stop",
        context_window=16,
        max_output_tokens=5,
        min_tokens_for_generation=5,
        context_warning_threshold=10,
    )
    runtime.initialize_conversation("x")

    tool_call = ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "hello"}))
    runtime.apply_assistant_message(ConversationMessage.assistant(tool_calls=[tool_call]))
    result = runtime.apply_batch_tool_messages([ConversationMessage.tool("0123456789", tool_call_id=tool_call.id, name="echo")])

    assert result.stage == ConversationStage.ASSISTANT_TERMINATED_TOKEN_EXCEED
    assert result.action == StepAction.DONE
    assert result.info["context_budget_stop"] is True
    assert result.info["context_limit_estimate"]["remaining_tokens"] < 5
    assert [message.content for message in result.appended_messages] == ["0123456789"]
    assert "stop" not in [message.content for message in result.visible_conversation]


def test_context_limit_endpoint_error_can_force_finalize() -> None:
    runtime = _runtime(
        system_prompt="s",
        user_prompt_template="T:{task}",
        early_stop_announcement_prompt="stop",
        context_window=200,
        max_output_tokens=10,
    )
    runtime.initialize_conversation("x")

    tool_call = ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "hello"}))
    runtime.apply_assistant_message(ConversationMessage.assistant(tool_calls=[tool_call]))
    runtime.apply_batch_tool_messages([ConversationMessage.tool("tool result", tool_call_id=tool_call.id, name="echo")])
    result = runtime.force_finalize_due_to_context_limit_after_error(
        error_info={"input_tokens": 195, "requested_output_tokens": 10, "context_window": 200}
    )

    assert result.stage == ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT
    assert result.info["finalization_trigger"] == "context_limit"
    assert result.info["rollback_message_count"] == 2
    assert result.info["context_limit_estimate"]["provider_error"]["input_tokens"] == 195


def test_turn_limit_force_finalization_rejects_tool_calls() -> None:
    runtime = _runtime(max_turns=1)

    initial = runtime.initialize_conversation("say hello")
    result = runtime.apply_assistant_message(
        ConversationMessage.assistant(tool_calls=[ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "hello"}))])
    )

    assert initial.stage == ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT
    assert result.stage == ConversationStage.ASSISTANT_TOOL_CALLS_REJECTED_FORCE_FINAL
    assert result.info["tool_request_count"] == 1
    assert result.info["stage_path"] == [
        ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT.value,
        ConversationStage.ASSISTANT_MESSAGE_RECEIVED.value,
        ConversationStage.ASSISTANT_TOOL_CALLS_REJECTED_FORCE_FINAL.value,
    ]


def test_turn_limit_force_finalization_can_complete_with_plain_answer() -> None:
    runtime = _runtime(max_turns=1)

    runtime.initialize_conversation("say hello")
    result = runtime.apply_assistant_message(ConversationMessage.assistant("final answer"))

    assert result.stage == ConversationStage.ASSISTANT_FORCE_COMPLETED
    assert result.info["stage_path"] == [
        ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT.value,
        ConversationStage.ASSISTANT_MESSAGE_RECEIVED.value,
        ConversationStage.ASSISTANT_FORCE_COMPLETED.value,
    ]


def test_context_limit_force_finalization_rejects_tool_calls() -> None:
    runtime = _runtime(
        system_prompt="s",
        user_prompt_template="T:{task}",
        early_stop_announcement_prompt="stop",
        context_window=25,
        max_output_tokens=1,
    )
    runtime.initialize_conversation("x")

    tc_1 = ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "hello"}))
    tc_2 = ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "world"}))
    runtime.apply_assistant_message(ConversationMessage.assistant(tool_calls=[tc_1, tc_2]))
    runtime.apply_batch_tool_messages(
        [
            ConversationMessage.tool("0123456789", tool_call_id=tc_1.id, name="echo"),
            ConversationMessage.tool("abcdefghij", tool_call_id=tc_2.id, name="echo"),
        ]
    )

    result = runtime.apply_assistant_message(
        ConversationMessage.assistant(tool_calls=[ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "again"}))])
    )

    assert result.stage == ConversationStage.ASSISTANT_TOOL_CALLS_REJECTED_FORCE_FINAL
    assert result.info["tool_request_count"] == 1
    assert result.info["stage_path"] == [
        ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT.value,
        ConversationStage.ASSISTANT_MESSAGE_RECEIVED.value,
        ConversationStage.ASSISTANT_TOOL_CALLS_REJECTED_FORCE_FINAL.value,
    ]


def test_context_limit_force_finalization_can_complete_with_plain_answer() -> None:
    runtime = _runtime(
        system_prompt="s",
        user_prompt_template="T:{task}",
        early_stop_announcement_prompt="stop",
        context_window=25,
        max_output_tokens=1,
    )
    runtime.initialize_conversation("x")

    tc_1 = ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "hello"}))
    tc_2 = ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "world"}))
    runtime.apply_assistant_message(ConversationMessage.assistant(tool_calls=[tc_1, tc_2]))
    runtime.apply_batch_tool_messages(
        [
            ConversationMessage.tool("0123456789", tool_call_id=tc_1.id, name="echo"),
            ConversationMessage.tool("abcdefghij", tool_call_id=tc_2.id, name="echo"),
        ]
    )

    result = runtime.apply_assistant_message(ConversationMessage.assistant("final answer"))

    assert result.stage == ConversationStage.ASSISTANT_FORCE_COMPLETED
    assert result.info["stage_path"] == [
        ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT.value,
        ConversationStage.ASSISTANT_MESSAGE_RECEIVED.value,
        ConversationStage.ASSISTANT_FORCE_COMPLETED.value,
    ]


def test_single_turn_without_early_stop_completes_normally() -> None:
    """Test the final stage in single-turn with None early stop annoucement.

    With early_stop_announcement_prompt=None and max_turns=1, the silent force-finalization stage is set but the assistant responds normally
    and completes as ASSISTANT_COMPLETED.
    """
    runtime = _runtime(early_stop_announcement_prompt=None, max_turns=1)

    init_result = runtime.initialize_conversation("say hello")
    # Silent force-finalization stage: no prompt appended, model gets a normal call
    assert init_result.stage == ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT

    result = runtime.apply_assistant_message(ConversationMessage.assistant("final answer"))
    # Silent stages are not in FORCE_FINALIZATION_STAGES, so the normal completion path applies
    assert result.stage == ConversationStage.ASSISTANT_COMPLETED


def test_silent_force_finalization_rejects_tool_calls() -> None:
    """With early_stop=None, silent force-finalization still rejects tool calls."""
    runtime = _runtime(early_stop_announcement_prompt=None, max_turns=1)

    runtime.initialize_conversation("say hello")
    result = runtime.apply_assistant_message(
        ConversationMessage.assistant(tool_calls=[ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "hello"}))])
    )

    assert result.stage == ConversationStage.ASSISTANT_TOOL_CALLS_REJECTED_FORCE_FINAL
    assert result.info["tool_request_count"] == 1


def test_valid_stage_transition_mapping_matches_runtime_immediate_hops() -> None:
    expected = {
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

    assert expected == VALID_STAGE_TRANSITION


def test_tool_call_arguments_roundtrip_through_to_model_message() -> None:
    r"""Tool call arguments must be serialized as JSON strings in to_model_message().

    The OpenAI API and HuggingFace chat templates both expect ``function.arguments``
    as a JSON string, not a raw dict.
    """
    import json

    original_args = {"query": "capital of France", "limit": 3}
    msg = ConversationMessage.assistant(
        "Let me search.",
        tool_calls=[ToolCall(function=ToolCallSpec(name="search", arguments=original_args))],
    )

    model_msg = msg.to_model_message()
    serialized_args = model_msg["tool_calls"][0]["function"]["arguments"]

    assert isinstance(serialized_args, str), f"Expected str, got {type(serialized_args).__name__}"
    assert json.loads(serialized_args) == original_args


# ---------------------------------------------------------------------------
# Refusal rollback tests
# ---------------------------------------------------------------------------

REFUSAL_KEYWORDS = ["I'm sorry, but I can't", "time constraint"]


def _refusal_runtime(**overrides: object) -> ConversationRuntime:
    """Build a runtime with refusal detection enabled."""
    params = {
        "assistant_rollback": AssistantRollbackConfig(refusal_keywords=REFUSAL_KEYWORDS, max_rollbacks=3),
    }
    params.update(overrides)
    return _runtime(**params)


def test_refusal_keyword_triggers_rollback_and_call_model() -> None:
    """When the assistant response contains a refusal keyword, the message is rolled back and the runtime returns CALL_MODEL."""
    runtime = _refusal_runtime()
    result = runtime.initialize_conversation("solve it")
    assert result.action == StepAction.CALL_MODEL

    # Assistant responds with a refusal
    refusal_msg = ConversationMessage.assistant("I'm sorry, but I can't solve this problem due to time constraint.")
    result = runtime.apply_assistant_message(refusal_msg)

    assert result.action == StepAction.CALL_MODEL
    assert ConversationStage.ASSISTANT_MESSAGE_ROLLBACK in [ConversationStage(s) for s in result.info["stage_path"]]
    assert result.info["refusal_keywords"]
    assert result.info["assistant_rollback_count"] == 1

    # The refusal message should NOT be in the visible conversation
    visible_assistant_msgs = [m for m in result.visible_conversation if m.role == MessageRole.ASSISTANT]
    assert len(visible_assistant_msgs) == 0


def test_refusal_rollback_respects_max_limit() -> None:
    """After hitting max_assistant_rollbacks, the runtime stops rolling back and completes normally."""
    runtime = _refusal_runtime(assistant_rollback=AssistantRollbackConfig(refusal_keywords=REFUSAL_KEYWORDS, max_rollbacks=2))
    result = runtime.initialize_conversation("solve it")

    # First refusal — rolled back
    result = runtime.apply_assistant_message(ConversationMessage.assistant("I'm sorry, but I can't do this."))
    assert result.action == StepAction.CALL_MODEL
    assert result.info.get("assistant_rollback_count") == 1

    # Second refusal — rolled back (hits limit)
    result = runtime.apply_assistant_message(ConversationMessage.assistant("I'm sorry, but I can't help."))
    assert result.action == StepAction.CALL_MODEL
    assert result.info.get("assistant_rollback_count") == 2

    # Third refusal — should NOT be rolled back (limit exceeded), completes normally
    result = runtime.apply_assistant_message(ConversationMessage.assistant("I'm sorry, but I can't solve this."))
    assert result.action == StepAction.DONE
    assert result.stage == ConversationStage.ASSISTANT_COMPLETED


def test_refusal_not_detected_when_tool_calls_present() -> None:
    """An assistant message with tool calls should proceed to tool execution even if text contains refusal keywords."""
    runtime = _refusal_runtime(
        tools=[{"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {}}}}],
    )
    result = runtime.initialize_conversation("solve it")

    tool_call = ToolCall(function=ToolCallSpec(name="search", arguments={"q": "test"}))
    msg = ConversationMessage.assistant("I'm sorry, but I can't do this directly. Let me search.", tool_calls=[tool_call])
    result = runtime.apply_assistant_message(msg)

    # Should dispatch to tool execution, not refusal rollback
    assert result.action == StepAction.EXECUTE_TOOLS
    assert result.stage == ConversationStage.ASSISTANT_TOOL_CALLS_AWAIT_TOOL


def test_refusal_not_detected_in_final_response_round() -> None:
    """After the final_response_prompt, refusal keywords should NOT trigger rollback."""
    runtime = _refusal_runtime(final_response_prompt="Give your final answer.")
    result = runtime.initialize_conversation("solve it")

    # First response triggers final_response_prompt injection
    result = runtime.apply_assistant_message(ConversationMessage.assistant("I think the answer is 42."))
    assert result.action == StepAction.CALL_MODEL
    assert result.stage == ConversationStage.FINAL_RESPONSE_PROMPT_AWAIT_ASSISTANT

    # Second response (in final-response round) — refusal keywords should be ignored
    result = runtime.apply_assistant_message(ConversationMessage.assistant("I'm sorry, but I can't be more specific. The answer is 42."))
    assert result.action == StepAction.DONE
    assert result.stage == ConversationStage.ASSISTANT_COMPLETED


def test_refusal_disabled_by_default() -> None:
    """With default config (empty refusal_keywords), no refusal detection occurs."""
    runtime = _runtime()
    result = runtime.initialize_conversation("solve it")

    # This should complete normally even though the text matches refusal patterns
    result = runtime.apply_assistant_message(ConversationMessage.assistant("I'm sorry, but I can't."))
    assert result.action == StepAction.DONE
    assert result.stage == ConversationStage.ASSISTANT_COMPLETED


def test_refusal_counter_resets_on_non_refusal() -> None:
    """The consecutive refusal counter resets on normal completion and after reset_rollback_counter()."""
    runtime = _refusal_runtime(
        assistant_rollback=AssistantRollbackConfig(refusal_keywords=REFUSAL_KEYWORDS, max_rollbacks=2),
        context_window=10000,
        tools=[{"type": "function", "function": {"name": "s", "parameters": {"type": "object", "properties": {}}}}],
    )
    result = runtime.initialize_conversation("solve")

    # Refusal 1 — rolled back (count → 1)
    result = runtime.apply_assistant_message(ConversationMessage.assistant("I'm sorry, but I can't."))
    assert result.info.get("assistant_rollback_count") == 1

    # Tool call response — NOT a refusal, but counter only resets after
    # successful tool execution (via reset_rollback_counter).
    tc = ToolCall(function=ToolCallSpec(name="s", arguments={}))
    result = runtime.apply_assistant_message(ConversationMessage.assistant("ok", tool_calls=[tc]))
    assert result.action == StepAction.EXECUTE_TOOLS

    # Simulate orchestrator calling reset after successful tool execution
    runtime.reset_rollback_counter()

    # Provide tool result
    result = runtime.apply_batch_tool_messages([ConversationMessage.tool("result", tool_call_id=tc.id, name="s")])
    assert result.action == StepAction.CALL_MODEL

    # Refusal again — counter should be back to 1 (was reset)
    result = runtime.apply_assistant_message(ConversationMessage.assistant("I'm sorry, but I can't."))
    assert result.info.get("assistant_rollback_count") == 1


def test_rollback_marker_includes_reason() -> None:
    """Rollback markers created by refusal detection should include the reason field."""
    runtime = _refusal_runtime()
    result = runtime.initialize_conversation("solve it")

    result = runtime.apply_assistant_message(ConversationMessage.assistant("I'm sorry, but I can't."))

    # Find rollback markers in full conversation
    import json

    rollback_markers = [m for m in result.full_conversation if m.role == MessageRole.CONVERSATION_RUNTIME and m.name == "rollback"]
    assert len(rollback_markers) == 1
    payload = json.loads(rollback_markers[0].content)
    assert payload["reason"] == RollbackReason.ASSISTANT_REFUSAL
    assert payload["rollback_message_count"] == 1


def test_format_error_rollback_hides_message_and_retries() -> None:
    """When format_error_strategy='rollback', format errors trigger rollback instead of re-prompt."""
    runtime = _runtime(
        format_error=FormatErrorConfig(strategy=FormatErrorStrategy.ROLLBACK),
        assistant_rollback=AssistantRollbackConfig(max_rollbacks=3),
        context_window=10000,
    )
    result = runtime.initialize_conversation("solve it")

    # Assistant responds with text containing tool_call keywords (format error)
    msg = ConversationMessage.assistant("Let me try <tool_call>search</tool_call>")
    result = runtime.apply_assistant_message(msg)

    assert result.action == StepAction.CALL_MODEL
    assert ConversationStage.ASSISTANT_MESSAGE_ROLLBACK in [ConversationStage(s) for s in result.info["stage_path"]]
    assert result.info["rollback_reason"] == RollbackReason.FORMAT_ERROR
    assert result.info["assistant_rollback_count"] == 1
    # The new metadata surfaces which strategy fired and what the offending message looked like,
    # so the viz tool can bucket rollouts by strategy without inspecting the full conversation.
    assert result.info["format_error_strategy"] == "rollback"
    assert result.info["offending_content_preview"] == "Let me try <tool_call>search</tool_call>"
    assert result.info["offending_content_length"] == len("Let me try <tool_call>search</tool_call>")

    # The bad message should NOT be in the visible conversation
    visible_assistant = [m for m in result.visible_conversation if m.role == MessageRole.ASSISTANT]
    assert len(visible_assistant) == 0


def test_format_error_default_sends_reprompt() -> None:
    """With default format_error_strategy='reprompt', format errors send a recovery prompt."""
    runtime = _runtime(context_window=10000)
    result = runtime.initialize_conversation("solve it")

    msg = ConversationMessage.assistant("Let me try <tool_call>search</tool_call>")
    result = runtime.apply_assistant_message(msg)

    assert result.action == StepAction.CALL_MODEL
    assert result.stage == ConversationStage.TOOL_CALL_FORMAT_ERROR_PROMPT_AWAIT_ASSISTANT


def test_format_error_and_refusal_share_rollback_counter() -> None:
    """Format errors and refusals share the same assistant rollback counter."""
    runtime = _runtime(
        format_error=FormatErrorConfig(strategy=FormatErrorStrategy.ROLLBACK),
        assistant_rollback=AssistantRollbackConfig(refusal_keywords=["I'm sorry"], max_rollbacks=3),
        context_window=10000,
    )
    result = runtime.initialize_conversation("solve it")

    # Refusal — count = 1
    result = runtime.apply_assistant_message(ConversationMessage.assistant("I'm sorry, I can't."))
    assert result.info.get("assistant_rollback_count") == 1

    # Format error — count = 2
    result = runtime.apply_assistant_message(ConversationMessage.assistant("<tool_call>bad</tool_call>"))
    assert result.info.get("assistant_rollback_count") == 2

    # Refusal — count = 3 (hits limit)
    result = runtime.apply_assistant_message(ConversationMessage.assistant("I'm sorry again."))
    assert result.info.get("assistant_rollback_count") == 3

    # Next attempt — limit reached, should complete normally
    result = runtime.apply_assistant_message(ConversationMessage.assistant("I'm sorry once more."))
    assert result.action == StepAction.DONE


def test_rollback_assistant_tool_calls_from_orchestrator() -> None:
    """Test the public rollback_assistant_tool_calls() method used by orchestrators."""
    runtime = _runtime(
        assistant_rollback=AssistantRollbackConfig(max_rollbacks=3),
        context_window=10000,
        tools=[{"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {}}}}],
    )
    result = runtime.initialize_conversation("solve it")

    # Assistant makes a tool call
    tc = ToolCall(function=ToolCallSpec(name="search", arguments={"q": "test"}))
    result = runtime.apply_assistant_message(ConversationMessage.assistant("Searching.", tool_calls=[tc]))
    assert result.action == StepAction.EXECUTE_TOOLS
    assert result.stage == ConversationStage.ASSISTANT_TOOL_CALLS_AWAIT_TOOL

    # Orchestrator decides to rollback (e.g. duplicate query)
    rollback_result = runtime.rollback_assistant_tool_calls(reason=RollbackReason.DUPLICATE_QUERY)
    assert rollback_result is not None
    assert rollback_result.action == StepAction.CALL_MODEL
    assert rollback_result.info["rollback_reason"] == RollbackReason.DUPLICATE_QUERY

    # The tool-call assistant message should be hidden
    visible_assistant = [m for m in rollback_result.visible_conversation if m.role == MessageRole.ASSISTANT]
    assert len(visible_assistant) == 0


def test_rollback_assistant_tool_calls_returns_none_at_limit() -> None:
    """rollback_assistant_tool_calls() returns None when rollback limit is reached."""
    runtime = _runtime(
        assistant_rollback=AssistantRollbackConfig(max_rollbacks=1),
        context_window=10000,
        tools=[{"type": "function", "function": {"name": "s", "parameters": {"type": "object", "properties": {}}}}],
    )
    _result = runtime.initialize_conversation("solve it")

    # First tool call + orchestrator rollback — succeeds (count → 1)
    tc = ToolCall(function=ToolCallSpec(name="s", arguments={}))
    runtime.apply_assistant_message(ConversationMessage.assistant("go", tool_calls=[tc]))
    rollback = runtime.rollback_assistant_tool_calls(reason=RollbackReason.TOOL_ERROR)
    assert rollback is not None
    assert rollback.info.get("assistant_rollback_count") == 1

    # Second tool call + rollback — should fail (limit=1, count=1)
    # Note: apply_assistant_message does NOT reset counter for tool calls;
    # only reset_rollback_counter() (called after successful execution) or
    # normal completion resets it.
    tc2 = ToolCall(function=ToolCallSpec(name="s", arguments={}))
    runtime.apply_assistant_message(ConversationMessage.assistant("go2", tool_calls=[tc2]))
    rollback2 = runtime.rollback_assistant_tool_calls(reason=RollbackReason.TOOL_ERROR)
    assert rollback2 is None  # Limit reached
