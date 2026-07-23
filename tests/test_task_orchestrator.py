import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agentic.config import AssistantRollbackConfig, ConversationConfig, FormatErrorConfig, OrchestrationConfig
from agentic.contracts import ConversationMessage, ConversationStage, FormatErrorStrategy, ModelResponse, ToolCall, ToolCallSpec
from agentic.model_clients import CallableModelClient, ModelContextLimitError
from agentic.observability import TaskLogger
from agentic.orchestration import TaskOrchestrator
from agentic.rewards import ToolCallRewardEvaluator
from agentic.tools import CallableTool, ToolManager, ToolResult


async def _complete(
    messages: list[ConversationMessage],
    _tools: list[dict[str, Any]] | None,
    _tool_choice: str | None,
) -> ModelResponse:
    if any(message.role.value == "tool" for message in messages):
        return ModelResponse(message=ConversationMessage.assistant("Final answer: tool completed"))
    return ModelResponse(message=ConversationMessage.assistant(tool_calls=[ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "done"}))]))


async def _run_runner(task_id: str | None = None) -> Any:
    tool = CallableTool(
        name="echo",
        description="Return the provided text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        fn=lambda text: ToolResult(content=text),
    )
    orchestrator = TaskOrchestrator(
        config=OrchestrationConfig(name="planner"),
        model_client=CallableModelClient(_complete, context_window=4096, max_output_tokens=512),
        tool_manager=ToolManager(tools=[tool]),
    )
    result = await orchestrator.run("solve it", task_id=task_id)
    return result


async def _complete_after_format_repair(
    messages: list[ConversationMessage],
    _tools: list[dict[str, Any]] | None,
    _tool_choice: str | None,
) -> ModelResponse:
    if any(message.role.value == "tool" for message in messages):
        return ModelResponse(message=ConversationMessage.assistant("Final answer: tool completed"))
    if messages and messages[-1].role.value == "user" and messages[-1].content == "Please emit a valid tool call.":
        return ModelResponse(
            message=ConversationMessage.assistant(tool_calls=[ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "done"}))])
        )
    return ModelResponse(message=ConversationMessage.assistant("<tool_call>bad format</tool_call>"))


async def _run_runner_with_wrong_tool_call_format() -> Any:
    tool = CallableTool(
        name="echo",
        description="Return the provided text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        fn=lambda text: ToolResult(content=text),
    )
    orchestrator = TaskOrchestrator(
        config=OrchestrationConfig(
            name="planner",
            conversation=ConversationConfig(
                format_error=FormatErrorConfig(reprompt="Please emit a valid tool call."),
            ),
        ),
        model_client=CallableModelClient(_complete_after_format_repair, context_window=4096, max_output_tokens=512),
        tool_manager=ToolManager(tools=[tool]),
        reward_evaluator=ToolCallRewardEvaluator(
            reward_per_success_tool_call=1.0,
            reward_per_failed_tool_call=-1.0,
            reward_per_rejected_tool_call=0.0,
            reward_for_wrong_tool_call_format=-1.0,
        ),
    )
    return await orchestrator.run("solve it")


async def _run_runner_with_rolled_back_assistant_answer() -> Any:
    calls = 0

    async def _complete_after_rollback(
        _messages: list[ConversationMessage],
        _tools: list[dict[str, Any]] | None,
        _tool_choice: str | None,
    ) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(message=ConversationMessage.assistant("<tool_call>bad</tool_call><answer>hidden bad</answer>"))
        return ModelResponse(message=ConversationMessage.assistant("<answer>visible final</answer>"))

    orchestrator = TaskOrchestrator(
        config=OrchestrationConfig(
            name="planner",
            conversation=ConversationConfig(
                format_error=FormatErrorConfig(strategy=FormatErrorStrategy.ROLLBACK),
                assistant_rollback=AssistantRollbackConfig(max_rollbacks=1),
            ),
        ),
        model_client=CallableModelClient(_complete_after_rollback, context_window=4096, max_output_tokens=512),
        tool_manager=ToolManager(tools=[]),
    )
    return await orchestrator.run("solve it")


async def _complete_after_final_response_prompt(
    messages: list[ConversationMessage],
    _tools: list[dict[str, Any]] | None,
    _tool_choice: str | None,
) -> ModelResponse:
    if messages and messages[-1].role.value == "user" and messages[-1].content == "Please provide the final answer directly.":
        return ModelResponse(message=ConversationMessage.assistant("Final answer: completed"))
    return ModelResponse(message=ConversationMessage.assistant("I have enough information."))


async def _run_runner_with_final_response_prompt() -> Any:
    orchestrator = TaskOrchestrator(
        config=OrchestrationConfig(
            name="planner",
            conversation=ConversationConfig(final_response_prompt="Please provide the final answer directly."),
        ),
        model_client=CallableModelClient(_complete_after_final_response_prompt, context_window=4096, max_output_tokens=512),
        tool_manager=ToolManager(tools=[]),
    )
    return await orchestrator.run("solve it")


async def _run_runner_with_captured_inputs() -> tuple[Any, list[list[ConversationMessage]], list[list[dict[str, Any]] | None]]:
    captured_messages: list[list[ConversationMessage]] = []
    captured_tools: list[list[dict[str, Any]] | None] = []

    tool = CallableTool(
        name="echo",
        description="Return the provided text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        fn=lambda text: ToolResult(content=text),
    )

    async def _complete_with_capture(
        messages: list[ConversationMessage],
        _tools: list[dict[str, Any]] | None,
        _tool_choice: str | None,
    ) -> ModelResponse:
        captured_messages.append(list(messages))
        captured_tools.append(_tools)
        if messages[-1].role.value == "user" and messages[-1].content == (
            "You cannot call any tool now. Generate your final answer to the given task based on the conversation so far.\n\n"
            "Please provide the final answer directly."
        ):
            return ModelResponse(message=ConversationMessage.assistant("forced final answer"))
        return ModelResponse(
            message=ConversationMessage.assistant(
                tool_calls=[ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "012345678901234567890123456789"}))]
            )
        )

    orchestrator = TaskOrchestrator(
        config=OrchestrationConfig(
            name="planner",
            conversation=ConversationConfig(
                final_response_prompt="Please provide the final answer directly.",
                context_window=25,
            ),
        ),
        model_client=CallableModelClient(_complete_with_capture, context_window=25, max_output_tokens=32),
        tool_manager=ToolManager(tools=[tool]),
    )
    result = await orchestrator.run("x")
    return result, captured_messages, captured_tools


async def _run_runner_with_endpoint_context_limit_recovery() -> Any:
    calls = 0

    async def _complete_with_context_limit(
        _messages: list[ConversationMessage],
        _tools: list[dict[str, Any]] | None,
        _tool_choice: str | None,
    ) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                message=ConversationMessage.assistant(tool_calls=[ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "done"}))])
            )
        if calls == 2:
            raise ModelContextLimitError(
                "input 95 + output 10 = 105 > 100",
                status_code=400,
                input_tokens=95,
                requested_output_tokens=10,
                total_tokens=105,
                context_window=100,
            )
        return ModelResponse(message=ConversationMessage.assistant("Final answer after rollback"))

    tool = CallableTool(
        name="echo",
        description="Return the provided text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        fn=lambda text: ToolResult(content=text),
    )
    orchestrator = TaskOrchestrator(
        config=OrchestrationConfig(
            name="planner",
            conversation=ConversationConfig(context_window=100, context_safety_margin=0),
        ),
        model_client=CallableModelClient(_complete_with_context_limit, context_window=100, max_output_tokens=10),
        tool_manager=ToolManager(tools=[tool]),
    )
    return await orchestrator.run("solve it")


async def _run_runner_with_user_role_tool_results() -> tuple[Any, list[list[ConversationMessage]]]:
    captured_messages: list[list[ConversationMessage]] = []

    tool = CallableTool(
        name="echo",
        description="Return the provided text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        fn=lambda text: ToolResult(content=text),
    )

    async def _complete_with_user_tool_result(
        messages: list[ConversationMessage],
        _tools: list[dict[str, Any]] | None,
        _tool_choice: str | None,
    ) -> ModelResponse:
        captured_messages.append(list(messages))
        if any(message.role.value == "user" and message.content == "done" for message in messages):
            return ModelResponse(message=ConversationMessage.assistant("Final answer: tool completed"))
        return ModelResponse(
            message=ConversationMessage.assistant(tool_calls=[ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "done"}))])
        )

    orchestrator = TaskOrchestrator(
        config=OrchestrationConfig(
            name="planner",
            conversation=ConversationConfig(tool_result_role="user"),
        ),
        model_client=CallableModelClient(_complete_with_user_tool_result, context_window=4096, max_output_tokens=512),
        tool_manager=ToolManager(tools=[tool]),
    )
    result = await orchestrator.run("solve it")
    return result, captured_messages


def test_task_orchestrator_runs_tool_loop() -> None:
    result = asyncio.run(_run_runner())
    assert result.output == "Final answer: tool completed"
    assert result.done is True
    assert result.reason == "assistant_final_answer"
    assert result.reward == 1.0
    assert result.info["reward_list"] == [1.0, 0.0]
    assert result.info["last_reward"] == 0.0
    assert result.info["total_reward"] == 1.0
    assert result.info["tool_metrics"]["echo"]["num_requested"] == 1
    assert result.info["tool_metrics"]["echo"]["num_success"] == 1
    assert len(result.info["tool_metrics"]["echo"]["latency_ms"]) == 1
    assert result.info["tool_metrics"]["echo"]["latency_ms"][0] >= 0.0
    assert [message.role.value for message in result.conversation] == ["system", "user", "assistant", "tool", "assistant"]
    assert result.visible_conversation[-1].content == "Final answer: tool completed"


def test_task_orchestrator_can_send_tool_results_as_user_messages() -> None:
    result, captured_messages = asyncio.run(_run_runner_with_user_role_tool_results())

    assert result.output == "Final answer: tool completed"
    assert [message.role.value for message in result.conversation] == ["system", "user", "assistant", "tool", "assistant"]
    assert [message.role.value for message in captured_messages[-1]] == ["system", "user", "assistant", "user"]
    assert captured_messages[-1][-1].content == "done"


def test_task_orchestrator_preserves_explicit_task_id() -> None:
    result = asyncio.run(_run_runner(task_id="task-123"))
    assert result.metadata["task_id"] == "task-123"


def test_task_orchestrator_punishes_wrong_tool_call_format_and_recovers() -> None:
    result = asyncio.run(_run_runner_with_wrong_tool_call_format())

    assert result.done is True
    assert result.reason == "assistant_final_answer"
    assert result.reward == 0.0
    assert result.info["reward_list"] == [-1.0, 1.0, 0.0]
    assert result.info["last_reward"] == 0.0
    assert result.info["total_reward"] == 0.0
    assert result.info["tool_metrics"]["echo"]["num_requested"] == 1
    assert result.info["tool_metrics"]["echo"]["num_success"] == 1
    # ``rollback_stats`` is the per-trace aggregate added so viz / dashboards
    # can bucket tasks by recovery strategy without replaying every step.
    # The reprompt path does not emit a ``rollback_reason`` (no assistant rollback)
    # but it DOES emit ``format_error_strategy`` and the offending keywords,
    # which is exactly what the format-error telemetry is meant to expose.
    stats = result.info["rollback_stats"]
    assert stats["format_error_by_strategy"] == {"reprompt": 1}
    assert stats["format_error_keywords"] == {"<tool_call>": 1, "</tool_call>": 1}
    assert stats["total"] == 0  # no rollbacks in the reprompt strategy
    assert [message.role.value for message in result.conversation] == ["system", "user", "assistant", "user", "assistant", "tool", "assistant"]


def test_task_orchestrator_uses_visible_conversation_for_final_output_after_rollback() -> None:
    result = asyncio.run(_run_runner_with_rolled_back_assistant_answer())

    assert result.reason == "assistant_final_answer"
    assert result.output == "<answer>visible final</answer>"
    assert any(message.role.value == "ConversationRuntime" and message.name == "rollback" for message in result.conversation)
    assert any("<answer>hidden bad</answer>" in (message.content or "") for message in result.conversation if message.role.value == "assistant")
    assert not any(
        "<answer>hidden bad</answer>" in (message.content or "") for message in result.visible_conversation if message.role.value == "assistant"
    )


def test_task_orchestrator_requests_final_response_before_completion() -> None:
    result = asyncio.run(_run_runner_with_final_response_prompt())

    assert result.done is True
    assert result.reason == "assistant_final_answer"
    assert result.output == "Final answer: completed"
    assert result.info["reward_list"] == [0.0, 0.0]
    assert result.info["last_reward"] == 0.0
    assert result.info["total_reward"] == 0.0
    assert [message.role.value for message in result.conversation] == ["system", "user", "assistant", "user", "assistant"]


def test_task_orchestrator_persists_termination_reason_to_task_logger() -> None:
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(log_dir=tmp_dir)
        orchestrator = TaskOrchestrator(
            config=OrchestrationConfig(
                name="planner",
                conversation=ConversationConfig(final_response_prompt="Please provide the final answer directly."),
            ),
            model_client=CallableModelClient(_complete_after_final_response_prompt, context_window=4096, max_output_tokens=512),
            tool_manager=ToolManager(tools=[]),
            task_logger=logger,
        )

        result = asyncio.run(orchestrator.run("solve it", task_id="task-logger-status"))

        payload = json.loads(Path(tmp_dir, "planner", "task-logger-status.json").read_text(encoding="utf-8"))
        assert result.reason == "assistant_final_answer"
        assert payload["status"] == "assistant_final_answer"
        assert payload["steps"][0]["step_name"].endswith("runtime.step")
        assert payload["steps"][0]["metadata"]["stage_path"] == [
            ConversationStage.NOT_INITIALIZED.value,
            ConversationStage.SYSTEM_MESSAGE_SET.value,
            ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT.value,
        ]
        assert any(
            step["metadata"].get("stage") == ConversationStage.ASSISTANT_COMPLETED.value
            for step in payload["steps"]
            if step["step_name"].endswith("runtime.step")
        )


def test_task_orchestrator_sends_visible_conversation_to_model_client() -> None:
    result, captured_messages, captured_tools = asyncio.run(_run_runner_with_captured_inputs())

    assert result.reason == "terminated_context_limit"
    assert len(captured_messages) == 2
    assert captured_tools[0] is not None
    assert captured_tools[1] is not None  # tools always passed for inference cache hit
    assert [message.role.value for message in captured_messages[1]] == ["system", "user", "user"]
    assert captured_messages[1][-1].content == (
        "You cannot call any tool now. Generate your final answer to the given task based on the conversation so far.\n\n"
        "Please provide the final answer directly."
    )


def test_task_orchestrator_recovers_endpoint_context_limit_error() -> None:
    result = asyncio.run(_run_runner_with_endpoint_context_limit_recovery())

    assert result.reason == "terminated_context_limit"
    assert result.output == "Final answer after rollback"
    assert result.info["model_context_limit_error"]["input_tokens"] == 95
    assert result.info["context_limit_estimate"]["provider_error"]["context_window"] == 100


def test_task_orchestrator_indents_log_messages_by_level() -> None:
    import logging

    captured: list[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(self.format(record))

    handler = _CaptureHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))

    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="indent-test", log_dir=tmp_dir)
        logger._logger.handlers.clear()
        logger._logger.addHandler(handler)

        orchestrator = TaskOrchestrator(
            config=OrchestrationConfig(
                name="planner",
                conversation=ConversationConfig(final_response_prompt="Please provide the final answer directly."),
            ),
            model_client=CallableModelClient(_complete_after_final_response_prompt, context_window=4096, max_output_tokens=512),
            tool_manager=ToolManager(tools=[]),
            task_logger=logger,
            level=2,
        )
        asyncio.run(orchestrator.run("solve it", task_id="indent-test"))

    assert len(captured) > 0
    for msg in captured:
        assert msg.startswith("    "), f"Expected 4-space indent (level=2), got: {msg!r}"


def test_task_orchestrator_writes_json_to_structured_subdirectory() -> None:
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="subdir-test", log_dir=tmp_dir)

        orchestrator = TaskOrchestrator(
            config=OrchestrationConfig(
                name="planner",
                conversation=ConversationConfig(final_response_prompt="Please provide the final answer directly."),
            ),
            model_client=CallableModelClient(_complete_after_final_response_prompt, context_window=4096, max_output_tokens=512),
            tool_manager=ToolManager(tools=[]),
            task_logger=logger,
        )
        # Simulate nested orchestrator with tool_path
        orchestrator.tool_path = ["delegate", "search"]
        asyncio.run(orchestrator.run("solve it", task_id="nested-task"))

        # JSON should be in the subdirectory matching the tool_path (orchestrator name prepended)
        expected_path = Path(tmp_dir) / "planner" / "delegate" / "search" / "nested-task.json"
        assert expected_path.exists(), f"Expected JSON at {expected_path}"
        payload = json.loads(expected_path.read_text(encoding="utf-8"))
        assert payload["tool_path"] == ["planner", "delegate", "search"]
