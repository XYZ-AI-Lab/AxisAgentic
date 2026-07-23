import asyncio
from typing import Any

from agentic.config import ConversationConfig, OrchestrationConfig
from agentic.contracts import ConversationMessage, ModelResponse, ToolCall, ToolCallSpec
from agentic.model_clients import CallableModelClient
from agentic.orchestration import TaskOrchestrator
from agentic.tools import CallableTool, ToolManager, ToolResult


async def _complete(messages: list[ConversationMessage], _tools: list[dict[str, Any]], _tool_choice: str | None) -> ModelResponse:
    last_message = messages[-1]
    if last_message.role.value == "user" and last_message.content in {
        "You cannot call any tool now. Generate your final answer to the given task based on the conversation so far.",
        "You cannot call any tool now. Generate your final answer to the given task based on the conversation so far.\n\n"
        "Please provide the final answer directly.",
    }:
        return ModelResponse(message=ConversationMessage.assistant("forced final answer"))
    return ModelResponse(
        message=ConversationMessage.assistant(
            tool_calls=[ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "012345678901234567890123456789"}))]
        )
    )


async def _run_forced_finalization() -> Any:
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
                final_response_prompt="Please provide the final answer directly.",
                context_window=25,
            ),
        ),
        model_client=CallableModelClient(_complete, context_window=25, max_output_tokens=32),
        tool_manager=ToolManager(tools=[tool]),
    )
    return await orchestrator.run("x")


async def _run_turn_limited_finalization() -> Any:
    orchestrator = TaskOrchestrator(
        config=OrchestrationConfig(
            name="planner",
            conversation=ConversationConfig(max_turns=1, context_window=64),
        ),
        model_client=CallableModelClient(_complete, context_window=64, max_output_tokens=32),
    )
    return await orchestrator.run("Short task.")


def test_task_orchestrator_generates_forced_final_answer_when_context_budget_is_exceeded() -> None:
    result = asyncio.run(_run_forced_finalization())
    assert result.output == "forced final answer"
    assert result.reason == "terminated_context_limit"
    assert result.done is True


def test_task_orchestrator_generates_forced_final_answer_when_turn_limit_is_reached() -> None:
    result = asyncio.run(_run_turn_limited_finalization())
    assert result.output == "forced final answer"
    assert result.reason == "terminated_turn_limit"
    assert result.done is True
