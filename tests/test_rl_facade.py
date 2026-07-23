import asyncio
from typing import Any

from agentic.config import ConversationConfig, FormatErrorConfig, OrchestrationConfig
from agentic.contracts import ConversationMessage, ConversationStage, ModelResponse, ToolCall, ToolCallSpec
from agentic.conversations import ConversationRuntime
from agentic.model_clients import CallableModelClient
from agentic.orchestration import TaskOrchestrator
from agentic.rl import RLEnvironmentFacade, RLPolicyFacade, RLRolloutFacade
from agentic.tools import CallableTool, ToolManager, ToolResult


def _estimate_tokens(messages: list[ConversationMessage]) -> int:
    return sum(len(message.content or "") for message in messages)


WRONG_TOOL_CALL_FORMAT_PROMPT = "Please emit a valid tool call."
TOOL_CALL_KEYWORDS = ["<tool_call>", "</tool_call>"]
EARLY_STOP_ANNOUNCEMENT_PROMPT = "You cannot call any tool now. Generate your final answer to the given task based on the conversation so far."


def test_rl_policy_facade_passes_tool_definitions_to_model_client() -> None:
    captured: dict[str, object] = {}

    async def _complete(messages: list[ConversationMessage], tools: list[dict[str, Any]], tool_choice: str | None) -> ModelResponse:
        captured["messages"] = messages
        captured["tools"] = tools
        captured["tool_choice"] = tool_choice
        return ModelResponse(message=ConversationMessage.assistant("ok"))

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
    facade = RLPolicyFacade(
        model_client=CallableModelClient(_complete, context_window=1024, max_output_tokens=128),
        tool_manager=ToolManager(tools=[tool]),
    )

    result = asyncio.run(facade.act([ConversationMessage.user("hello")]))

    assert result.message.content == "ok"
    assert captured["tool_choice"] == "auto"
    assert isinstance(captured["tools"], list)
    assert captured["tools"][0]["function"]["name"] == "echo"


def test_rl_environment_facade_wraps_conversation_runtime_reset_and_step() -> None:
    runtime = ConversationRuntime(
        config=ConversationConfig(
            system_prompt="system",
            user_prompt_template="Task: {task}",
            early_stop_announcement_prompt=EARLY_STOP_ANNOUNCEMENT_PROMPT,
            format_error=FormatErrorConfig(keywords=TOOL_CALL_KEYWORDS, reprompt=WRONG_TOOL_CALL_FORMAT_PROMPT),
            max_turns=12,
            context_window=200,
        ),
        tools=None,
        max_output_tokens=20,
        token_estimator=_estimate_tokens,
    )
    facade = RLEnvironmentFacade(runtime=runtime)

    initial = facade.reset("say hello")
    follow_up = facade.step(ConversationMessage.assistant(tool_calls=[ToolCall(function=ToolCallSpec(name="echo", arguments={"text": "hello"}))]))
    after_tool = facade.step(ConversationMessage.tool("hello", tool_call_id=follow_up.full_conversation[-1].tool_calls[0].id, name="echo"))

    assert initial.full_conversation[1].content == "Task: say hello"
    assert [message.role.value for message in initial.appended_messages] == ["system", "user"]
    assert follow_up.stage == ConversationStage.ASSISTANT_TOOL_CALLS_AWAIT_TOOL
    assert follow_up.info["tool_request_count"] == 1
    assert [message.role.value for message in follow_up.appended_messages] == ["assistant"]
    assert after_tool.stage == ConversationStage.TOOL_MESSAGES_AWAIT_ASSISTANT
    assert after_tool.full_conversation[-1].content == "hello"


def test_rl_rollout_facade_wraps_task_orchestrator() -> None:
    async def _complete(_messages: list[ConversationMessage], _tools: list[dict[str, Any]], _tool_choice: str | None) -> ModelResponse:
        return ModelResponse(message=ConversationMessage.assistant("done"))

    orchestrator = TaskOrchestrator(
        config=OrchestrationConfig(name="planner"),
        model_client=CallableModelClient(_complete, context_window=1024, max_output_tokens=128),
    )
    facade = RLRolloutFacade(orchestrator=orchestrator)

    result = asyncio.run(facade.rollout("finish this"))

    assert result.output == "done"
    assert result.done is True
