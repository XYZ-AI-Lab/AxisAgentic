import asyncio
from typing import Any

from agentic.config import OrchestrationConfig
from agentic.contracts import ConversationMessage, ModelResponse, ToolRequest
from agentic.model_clients import CallableModelClient
from agentic.orchestration import OrchestratorTool, TaskOrchestrator
from agentic.tools import ToolContext, ToolManager


async def _complete(_messages: list[ConversationMessage], _tools: list[dict[str, Any]], _tool_choice: str | None) -> ModelResponse:
    return ModelResponse(message=ConversationMessage.assistant("subtask done"))


async def _run_orchestrator_tool() -> str:
    orchestrator = TaskOrchestrator(
        config=OrchestrationConfig(name="sub-runner"),
        model_client=CallableModelClient(_complete, context_window=1024, max_output_tokens=128),
        tool_manager=ToolManager(tools=[]),
    )
    orchestrator_tool = OrchestratorTool(
        orchestrator=orchestrator,
        name="delegate",
        description="Delegate the task to a nested orchestrator.",
        parameters={
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
            "additionalProperties": False,
        },
    )
    result = await orchestrator_tool.arun(
        {"task": "nested"},
        context=ToolContext(request=ToolRequest(tool_name="delegate", arguments={"task": "nested"}, call_id="call_1")),
    )
    return result.content


def test_orchestrator_tool_wraps_task_orchestrator() -> None:
    content = asyncio.run(_run_orchestrator_tool())
    assert content == "subtask done"


async def _run_orchestrator_tool_with_task_id() -> tuple[str, str]:
    captured_task_ids: list[str] = []
    original_run = TaskOrchestrator.run

    async def _patched_run(self: Any, task: Any, task_id: str | None = None) -> Any:
        captured_task_ids.append(task_id or "auto")
        return await original_run(self, task, task_id=task_id)

    TaskOrchestrator.run = _patched_run  # type: ignore[assignment]
    try:
        orchestrator = TaskOrchestrator(
            config=OrchestrationConfig(name="sub-runner"),
            model_client=CallableModelClient(_complete, context_window=1024, max_output_tokens=128),
            tool_manager=ToolManager(tools=[]),
        )
        orchestrator_tool = OrchestratorTool(
            orchestrator=orchestrator,
            name="delegate",
            description="Delegate the task.",
            parameters={
                "type": "object",
                "properties": {"task": {"type": "string"}, "task_id": {"type": "string"}},
                "required": ["task"],
                "additionalProperties": False,
            },
        )
        result = await orchestrator_tool.arun(
            {"task": "nested", "task_id": "custom-id-123"},
            context=ToolContext(request=ToolRequest(tool_name="delegate", arguments={}, call_id="call_2")),
        )
        return result.content, captured_task_ids[0]
    finally:
        TaskOrchestrator.run = original_run  # type: ignore[assignment]


def test_orchestrator_tool_extracts_task_id_from_arguments() -> None:
    content, task_id = asyncio.run(_run_orchestrator_tool_with_task_id())
    assert content == "subtask done"
    assert task_id == "custom-id-123"


async def _run_orchestrator_tool_without_task_key() -> str:
    """When arguments has no 'task' key, the full dict is passed as structured input."""
    orchestrator = TaskOrchestrator(
        config=OrchestrationConfig(name="sub-runner", allow_structured_task_input=True),
        model_client=CallableModelClient(_complete, context_window=1024, max_output_tokens=128),
        tool_manager=ToolManager(tools=[]),
    )
    orchestrator_tool = OrchestratorTool(
        orchestrator=orchestrator,
        name="delegate",
        description="Delegate the task.",
        parameters={
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
            "additionalProperties": False,
        },
    )
    result = await orchestrator_tool.arun(
        {"question": "what is 1+1?"},
        context=ToolContext(request=ToolRequest(tool_name="delegate", arguments={}, call_id="call_3")),
    )
    return result.content


def test_orchestrator_tool_passes_full_dict_when_no_task_key() -> None:
    content = asyncio.run(_run_orchestrator_tool_without_task_key())
    assert content == "subtask done"


def test_tool_manager_propagates_level_to_orchestrator_tool() -> None:
    inner_orchestrator = TaskOrchestrator(
        config=OrchestrationConfig(name="sub-runner"),
        model_client=CallableModelClient(_complete, context_window=1024, max_output_tokens=128),
        tool_manager=ToolManager(tools=[]),
    )
    assert inner_orchestrator.level == 0
    assert inner_orchestrator.tool_manager._level == 1

    orchestrator_tool = OrchestratorTool(
        orchestrator=inner_orchestrator,
        name="delegate",
        description="Delegate.",
        parameters={"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"], "additionalProperties": False},
    )

    # Registering into a ToolManager at level=2 should set the inner orchestrator to level=2
    parent_manager = ToolManager(tools=[orchestrator_tool], level=2)
    assert inner_orchestrator.level == 2
    assert inner_orchestrator.tool_manager._level == 3
    assert inner_orchestrator.tool_path == ["delegate"]
    assert inner_orchestrator.tool_manager._tool_path == ["delegate"]

    # set_level should also propagate
    parent_manager.set_level(5)
    assert inner_orchestrator.level == 5
    assert inner_orchestrator.tool_manager._level == 6


def test_tool_default_emoji_and_orchestrator_tool_emoji() -> None:
    from agentic.tools import CallableTool, ToolResult

    tool = CallableTool(
        name="echo",
        description="Echo.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        fn=lambda: ToolResult(content="ok"),
    )
    assert tool.emoji == "🔧"

    orch_tool = OrchestratorTool(
        orchestrator=TaskOrchestrator(
            config=OrchestrationConfig(name="sub"),
            model_client=CallableModelClient(_complete, context_window=1024, max_output_tokens=128),
            tool_manager=ToolManager(tools=[]),
        ),
        name="delegate",
        description="Delegate.",
    )
    assert orch_tool.emoji == "🤖"

    custom_tool = CallableTool(
        name="search",
        description="Search.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        fn=lambda: ToolResult(content="found"),
        emoji="🔍",
    )
    assert custom_tool.emoji == "🔍"
