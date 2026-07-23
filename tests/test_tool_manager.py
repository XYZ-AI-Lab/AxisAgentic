import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from agentic.config import ToolArgumentRepairConfig, ToolManagerConfig
from agentic.contracts import ToolRequest, ToolResultReason, ToolResultStatus
from agentic.observability import TaskLogger
from agentic.tools import CallableTool, ToolManager, ToolResult
from agentic.tools.manager import UNKNOWN_TOOL_METRICS_KEY


async def _run_tool() -> tuple[ToolResult, dict[str, int]]:
    tool = CallableTool(
        name="echo",
        description="Return the provided text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        fn=lambda text: ToolResult(content=text.upper()),
    )
    manager = ToolManager(tools=[tool])
    result = await manager._execute(ToolRequest(tool_name="echo", arguments={"text": "agentic"}))
    metrics = manager.metrics_snapshot()["echo"]
    return result, {
        "num_requested": metrics.num_requested,
        "num_success": metrics.num_success,
        "num_failed": metrics.num_failed,
        "num_rejected": metrics.num_rejected,
    }


def test_tool_manager_executes_callable_tool() -> None:
    result, metrics = asyncio.run(_run_tool())
    assert result.content == "AGENTIC"
    assert result.status == ToolResultStatus.SUCCESS
    assert metrics == {"num_requested": 1, "num_success": 1, "num_failed": 0, "num_rejected": 0}


def test_tool_manager_snapshot_restore_task_state() -> None:
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
    manager = ToolManager(tools=[tool])

    asyncio.run(manager._execute(ToolRequest(tool_name="echo", arguments={"text": "one"})))
    snapshot = manager.snapshot_task_state()
    asyncio.run(manager._execute(ToolRequest(tool_name="echo", arguments={"text": "two"})))

    assert manager.metrics_snapshot()["echo"].num_success == 2

    manager.restore_task_state(snapshot)

    assert manager.metrics_snapshot()["echo"].num_success == 1
    assert manager.unknown_tool_names_snapshot() == {}


def test_tool_manager_rejects_duplicate_tool_registration() -> None:
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
    manager = ToolManager(tools=[tool])

    with pytest.raises(ValueError, match="already registered") as exc_info:
        manager.register(tool)

    assert "already registered" in str(exc_info.value)


def test_tool_manager_rejects_unknown_tool() -> None:
    manager = ToolManager(tools=[])
    result = asyncio.run(manager._execute(ToolRequest(tool_name="nonexistent", arguments={})))
    assert result.status == ToolResultStatus.REJECTED
    assert result.reason == ToolResultReason.UNKNOWN_TOOL
    assert "Unknown tool 'nonexistent'" in result.content
    metrics = manager.metrics_snapshot()[UNKNOWN_TOOL_METRICS_KEY]
    assert metrics.num_requested == 1
    assert metrics.num_rejected == 1
    assert metrics.rejection_reasons == {ToolResultReason.UNKNOWN_TOOL: 1}
    assert manager.unknown_tool_names_snapshot() == {"nonexistent": 1}


def test_tool_manager_returns_failed_for_execution_exception() -> None:
    def _raise(**_kwargs: object) -> ToolResult:
        msg = "something went wrong"
        raise RuntimeError(msg)

    tool = CallableTool(
        name="bad_tool",
        description="Always fails.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        fn=_raise,
    )
    manager = ToolManager(tools=[tool])
    result = asyncio.run(manager._execute(ToolRequest(tool_name="bad_tool", arguments={})))
    assert result.status == ToolResultStatus.FAILED
    assert result.reason == ToolResultReason.EXECUTION_ERROR
    assert "something went wrong" in result.content
    metrics = manager.metrics_snapshot()["bad_tool"]
    assert metrics.num_requested == 1
    assert metrics.num_failed == 1
    assert metrics.failure_reasons == {ToolResultReason.EXECUTION_ERROR: 1}
    assert len(metrics.latency_ms) == 1


def test_tool_manager_parameter_validation_rejects_missing_required() -> None:
    tool = CallableTool(
        name="echo",
        description="Requires text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        fn=lambda text: ToolResult(content=text),
    )
    manager = ToolManager(tools=[tool])
    result = asyncio.run(manager._execute(ToolRequest(tool_name="echo", arguments={})))
    assert result.status == ToolResultStatus.FAILED
    assert result.reason == ToolResultReason.PARAMETER_ERROR
    assert "Parameter validation error" in result.content
    metrics = manager.metrics_snapshot()["echo"]
    assert metrics.num_requested == 1
    assert metrics.num_failed == 1
    assert metrics.failure_reasons == {ToolResultReason.PARAMETER_ERROR: 1}


def test_tool_manager_argument_repair_is_opt_in() -> None:
    tool = CallableTool(
        name="python",
        description="Runs code.",
        parameters={
            "type": "object",
            "properties": {
                "code_block": {"type": "string", "x-agentic-aliases": ["code"]},
                "sandbox_id": {"type": "string", "default": "default"},
            },
            "required": ["code_block"],
        },
        strict_mode=False,
        fn=lambda code_block, sandbox_id="default": ToolResult(content=f"{code_block}|{sandbox_id}"),
    )
    manager = ToolManager(tools=[tool])

    result = asyncio.run(manager._execute(ToolRequest(tool_name="python", arguments={"code": "print(1)", "sandbox_id": 1})))

    assert result.status == ToolResultStatus.FAILED
    assert result.reason == ToolResultReason.PARAMETER_ERROR


def test_tool_manager_repairs_arguments_when_enabled() -> None:
    tool = CallableTool(
        name="python",
        description="Runs code.",
        parameters={
            "type": "object",
            "properties": {
                "code_block": {"type": "string", "x-agentic-aliases": ["code"]},
                "sandbox_id": {"type": "string", "default": "default"},
            },
            "required": ["code_block"],
        },
        strict_mode=False,
        fn=lambda code_block, sandbox_id="default": ToolResult(content=f"{code_block}|{sandbox_id}"),
    )
    manager = ToolManager(
        tools=[tool],
        config=ToolManagerConfig(argument_repair=ToolArgumentRepairConfig(enabled=True)),
    )

    request = ToolRequest(tool_name="python", arguments={" code ": "print(1)", "sandbox_id": 1, "ignored": "x"})
    repaired = manager.repair_tool_request(request)
    result = asyncio.run(manager._execute(request))

    assert repaired.arguments["code_block"] == "print(1)"
    assert repaired.arguments["sandbox_id"] == "1"
    assert repaired.metadata["argument_repair"]["applied_rules"] == ["normalize_key_names", "schema_aliases", "scalar_type_coercion"]
    assert result.status == ToolResultStatus.SUCCESS
    assert result.content == "print(1)|1"


def test_tool_manager_argument_repair_skips_invalid_none_defaults() -> None:
    tool = CallableTool(
        name="search",
        description="Searches.",
        parameters={
            "type": "object",
            "properties": {
                "q": {"type": "string"},
                "location": {"type": "string", "default": None},
                "num": {"type": "integer", "default": None},
                "gl": {"type": "string", "default": "us"},
            },
            "required": ["q"],
        },
        strict_mode=False,
        fn=lambda q, gl="us", location=None, num=None: ToolResult(content=f"{q}|{gl}|{location}|{num}"),
    )
    manager = ToolManager(
        tools=[tool],
        config=ToolManagerConfig(argument_repair=ToolArgumentRepairConfig(enabled=True)),
    )

    repaired = manager.repair_tool_request(ToolRequest(tool_name="search", arguments={"q": "agentic"}))
    result = asyncio.run(manager._execute(ToolRequest(tool_name="search", arguments={"q": "agentic"})))

    assert repaired.arguments == {"q": "agentic", "gl": "us"}
    assert result.status == ToolResultStatus.SUCCESS
    assert result.content == "agentic|us|None|None"


def test_tool_manager_argument_repair_hook_can_be_registered() -> None:
    tool = CallableTool(
        name="echo",
        description="Echoes.",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        strict_mode=False,
        fn=lambda text: ToolResult(content=text),
    )
    manager = ToolManager(
        tools=[tool],
        config=ToolManagerConfig(argument_repair=ToolArgumentRepairConfig(enabled=True)),
    )
    manager.register_argument_repair_hook("echo", lambda _request, _tool, arguments: {"text": arguments.get("message", "")})

    result = asyncio.run(manager._execute(ToolRequest(tool_name="echo", arguments={"message": "hooked"})))

    assert result.status == ToolResultStatus.SUCCESS
    assert result.content == "hooked"


def test_tool_manager_call_budget_enforcement() -> None:
    tool = CallableTool(
        name="echo",
        description="Echoes.",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": False},
        fn=lambda text: ToolResult(content=text),
        max_calls_per_task=2,
        call_budget_exceeded_message="Budget exceeded.",
    )
    manager = ToolManager(tools=[tool])
    asyncio.run(manager._execute(ToolRequest(tool_name="echo", arguments={"text": "1"})))
    asyncio.run(manager._execute(ToolRequest(tool_name="echo", arguments={"text": "2"})))
    result = asyncio.run(manager._execute(ToolRequest(tool_name="echo", arguments={"text": "3"})))

    assert result.status == ToolResultStatus.REJECTED
    assert result.reason == ToolResultReason.CALL_BUDGET_EXCEEDED
    assert result.content == "Budget exceeded."
    metrics = manager.metrics_snapshot()["echo"]
    assert metrics.num_requested == 3
    assert metrics.num_success == 2
    assert metrics.num_rejected == 1
    assert metrics.rejection_reasons == {ToolResultReason.CALL_BUDGET_EXCEEDED: 1}


def test_tool_manager_reset_task_state_clears_budget() -> None:
    tool = CallableTool(
        name="echo",
        description="Echoes.",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": False},
        fn=lambda text: ToolResult(content=text),
        max_calls_per_task=1,
    )
    manager = ToolManager(tools=[tool])
    asyncio.run(manager._execute(ToolRequest(tool_name="echo", arguments={"text": "1"})))
    manager.reset_task_state()
    result = asyncio.run(manager._execute(ToolRequest(tool_name="echo", arguments={"text": "2"})))
    assert result.content == "2"
    assert result.status == ToolResultStatus.SUCCESS


def test_tool_manager_reset_task_state_clears_metrics() -> None:
    """Metrics must not bleed across tasks after reset_task_state()."""
    tool = CallableTool(
        name="echo",
        description="Echoes.",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": False},
        fn=lambda text: ToolResult(content=text),
    )
    manager = ToolManager(tools=[tool])
    asyncio.run(manager._execute(ToolRequest(tool_name="echo", arguments={"text": "1"})))
    metrics_before = manager.metrics_snapshot()["echo"]
    assert metrics_before.num_requested == 1
    assert metrics_before.num_success == 1
    assert len(metrics_before.latency_ms) == 1

    manager.reset_task_state()

    metrics_after = manager.metrics_snapshot()["echo"]
    assert metrics_after.num_requested == 0
    assert metrics_after.num_success == 0
    assert metrics_after.num_failed == 0
    assert metrics_after.num_rejected == 0
    assert metrics_after.latency_ms == []
    assert metrics_after.rejection_reasons == {}
    assert metrics_after.failure_reasons == {}

    # Second task: metrics reflect only the new task.
    asyncio.run(manager._execute(ToolRequest(tool_name="echo", arguments={"text": "2"})))
    metrics_task2 = manager.metrics_snapshot()["echo"]
    assert metrics_task2.num_requested == 1
    assert metrics_task2.num_success == 1
    assert len(metrics_task2.latency_ms) == 1


def test_tool_manager_propagates_tool_result_metadata_to_task_trace() -> None:
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="test-tool-metadata", log_dir=tmp_dir)
        logger.start_task("task-1", "inspect metadata")

        tool = CallableTool(
            name="echo",
            description="Echoes.",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": False},
            fn=lambda text: ToolResult(content=text, metadata={"timing_ms": {"jina": 12.5}}),
        )
        manager = ToolManager(tools=[tool], task_logger=logger)

        asyncio.run(manager._execute(ToolRequest(tool_name="echo", arguments={"text": "hi"}), task_id="task-1"))
        logger.finish_task("task-1", status="completed")

        payload = json.loads((Path(tmp_dir) / "task-1.json").read_text(encoding="utf-8"))
        assert payload["tool_calls"][0]["metadata"]["timing_ms"]["jina"] == 12.5
