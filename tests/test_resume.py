"""Tests for resume-from-trace capability: OrchestrationResult.from_trace() and conversation parsing."""

from __future__ import annotations

from typing import Any

from agentic.contracts import ConversationMessage, MessageRole
from agentic.contracts.messages import ToolCall, ToolCallSpec
from agentic.observability.task_logger import TaskTrace, TokenUsageRecord
from agentic.orchestration.task_orchestrator import OrchestrationResult, _conversation_dicts_to_messages

# ---------------------------------------------------------------------------
# _conversation_dicts_to_messages tests
# ---------------------------------------------------------------------------


def test_conversation_dicts_to_messages_basic() -> None:
    """Round-trip: ConversationMessage → to_model_message() → _conversation_dicts_to_messages()."""
    original = [
        ConversationMessage.system("You are helpful."),
        ConversationMessage.user("Hello"),
        ConversationMessage.assistant("Hi there!"),
    ]
    dicts = [m.to_model_message() for m in original]
    restored = _conversation_dicts_to_messages(dicts)

    assert len(restored) == 3
    assert restored[0].role == MessageRole.SYSTEM
    assert restored[0].content == "You are helpful."
    assert restored[1].role == MessageRole.USER
    assert restored[1].content == "Hello"
    assert restored[2].role == MessageRole.ASSISTANT
    assert restored[2].content == "Hi there!"


def test_conversation_dicts_to_messages_with_tool_calls() -> None:
    """Tool call arguments (JSON strings in traces) should be parsed back to dicts."""
    msg = ConversationMessage.assistant(
        content=None,
        reasoning_content="Need to call the search tool.",
        tool_calls=[
            ToolCall(id="call_1", function=ToolCallSpec(name="search", arguments={"query": "test"})),
        ],
    )
    dicts = [msg.to_model_message()]
    restored = _conversation_dicts_to_messages(dicts)

    assert len(restored) == 1
    assert restored[0].reasoning_content == "Need to call the search tool."
    assert restored[0].tool_calls is not None
    assert len(restored[0].tool_calls) == 1
    tc = restored[0].tool_calls[0]
    assert tc.id == "call_1"
    assert tc.function.name == "search"
    assert tc.function.arguments == {"query": "test"}


def test_conversation_dicts_to_messages_skips_runtime_role() -> None:
    """Messages with the ConversationRuntime role should be skipped."""
    dicts = [
        {"role": "user", "content": "hi"},
        {"role": "ConversationRuntime", "content": "internal note"},
        {"role": "assistant", "content": "hello"},
    ]
    restored = _conversation_dicts_to_messages(dicts)
    assert len(restored) == 2
    assert restored[0].role == MessageRole.USER
    assert restored[1].role == MessageRole.ASSISTANT


def test_conversation_dicts_to_messages_with_tool_message() -> None:
    """Tool result messages should preserve tool_call_id and name."""
    dicts = [
        {"role": "tool", "content": "search results here", "tool_call_id": "call_1", "name": "search"},
    ]
    restored = _conversation_dicts_to_messages(dicts)
    assert len(restored) == 1
    assert restored[0].role == MessageRole.TOOL
    assert restored[0].content == "search results here"
    assert restored[0].tool_call_id == "call_1"
    assert restored[0].name == "search"


# ---------------------------------------------------------------------------
# OrchestrationResult.from_trace() tests
# ---------------------------------------------------------------------------


def _make_trace(**overrides: Any) -> TaskTrace:
    """Create a minimal completed TaskTrace for testing."""
    defaults = {
        "task_id": "test-1",
        "task_input": "What is 2+2?",
        "status": "assistant_final_answer",
        "started_at": "2026-01-01T00:00:00+08:00",
        "ended_at": "2026-01-01T00:01:00+08:00",
        "metadata": {
            "output": "4",
            "turn_used": 3,
            "orchestrator_name": "test-orch",
            "run_info": {"total_reward": 1.5, "reward_list": [0.5, 0.5, 0.5]},
        },
        "conversation": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        "token_usage": TokenUsageRecord(input_tokens=100, output_tokens=10, total_tokens=110),
    }
    defaults.update(overrides)
    return TaskTrace(**defaults)


def test_from_trace_reconstructs_result() -> None:
    """from_trace should correctly map all fields."""
    trace = _make_trace()
    result = OrchestrationResult.from_trace(trace)

    assert result.output == "4"
    assert result.reason == "assistant_final_answer"
    assert result.num_turns == 3
    assert result.reward == 1.5
    assert result.done is True
    assert result.metadata["orchestrator_name"] == "test-orch"
    assert result.info == {"total_reward": 1.5, "reward_list": [0.5, 0.5, 0.5]}


def test_from_trace_reconstructs_conversation() -> None:
    """from_trace with reconstruct_conversation=True should parse messages."""
    trace = _make_trace()
    result = OrchestrationResult.from_trace(trace, reconstruct_conversation=True)

    assert len(result.conversation) == 2
    assert result.conversation[0].role == MessageRole.USER
    assert result.conversation[0].content == "What is 2+2?"
    assert result.conversation[1].role == MessageRole.ASSISTANT
    assert result.conversation[1].content == "4"


def test_from_trace_without_conversation_reconstruction() -> None:
    """from_trace with reconstruct_conversation=False should leave conversation empty."""
    trace = _make_trace()
    result = OrchestrationResult.from_trace(trace, reconstruct_conversation=False)

    assert result.conversation == []
    assert result.output == "4"  # output still comes from metadata


def test_from_trace_extracts_output_from_conversation_fallback() -> None:
    """When metadata has no 'output' field, fall back to last assistant message."""
    metadata = {"turn_used": 2, "orchestrator_name": "test"}
    trace = _make_trace(
        metadata=metadata,
        conversation=[
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "first response"},
            {"role": "user", "content": "follow up"},
            {"role": "assistant", "content": "final answer"},
        ],
    )
    result = OrchestrationResult.from_trace(trace)
    assert result.output == "final answer"


def test_from_trace_preserves_metadata_output_after_rollback() -> None:
    trace = _make_trace(
        metadata={
            "output": "No \\boxed{} content found in final response.",
            "turn_used": 2,
            "orchestrator_name": "test",
            "run_info": {},
        },
        conversation=[
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "<answer>hidden bad</answer>"},
            {
                "role": "ConversationRuntime",
                "name": "rollback",
                "content": '{"rollback_message_count":1,"reason":"format_error"}',
            },
            {"role": "assistant", "content": "<answer>visible final</answer>"},
        ],
    )

    result = OrchestrationResult.from_trace(trace)

    assert result.output == "No \\boxed{} content found in final response."
    assert [message.role for message in result.visible_conversation] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert result.visible_conversation[-1].content == "<answer>visible final</answer>"
    assert any(message.role == MessageRole.CONVERSATION_RUNTIME for message in result.conversation)
    assert "<answer>hidden bad</answer>" not in [message.content for message in result.visible_conversation]


def test_from_trace_uses_visible_conversation_fallback_after_rollback_when_metadata_output_missing() -> None:
    trace = _make_trace(
        metadata={"turn_used": 2, "orchestrator_name": "test", "run_info": {}},
        conversation=[
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "<answer>hidden bad</answer>"},
            {
                "role": "ConversationRuntime",
                "name": "rollback",
                "content": '{"rollback_message_count":1,"reason":"format_error"}',
            },
            {"role": "assistant", "content": "<answer>visible final</answer>"},
        ],
    )

    result = OrchestrationResult.from_trace(trace)

    assert result.output == "<answer>visible final</answer>"
    assert result.visible_conversation[-1].content == "<answer>visible final</answer>"


def test_from_trace_handles_empty_conversation() -> None:
    """from_trace should handle traces with no conversation."""
    trace = _make_trace(conversation=[], metadata={"output": "cached", "turn_used": 0})
    result = OrchestrationResult.from_trace(trace)

    assert result.output == "cached"
    assert result.conversation == []
    assert result.num_turns == 0


def test_from_trace_handles_missing_run_info() -> None:
    """from_trace should default reward and info when run_info is absent."""
    trace = _make_trace(metadata={"output": "answer", "turn_used": 1})
    result = OrchestrationResult.from_trace(trace)

    assert result.reward == 0.0
    assert result.info == {}
