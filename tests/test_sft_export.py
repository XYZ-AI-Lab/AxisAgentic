from __future__ import annotations

import json

from agentic.sft_export import conversation_to_swift_agent_sample, validate_swift_agent_sample


def test_swift_agent_export_preserves_reasoning_and_tool_flow() -> None:
    conversation = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "QUESTION"},
        {
            "role": "assistant",
            "content": "I will search.",
            "reasoning_content": "Need evidence.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": {"query": "QUESTION"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "web_search", "content": '{"result":"OK"}'},
        {"role": "assistant", "content": "ANSWER"},
    ]
    tools = [{"type": "function", "function": {"name": "web_search", "description": "Search", "parameters": {"type": "object"}}}]

    result = conversation_to_swift_agent_sample(conversation=conversation, tools=tools)

    validate_swift_agent_sample(result.sample)
    assert json.loads(result.sample["tools"])[0]["function"]["name"] == "web_search"
    assert result.sample["messages"][2]["role"] == "assistant"
    assert "<think>\nNeed evidence.\n</think>" in result.sample["messages"][2]["content"]
    assert result.sample["messages"][3] == {"role": "tool_call", "content": '{"name":"web_search","arguments":{"query":"QUESTION"}}'}
    assert result.sample["messages"][4] == {"role": "tool_response", "content": '{"result":"OK"}'}


def test_swift_agent_export_wraps_non_json_tool_response() -> None:
    result = conversation_to_swift_agent_sample(
        conversation=[
            {"role": "user", "content": "QUESTION"},
            {"role": "tool", "content": "plain text", "tool_call_id": "call_1", "name": "search"},
        ],
        tools=[],
    )

    validate_swift_agent_sample(result.sample)
    assert result.sample["messages"][1]["content"] == '{"content":"plain text"}'
    assert result.warnings[0].code == "wrapped_non_json_tool_response"


def test_swift_agent_export_applies_runtime_rollback_visibility() -> None:
    result = conversation_to_swift_agent_sample(
        conversation=[
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "QUESTION"},
            {
                "role": "assistant",
                "reasoning_content": "Bad attempt.",
                "tool_calls": [
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": {"query": "bad"}},
                    }
                ],
            },
            {
                "role": "ConversationRuntime",
                "name": "rollback",
                "content": '{"rollback_message_count":1,"reason":"tool_error"}',
            },
            {
                "role": "assistant",
                "reasoning_content": "Good attempt.",
                "tool_calls": [
                    {
                        "id": "call_good",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": {"query": "good"}},
                    }
                ],
            },
        ],
        tools=[],
    )

    validate_swift_agent_sample(result.sample)
    serialized = json.dumps(result.sample["messages"], ensure_ascii=False)
    assert "Bad attempt." not in serialized
    assert "Good attempt." in serialized
    assert any(warning.code == "runtime_rollback_applied" for warning in result.warnings)


def test_swift_agent_export_applies_runtime_compaction_visibility() -> None:
    result = conversation_to_swift_agent_sample(
        conversation=[
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "QUESTION"},
            {"role": "assistant", "content": "Old turn 1."},
            {"role": "assistant", "content": "Old turn 2."},
            {"role": "assistant", "content": "Recent turn."},
            {
                "role": "ConversationRuntime",
                "name": "compaction",
                "content": json.dumps({"summary": "COMPACTED SUMMARY", "prefix_len": 2, "compressed_count": 2}),
            },
            {"role": "assistant", "content": "Post-compaction turn."},
        ],
        tools=[],
    )

    validate_swift_agent_sample(result.sample)
    serialized = json.dumps(result.sample["messages"], ensure_ascii=False)
    assert "Old turn 1." not in serialized
    assert "Old turn 2." not in serialized
    assert "COMPACTED SUMMARY" in serialized
    assert "Recent turn." in serialized
    assert "Post-compaction turn." in serialized
    assert any(warning.code == "runtime_compaction_applied" for warning in result.warnings)
