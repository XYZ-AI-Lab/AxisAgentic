"""Append-only context compaction: marker semantics, replay, and trace reconstruction.

The contract under test: ``_full_conversation`` only ever grows. A context
compaction is recorded as a CONVERSATION_RUNTIME marker (``name="compaction"``)
whose payload describes the visible splice; the model-visible conversation is
derived by replaying markers, both live (runtime) and offline (persisted trace).
"""

import asyncio
import json
from typing import Any, cast

from agentic.config import ConversationConfig
from agentic.contracts import ConversationMessage, MessageRole, build_compaction_marker, parse_compaction_marker
from agentic.conversations import ConversationRuntime
from agentic.orchestration.task_orchestrator import (
    _conversation_dicts_to_messages,
    _materialize_visible_conversation,
)
from recipe.web_search.agent.context_compression_manager import (
    ContextCompressionConfig,
    ContextCompressionManager,
)
from recipe.web_search.agent.runtime import (
    TOOL_RESULT_OMITTED_PLACEHOLDER,
    WebSearchConversationConfig,
    WebSearchConversationRuntime,
)


def _estimate_tokens(messages: list[ConversationMessage]) -> int:
    return sum(len(message.content or "") for message in messages)


def _runtime() -> ConversationRuntime:
    return ConversationRuntime(
        config=ConversationConfig(system_prompt="system", user_prompt_template="Task: {task}"),
        tools=None,
        max_output_tokens=20,
        token_estimator=_estimate_tokens,
    )


def _compaction_marker(summary: str, prefix_len: int, compressed_count: int) -> ConversationMessage:
    return build_compaction_marker(summary=summary, prefix_len=prefix_len, compressed_count=compressed_count)


def _seed_conversation(runtime: ConversationRuntime, turns: int) -> None:
    runtime._append_messages([ConversationMessage.system("SYS"), ConversationMessage.user("TASK")])
    for i in range(turns):
        runtime._append_messages(
            [
                ConversationMessage.assistant(content=f"think-{i}"),
                ConversationMessage.tool(f"result-{i}", tool_call_id=f"c{i}", name="t"),
            ]
        )


class TestCompactionMarkerReplay:
    def test_full_conversation_is_append_only(self) -> None:
        runtime = _runtime()
        _seed_conversation(runtime, turns=6)
        full_len_before = len(runtime._full_conversation)

        runtime._append_messages([_compaction_marker("SUMMARY", prefix_len=2, compressed_count=8)])

        assert len(runtime._full_conversation) == full_len_before + 1
        contents = [m.content for m in runtime._full_conversation]
        for i in range(6):
            assert f"think-{i}" in contents
            assert f"result-{i}" in contents

    def test_visible_conversation_is_spliced(self) -> None:
        runtime = _runtime()
        _seed_conversation(runtime, turns=6)

        runtime._append_messages([_compaction_marker("SUMMARY", prefix_len=2, compressed_count=8)])

        visible = runtime._build_visible_conversation()
        assert [m.role for m in visible] == [
            MessageRole.SYSTEM,
            MessageRole.USER,
            MessageRole.ASSISTANT,  # summary
            MessageRole.ASSISTANT,
            MessageRole.TOOL,
            MessageRole.ASSISTANT,
            MessageRole.TOOL,
        ]
        assert visible[2].content == "SUMMARY"
        assert visible[3].content == "think-4"

    def test_rebuild_visible_context_matches_incremental(self) -> None:
        runtime = _runtime()
        _seed_conversation(runtime, turns=6)
        runtime._append_messages([_compaction_marker("SUMMARY", prefix_len=2, compressed_count=8)])
        live = runtime._build_visible_conversation()

        runtime._rebuild_visible_context()

        rebuilt = runtime._build_visible_conversation()
        assert [(m.role, m.content) for m in rebuilt] == [(m.role, m.content) for m in live]

    def test_messages_after_compaction_keep_appending(self) -> None:
        runtime = _runtime()
        _seed_conversation(runtime, turns=6)
        runtime._append_messages([_compaction_marker("SUMMARY", prefix_len=2, compressed_count=8)])

        runtime._append_messages([ConversationMessage.assistant(content="after")])

        assert runtime._build_visible_conversation()[-1].content == "after"
        assert runtime._full_conversation[-1].content == "after"

    def test_out_of_range_marker_is_noop_for_visible(self) -> None:
        runtime = _runtime()
        _seed_conversation(runtime, turns=2)
        visible_before = runtime._build_visible_conversation()

        runtime._append_messages([_compaction_marker("SUMMARY", prefix_len=2, compressed_count=99)])

        assert runtime._build_visible_conversation() == visible_before
        assert runtime._full_conversation[-1].name == "compaction"

    def test_rollback_after_compaction(self) -> None:
        runtime = _runtime()
        _seed_conversation(runtime, turns=6)
        runtime._append_messages([_compaction_marker("SUMMARY", prefix_len=2, compressed_count=8)])

        runtime._append_messages([runtime._build_rollback_marker(1, reason="tool_error")])

        visible = runtime._build_visible_conversation()
        assert visible[-1].content == "think-5"


class TestMarkerRobustness:
    def test_malformed_marker_is_tolerated_not_fatal(self) -> None:
        runtime = _runtime()
        _seed_conversation(runtime, turns=2)
        visible_before = runtime._build_visible_conversation()

        runtime._append_messages([ConversationMessage.runtime("{not json", name="compaction")])

        assert runtime._build_visible_conversation() == visible_before
        assert runtime._full_conversation[-1].name == "compaction"
        runtime._rebuild_visible_context()
        assert [m.content for m in runtime._build_visible_conversation()] == [m.content for m in visible_before]

    def test_bool_payload_is_rejected(self) -> None:
        marker = ConversationMessage.runtime(
            json.dumps({"summary": "S", "prefix_len": True, "compressed_count": 2}),
            name="compaction",
        )
        assert parse_compaction_marker(marker) is None

    def test_compaction_preserves_token_calibration(self) -> None:
        runtime = ConversationRuntime(
            config=ConversationConfig(system_prompt="system", user_prompt_template="Task: {task}"),
            tools=None,
            max_output_tokens=20,
            token_estimator=_estimate_tokens,
            token_estimator_is_additive=True,
        )
        _seed_conversation(runtime, turns=6)
        runtime._visible_context.calibrate(observed_tokens=300, raw_estimated_tokens=100)
        ratio_before = runtime._visible_context.calibration_ratio
        assert ratio_before != 1.0

        runtime._append_messages([_compaction_marker("SUMMARY", prefix_len=2, compressed_count=8)])

        assert runtime._visible_context.calibration_ratio == ratio_before


class TestUntruncatedCompressionInput:
    def test_untruncated_view_keeps_tool_content_that_prompt_rendering_omits(self) -> None:
        config = WebSearchConversationConfig(
            system_prompt="system",
            user_prompt_template="Task: {task}",
            keep_tool_result=1,
        )
        runtime = WebSearchConversationRuntime(
            config=config,
            tools=None,
            max_output_tokens=20,
            token_estimator=_estimate_tokens,
        )
        _seed_conversation(runtime, turns=3)

        rendered = runtime._build_visible_conversation()
        rendered_tool_contents = [m.content for m in rendered if m.role == MessageRole.TOOL]
        assert rendered_tool_contents == [TOOL_RESULT_OMITTED_PLACEHOLDER, TOOL_RESULT_OMITTED_PLACEHOLDER, "result-2"]

        untruncated = runtime.untruncated_visible_conversation()
        untruncated_tool_contents = [m.content for m in untruncated if m.role == MessageRole.TOOL]
        assert untruncated_tool_contents == ["result-0", "result-1", "result-2"]
        # Same shape: compaction splice indices computed on either view line up.
        assert len(untruncated) == len(rendered)


class TestTraceReconstruction:
    def test_offline_materialization_matches_live_visible(self) -> None:
        runtime = _runtime()
        _seed_conversation(runtime, turns=6)
        runtime._append_messages([_compaction_marker("SUMMARY", prefix_len=2, compressed_count=8)])
        runtime._append_messages([ConversationMessage.assistant(content="after")])
        live = runtime._build_visible_conversation()

        dicts = [m.to_model_message() for m in runtime._full_conversation]
        restored = _materialize_visible_conversation(_conversation_dicts_to_messages(dicts, include_runtime=True))

        assert [(m.role, m.content) for m in restored] == [(m.role, m.content) for m in live]


class _StubCompressionTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def call(self, *, previous_summary: str, task_context: str, new_context: str) -> dict[str, object]:
        self.calls.append({"previous_summary": previous_summary, "task_context": task_context, "new_context": new_context})
        return {"success": True, "summary": f"summary-{len(self.calls)}"}


def _make_manager(stub: _StubCompressionTool) -> ContextCompressionManager:
    manager = ContextCompressionManager(
        ContextCompressionConfig(enabled=True, interval=2, recent_window=1),
        base_url="http://unused",
        model_name="stub",
        api_key="stub",
    )
    cast("Any", manager).tool = stub
    return manager


class TestManagerProducesMarker:
    def test_marker_records_visible_splice(self) -> None:
        manager = _make_manager(_StubCompressionTool())
        runtime = _runtime()
        _seed_conversation(runtime, turns=4)

        marker, summary_text = asyncio.run(
            manager.maybe_compress(
                turn_count=4,
                visible_conversation=runtime._build_visible_conversation(),
                task_text="TASK",
            )
        )

        assert summary_text == "summary-1"
        assert marker is not None
        assert marker.role == MessageRole.CONVERSATION_RUNTIME
        assert marker.name == "compaction"
        assert marker.content is not None
        payload = json.loads(marker.content)
        # prefix = system + first user; recent_window=1 keeps the last 2 messages.
        assert payload["prefix_len"] == 2
        assert payload["compressed_count"] == 6
        assert "summary-1" in payload["summary"]

        runtime._append_messages([marker])
        visible = runtime._build_visible_conversation()
        assert len(visible) == 5  # system, user, summary, last assistant, last tool
        assert "summary-1" in (visible[2].content or "")

    def test_second_compaction_excludes_prior_summary_from_input(self) -> None:
        stub = _StubCompressionTool()
        manager = _make_manager(stub)
        runtime = _runtime()
        _seed_conversation(runtime, turns=4)
        marker, _ = asyncio.run(
            manager.maybe_compress(
                turn_count=4,
                visible_conversation=runtime._build_visible_conversation(),
                task_text="TASK",
            )
        )
        assert marker is not None
        runtime._append_messages([marker])
        runtime._append_messages(
            [
                ConversationMessage.assistant(content="think-late"),
                ConversationMessage.tool("result-late", tool_call_id="cl", name="t"),
            ]
        )

        second_marker, _ = asyncio.run(
            manager.maybe_compress(
                turn_count=6,
                visible_conversation=runtime._build_visible_conversation(),
                task_text="TASK",
            )
        )

        assert second_marker is not None
        assert second_marker.content is not None
        # Prior summary flows via previous_summary, not duplicated in new_context.
        assert "summary-1" in stub.calls[1]["previous_summary"]
        assert "summary-1" not in stub.calls[1]["new_context"]
        payload = json.loads(second_marker.content)
        # The prior summary message itself is still spliced out of the visible list.
        runtime._append_messages([second_marker])
        visible = runtime._build_visible_conversation()
        assert payload["prefix_len"] == 2
        assert all("summary-1" not in (m.content or "") for m in visible)
        assert "summary-2" in (visible[2].content or "")
