"""Append-only discard-all context management: marker semantics, replay, reconstruction.

The contract under test: ``_full_conversation`` only ever grows. A discard-all
reset is recorded as a CONVERSATION_RUNTIME marker (``name="discard_all"``) whose
payload records the retained prefix length; the model-visible conversation is
derived by replaying markers, both live (runtime) and offline (persisted trace,
SFT export). Tool-response content is never mutated, so SFT export stays
byte-identical to what the model saw.
"""

import asyncio
import json

import pytest
from pydantic import ValidationError

from agentic.config import ConversationConfig
from agentic.contracts import (
    ConversationMessage,
    ConversationStage,
    ConversationStepResult,
    FinalizationTrigger,
    MessageRole,
    StepAction,
    build_discard_all_marker,
    parse_discard_all_marker,
    parse_discard_all_marker_dict,
    splice_discard_all,
)
from agentic.contracts.markers import DiscardAllMarker
from agentic.conversations import ConversationRuntime
from agentic.model_clients.errors import ModelContextLimitError
from agentic.orchestration.task_orchestrator import (
    _conversation_dicts_to_messages,
    _materialize_visible_conversation,
)
from agentic.sft_export import conversation_to_swift_agent_sample
from recipe.web_search.agent.discard_all_manager import DiscardAllConfig, DiscardAllManager
from recipe.web_search.config import AgentConfig, WebSearchEvalConfig


def _estimate_tokens(messages: list[ConversationMessage]) -> int:
    return sum(len(message.content or "") for message in messages)


def _runtime() -> ConversationRuntime:
    return ConversationRuntime(
        config=ConversationConfig(system_prompt="system", user_prompt_template="Task: {task}"),
        tools=None,
        max_output_tokens=20,
        token_estimator=_estimate_tokens,
    )


def _discard_marker(prefix_len: int) -> ConversationMessage:
    return build_discard_all_marker(prefix_len=prefix_len)


def _seed_conversation(runtime: ConversationRuntime, turns: int) -> None:
    runtime._append_messages([ConversationMessage.system("SYS"), ConversationMessage.user("TASK")])
    for i in range(turns):
        runtime._append_messages(
            [
                ConversationMessage.assistant(content=f"think-{i}"),
                ConversationMessage.tool(f"result-{i}", tool_call_id=f"c{i}", name="t"),
            ]
        )


class TestDiscardAllMarkerContract:
    def test_build_parse_round_trip(self) -> None:
        marker = _discard_marker(prefix_len=2)
        assert marker.role == MessageRole.CONVERSATION_RUNTIME
        assert marker.name == "discard_all"
        parsed = parse_discard_all_marker(marker)
        assert parsed == DiscardAllMarker(prefix_len=2)

    def test_parse_dict_form(self) -> None:
        marker = _discard_marker(prefix_len=3)
        parsed = parse_discard_all_marker_dict(marker.to_model_message())
        assert parsed == DiscardAllMarker(prefix_len=3)

    def test_bool_payload_is_rejected(self) -> None:
        marker = ConversationMessage.runtime(json.dumps({"prefix_len": True}), name="discard_all")
        assert parse_discard_all_marker(marker) is None

    def test_negative_prefix_is_rejected(self) -> None:
        marker = ConversationMessage.runtime(json.dumps({"prefix_len": -1}), name="discard_all")
        assert parse_discard_all_marker(marker) is None

    def test_malformed_payload_is_rejected(self) -> None:
        assert parse_discard_all_marker(ConversationMessage.runtime("{not json", name="discard_all")) is None

    def test_compaction_marker_is_not_a_discard_marker(self) -> None:
        marker = ConversationMessage.runtime(json.dumps({"prefix_len": 2}), name="compaction")
        assert parse_discard_all_marker(marker) is None

    def test_splice_truncates_to_prefix(self) -> None:
        visible = [ConversationMessage.user(str(i)) for i in range(6)]
        spliced = splice_discard_all(visible, DiscardAllMarker(prefix_len=2))
        assert spliced is not None
        assert [m.content for m in spliced] == ["0", "1"]

    def test_splice_prefix_equal_len_is_noop(self) -> None:
        visible = [ConversationMessage.user(str(i)) for i in range(3)]
        spliced = splice_discard_all(visible, DiscardAllMarker(prefix_len=3))
        assert spliced is not None
        assert [m.content for m in spliced] == ["0", "1", "2"]

    def test_splice_out_of_range_returns_none(self) -> None:
        visible = [ConversationMessage.user(str(i)) for i in range(3)]
        assert splice_discard_all(visible, DiscardAllMarker(prefix_len=4)) is None


class TestDiscardAllMarkerReplay:
    def test_full_conversation_is_append_only(self) -> None:
        runtime = _runtime()
        _seed_conversation(runtime, turns=6)
        full_len_before = len(runtime._full_conversation)

        runtime._append_messages([_discard_marker(prefix_len=2)])

        assert len(runtime._full_conversation) == full_len_before + 1
        contents = [m.content for m in runtime._full_conversation]
        for i in range(6):
            assert f"think-{i}" in contents
            assert f"result-{i}" in contents

    def test_visible_conversation_is_truncated_to_prefix(self) -> None:
        runtime = _runtime()
        _seed_conversation(runtime, turns=6)

        runtime._append_messages([_discard_marker(prefix_len=2)])

        visible = runtime._build_visible_conversation()
        assert [(m.role, m.content) for m in visible] == [
            (MessageRole.SYSTEM, "SYS"),
            (MessageRole.USER, "TASK"),
        ]

    def test_messages_after_discard_keep_appending(self) -> None:
        runtime = _runtime()
        _seed_conversation(runtime, turns=6)
        runtime._append_messages([_discard_marker(prefix_len=2)])

        runtime._append_messages(
            [
                ConversationMessage.assistant(content="after-think"),
                ConversationMessage.tool("after-result", tool_call_id="cx", name="t"),
            ]
        )

        visible = runtime._build_visible_conversation()
        assert [m.content for m in visible] == ["SYS", "TASK", "after-think", "after-result"]
        assert runtime._full_conversation[-1].content == "after-result"

    def test_rebuild_visible_context_matches_incremental(self) -> None:
        runtime = _runtime()
        _seed_conversation(runtime, turns=6)
        runtime._append_messages([_discard_marker(prefix_len=2)])
        runtime._append_messages([ConversationMessage.assistant(content="after")])
        live = runtime._build_visible_conversation()

        runtime._rebuild_visible_context()

        rebuilt = runtime._build_visible_conversation()
        assert [(m.role, m.content) for m in rebuilt] == [(m.role, m.content) for m in live]

    def test_out_of_range_marker_is_noop_for_visible(self) -> None:
        runtime = _runtime()
        _seed_conversation(runtime, turns=2)
        visible_before = runtime._build_visible_conversation()

        runtime._append_messages([_discard_marker(prefix_len=99)])

        assert runtime._build_visible_conversation() == visible_before
        assert runtime._full_conversation[-1].name == "discard_all"

    def test_discard_preserves_token_calibration(self) -> None:
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

        runtime._append_messages([_discard_marker(prefix_len=2)])

        assert runtime._visible_context.calibration_ratio == ratio_before

    def test_second_discard_after_regrowth(self) -> None:
        runtime = _runtime()
        _seed_conversation(runtime, turns=4)
        runtime._append_messages([_discard_marker(prefix_len=2)])
        # New trajectory after the reset.
        runtime._append_messages(
            [
                ConversationMessage.assistant(content="new-think"),
                ConversationMessage.tool("new-result", tool_call_id="cn", name="t"),
            ]
        )
        runtime._append_messages([_discard_marker(prefix_len=2)])

        visible = runtime._build_visible_conversation()
        assert [m.content for m in visible] == ["SYS", "TASK"]


class TestTraceReconstruction:
    def test_offline_materialization_matches_live_visible(self) -> None:
        runtime = _runtime()
        _seed_conversation(runtime, turns=6)
        runtime._append_messages([_discard_marker(prefix_len=2)])
        runtime._append_messages(
            [
                ConversationMessage.assistant(content="after-think"),
                ConversationMessage.tool("after-result", tool_call_id="cx", name="t"),
            ]
        )
        live = runtime._build_visible_conversation()

        dicts = [m.to_model_message() for m in runtime._full_conversation]
        restored = _materialize_visible_conversation(_conversation_dicts_to_messages(dicts, include_runtime=True))

        assert [(m.role, m.content) for m in restored] == [(m.role, m.content) for m in live]


class TestSftExportParity:
    def test_sft_export_discards_the_recorded_range(self) -> None:
        conversation = [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "QUESTION"},
            {
                "role": "assistant",
                "content": "first turn",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": {"query": "first"}}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "web_search", "content": '{"result":"first-evidence"}'},
            {"role": "ConversationRuntime", "name": "discard_all", "content": json.dumps({"prefix_len": 2})},
            {
                "role": "assistant",
                "content": "second turn",
                "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "web_search", "arguments": {"query": "second"}}}],
            },
            {"role": "tool", "tool_call_id": "call_2", "name": "web_search", "content": '{"result":"second-evidence"}'},
            {"role": "assistant", "content": "FINAL"},
        ]
        tools = [{"type": "function", "function": {"name": "web_search", "description": "Search", "parameters": {"type": "object"}}}]

        result = conversation_to_swift_agent_sample(conversation=conversation, tools=tools)

        messages = result.sample["messages"]
        contents = [m["content"] for m in messages]
        # The pre-discard turn is gone; the post-discard turn and prefix remain.
        assert all("first turn" not in c for c in contents)
        tool_responses = [m["content"] for m in messages if m["role"] == "tool_response"]
        assert tool_responses == ['{"result":"second-evidence"}']
        assert any("second turn" in c for c in contents)
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "tool_call", "tool_response", "assistant"]
        assert any(w.code == "runtime_discard_all_applied" for w in result.warnings)


class TestDiscardAllManagerTrigger:
    def _manager(self, **overrides: object) -> DiscardAllManager:
        cfg = DiscardAllConfig(enabled=True, trigger_ratio=0.80, min_turns_between=3, max_tool_calls=1800)
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return DiscardAllManager(cfg)

    def test_fires_above_threshold(self) -> None:
        manager = self._manager()
        assert manager.should_trigger(observed_prompt_tokens=900, context_window=1000, turn_count=10, last_trigger_turn=0)

    def test_does_not_fire_at_exact_threshold(self) -> None:
        manager = self._manager()
        assert not manager.should_trigger(observed_prompt_tokens=800, context_window=1000, turn_count=10, last_trigger_turn=0)

    def test_does_not_fire_below_threshold(self) -> None:
        manager = self._manager()
        assert not manager.should_trigger(observed_prompt_tokens=799, context_window=1000, turn_count=10, last_trigger_turn=0)

    def test_min_turns_between_blocks_thrash(self) -> None:
        manager = self._manager(min_turns_between=3)
        assert not manager.should_trigger(observed_prompt_tokens=900, context_window=1000, turn_count=5, last_trigger_turn=3)
        assert manager.should_trigger(observed_prompt_tokens=900, context_window=1000, turn_count=6, last_trigger_turn=3)

    def test_first_discard_bypasses_cooldown(self) -> None:
        # No prior reset (last_trigger_turn=None): the first reset must be allowed
        # immediately, even on turn 1, regardless of min_turns_between.
        manager = self._manager(min_turns_between=3)
        assert manager.should_trigger(observed_prompt_tokens=900, context_window=1000, turn_count=1, last_trigger_turn=None)
        assert manager.should_trigger(observed_prompt_tokens=900, context_window=1000, turn_count=2, last_trigger_turn=None)

    def test_disabled_never_fires(self) -> None:
        manager = self._manager(enabled=False)
        assert not manager.should_trigger(observed_prompt_tokens=999, context_window=1000, turn_count=10, last_trigger_turn=0)

    def test_guards_missing_signals(self) -> None:
        manager = self._manager()
        assert not manager.should_trigger(observed_prompt_tokens=None, context_window=1000, turn_count=10, last_trigger_turn=0)
        assert not manager.should_trigger(observed_prompt_tokens=900, context_window=None, turn_count=10, last_trigger_turn=0)
        assert not manager.should_trigger(observed_prompt_tokens=0, context_window=1000, turn_count=10, last_trigger_turn=0)
        assert not manager.should_trigger(observed_prompt_tokens=900, context_window=0, turn_count=10, last_trigger_turn=0)

    def test_build_marker_keeps_system_and_first_user(self) -> None:
        manager = self._manager()
        visible = [
            ConversationMessage.system("S1"),
            ConversationMessage.system("S2"),
            ConversationMessage.user("TASK"),
            ConversationMessage.assistant(content="a"),
            ConversationMessage.tool("r", tool_call_id="c", name="t"),
        ]
        marker = manager.build_marker(visible)
        assert marker is not None
        assert parse_discard_all_marker(marker) == DiscardAllMarker(prefix_len=3)

    def test_build_marker_none_when_nothing_to_discard(self) -> None:
        manager = self._manager()
        visible = [ConversationMessage.system("S"), ConversationMessage.user("TASK")]
        assert manager.build_marker(visible) is None

    def test_invalid_config_rejected(self) -> None:
        with pytest.raises(ValueError, match="trigger_ratio"):
            DiscardAllManager(DiscardAllConfig(enabled=True, trigger_ratio=0.0))
        with pytest.raises(ValueError, match="max_tool_calls"):
            DiscardAllManager(DiscardAllConfig(enabled=True, max_tool_calls=0))


class TestConfigMutualExclusion:
    def test_both_enabled_raises(self) -> None:
        with pytest.raises(ValidationError, match="mutually exclusive"):
            AgentConfig(
                discard_all={"enabled": True},
                context_compression={"enabled": True},
            )

    def test_only_discard_all_is_allowed(self) -> None:
        cfg = AgentConfig(discard_all={"enabled": True})
        assert cfg.discard_all.enabled
        assert not cfg.context_compression.enabled

    def test_only_compression_is_allowed(self) -> None:
        cfg = AgentConfig(context_compression={"enabled": True})
        assert cfg.context_compression.enabled
        assert not cfg.discard_all.enabled

    def test_discard_all_rejects_estimated_detection(self) -> None:
        with pytest.raises(ValidationError, match="provider_error"):
            WebSearchEvalConfig.model_validate(
                {
                    "agent": {"discard_all": {"enabled": True}},
                    "model": {"context": {"limit_detection": "estimated"}},
                }
            )

    def test_discard_all_allows_provider_error_detection(self) -> None:
        cfg = WebSearchEvalConfig.model_validate(
            {
                "agent": {"discard_all": {"enabled": True}},
                "model": {"context": {"limit_detection": "provider_error"}},
            }
        )
        assert cfg.agent.discard_all.enabled
        assert cfg.model.context.limit_detection == "provider_error"


class TestCliMutualExclusion:
    _REQUIRED = ["--base_url", "http://x", "--data_path", "x.jsonl", "--disable_tools"]

    def _parse(self, extra: list[str]):
        from recipe.web_search.runners.evaluate_benchmark import parse_args

        return parse_args([*self._REQUIRED, *extra])

    def test_cli_rejects_both_strategies(self) -> None:
        from recipe.web_search.runners.evaluate_benchmark import build_orchestrator

        args = self._parse(
            [
                "--context_limit_detection",
                "provider_error",
                "--context_compression_enabled",
                "true",
                "--context_compression_llm_base_url",
                "http://compression",
                "--context_compression_llm_model_name",
                "compression-model",
                "--discard_all_enabled",
                "true",
            ]
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_orchestrator(object(), None, args)

    def test_cli_rejects_discard_all_with_estimated_detection(self) -> None:
        from recipe.web_search.runners.evaluate_benchmark import build_orchestrator

        args = self._parse(["--context_limit_detection", "estimated", "--discard_all_enabled", "true"])
        with pytest.raises(ValueError, match="estimated"):
            build_orchestrator(object(), None, args)


class TestToolManagerCapWiring:
    _REQUIRED = ["--base_url", "http://x", "--data_path", "x.jsonl", "--disable_tools", "--context_limit_detection", "provider_error"]

    def _build(self, extra: list[str]):
        from recipe.web_search.runners.evaluate_benchmark import build_orchestrator, parse_args

        args = parse_args([*self._REQUIRED, *extra])
        return build_orchestrator(object(), None, args)

    def test_cap_not_set_on_tool_manager_when_enabled(self) -> None:
        orchestrator = self._build(["--discard_all_enabled", "true", "--discard_all_max_tool_calls", "1234"])
        assert orchestrator.tool_manager._config.max_tool_calls_per_task is None
        assert orchestrator._discard_all_manager is not None
        assert orchestrator._discard_all_manager.max_tool_calls == 1234

    def test_no_cap_when_disabled(self) -> None:
        orchestrator = self._build([])
        assert orchestrator.tool_manager._config.max_tool_calls_per_task is None

    def test_max_turns_superseded_by_cap(self) -> None:
        orchestrator = self._build(["--max_turns", "300", "--discard_all_enabled", "true", "--discard_all_max_tool_calls", "1234"])
        assert orchestrator.config.conversation.max_turns == 1234
        assert orchestrator._discard_all_last_attempt_max_turns == 300


class TestDiscardAllOrchestratorFlow:
    _REQUIRED = ["--base_url", "http://x", "--data_path", "x.jsonl", "--disable_tools", "--context_limit_detection", "provider_error"]

    def _build(self, extra: list[str]):
        from recipe.web_search.runners.evaluate_benchmark import build_orchestrator, parse_args

        args = parse_args([*self._REQUIRED, *extra])
        return build_orchestrator(object(), None, args)

    def _step_result(self, runtime: ConversationRuntime) -> ConversationStepResult:
        return ConversationStepResult(
            stage=runtime.stage,
            action=StepAction.CALL_MODEL,
            visible_conversation=runtime._build_visible_conversation(),
            full_conversation=list(runtime._full_conversation),
        )

    def _runtime_with_discardable_context(self) -> ConversationRuntime:
        runtime = _runtime()
        runtime.initialize_conversation("question")
        runtime._append_messages(
            [
                ConversationMessage.assistant(content="searching"),
                ConversationMessage.tool("evidence", tool_call_id="c1", name="web_search"),
            ]
        )
        runtime._assistant_turn_count = 7
        return runtime

    def test_last_attempt_starts_at_tool_budget_and_extends_from_current_turn(self) -> None:
        orchestrator = self._build(["--max_turns", "20", "--discard_all_enabled", "true", "--discard_all_max_tool_calls", "5"])
        runtime = self._runtime_with_discardable_context()
        runtime.max_turns = 5
        orchestrator.tool_manager._task_total_calls = 5

        info = orchestrator._maybe_enter_discard_all_last_attempt(
            runtime=runtime,
            task_id="task",
            turn_count=7,
            reason="test",
        )

        assert info is not None
        assert orchestrator._discard_all_last_attempt_mode
        assert runtime.max_turns == 27
        assert info["runtime_max_turns_before"] == 5
        assert info["runtime_max_turns_after"] == 27
        assert orchestrator._sync_loop_limits_for_discard_all_last_attempt(
            runtime=runtime,
            max_turns=5,
            max_attempts=205,
        ) == (27, 227)

    def test_provider_context_error_discards_when_budget_remains(self) -> None:
        orchestrator = self._build(["--max_turns", "20", "--discard_all_enabled", "true", "--discard_all_max_tool_calls", "5"])
        runtime = self._runtime_with_discardable_context()
        orchestrator.tool_manager._task_total_calls = 4

        result, reward, done, reason, info = orchestrator._handle_web_search_context_limit_error(
            runtime=runtime,
            step_result=self._step_result(runtime),
            task_id="task",
            turn_idx=8,
            error=ModelContextLimitError("context too long", context_window=100),
        )

        assert reward == 0.0
        assert not done
        assert reason is None
        assert result.action == StepAction.CALL_MODEL
        assert result.stage == ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT
        assert info["discarded_context_limit_error_with_discard_all"] is True
        assert info["discard_all"]["reason"] == "provider_context_limit_error"
        assert runtime._full_conversation[-1].name == "discard_all"
        assert [message.content for message in result.visible_conversation] == ["system", "Task: question"]

    def test_discard_preempts_force_final_before_last_attempt(self) -> None:
        orchestrator = self._build(["--max_turns", "20", "--discard_all_enabled", "true", "--discard_all_max_tool_calls", "5"])
        runtime = self._runtime_with_discardable_context()
        runtime.stage = ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT
        runtime._force_final_trigger = FinalizationTrigger.TURN_LIMIT
        runtime._force_final_has_prompt = True
        orchestrator.tool_manager._task_total_calls = 4

        info = asyncio.run(
            orchestrator._maybe_apply_discard_all(
                runtime=runtime,
                task_id="task",
                turn_count=7,
                done=False,
            )
        )

        assert info is not None
        assert info["reason"] == "preempt_turn_limit"
        assert runtime.stage == ConversationStage.INITIAL_USER_MESSAGE_AWAIT_ASSISTANT
        assert runtime._force_final_trigger is None
        assert runtime._full_conversation[-1].name == "discard_all"

    def test_provider_context_error_uses_last_attempt_after_budget(self) -> None:
        orchestrator = self._build(["--max_turns", "20", "--discard_all_enabled", "true", "--discard_all_max_tool_calls", "5"])
        runtime = self._runtime_with_discardable_context()
        runtime.max_turns = 5
        orchestrator.tool_manager._task_total_calls = 5

        result, reward, done, reason, info = orchestrator._handle_web_search_context_limit_error(
            runtime=runtime,
            step_result=self._step_result(runtime),
            task_id="task",
            turn_idx=8,
            error=ModelContextLimitError("context too long", context_window=100),
        )

        assert result.stage == runtime.stage
        assert reward == 0.0
        assert done
        assert reason == "terminated_context_limit"
        assert "discard_all" not in info
        assert info["discard_all_last_attempt"]["runtime_max_turns_after"] == 27
        assert runtime._full_conversation[-1].name != "discard_all"
