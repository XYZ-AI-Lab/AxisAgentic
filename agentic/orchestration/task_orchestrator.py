# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agentic.contracts import (
    ConversationMessage,
    ConversationStage,
    ConversationStepResult,
    FinalizationTrigger,
    MessageRole,
    StepAction,
    TokenUsage,
    ToolRequest,
)
from agentic.contracts.markers import (
    parse_compaction_marker,
    parse_discard_all_marker,
    splice_compaction,
    splice_discard_all,
)
from agentic.contracts.messages import ToolCall, ToolCallSpec, ToolResultStatus
from agentic.conversations.conversation_runtime import ConversationRuntime
from agentic.model_clients.errors import ModelContextLimitError
from agentic.rewards import RewardContext, RewardEvaluator, ToolCallRewardEvaluator

if TYPE_CHECKING:
    from agentic.config import OrchestrationConfig
    from agentic.model_clients.base import ModelClient
    from agentic.observability.task_logger import TaskLogger, TaskTrace
    from agentic.tools.manager import ToolManager

logger = logging.getLogger(__name__)

_ROLE_MAP: dict[str, MessageRole] = {r.value: r for r in MessageRole}


def _summarize_samples(values: list[float]) -> dict[str, float]:
    """Return generic avg/sum/min/max/p50/p95 stats for a numeric sample."""
    if not values:
        return {"avg": 0.0, "sum": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0}
    sorted_v = sorted(values)
    n = len(sorted_v)

    def _percentile(q: float) -> float:
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return float(sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac)

    total = float(sum(sorted_v))
    return {
        "avg": total / n,
        "sum": total,
        "min": float(sorted_v[0]),
        "max": float(sorted_v[-1]),
        "p50": _percentile(0.5),
        "p95": _percentile(0.95),
    }


def _summarize_latency_ms(latencies: list[float]) -> dict[str, float]:
    """Return latency summary stats in milliseconds, with zero defaults for empty samples."""
    if not latencies:
        return {
            "avg_latency_ms": 0.0,
            "sum_latency_ms": 0.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    def _percentile(q: float) -> float:
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return float(sorted_lat[lo] * (1 - frac) + sorted_lat[hi] * frac)

    total = float(sum(sorted_lat))
    return {
        "avg_latency_ms": total / n,
        "sum_latency_ms": total,
        "min_latency_ms": float(sorted_lat[0]),
        "max_latency_ms": float(sorted_lat[-1]),
        "p50_latency_ms": _percentile(0.5),
        "p95_latency_ms": _percentile(0.95),
    }


def _conversation_dicts_to_messages(dicts: list[dict[str, Any]], *, include_runtime: bool = False) -> list[ConversationMessage]:
    """Best-effort reconstruction of :class:`ConversationMessage` objects from trace dict format.

    Handles the ``role``, ``content``, ``reasoning_content``, ``visible_thought``,
    ``tool_calls`` (with JSON-string arguments), ``name``, and ``tool_call_id``
    fields produced by :meth:`ConversationMessage.to_model_message`.

    ``include_runtime=False`` is for normal model-facing transcripts: internal
    runtime messages are skipped because they are never sent to the model.
    ``include_runtime=True`` is used by :meth:`OrchestrationResult.from_trace`
    while loading persisted traces, where rollback markers must be replayed to
    reconstruct the exact visible conversation.
    """
    messages: list[ConversationMessage] = []
    for d in dicts:
        role_str = d.get("role", "")
        if role_str == MessageRole.CONVERSATION_RUNTIME.value and not include_runtime:
            continue
        role = _ROLE_MAP.get(role_str, MessageRole.USER)

        tool_calls: list[ToolCall] | None = None
        raw_tcs = d.get("tool_calls")
        if raw_tcs:
            tool_calls = []
            for tc in raw_tcs:
                fn = tc.get("function", {})
                raw_args = fn.get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except (json.JSONDecodeError, ValueError):
                        raw_args = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        type=tc.get("type", "function"),
                        function=ToolCallSpec(name=fn.get("name", ""), arguments=raw_args),
                    )
                )

        messages.append(
            ConversationMessage(
                role=role,
                content=d.get("content"),
                reasoning_content=d.get("reasoning_content"),
                visible_thought=d.get("visible_thought"),
                tool_calls=tool_calls,
                name=d.get("name"),
                tool_call_id=d.get("tool_call_id"),
            )
        )
    return messages


def _parse_rollback_marker(message: ConversationMessage) -> tuple[int, str | None]:
    if message.role != MessageRole.CONVERSATION_RUNTIME or message.name != "rollback" or message.content is None:
        return 0, None
    try:
        payload = json.loads(message.content)
    except (TypeError, json.JSONDecodeError):
        return 0, None
    if not isinstance(payload, dict):
        return 0, None
    rollback_count = payload.get("rollback_message_count", 0)
    reason = payload.get("reason")
    count = rollback_count if isinstance(rollback_count, int) and rollback_count > 0 else 0
    return count, reason if isinstance(reason, str) else None


def _materialize_visible_conversation(messages: list[ConversationMessage]) -> list[ConversationMessage]:
    """Apply runtime rollback and compaction markers to reconstruct the model-visible conversation.

    Marker parsing and splicing go through ``agentic.contracts.markers`` — the
    same helpers the live runtime replay uses — so a persisted append-only trace
    replays to the same compacted visible conversation the model actually saw.
    """
    visible_conversation: list[ConversationMessage] = []
    for message in messages:
        rollback_message_count, _reason = _parse_rollback_marker(message)
        if rollback_message_count:
            while rollback_message_count > 0 and visible_conversation:
                if visible_conversation[-1].role == MessageRole.SYSTEM:
                    break
                visible_conversation.pop()
                rollback_message_count -= 1
            continue
        compaction = parse_compaction_marker(message)
        if compaction is not None:
            spliced = splice_compaction(visible_conversation, compaction)
            if spliced is not None:
                visible_conversation = spliced
            continue
        discard = parse_discard_all_marker(message)
        if discard is not None:
            truncated = splice_discard_all(visible_conversation, discard)
            if truncated is not None:
                visible_conversation = truncated
            continue
        if message.role != MessageRole.CONVERSATION_RUNTIME:
            visible_conversation.append(message)
    return visible_conversation


def _extract_last_assistant_content(messages: list[ConversationMessage]) -> str:
    for message in reversed(messages):
        if message.role == MessageRole.ASSISTANT and message.content:
            return message.content
    return ""


def _serialize_tool_results_as_user(messages: list[ConversationMessage]) -> list[ConversationMessage]:
    """Return a model-facing copy with tool-result messages rendered as user messages."""
    rendered: list[ConversationMessage] = []
    for message in messages:
        if message.role == MessageRole.TOOL:
            rendered.append(ConversationMessage.user(message.content or ""))
        else:
            rendered.append(message)
    return rendered


class OrchestrationResult(BaseModel):
    output: Any
    reason: str
    conversation: list[ConversationMessage] = Field(default_factory=list)
    visible_conversation: list[ConversationMessage] = Field(default_factory=list)
    num_turns: int = 0
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_trace(cls, trace: TaskTrace, *, reconstruct_conversation: bool = True) -> OrchestrationResult:
        """Reconstruct an :class:`OrchestrationResult` from a persisted :class:`TaskTrace`.

        Useful for resuming evaluation runs — callers can load a completed trace
        and obtain a result object without re-running inference.

        Args:
            trace: A loaded TaskTrace (from :meth:`TaskLogger.load_trace` or
                :meth:`TaskTrace.from_dict`).
            reconstruct_conversation: If ``True``, parse conversation dicts back
                into :class:`ConversationMessage` objects. If ``False``, leave
                conversation and visible_conversation empty.
        """
        metadata = dict(trace.metadata)
        conversation: list[ConversationMessage] = []
        visible_conversation: list[ConversationMessage] = []
        if reconstruct_conversation and trace.conversation:
            conversation = _conversation_dicts_to_messages(trace.conversation, include_runtime=True)
            visible_conversation = _materialize_visible_conversation(conversation)

        output = metadata.get("output")
        if output is None and visible_conversation:
            output = _extract_last_assistant_content(visible_conversation)
        if output is None and conversation:
            output = _extract_last_assistant_content(conversation)

        run_info = metadata.get("run_info", {})
        return cls(
            output=output,
            reason=trace.status,
            conversation=conversation,
            visible_conversation=visible_conversation,
            num_turns=metadata.get("turn_used", 0),
            reward=run_info.get("total_reward", 0.0) if isinstance(run_info, dict) else 0.0,
            done=True,
            info=run_info if isinstance(run_info, dict) else {},
            metadata=metadata,
        )


class TaskOrchestrator:
    def __init__(
        self,
        *,
        config: OrchestrationConfig,
        model_client: ModelClient,
        tool_manager: ToolManager | None = None,
        task_logger: TaskLogger | None = None,
        reward_evaluator: RewardEvaluator | None = None,
        conversation_runtime_class: type[ConversationRuntime] = ConversationRuntime,
        level: int = 0,
    ) -> None:
        if tool_manager is None:
            from agentic.tools.manager import ToolManager as RuntimeToolManager

            tool_manager = RuntimeToolManager(config=config.tool_manager, task_logger=task_logger, level=level + 1)
        self.config = config
        self.model_client = model_client
        self.task_logger = task_logger
        self.level = level
        self.tool_path: list[str] = []
        self.tool_manager = tool_manager
        self.tool_manager.set_task_logger(task_logger)
        self.tool_manager.set_level(level + 1)
        self._conversation_runtime_class = conversation_runtime_class
        self._run_config_saved = False
        if reward_evaluator is not None:
            self.reward_evaluator = reward_evaluator
        else:
            from agentic.config import RewardConfig

            self.reward_evaluator = ToolCallRewardEvaluator(**RewardConfig().model_dump())

    def _model_visible_conversation(self, messages: list[ConversationMessage]) -> list[ConversationMessage]:
        if self.config.conversation.tool_result_role == "user":
            return _serialize_tool_results_as_user(messages)
        return messages

    def set_hierarchy_context(self, *, level: int, tool_path: list[str]) -> None:
        """Update the nesting level and tool path for this orchestrator and its tool manager."""
        self.level = level
        self.tool_path = list(tool_path)
        self.tool_manager.set_tool_path(tool_path)
        self.tool_manager.set_level(level + 1)

    async def run(
        self,
        task: str | dict[str, Any],
        task_id: str | None = None,
        *,
        extra_trace_metadata: dict[str, Any] | None = None,
    ) -> OrchestrationResult:
        # Initialize task running
        task_id = task_id or uuid4().hex
        normalized_task = self.normalize_task(task)
        self._maybe_save_run_config()
        self._start_trace(task_id, normalized_task, extra_metadata=extra_trace_metadata)
        self.tool_manager.reset_task_state()
        await self.tool_manager.begin_task(task_id=task_id)
        try:
            tools = self.tool_manager.list_tool_definitions()
            conv_runtime = self.build_conversation_runtime(tools=tools)

            # Initialize conversation runtime
            turn_count = 0
            t0 = time.perf_counter()
            step_result = conv_runtime.initialize_conversation(normalized_task)
            runtime_elapsed_ms = (time.perf_counter() - t0) * 1000
            self._log_runtime_step(task_id, turn_count, step_result, elapsed_ms=runtime_elapsed_ms)
            self._sync_trace(task_id, step_result.appended_messages)

            # Iterative turn loop
            done = False
            reason = None
            step_reward = 0.0
            total_reward = 0.0
            # Safety buffer: the orchestrator's turn_count can exceed max_turns when
            # _run_turn() returns without advancing the runtime's internal turn counter
            # (e.g. empty-response retries or rollbacks that undo the assistant turn).
            # The extra margin prevents the loop from running unbounded while still
            # allowing the ConversationRuntime to reach its own turn limit naturally.
            _LOOP_SAFETY_MARGIN = 200
            max_loop_iterations = (self.config.conversation.max_turns or 10_000) + _LOOP_SAFETY_MARGIN
            run_info: dict[str, Any] = self._update_run_info({}, dict(step_result.info))
            while not done:
                turn_count += 1
                step_result, step_reward, done, reason, step_info = await self._run_turn(
                    runtime=conv_runtime,
                    tools=tools,
                    step_result=step_result,
                    task_id=task_id,
                    turn_idx=turn_count,
                )
                total_reward += step_reward
                run_info = self._update_run_info(run_info, step_info)
                if turn_count >= max_loop_iterations and not done:
                    self._log_step(task_id, turn_count, "orchestrator.error", f"Safety limit reached ({turn_count} iterations)", emoji="❌")
                    reason = "terminated_loop_safety_limit"
                    done = True
            if reason is None:
                msg = "Orchestration loop ended without a termination reason."
                raise RuntimeError(msg)

            runtime_timing = run_info.setdefault(
                "inference_timing",
                {"model_client_elapsed_ms_sum": 0.0, "engine_e2e_elapsed_s_sum": 0.0, "num_inference_calls": 0},
            )
            runtime_timing["tokenizer_elapsed_ms_sum"] = float(conv_runtime.tokenizer_elapsed_ms)
            runtime_timing["runtime_step_elapsed_ms_sum"] = float(conv_runtime.runtime_step_elapsed_ms)

            # Formulate final solution
            output = self.extract_final_output(step_result.visible_conversation)
            metadata = {
                "task_id": task_id,
                "orchestrator_name": self.config.name,
                "output": output,
                "max_turns": self.config.conversation.max_turns,
                "tool_names": self.tool_manager.list_tool_names(),
                "tools": tools,
                "turn_used": turn_count,
                "done_reason": reason,
                "run_info": run_info,
                "unknown_tool_names": self.tool_manager.unknown_tool_names_snapshot(),
            }
            if self.task_logger is not None:
                task_logger_timing = self.task_logger.finish_task(task_id, status=reason, metadata=metadata, level=self.level)
                metadata.update(task_logger_timing)
                run_info["task_logger_timing"] = task_logger_timing
            return OrchestrationResult(
                output=output,
                reason=reason,
                conversation=step_result.full_conversation,
                visible_conversation=step_result.visible_conversation,
                num_turns=turn_count,
                reward=total_reward,
                done=done,
                info=run_info,
                metadata=metadata,
            )
        finally:
            await self.tool_manager.end_task(task_id=task_id)

    async def _run_turn(
        self,
        *,
        runtime: ConversationRuntime,
        tools: list[dict[str, Any]] | None,
        step_result: ConversationStepResult,
        task_id: str,
        turn_idx: int,
    ) -> tuple[ConversationStepResult, float, bool, str | None, dict[str, Any]]:
        # Call model_client to generate response
        # NOTE: To keep tracking the same conversation -- inference sequence cache hit,
        # `tools` always be passed here no matter it is force final generation or not.
        try:
            t0 = time.perf_counter()
            model_response = await self.model_client.acomplete(
                messages=self._model_visible_conversation(step_result.visible_conversation),
                tools=tools,
                tool_choice="auto" if tools else None,
            )
            model_elapsed_ms = (time.perf_counter() - t0) * 1000
        except (ValueError, RuntimeError) as exc:
            return self._handle_model_error(runtime=runtime, tools=tools, step_result=step_result, task_id=task_id, turn_idx=turn_idx, error=exc)
        model_extra: dict[str, Any] | None = None
        raw = getattr(model_response, "raw_response", None)
        if isinstance(raw, dict):
            logged_elapsed = raw.get("client_elapsed_ms_excluding_request_logging")
            if isinstance(logged_elapsed, int | float):
                model_elapsed_ms = float(logged_elapsed)
            # ``RolloutModelClient`` stashes its ``GenerationResult.extra`` here —
            # currently carries SGLang-internal ``e2e_elapsed_seconds``, ``cached_tokens``, ``retry`` and ``stop_reason``.
            # Surface these so dashboards can separate SGLang-internal queue/decode time from controller wall-clock.
            model_extra = {k: raw[k] for k in ("e2e_elapsed_seconds", "cached_tokens", "retry", "stop_reason") if k in raw}
        self._log_token_usage(task_id, turn_idx, model_response.usage, elapsed_ms=model_elapsed_ms, extra=model_extra)
        if model_response.usage is not None:
            runtime.record_observed_prompt_tokens(
                step_result.visible_conversation,
                input_tokens=model_response.usage.input_tokens,
            )

        inference_timing: dict[str, float] = {"model_client_elapsed_ms": model_elapsed_ms}
        if model_extra and "e2e_elapsed_seconds" in model_extra:
            inference_timing["engine_e2e_elapsed_s"] = float(model_extra["e2e_elapsed_seconds"])

        # Apply assistant message and dispatch on the returned action.
        t0 = time.perf_counter()
        step_result = runtime.apply_assistant_message(model_response.message, finish_reason=model_response.finish_reason)
        runtime_elapsed_ms = (time.perf_counter() - t0) * 1000
        self._sync_trace(task_id, step_result.appended_messages)
        self._log_runtime_step(task_id, turn_idx, step_result, elapsed_ms=runtime_elapsed_ms)

        if step_result.action == StepAction.EXECUTE_TOOLS:
            return await self._run_tool_phase(
                runtime=runtime,
                step_result=step_result,
                task_id=task_id,
                turn_idx=turn_idx,
                assistant_message=model_response.message,
                usage=model_response.usage,
                inference_timing=inference_timing,
            )

        if step_result.action == StepAction.DONE:
            done_reason = self._done_reason_for_stage(step_result.stage, info=dict(step_result.info))
            info, reward = self._build_step_info(
                step_result=step_result,
                assistant_message=model_response.message,
                tool_outcomes=[],
                usage=model_response.usage,
                inference_timing=inference_timing,
            )
            return step_result, reward, True, done_reason, info

        # StepAction.CALL_MODEL — continue the loop.
        info, reward = self._build_step_info(
            step_result=step_result,
            assistant_message=model_response.message,
            tool_outcomes=[],
            usage=model_response.usage,
            inference_timing=inference_timing,
        )
        return step_result, reward, False, None, info

    async def _run_tool_phase(
        self,
        *,
        runtime: ConversationRuntime,
        step_result: ConversationStepResult,
        task_id: str,
        turn_idx: int,
        assistant_message: ConversationMessage,
        usage: TokenUsage | None = None,
        inference_timing: dict[str, float] | None = None,
    ) -> tuple[ConversationStepResult, float, bool, str | None, dict[str, Any]]:
        tool_requests = self._build_tool_requests(assistant_message)

        t0 = time.perf_counter()
        tool_outcomes = await self.tool_manager.execute_tool_calls(tool_requests, task_id=task_id)
        tool_elapsed_ms = (time.perf_counter() - t0) * 1000

        self._log_step(
            task_id,
            turn_idx,
            "tool.execution",
            f"processed {len(tool_outcomes)} tool call(s)",
            metadata={"tool_names": [outcome.request.tool_name for outcome in tool_outcomes]},
            emoji="⚒️",
            elapsed_ms=tool_elapsed_ms,
        )
        tool_messages = [outcome.message for outcome in tool_outcomes]
        rejected_call_ids = frozenset(outcome.request.call_id for outcome in tool_outcomes if outcome.result.status == ToolResultStatus.REJECTED)
        tools_exhausted = self.tool_manager.all_tools_exhausted()
        t0 = time.perf_counter()
        step_result = runtime.apply_batch_tool_messages(tool_messages, tools_exhausted=tools_exhausted, rejected_call_ids=rejected_call_ids)
        runtime_elapsed_ms = (time.perf_counter() - t0) * 1000
        self._sync_trace(task_id, step_result.appended_messages)
        self._log_runtime_step(task_id, turn_idx, step_result, elapsed_ms=runtime_elapsed_ms)
        info, reward = self._build_step_info(
            step_result=step_result,
            assistant_message=assistant_message,
            tool_outcomes=tool_outcomes,
            usage=usage,
            inference_timing=inference_timing,
        )

        return step_result, reward, False, None, info

    def build_conversation_runtime(self, tools: list[dict[str, Any]] | None) -> ConversationRuntime:
        return self._conversation_runtime_class(
            config=self.config.conversation,
            tools=tools,
            max_output_tokens=self.model_client.max_output_tokens,
            token_estimator=self.model_client.estimate_tokens,
        )

    def _build_tool_requests(self, assistant_message: ConversationMessage) -> list[ToolRequest]:
        if assistant_message.role != MessageRole.ASSISTANT or not assistant_message.tool_calls:
            return []
        requests = [
            ToolRequest(
                tool_name=tool_call.function.name,
                arguments=tool_call.function.arguments,
                call_id=tool_call.id,
            )
            for tool_call in assistant_message.tool_calls
        ]
        return [self.tool_manager.repair_tool_request(request) for request in requests]

    def normalize_task(self, task: str | dict[str, Any]) -> str:
        if isinstance(task, str):
            return task
        if not self.config.allow_structured_task_input:
            msg = f"Orchestrator '{self.config.name}' does not accept structured task input."
            raise TypeError(msg)
        pairs = [f"{key}: {value}" for key, value in task.items()]
        return "\n".join(pairs)

    def extract_final_output(self, conversation: list[ConversationMessage]) -> Any:
        for message in reversed(conversation):
            if message.role == MessageRole.ASSISTANT and message.content:
                return message.content
        return ""

    def _log_runtime_step(self, task_id: str, turn_idx: int, step_result: ConversationStepResult, *, elapsed_ms: float | None = None) -> None:
        """Log a conversation runtime step using its stage path and action."""
        stage_path = step_result.info.get("stage_path", [])
        message = " -> ".join(stage_path) if stage_path else step_result.stage.value
        self._log_step(
            task_id,
            turn_idx,
            "runtime.step",
            message,
            metadata=dict(step_result.info),
            emoji=step_result.info.get("emoji", "") or None,
            elapsed_ms=elapsed_ms,
        )

    def _evaluate_reward(
        self,
        *,
        step_result: ConversationStepResult,
        assistant_message: ConversationMessage | None,
        tool_outcomes: list[Any],
    ) -> float:
        return self.reward_evaluator.evaluate(
            RewardContext(
                conversation=step_result.full_conversation,
                assistant_message=assistant_message,
                tool_outcomes=tool_outcomes,
                stage=step_result.stage,
                info=step_result.info,
            )
        )

    def _handle_model_error(
        self,
        *,
        runtime: ConversationRuntime,
        tools: list[dict[str, Any]] | None,
        step_result: ConversationStepResult,
        task_id: str,
        turn_idx: int,
        error: Exception,
    ) -> tuple[ConversationStepResult, float, bool, str | None, dict[str, Any]]:
        """Handle an error raised by the model client by terminating the turn."""
        del tools
        if isinstance(error, ModelContextLimitError):
            return self._handle_model_context_limit_error(
                runtime=runtime,
                task_id=task_id,
                turn_idx=turn_idx,
                error=error,
            )
        reason = f"model_error: {error}"
        self._log_step(task_id, turn_idx, "orchestrator.error", reason, emoji="❌")
        info: dict[str, Any] = {"model_error": str(error)}
        return step_result, 0.0, True, reason, info

    def _handle_model_context_limit_error(
        self,
        *,
        runtime: ConversationRuntime,
        task_id: str,
        turn_idx: int,
        error: ModelContextLimitError,
    ) -> tuple[ConversationStepResult, float, bool, str | None, dict[str, Any]]:
        self._log_step(task_id, turn_idx, "orchestrator.context_limit_error", str(error), emoji="🧱")
        t0 = time.perf_counter()
        recovered = runtime.force_finalize_due_to_context_limit_after_error(error_info=error.to_info())
        runtime_elapsed_ms = (time.perf_counter() - t0) * 1000
        self._sync_trace(task_id, recovered.appended_messages)
        self._log_runtime_step(task_id, turn_idx, recovered, elapsed_ms=runtime_elapsed_ms)
        info: dict[str, Any] = dict(recovered.info)
        info["model_context_limit_error"] = error.to_info()
        if recovered.action == StepAction.CALL_MODEL and not info.get("context_limit_error_during_force_final"):
            return recovered, 0.0, False, None, info
        return recovered, 0.0, True, "terminated_context_limit", info

    def _done_reason_for_stage(self, stage: ConversationStage, info: dict[str, Any] | None = None) -> str:
        if stage == ConversationStage.ASSISTANT_COMPLETED:
            return "assistant_final_answer"
        if stage == ConversationStage.ASSISTANT_FORCE_COMPLETED:
            trigger = (info or {}).get("finalization_trigger", "")
            if trigger == FinalizationTrigger.CONTEXT_LIMIT:
                return "terminated_context_limit"
            if trigger == FinalizationTrigger.TURN_LIMIT:
                return "terminated_turn_limit"
            if trigger == FinalizationTrigger.TOOLS_EXHAUSTED:
                return "terminated_tools_exhausted"
            return "terminated_force_completed"
        if stage == ConversationStage.ASSISTANT_TOOL_CALLS_REJECTED_FORCE_FINAL:
            return "assistant_tool_calls_rejected_due_to_force_finalization"
        if stage == ConversationStage.ASSISTANT_TERMINATED_TOKEN_EXCEED:
            return "terminated_token_exceed"
        return stage.value

    def _build_step_info(
        self,
        *,
        step_result: ConversationStepResult,
        assistant_message: ConversationMessage | None,
        tool_outcomes: list[Any],
        usage: TokenUsage | None = None,
        inference_timing: dict[str, float] | None = None,
    ) -> tuple[dict[str, Any], float]:
        reward = self._evaluate_reward(
            step_result=step_result,
            assistant_message=assistant_message,
            tool_outcomes=tool_outcomes,
        )
        info: dict[str, Any] = dict(step_result.info)
        info["reward"] = reward
        if usage is not None:
            info["token_usage"] = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            }
        if inference_timing:
            info["inference_timing"] = dict(inference_timing)
        if tool_outcomes:
            info["tool_metrics"] = {
                name: {
                    "num_requested": metrics.num_requested,
                    "num_success": metrics.num_success,
                    "num_failed": metrics.num_failed,
                    "num_rejected": metrics.num_rejected,
                    "latency_ms": metrics.latency_ms,
                    "rejection_reasons": dict(metrics.rejection_reasons),
                    "failure_reasons": dict(metrics.failure_reasons),
                    "metadata_samples": {k: list(v) for k, v in metrics.metadata_samples.items()},
                }
                for name, metrics in self.tool_manager.metrics_snapshot().items()
            }
            info["tool_results"] = [
                {
                    "tool_name": outcome.request.tool_name,
                    "tool_call_id": outcome.request.call_id,
                    "status": outcome.result.status.value,
                    "reason": outcome.result.reason,
                    "content": outcome.result.content,
                }
                for outcome in tool_outcomes
            ]
        return info, reward

    def _update_run_info(self, run_info: dict[str, Any], step_info: dict[str, Any]) -> dict[str, Any]:
        if "reward" in step_info:
            run_info["reward_list"] = [*run_info.get("reward_list", []), step_info["reward"]]
            run_info["last_reward"] = step_info["reward"]
            run_info["total_reward"] = run_info.get("total_reward", 0) + step_info["reward"]

        # token_usage is already tracked in the top-level token_usage field
        # (via TaskLogger.add_token_usage) and in model_client.inference steps,
        # so skip duplicating it into run_info metadata.

        step_timing = step_info.get("inference_timing")
        if step_timing:
            run_timing = run_info.setdefault(
                "inference_timing",
                {"model_client_elapsed_ms_sum": 0.0, "engine_e2e_elapsed_s_sum": 0.0, "num_inference_calls": 0},
            )
            run_timing["model_client_elapsed_ms_sum"] += float(step_timing.get("model_client_elapsed_ms", 0.0))
            if "engine_e2e_elapsed_s" in step_timing:
                run_timing["engine_e2e_elapsed_s_sum"] += float(step_timing["engine_e2e_elapsed_s"])
            run_timing["num_inference_calls"] += 1

        if "tool_metrics" in step_info:
            # metrics_snapshot() is cumulative, so replace rather than accumulate.
            per_tool = step_info["tool_metrics"]
            overall_latency: list[float] = []
            overall_req = overall_succ = overall_fail = overall_rej = 0
            overall_metadata_samples: dict[str, list[float]] = {}
            for metrics in per_tool.values():
                latency = metrics.get("latency_ms", [])
                metrics.update(_summarize_latency_ms(latency))
                overall_req += metrics["num_requested"]
                overall_succ += metrics["num_success"]
                overall_fail += metrics["num_failed"]
                overall_rej += metrics["num_rejected"]
                overall_latency.extend(latency)
                samples = metrics.get("metadata_samples", {})
                if isinstance(samples, dict):
                    metrics["metadata_stats"] = {key: _summarize_samples(values) for key, values in samples.items()}
                    for key, values in samples.items():
                        overall_metadata_samples.setdefault(key, []).extend(values)
            per_tool["overall_metrics"] = {
                "num_requested": overall_req,
                "num_success": overall_succ,
                "num_failed": overall_fail,
                "num_rejected": overall_rej,
                "latency_ms": overall_latency,
                **_summarize_latency_ms(overall_latency),
                "metadata_stats": {key: _summarize_samples(values) for key, values in overall_metadata_samples.items()},
            }
            run_info["tool_metrics"] = per_tool

        # Per-trace format-error / rollback counters.
        # ``wrong_tool_call_format_keywords`` is only attached to step_info when the
        # runtime detected a format error on THIS turn (either reprompt or rollback path);
        # ``rollback_reason`` is attached to every rollback (format + refusal + orchestrator-initiated).
        # We surface cumulative counts in run_info so viz / aggregators can bucket tasks without replaying every step.
        rollback_stats = run_info.setdefault(
            "rollback_stats",
            {"total": 0, "by_reason": {}, "format_error_by_strategy": {}, "format_error_keywords": {}},
        )
        reason = step_info.get("rollback_reason")
        if reason:
            rollback_stats["total"] += 1
            rollback_stats["by_reason"][reason] = rollback_stats["by_reason"].get(reason, 0) + 1
        strategy = step_info.get("format_error_strategy")
        if strategy:
            rollback_stats["format_error_by_strategy"][strategy] = rollback_stats["format_error_by_strategy"].get(strategy, 0) + 1
        for keyword in step_info.get("wrong_tool_call_format_keywords") or []:
            rollback_stats["format_error_keywords"][keyword] = rollback_stats["format_error_keywords"].get(keyword, 0) + 1

        for key in ("model_context_limit_error", "context_limit_estimate"):
            if key in step_info:
                run_info[key] = step_info[key]

        return run_info

    def _log_token_usage(
        self,
        task_id: str,
        turn_idx: int,
        usage: TokenUsage | None,
        *,
        elapsed_ms: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log token usage from a model response as a trace step.

        ``extra`` carries provider-specific fields (e.g. SGLang's ``e2e_elapsed_seconds``, ``cached_tokens``)
        that the caller wants merged into the step metadata —
        ``RolloutModelClient`` stashes these in ``ModelResponse.raw_response`` and the orchestrator forwards them here
        so the viz tool can separate SGLang-internal queue/decode time from controller wall-clock.
        """
        if usage is None:
            return
        usage_dict: dict[str, Any] = {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "total_tokens": usage.total_tokens}
        if extra:
            usage_dict.update(extra)
        self._log_step(
            task_id,
            turn_idx,
            "model_client.inference",
            f"in={usage.input_tokens:,} out={usage.output_tokens:,} total={usage.total_tokens:,}",
            metadata=usage_dict,
            emoji="🧠",
            elapsed_ms=elapsed_ms,
        )
        if self.task_logger is not None:
            self.task_logger.add_token_usage(
                task_id,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            )

    def _start_trace(
        self,
        task_id: str,
        normalized_task: str,
        *,
        extra_metadata: dict[str, Any] | None = None,
        extra_tool_path: list[str] | None = None,
    ) -> None:
        if self.task_logger is None:
            return
        metadata: dict[str, Any] = {
            "orchestrator": self.config.name,
            "model": getattr(self.model_client, "model", None),
            "context_window": self.model_client.context_window,
            "max_output_tokens": self.model_client.max_output_tokens,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        self.task_logger.start_task(
            task_id,
            normalized_task,
            metadata=metadata,
            level=self.level,
            tool_path=[self.config.name, *self.tool_path, *(extra_tool_path or [])],
        )

    def _maybe_save_run_config(self) -> None:
        """Persist orchestration config and system prompt to the log dir (once per orchestrator)."""
        if self._run_config_saved or self.task_logger is None:
            return
        self._run_config_saved = True

        run_config = {
            "orchestrator_name": self.config.name,
            "max_turns": self.config.conversation.max_turns,
            "context_window": self.config.conversation.context_window,
            "tool_names": self.tool_manager.list_tool_names(),
            "model_context_window": self.model_client.context_window,
            "model_max_output_tokens": self.model_client.max_output_tokens,
        }
        self.task_logger.save_config(run_config, "run_config.json", level=self.level)

        system_prompt = self.config.conversation.system_prompt
        if system_prompt:
            self.task_logger.save_text(system_prompt, "system_prompt.txt")

    def _sync_trace(self, task_id: str, conversation: list[ConversationMessage]) -> None:
        if self.task_logger is None:
            return
        self.task_logger.append_conversation(task_id, [message.to_model_message() for message in conversation])

    def _log_step(
        self,
        task_id: str,
        turn_idx: int,
        step_name: str,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
        emoji: str | None = None,
        elapsed_ms: float | None = None,
    ) -> None:
        if self.task_logger is None:
            return

        if emoji:
            step_name = f"{emoji} {step_name}"

        if elapsed_ms is not None:
            metadata = {**(metadata or {}), "elapsed_ms": round(elapsed_ms, 1)}
            message = f"({elapsed_ms:.0f} ms) {message}"

        self.task_logger.add_step(task_id, turn_idx, step_name, message, metadata=metadata, level=self.level)
