# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import jsonschema

from agentic.config.models import ToolManagerConfig
from agentic.contracts import ConversationMessage, ToolRequest
from agentic.contracts.messages import ToolResultReason, ToolResultStatus
from agentic.observability.task_logger import TaskLogger, ToolTrace
from agentic.tools.argument_repair import ToolArgumentRepairer, ToolArgumentRepairHook
from agentic.tools.base import Tool, ToolContext, ToolMetrics, ToolResult

UNKNOWN_TOOL_METRICS_KEY = "__unknown__"


@dataclass(slots=True)
class ToolExecutionOutcome:
    request: ToolRequest
    result: ToolResult
    message: ConversationMessage


class ToolManager:
    def __init__(
        self,
        *,
        tools: list[Tool] | None = None,
        config: ToolManagerConfig | None = None,
        task_logger: TaskLogger | None = None,
        level: int = 0,
        tool_path: list[str] | None = None,
    ) -> None:
        self._config = config or ToolManagerConfig()
        repair_config = self._config.argument_repair
        self._argument_repairer = ToolArgumentRepairer(**repair_config.model_dump())
        self._tools: dict[str, Tool] = {}
        self._metrics: dict[str, ToolMetrics] = {}
        self._task_logger = task_logger
        self._level = level
        self._tool_path: list[str] = list(tool_path or [])
        # Per-task state (reset via reset_task_state)
        self._task_call_counts: dict[str, int] = {}
        self._task_total_calls: int = 0
        self._task_unknown_tool_names: dict[str, int] = {}
        # Cross-round consecutive-args tracking: tool_name → (args, consecutive_count)
        self._prev_round_args: dict[str, tuple[dict[str, Any], int]] = {}
        for tool in tools or []:
            self.register(tool)

    def set_task_logger(self, task_logger: TaskLogger | None) -> None:
        self._task_logger = task_logger

    def set_level(self, level: int) -> None:
        self._level = level
        for tool in self._tools.values():
            self._propagate_level_to_tool(tool)

    def set_tool_path(self, tool_path: list[str]) -> None:
        self._tool_path = list(tool_path)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._propagate_level_to_tool(tool)
        self._tools[tool.name] = tool
        self._metrics.setdefault(tool.name, ToolMetrics())
        self._task_call_counts.setdefault(tool.name, 0)

    def register_argument_repair_hook(self, tool_name: str, hook: ToolArgumentRepairHook) -> None:
        """Register a domain-specific argument repair hook for one tool."""
        self._argument_repairer.register_hook(tool_name, hook)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def list_tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def list_tool_definitions(self, *, strict: bool | None = None) -> list[dict[str, Any]]:
        return [tool.to_tool_definition(strict=strict) for tool in self._tools.values()]

    def list_server_tool_groups(self) -> list[dict[str, Any]]:
        """Return tools grouped by ``server_name`` in MCP server format.

        Each entry is ``{"name": server_name, "tools": [{"name", "description", "schema"}, ...]}``.
        Tools without a ``server_name`` get their own singleton group keyed by tool name.
        """
        from collections import OrderedDict

        groups: OrderedDict[str, list[Tool]] = OrderedDict()
        for tool in self._tools.values():
            key = tool.server_name or tool.name
            groups.setdefault(key, []).append(tool)

        result: list[dict[str, Any]] = []
        for server_name, tools in groups.items():
            server_tools: list[dict[str, Any]] = []
            for tool in tools:
                tool_def = tool.to_tool_definition()
                func_def = tool_def.get("function", {})
                schema = func_def.get("parameters", {})
                server_tools.append(
                    {
                        "name": func_def.get("name", tool.name),
                        "description": func_def.get("description", tool.description),
                        "schema": str(schema),
                    }
                )
            result.append({"name": server_name, "tools": server_tools})
        return result

    def reset_task_state(self) -> None:
        """Reset per-task state. Call at the start of each new task."""
        for name in self._task_call_counts:
            self._task_call_counts[name] = 0
        self._task_total_calls = 0
        self._task_unknown_tool_names.clear()
        self._prev_round_args.clear()
        # Reset cumulative metrics so they don't bleed across tasks.
        for m in self._metrics.values():
            m.num_requested = 0
            m.num_success = 0
            m.num_failed = 0
            m.num_rejected = 0
            m.latency_ms.clear()
            m.rejection_reasons.clear()
            m.failure_reasons.clear()
            m.metadata_samples.clear()

    async def begin_task(self, *, task_id: str | None = None) -> None:
        """Start per-task state for tools that own external resources."""
        for tool in self._tools.values():
            await tool.begin_task(task_id=task_id)

    async def end_task(self, *, task_id: str | None = None) -> None:
        """Clean up per-task state for tools that own external resources."""
        for tool in reversed(list(self._tools.values())):
            await tool.end_task(task_id=task_id)

    def snapshot_task_state(self) -> dict[str, Any]:
        return {
            "task_call_counts": dict(self._task_call_counts),
            "task_total_calls": self._task_total_calls,
            "task_unknown_tool_names": dict(self._task_unknown_tool_names),
            "prev_round_args": copy.deepcopy(self._prev_round_args),
            "metrics": copy.deepcopy(self._metrics),
        }

    def restore_task_state(self, snapshot: dict[str, Any]) -> None:
        self._task_call_counts = dict(snapshot.get("task_call_counts", {}))
        for name in self._tools:
            self._task_call_counts.setdefault(name, 0)
        self._task_total_calls = int(snapshot.get("task_total_calls", 0) or 0)
        self._task_unknown_tool_names = dict(snapshot.get("task_unknown_tool_names", {}))
        self._prev_round_args = copy.deepcopy(snapshot.get("prev_round_args", {}))
        metrics = snapshot.get("metrics")
        if isinstance(metrics, dict):
            self._metrics = copy.deepcopy(metrics)
        for name in self._tools:
            self._metrics.setdefault(name, ToolMetrics())

    def unknown_tool_names_snapshot(self) -> dict[str, int]:
        """Return per-task counts of unknown tool names requested."""
        return dict(self._task_unknown_tool_names)

    @property
    def task_total_calls(self) -> int:
        """Total tool calls executed in the current task (excludes rejections)."""
        return self._task_total_calls

    def all_tools_exhausted(self) -> bool:
        """Return True if no registered tool can accept another call in the current task.

        This is True when either the shared task budget is reached, or every
        registered tool has individually reached its per-tool call limit.
        """
        if self._config.max_tool_calls_per_task is not None and self._task_total_calls >= self._config.max_tool_calls_per_task:
            return True
        if not self._tools:
            return False
        for name, tool in self._tools.items():
            effective_max = tool.max_calls_per_task if tool.max_calls_per_task is not None else self._config.default_max_calls_per_tool_per_task
            if effective_max is None:
                return False
            if self._task_call_counts.get(name, 0) < effective_max:
                return False
        return True

    def record_rejection(self, tool_name: str, *, reason: ToolResultReason) -> None:
        """Record a rejection that happened outside the ToolManager (e.g. runtime-level)."""
        metrics = self._metrics.setdefault(tool_name, ToolMetrics())
        metrics.num_requested += 1
        metrics.num_rejected += 1
        metrics.rejection_reasons[reason] = metrics.rejection_reasons.get(reason, 0) + 1

    def metrics_snapshot(self) -> dict[str, ToolMetrics]:
        return {
            name: ToolMetrics(
                num_requested=m.num_requested,
                num_success=m.num_success,
                num_failed=m.num_failed,
                num_rejected=m.num_rejected,
                latency_ms=list(m.latency_ms),
                rejection_reasons=dict(m.rejection_reasons),
                failure_reasons=dict(m.failure_reasons),
                metadata_samples={k: list(v) for k, v in m.metadata_samples.items()},
            )
            for name, m in self._metrics.items()
        }

    @staticmethod
    def _record_metadata_samples(metrics: ToolMetrics, latency_ms: float, result: ToolResult) -> None:
        metadata = result.metadata if isinstance(result.metadata, dict) else None
        if not metadata:
            return

        def _record(key: str, value: Any) -> None:
            if isinstance(value, bool):
                return
            if isinstance(value, (int, float)):
                metrics.metadata_samples.setdefault(key, []).append(float(value))

        for key, value in metadata.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    _record(f"{key}_{sub_key}", sub_value)
            else:
                _record(key, value)
        request_latency_ms = metadata.get("request_latency_ms")
        if isinstance(request_latency_ms, (int, float)) and not isinstance(request_latency_ms, bool):
            queue_ms = max(0.0, float(latency_ms) - float(request_latency_ms))
            metrics.metadata_samples.setdefault("client_queue_ms", []).append(queue_ms)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute_tool_calls(self, tool_requests: list[ToolRequest], *, task_id: str | None = None) -> list[ToolExecutionOutcome]:
        """Execute a batch of tool requests (one round), applying per-step limits.

        Each call to ``execute_tool_calls`` is one round.  Consecutive same-args
        detection compares each request against the *previous* round's recorded
        args.  After processing, the round's successfully executed args replace
        the previous round's tracking.  Tools not called this round have their
        consecutive count reset.
        """
        max_per_step = self._config.max_tool_calls_per_step
        outcomes: list[ToolExecutionOutcome] = []
        # Collect this round's successfully executed args: tool_name → args
        current_round_args: dict[str, dict[str, Any]] = {}
        step_count = 0
        for raw_request in tool_requests:
            request = self.repair_tool_request(raw_request)
            if max_per_step is not None and step_count >= max_per_step:
                result = self._build_rejection(
                    request,
                    reason=ToolResultReason.PER_STEP_LIMIT_EXCEEDED,
                    message=self._config.tool_call_limit_exceeded_message,
                )
                self._trace_tool(task_id, request, result)
            else:
                result = await self._execute(request, task_id=task_id)
                if result.status != ToolResultStatus.REJECTED:
                    step_count += 1
                if result.status == ToolResultStatus.SUCCESS:
                    current_round_args[request.tool_name] = request.arguments
            message = ConversationMessage.tool(result.content, tool_call_id=request.call_id, name=request.tool_name)
            outcomes.append(ToolExecutionOutcome(request=request, result=result, message=message))

        # Build next round's tracking from this round's successful calls
        new_prev: dict[str, tuple[dict[str, Any], int]] = {}
        for tool_name, args in current_round_args.items():
            prev_entry = self._prev_round_args.get(tool_name)
            if prev_entry is not None and prev_entry[0] == args:
                new_prev[tool_name] = (args, prev_entry[1] + 1)
            else:
                new_prev[tool_name] = (args, 1)
        self._prev_round_args = new_prev

        return outcomes

    async def _execute(self, request: ToolRequest, *, task_id: str | None = None) -> ToolResult:
        """Execute a single tool request with full policy checks."""
        request = self.repair_tool_request(request)

        # Unknown tool
        if not self.has_tool(request.tool_name):
            return self._handle_unknown_tool(request, task_id=task_id)

        tool = self._tools[request.tool_name]
        metrics = self._metrics[tool.name]
        metrics.num_requested += 1

        # Shared task budget
        if self._config.max_tool_calls_per_task is not None and self._task_total_calls >= self._config.max_tool_calls_per_task:
            return self._reject_and_trace(
                request,
                tool,
                metrics,
                reason=ToolResultReason.TASK_BUDGET_EXCEEDED,
                message=self._config.task_budget_exceeded_message,
                task_id=task_id,
            )

        # Per-tool budget
        effective_max = tool.max_calls_per_task if tool.max_calls_per_task is not None else self._config.default_max_calls_per_tool_per_task
        if effective_max is not None and self._task_call_counts.get(tool.name, 0) >= effective_max:
            effective_msg = tool.call_budget_exceeded_message or self._config.default_call_budget_exceeded_message
            return self._reject_and_trace(
                request,
                tool,
                metrics,
                reason=ToolResultReason.CALL_BUDGET_EXCEEDED,
                message=effective_msg,
                task_id=task_id,
            )

        # Consecutive same-args (compared against previous round)
        effective_max_consecutive = (
            tool.max_consecutive_calls_with_same_args
            if tool.max_consecutive_calls_with_same_args is not None
            else self._config.default_max_consecutive_calls_with_same_args
        )
        if effective_max_consecutive is not None:
            prev_entry = self._prev_round_args.get(tool.name)
            if prev_entry is not None:
                prev_args, consecutive_count = prev_entry
                if request.arguments == prev_args and consecutive_count >= effective_max_consecutive:
                    effective_msg = tool.consecutive_call_exceeded_message or self._config.default_consecutive_call_exceeded_message
                    return self._reject_and_trace(
                        request,
                        tool,
                        metrics,
                        reason=ToolResultReason.CONSECUTIVE_SAME_ARGS,
                        message=effective_msg,
                        task_id=task_id,
                    )

        # Parameter validation
        param_error = self._validate_parameters(tool, request.arguments)
        if param_error is not None:
            metrics.num_failed += 1
            metrics.failure_reasons[ToolResultReason.PARAMETER_ERROR] = metrics.failure_reasons.get(ToolResultReason.PARAMETER_ERROR, 0) + 1
            result = ToolResult(content=param_error, status=ToolResultStatus.FAILED, reason=ToolResultReason.PARAMETER_ERROR)
            self._trace_tool(task_id, request, result, latency_ms=0.0, emoji=tool.emoji)
            return result

        # Execute
        start = time.perf_counter()
        try:
            result = await tool.arun(request.arguments, context=ToolContext(request=request))
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            metrics.num_failed += 1
            metrics.failure_reasons[ToolResultReason.EXECUTION_ERROR] = metrics.failure_reasons.get(ToolResultReason.EXECUTION_ERROR, 0) + 1
            metrics.latency_ms.append(latency_ms)
            result = ToolResult(content=str(exc), status=ToolResultStatus.FAILED, reason=ToolResultReason.EXECUTION_ERROR)
            self._record_metadata_samples(metrics, latency_ms, result)
            self._trace_tool(task_id, request, result, latency_ms=latency_ms, emoji=tool.emoji)
            return result

        latency_ms = (time.perf_counter() - start) * 1000.0
        metrics.latency_ms.append(latency_ms)
        if result.status == ToolResultStatus.FAILED:
            metrics.num_failed += 1
            failure_reason = result.reason or ToolResultReason.EXECUTION_ERROR
            metrics.failure_reasons[failure_reason] = metrics.failure_reasons.get(failure_reason, 0) + 1
        else:
            metrics.num_success += 1

        self._record_metadata_samples(metrics, latency_ms, result)

        # Update task-level counters
        self._task_call_counts[tool.name] = self._task_call_counts.get(tool.name, 0) + 1
        self._task_total_calls += 1

        self._trace_tool(task_id, request, result, latency_ms=latency_ms, emoji=tool.emoji)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def repair_tool_request(self, request: ToolRequest) -> ToolRequest:
        """Return a request with opt-in pre-tool argument repair applied."""
        tool = self._tools.get(request.tool_name)
        if tool is None:
            return request
        repair = self._argument_repairer.repair(request, tool)
        if not repair.changed:
            return request
        metadata = dict(request.metadata or {})
        metadata["argument_repair"] = {"applied_rules": repair.applied_rules}
        return request.model_copy(update={"arguments": repair.arguments, "metadata": metadata})

    def _propagate_level_to_tool(self, tool: Tool) -> None:
        from agentic.orchestration.orchestrator_tool import OrchestratorTool as _OrchestratorTool

        if isinstance(tool, _OrchestratorTool):
            tool.set_hierarchy_context(level=self._level, tool_path=self._tool_path)

    def _handle_unknown_tool(self, request: ToolRequest, *, task_id: str | None) -> ToolResult:
        metrics = self._metrics.setdefault(UNKNOWN_TOOL_METRICS_KEY, ToolMetrics())
        metrics.num_requested += 1
        metrics.num_rejected += 1
        metrics.rejection_reasons[ToolResultReason.UNKNOWN_TOOL] = metrics.rejection_reasons.get(ToolResultReason.UNKNOWN_TOOL, 0) + 1
        self._task_unknown_tool_names[request.tool_name] = self._task_unknown_tool_names.get(request.tool_name, 0) + 1
        error_content = f"Unknown tool '{request.tool_name}'. Available tools: {self.list_tool_names()}"
        result = ToolResult(content=error_content, status=ToolResultStatus.REJECTED, reason=ToolResultReason.UNKNOWN_TOOL)
        self._trace_tool(task_id, request, result)
        return result

    def _reject_and_trace(
        self,
        request: ToolRequest,
        tool: Tool,
        metrics: ToolMetrics,
        *,
        reason: ToolResultReason,
        message: str,
        task_id: str | None,
    ) -> ToolResult:
        metrics.num_rejected += 1
        metrics.rejection_reasons[reason] = metrics.rejection_reasons.get(reason, 0) + 1
        result = ToolResult(content=message, status=ToolResultStatus.REJECTED, reason=reason)
        self._trace_tool(task_id, request, result, emoji=tool.emoji)
        return result

    def _build_rejection(self, request: ToolRequest, *, reason: ToolResultReason, message: str) -> ToolResult:
        """Build a rejection result and update metrics (used for per-step rejections)."""
        tool_name = request.tool_name if self.has_tool(request.tool_name) else UNKNOWN_TOOL_METRICS_KEY
        metrics = self._metrics.setdefault(tool_name, ToolMetrics())
        metrics.num_requested += 1
        metrics.num_rejected += 1
        metrics.rejection_reasons[reason] = metrics.rejection_reasons.get(reason, 0) + 1
        return ToolResult(content=message, status=ToolResultStatus.REJECTED, reason=reason)

    def _validate_parameters(self, tool: Tool, arguments: dict[str, Any]) -> str | None:
        """Validate arguments against the tool's JSON Schema. Returns error string or None."""
        schema = tool.parameters
        if not schema:
            return None
        try:
            jsonschema.validate(instance=arguments, schema=schema)
        except jsonschema.ValidationError as exc:
            return f"Parameter validation error: {exc.message}"
        return None

    def _trace_tool(
        self,
        task_id: str | None,
        request: ToolRequest,
        result: ToolResult,
        *,
        latency_ms: float = 0.0,
        emoji: str | None = None,
    ) -> None:
        if task_id is None or self._task_logger is None:
            return
        self._task_logger.add_tool_trace(
            task_id,
            ToolTrace(
                tool_name=request.tool_name,
                call_id=request.call_id,
                arguments=request.arguments,
                status=result.status,
                latency_ms=latency_ms,
                reason=result.reason,
                content=result.content,
                emoji=emoji or "🔧",
                metadata=result.metadata,
            ),
            level=self._level,
        )
