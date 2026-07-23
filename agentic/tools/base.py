# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from agentic.contracts.messages import ToolResultReason, ToolResultStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agentic.contracts import ToolRequest


@dataclass(slots=True)
class ToolContext:
    request: ToolRequest


@dataclass(slots=True)
class ToolMetrics:
    """Cumulative metrics for a single tool (or the ``__unknown__`` bucket)."""

    num_requested: int = 0
    num_success: int = 0
    num_failed: int = 0
    num_rejected: int = 0
    latency_ms: list[float] = field(default_factory=list)
    rejection_reasons: dict[ToolResultReason, int] = field(default_factory=dict)
    failure_reasons: dict[ToolResultReason, int] = field(default_factory=dict)
    metadata_samples: dict[str, list[float]] = field(default_factory=dict)


class ToolResult(BaseModel):
    """Output of a tool invocation.

    ``metadata`` is copied into trace JSON. Numeric top-level values, plus one-level
    nested numeric values flattened with ``_``, are sampled by ``ToolManager`` and
    reduced into TensorBoard metrics. Keys using ``_ms`` as a token get an additional
    ``_seconds`` alias downstream; ``request_latency_ms`` also enables synthesized
    ``client_queue_ms = total_tool_latency_ms - request_latency_ms``.
    """

    content: str
    status: ToolResultStatus = ToolResultStatus.SUCCESS
    reason: ToolResultReason | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


DEFAULT_TOOL_EMOJI = "🔧"


class Tool(ABC):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        strict_mode: bool = True,
        emoji: str = DEFAULT_TOOL_EMOJI,
        server_name: str | None = None,
        max_calls_per_task: int | None = None,
        call_budget_exceeded_message: str | None = None,
        max_consecutive_calls_with_same_args: int | None = None,
        consecutive_call_exceeded_message: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters or {"type": "object", "properties": {}, "additionalProperties": False}
        self.strict_mode = strict_mode
        self.emoji = emoji
        self.server_name = server_name
        self.max_calls_per_task = max_calls_per_task
        self.call_budget_exceeded_message = call_budget_exceeded_message
        self.max_consecutive_calls_with_same_args = max_consecutive_calls_with_same_args
        self.consecutive_call_exceeded_message = consecutive_call_exceeded_message

    @abstractmethod
    async def arun(self, arguments: dict[str, Any], *, context: ToolContext) -> ToolResult:
        raise NotImplementedError

    async def begin_task(self, *, task_id: str | None = None) -> None:
        """Start per-task tool state, if the tool owns any."""
        # Keep the public keyword and satisfy CI/lint in this no-op default.
        del task_id

    async def end_task(self, *, task_id: str | None = None) -> None:
        """Clean up per-task tool state, if the tool owns any."""
        # Keep the public keyword and satisfy CI/lint in this no-op default.
        del task_id

    def to_tool_definition(self, *, strict: bool | None = None) -> dict[str, Any]:
        effective_strict = self.strict_mode if strict is None else strict
        function_payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        if effective_strict:
            function_payload["strict"] = True
        return {"type": "function", "function": function_payload}


class CallableTool(Tool):
    """A :class:`Tool` backed by an async/sync callable function.

    The classmethod :meth:`from_mcp_definition` provides a convenient way to create
    a ``CallableTool`` directly from an MCP-style tool definition dict (the same
    format returned by ``session.list_tools()`` in the MCP protocol or by
    ``FastMCP``). This is useful for copying tool definitions from MCP servers
    so that the system-prompt rendering is identical.
    """

    @classmethod
    def from_mcp_definition(
        cls,
        mcp_def: dict[str, Any],
        fn: Callable[..., ToolResult | Awaitable[ToolResult] | str | dict[str, Any]],
        *,
        server_name: str | None = None,
        strict_mode: bool = False,
        emoji: str = DEFAULT_TOOL_EMOJI,
        max_calls_per_task: int | None = None,
        call_budget_exceeded_message: str | None = None,
        max_consecutive_calls_with_same_args: int | None = None,
        consecutive_call_exceeded_message: str | None = None,
    ) -> CallableTool:
        """Create a :class:`CallableTool` from an MCP tool definition.

        Args:
            mcp_def: MCP tool definition dict.  Expected keys:

                * ``name`` (str) — tool name.
                * ``description`` (str) — tool description (full docstring).
                * ``inputSchema`` (dict) — JSON-Schema for the tool parameters
                  (Pydantic / FastMCP style with ``title`` fields is fine).

            fn: The async or sync callable that implements the tool.
            server_name: MCP server name this tool belongs to.
            strict_mode: Defaults to ``False`` because MCP schemas from FastMCP
                typically lack ``additionalProperties`` and other strict-mode
                requirements.
            emoji: Logging emoji.
            max_calls_per_task: Passed through to :class:`Tool`.
            call_budget_exceeded_message: Passed through to :class:`Tool`.
            max_consecutive_calls_with_same_args: Passed through to :class:`Tool`.
            consecutive_call_exceeded_message: Passed through to :class:`Tool`.
        """
        return cls(
            name=mcp_def["name"],
            description=mcp_def["description"],
            parameters=mcp_def.get("inputSchema"),
            fn=fn,
            server_name=server_name,
            strict_mode=strict_mode,
            emoji=emoji,
            max_calls_per_task=max_calls_per_task,
            call_budget_exceeded_message=call_budget_exceeded_message,
            max_consecutive_calls_with_same_args=max_consecutive_calls_with_same_args,
            consecutive_call_exceeded_message=consecutive_call_exceeded_message,
        )

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any] | None,
        fn: Callable[..., ToolResult | Awaitable[ToolResult] | str | dict[str, Any]],
        strict_mode: bool = True,
        emoji: str = DEFAULT_TOOL_EMOJI,
        server_name: str | None = None,
        max_calls_per_task: int | None = None,
        call_budget_exceeded_message: str | None = None,
        max_consecutive_calls_with_same_args: int | None = None,
        consecutive_call_exceeded_message: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            strict_mode=strict_mode,
            emoji=emoji,
            server_name=server_name,
            max_calls_per_task=max_calls_per_task,
            call_budget_exceeded_message=call_budget_exceeded_message,
            max_consecutive_calls_with_same_args=max_consecutive_calls_with_same_args,
            consecutive_call_exceeded_message=consecutive_call_exceeded_message,
        )
        self._fn = fn
        self._signature = inspect.signature(fn)
        self._accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in self._signature.parameters.values())
        self._accepted_kwargs = {
            name
            for name, param in self._signature.parameters.items()
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

    async def arun(self, arguments: dict[str, Any], *, context: ToolContext) -> ToolResult:
        del context
        if not self._accepts_var_kwargs:
            arguments = {key: value for key, value in arguments.items() if key in self._accepted_kwargs}
        value = self._fn(**arguments)
        if inspect.isawaitable(value):
            value = await value
        if isinstance(value, ToolResult):
            return value
        if isinstance(value, dict):
            return ToolResult(content=json.dumps(value, ensure_ascii=True))
        return ToolResult(content=str(value))


class MCPToolAdapter(Tool):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any] | None,
        executor: Callable[[ToolRequest], Awaitable[ToolResult]],
        backend_name: str,
        strict_mode: bool = True,
        emoji: str = DEFAULT_TOOL_EMOJI,
        server_name: str | None = None,
        max_calls_per_task: int | None = None,
        call_budget_exceeded_message: str | None = None,
        max_consecutive_calls_with_same_args: int | None = None,
        consecutive_call_exceeded_message: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            strict_mode=strict_mode,
            emoji=emoji,
            server_name=server_name,
            max_calls_per_task=max_calls_per_task,
            call_budget_exceeded_message=call_budget_exceeded_message,
            max_consecutive_calls_with_same_args=max_consecutive_calls_with_same_args,
            consecutive_call_exceeded_message=consecutive_call_exceeded_message,
        )
        self._executor = executor
        self.backend_name = backend_name

    async def arun(self, arguments: dict[str, Any], *, context: ToolContext) -> ToolResult:
        request = context.request.model_copy(update={"arguments": arguments, "backend_name": self.backend_name})
        return await self._executor(request)
