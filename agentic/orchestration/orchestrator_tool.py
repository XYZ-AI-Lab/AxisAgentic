# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentic.tools.base import Tool, ToolContext, ToolResult

if TYPE_CHECKING:
    from agentic.orchestration.task_orchestrator import TaskOrchestrator


class OrchestratorTool(Tool):
    def __init__(
        self,
        *,
        orchestrator: TaskOrchestrator,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        strict_mode: bool = True,
        emoji: str = "🤖",
    ) -> None:
        super().__init__(name=name, description=description, parameters=parameters, strict_mode=strict_mode, emoji=emoji)
        self.orchestrator = orchestrator

    def set_hierarchy_context(self, *, level: int, tool_path: list[str]) -> None:
        """Propagate nesting level and tool path to the wrapped orchestrator."""
        self.orchestrator.set_hierarchy_context(level=level, tool_path=[*tool_path, self.name])

    async def arun(self, arguments: dict[str, Any], *, context: ToolContext) -> ToolResult:
        del context
        task: str | dict[str, Any] = arguments.get("task", arguments)
        task_id: str | None = arguments.get("task_id")
        result = await self.orchestrator.run(task, task_id=task_id)
        return ToolResult(content=str(result.output))
