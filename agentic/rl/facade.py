# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentic.contracts import MessageRole

if TYPE_CHECKING:
    from agentic.contracts import ConversationMessage, ConversationStepResult, ModelResponse
    from agentic.conversations.conversation_runtime import ConversationRuntime
    from agentic.model_clients.base import ModelClient
    from agentic.orchestration.task_orchestrator import OrchestrationResult, TaskOrchestrator
    from agentic.tools.manager import ToolManager


class RLPolicyFacade:
    def __init__(
        self,
        *,
        model_client: ModelClient,
        tool_manager: ToolManager | None = None,
    ) -> None:
        self.model_client = model_client
        self.tool_manager = tool_manager

    async def act(self, conversation: list[ConversationMessage]) -> ModelResponse:
        tools = None
        tool_choice = None
        if self.tool_manager is not None:
            tools = self.tool_manager.list_tool_definitions()
            tool_choice = "auto" if tools else None
        return await self.model_client.acomplete(conversation, tools=tools, tool_choice=tool_choice)


class RLEnvironmentFacade:
    def __init__(self, *, runtime: ConversationRuntime) -> None:
        self.runtime = runtime

    def reset(self, task_description: str) -> ConversationStepResult:
        self.runtime.reset()
        return self.runtime.initialize_conversation(task_description)

    def step(self, message: ConversationMessage) -> ConversationStepResult:
        if message.role == MessageRole.ASSISTANT:
            return self.runtime.apply_assistant_message(message)
        if message.role == MessageRole.TOOL:
            return self.runtime.apply_batch_tool_messages([message])
        msg = "RLEnvironmentFacade.step only supports assistant or tool messages."
        raise ValueError(msg)


class RLRolloutFacade:
    def __init__(self, *, orchestrator: TaskOrchestrator) -> None:
        self.orchestrator = orchestrator

    async def rollout(self, task: str | dict[str, Any]) -> OrchestrationResult:
        return await self.orchestrator.run(task)
