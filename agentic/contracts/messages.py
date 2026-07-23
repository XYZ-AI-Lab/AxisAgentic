# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class MessageRole(StrEnum):
    """Conversation role used by chat-completion style APIs."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    CONVERSATION_RUNTIME = "ConversationRuntime"


class ToolResultStatus(StrEnum):
    """Outcome status for a tool execution attempt."""

    SUCCESS = "success"
    FAILED = "failed"
    REJECTED = "rejected"


class ToolResultReason(StrEnum):
    """Reason for a non-success tool result."""

    # Failure reasons
    PARAMETER_ERROR = "parameter_error"
    EXECUTION_ERROR = "execution_error"

    # Rejection reasons
    UNKNOWN_TOOL = "unknown_tool"
    PER_STEP_LIMIT_EXCEEDED = "per_step_limit_exceeded"
    TASK_BUDGET_EXCEEDED = "task_budget_exceeded"
    CALL_BUDGET_EXCEEDED = "call_budget_exceeded"
    CONSECUTIVE_SAME_ARGS = "consecutive_same_args"


class ToolCallSpec(BaseModel):
    """Function-style tool call emitted by an assistant message."""

    name: str = Field(description="Registered tool name to invoke.")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Parsed arguments payload for the tool call.")


class ToolCall(BaseModel):
    """One tool invocation request attached to an assistant message."""

    id: str = Field(
        default_factory=lambda: f"call_{uuid4().hex}",
        description="Unique identifier for this tool call. Tool result messages reference this ID via ConversationMessage.tool_call_id.",
    )
    type: Literal["function"] = "function"
    function: ToolCallSpec


class ConversationMessage(BaseModel):
    """Single conversation item shared across all runtime roles."""

    role: MessageRole = Field(description="Role of the message sender.")
    content: str | None = Field(default=None, description="Natural-language or structured text payload of the message.")
    reasoning_content: str | None = Field(
        default=None,
        description="Hidden assistant reasoning returned by providers that expose it. This is separate from visible content.",
    )
    visible_thought: str | None = Field(
        default=None,
        description="Visible prompted rationale, such as xLAM thought fields. This is not hidden reasoning.",
    )
    tool_calls: list[ToolCall] | None = Field(
        default=None,
        description="Tool calls requested by an assistant message.",
    )
    name: str | None = Field(
        default=None,
        description="Optional sender name. For tool messages this is typically the tool name.",
    )
    tool_call_id: str | None = Field(
        default=None,
        description="ID of the originating assistant tool call that this tool message answers.",
    )

    @classmethod
    def system(cls, content: str) -> ConversationMessage:
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> ConversationMessage:
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(
        cls,
        content: str | None = None,
        *,
        reasoning_content: str | None = None,
        visible_thought: str | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> ConversationMessage:
        return cls(
            role=MessageRole.ASSISTANT,
            content=content,
            reasoning_content=reasoning_content,
            visible_thought=visible_thought,
            tool_calls=tool_calls,
        )

    @classmethod
    def tool(cls, content: str, *, tool_call_id: str, name: str) -> ConversationMessage:
        return cls(role=MessageRole.TOOL, content=content, tool_call_id=tool_call_id, name=name)

    @classmethod
    def runtime(cls, content: str, *, name: str | None = None) -> ConversationMessage:
        return cls(role=MessageRole.CONVERSATION_RUNTIME, content=content, name=name)

    def to_model_message(self) -> dict[str, Any]:
        role_value = self.role.value if isinstance(self.role, MessageRole) else str(self.role)
        payload: dict[str, Any] = {"role": role_value}
        if self.content is not None:
            payload["content"] = self.content
        if self.reasoning_content is not None:
            payload["reasoning_content"] = self.reasoning_content
        if self.visible_thought is not None:
            payload["visible_thought"] = self.visible_thought
        if self.tool_calls is not None:
            payload["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": json.dumps(tc.function.arguments) if isinstance(tc.function.arguments, dict) else tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload


class TokenUsage(BaseModel):
    """Token accounting returned by the underlying model provider."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ModelResponse(BaseModel):
    """Normalized model response object used by runtime code."""

    message: ConversationMessage
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    raw_response: Any | None = None


class ToolRequest(BaseModel):
    """Internal runtime representation of a tool execution request."""

    tool_name: str = Field(description="Target tool name registered in the tool manager.")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Parsed arguments passed to the tool.")
    call_id: str = Field(
        default_factory=lambda: f"call_{uuid4().hex}",
        description="Stable call identifier propagated into tool result messages.",
    )
    backend_name: str | None = Field(default=None, description="Optional MCP server name or remote backend identifier.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra execution metadata recorded by the runtime.")
