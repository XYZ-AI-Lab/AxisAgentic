# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agentic.contracts import ConversationMessage, ModelResponse
    from agentic.model_clients.request_logger import ModelRequestLogger


class ModelClient(ABC):
    def __init__(
        self,
        *,
        context_window: int,
        max_output_tokens: int,
        enable_tool_calls: bool = True,
    ) -> None:
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.enable_tool_calls = enable_tool_calls
        self.request_logger: ModelRequestLogger | None = None

    def set_request_logger(self, request_logger: ModelRequestLogger | None) -> None:
        self.request_logger = request_logger

    @abstractmethod
    async def acomplete(
        self,
        messages: list[ConversationMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> ModelResponse:
        raise NotImplementedError

    async def acomplete_raw(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> ModelResponse:
        """Call the model with pre-formatted message dicts.

        Subclasses with direct API access (e.g. :class:`OpenAICompatibleModelClient`)
        override this for efficiency. The default implementation converts dicts back
        to :class:`ConversationMessage` and delegates to :meth:`acomplete`.
        """
        from agentic.contracts import ConversationMessage as CM

        conv_messages = [CM(**m) for m in messages]
        return await self.acomplete(conv_messages, tools=tools, tool_choice=tool_choice)

    def estimate_tokens(self, messages: list[ConversationMessage]) -> int:
        text = "\n".join(message.content or "" for message in messages)
        return max(1, len(text) // 4)


class CallableModelClient(ModelClient):
    def __init__(
        self,
        complete_fn: Callable[[list[ConversationMessage], list[dict[str, Any]] | None, str | None], Awaitable[ModelResponse]],
        *,
        context_window: int,
        max_output_tokens: int,
        enable_tool_calls: bool = True,
    ) -> None:
        super().__init__(
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            enable_tool_calls=enable_tool_calls,
        )
        self._complete_fn = complete_fn

    async def acomplete(
        self,
        messages: list[ConversationMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> ModelResponse:
        return await self._complete_fn(messages, tools, tool_choice)
