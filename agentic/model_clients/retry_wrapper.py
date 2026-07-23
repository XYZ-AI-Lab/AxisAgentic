# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from agentic.model_clients.base import ModelClient

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentic.contracts import ConversationMessage, ModelResponse
    from agentic.model_clients.request_logger import ModelRequestLogger

logger = logging.getLogger(__name__)


class RetryingModelClient(ModelClient):
    """Model-client wrapper for provider-neutral response retries.

    The wrapper keeps native structured tool calling intact. It retries responses
    that end due to a length limit and responses with severe repeated text. When
    the wrapped client supports ``acomplete_raw`` keyword overrides, length
    retries request progressively larger ``max_tokens``.
    """

    def __init__(
        self,
        inner: ModelClient,
        *,
        max_retries: int = 10,
        retry_wait_seconds: float = 30.0,
        length_retry_scale: float = 1.1,
        repeat_tail_chars: int = 50,
        repeat_threshold: int = 5,
        continue_final_message: bool = True,
        context_safety_margin: int = 0,
        context_token_estimator: Callable[[list[ConversationMessage]], int] | None = None,
        cap_max_tokens_for_context: bool = True,
    ) -> None:
        super().__init__(
            context_window=inner.context_window,
            max_output_tokens=inner.max_output_tokens,
            enable_tool_calls=inner.enable_tool_calls,
        )
        self._inner = inner
        self.max_retries = max(1, max_retries)
        self.retry_wait_seconds = retry_wait_seconds
        self.length_retry_scale = length_retry_scale
        self.repeat_tail_chars = repeat_tail_chars
        self.repeat_threshold = repeat_threshold
        self.continue_final_message = continue_final_message
        self.context_safety_margin = max(0, context_safety_margin)
        self.context_token_estimator = context_token_estimator
        self.cap_max_tokens_for_context = cap_max_tokens_for_context

    def set_request_logger(self, request_logger: ModelRequestLogger | None) -> None:
        super().set_request_logger(request_logger)
        self._inner.set_request_logger(request_logger)

    def set_context_token_estimator(self, estimator: Callable[[list[ConversationMessage]], int] | None) -> None:
        self.context_token_estimator = estimator

    async def acomplete(
        self,
        messages: list[ConversationMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> ModelResponse:
        raw_messages = [message.to_model_message() for message in messages]
        return await self._call_with_retries(raw_messages, tools=tools, tool_choice=tool_choice)

    async def acomplete_raw(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        return await self._call_with_retries(messages, tools=tools, tool_choice=tool_choice, passthrough_kwargs=kwargs)

    async def _call_with_retries(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        passthrough_kwargs: dict[str, Any] | None = None,
    ) -> ModelResponse:
        current_max_tokens = self._cap_max_tokens_for_context(messages, self._inner.max_output_tokens)
        extra_body_override: dict[str, Any] = {}
        if self.continue_final_message and messages and messages[-1].get("role") == "assistant":
            extra_body_override["continue_final_message"] = True
            extra_body_override["add_generation_prompt"] = False

        response: ModelResponse | None = None
        for attempt in range(self.max_retries):
            kwargs = dict(passthrough_kwargs or {})
            kwargs.setdefault("max_tokens_override", current_max_tokens)
            if extra_body_override:
                caller_extra = kwargs.get("extra_body_override")
                merged_extra = dict(caller_extra) if isinstance(caller_extra, dict) else {}
                merged_extra.update(extra_body_override)
                kwargs["extra_body_override"] = merged_extra

            response = await self._inner.acomplete_raw(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )

            if response.finish_reason == "length":
                if attempt < self.max_retries - 1:
                    requested_max_tokens = max(current_max_tokens + 1, int(current_max_tokens * self.length_retry_scale))
                    current_max_tokens = self._cap_max_tokens_for_context(messages, requested_max_tokens)
                    logger.warning(
                        "Response truncated (finish_reason=length), retrying with max_tokens=%d (attempt %d/%d)",
                        current_max_tokens,
                        attempt + 1,
                        self.max_retries,
                    )
                    await asyncio.sleep(self.retry_wait_seconds)
                    continue
                logger.warning("Response still truncated after %d attempts, returning as-is", self.max_retries)
                return response

            text = response.message.content or ""
            if self._has_severe_repeat(text):
                if attempt < self.max_retries - 1:
                    logger.warning(
                        "Severe repeat detected, retrying response (attempt %d/%d)",
                        attempt + 1,
                        self.max_retries,
                    )
                    await asyncio.sleep(self.retry_wait_seconds)
                    continue
                logger.warning("Severe repeat persists after %d attempts, returning as-is", self.max_retries)

            return response

        assert response is not None
        return response

    def _has_severe_repeat(self, text: str) -> bool:
        if self.repeat_tail_chars <= 0 or self.repeat_threshold <= 0:
            return False
        if len(text) < self.repeat_tail_chars:
            return False
        tail = text[-self.repeat_tail_chars :]
        return text.count(tail) > self.repeat_threshold

    def _cap_max_tokens_for_context(self, messages: list[dict[str, Any]], requested_max_tokens: int) -> int:
        if not self.cap_max_tokens_for_context:
            return max(1, requested_max_tokens)
        estimated_prompt_tokens = self._estimate_raw_messages_tokens(messages)
        remaining = self._inner.context_window - estimated_prompt_tokens - self.context_safety_margin
        return max(1, min(requested_max_tokens, remaining))

    def _estimate_raw_messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        try:
            from agentic.contracts import ConversationMessage, ToolCall, ToolCallSpec

            converted: list[ConversationMessage] = []
            for raw in messages:
                payload = dict(raw)
                raw_tool_calls = payload.get("tool_calls")
                if raw_tool_calls:
                    tool_calls: list[ToolCall] = []
                    for tool_call in raw_tool_calls:
                        function = dict(tool_call.get("function", {}))
                        arguments = function.get("arguments", {})
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                arguments = {}
                        tool_calls.append(
                            ToolCall(
                                id=tool_call.get("id", ""),
                                type=tool_call.get("type", "function"),
                                function=ToolCallSpec(name=function.get("name", ""), arguments=arguments if isinstance(arguments, dict) else {}),
                            )
                        )
                    payload["tool_calls"] = tool_calls
                converted.append(ConversationMessage(**payload))
            if self.context_token_estimator is not None:
                return self.context_token_estimator(converted)
            return self._inner.estimate_tokens(converted)
        except Exception:
            serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            return max(1, len(serialized) // 3)

    def estimate_tokens(self, messages: list[ConversationMessage]) -> int:
        return self._inner.estimate_tokens(messages)
