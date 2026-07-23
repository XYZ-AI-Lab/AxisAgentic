# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any

from pydantic import Field

from agentic.config.models import ModelClientConfig
from agentic.contracts import ConversationMessage, ModelResponse, TokenUsage, ToolCall, ToolCallSpec
from agentic.model_assets import infer_endpoint_profile
from agentic.model_clients.base import ModelClient
from agentic.model_clients.errors import ModelContextLimitError

DEFAULT_REASONING_RESPONSE_FIELDS = ("reasoning_content", "reasoning", "reasoning_details")
DEFAULT_TRANSIENT_ENDPOINT_ERROR_STATUS_CODES = (408, 429, 500, 502, 503, 504)
STATUS_ERROR_RETRY_BACKOFF_MAX_SECONDS = 300.0
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelEndpointProfile:
    name: str
    response_reasoning_fields: tuple[str, ...] = DEFAULT_REASONING_RESPONSE_FIELDS
    message_reasoning_field: str = "reasoning_content"
    max_tokens_field: str = "max_tokens"
    request_extra_body: dict[str, Any] = field(default_factory=dict)
    preserve_reasoning_content: bool = True
    parse_embedded_thinking: bool = False
    parallel_tool_calls: bool | None = None


_ENDPOINT_PROFILES: dict[str, ModelEndpointProfile] = {
    "deepseek-v4-pro": ModelEndpointProfile(name="deepseek-v4-pro"),
    "kimi-k2.6": ModelEndpointProfile(name="kimi-k2.6"),
    "glm-5.1": ModelEndpointProfile(name="glm-5.1"),
    "qwen3-thinking": ModelEndpointProfile(name="qwen3-thinking"),
    "minimax": ModelEndpointProfile(name="minimax"),
    "nemotron": ModelEndpointProfile(name="nemotron"),
    "xlam": ModelEndpointProfile(name="xlam"),
}

_EXTRA_BODY_RESERVED_FIELDS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "parallel_tool_calls",
        "stream",
        "stop",
        "response_format",
        "seed",
    }
)


class OpenAICompatibleModelClientConfig(ModelClientConfig):
    """Configuration for OpenAI-compatible API model clients."""

    model: str = "gpt-4.1-mini"
    api_base: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 120.0
    top_p: float | None = None
    endpoint_profile: str | None = None
    preserve_reasoning_content: bool | None = None
    response_reasoning_fields: list[str] | None = None
    message_reasoning_field: str | None = None
    max_tokens_field: str | None = None
    parallel_tool_calls: bool | None = None
    parse_embedded_thinking: bool | None = None
    endpoint_error_exit_status_codes: list[int] = Field(default_factory=lambda: list(DEFAULT_TRANSIENT_ENDPOINT_ERROR_STATUS_CODES))
    endpoint_error_exit_after_seconds: float | None = 300.0
    endpoint_connection_error_retry_wait_seconds: float = 60.0
    endpoint_error_retry_backoff_multiplier: float = 2.0
    endpoint_error_retry_backoff_max_seconds: float = STATUS_ERROR_RETRY_BACKOFF_MAX_SECONDS
    endpoint_error_retry_jitter: bool = True
    endpoint_error_respect_retry_after: bool = True
    endpoint_retry_on_timeout: bool = True
    endpoint_retry_on_connection_error: bool = True
    token_estimation_chars_per_token: float = Field(default=3.0, gt=0)
    extra_body: dict[str, Any] = Field(default_factory=dict)


class OpenAICompatibleModelClient(ModelClient):
    def __init__(self, config: OpenAICompatibleModelClientConfig) -> None:
        super().__init__(
            context_window=config.context_window,
            max_output_tokens=config.max_output_tokens,
            enable_tool_calls=config.enable_tool_calls,
        )
        self._config = config
        self._client: Any = None
        self._profile = self._resolve_profile()
        self._endpoint_error_window_started_at: float | None = None

    def _resolve_profile(self) -> ModelEndpointProfile:
        profile_name = self._config.endpoint_profile or infer_endpoint_profile(self._config.model, self._config.api_base)
        if profile_name:
            base = _ENDPOINT_PROFILES.get(profile_name)
            if base is None:
                valid = ", ".join(sorted(_ENDPOINT_PROFILES))
                msg = f"Unknown endpoint_profile {profile_name!r}; expected one of: {valid}"
                raise ValueError(msg)
        else:
            base = ModelEndpointProfile(name="generic")
        return ModelEndpointProfile(
            name=base.name,
            response_reasoning_fields=tuple(self._config.response_reasoning_fields or base.response_reasoning_fields),
            message_reasoning_field=self._config.message_reasoning_field or base.message_reasoning_field,
            max_tokens_field=self._config.max_tokens_field or base.max_tokens_field,
            request_extra_body=dict(base.request_extra_body),
            preserve_reasoning_content=base.preserve_reasoning_content
            if self._config.preserve_reasoning_content is None
            else self._config.preserve_reasoning_content,
            parse_embedded_thinking=base.parse_embedded_thinking
            if self._config.parse_embedded_thinking is None
            else self._config.parse_embedded_thinking,
            parallel_tool_calls=base.parallel_tool_calls if self._config.parallel_tool_calls is None else self._config.parallel_tool_calls,
        )

    def _get_client(self) -> Any:
        """Return the shared AsyncOpenAI client, creating it lazily on first use."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=os.environ.get(self._config.api_key_env) if self._config.api_key_env else None,
                base_url=self._config.api_base,
                timeout=self._config.timeout_seconds,
                max_retries=0,
            )
        return self._client

    async def acomplete(
        self,
        messages: list[ConversationMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> ModelResponse:
        raw_messages = [self._to_model_message(message) for message in messages]
        return await self.acomplete_raw(raw_messages, tools=tools, tool_choice=tool_choice)

    async def acomplete_raw(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        max_tokens_override: int | None = None,
        extra_body_override: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Call the API with pre-formatted message dicts (bypassing ConversationMessage serialization)."""
        client = self._get_client()
        # Merge extra_body: config defaults + caller overrides.
        extra_body = self._merged_extra_body(extra_body_override)
        max_tokens_field = self._profile.max_tokens_field
        request_payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [self._normalize_outbound_message(message) for message in messages],
            "tools": tools if self.enable_tool_calls else None,
            "tool_choice": tool_choice if self.enable_tool_calls else None,
            "temperature": self._config.temperature,
            "extra_body": extra_body or None,
        }
        request_payload[max_tokens_field] = max_tokens_override or self._config.max_output_tokens
        if self._profile.parallel_tool_calls is not None:
            request_payload["parallel_tool_calls"] = self._profile.parallel_tool_calls
        if self._config.top_p is not None:
            request_payload["top_p"] = self._config.top_p
        (
            response,
            request_id,
            request_started_at,
            request_start,
            model_call_start,
        ) = await self._create_chat_completion_with_endpoint_retries(
            client,
            request_payload,
        )
        self._reset_endpoint_error_window()
        raw_response = response.model_dump() if hasattr(response, "model_dump") else response
        if not response.choices:
            if self.request_logger is not None and request_id is not None and request_started_at is not None:
                self.request_logger.log(
                    request_id=request_id,
                    started_at=request_started_at,
                    elapsed_ms=(time.perf_counter() - request_start) * 1000.0,
                    request=request_payload,
                    response=raw_response,
                    error={"type": "ValueError", "message": f"Model '{self._config.model}' returned no choices."},
                    metadata={"client": "OpenAICompatibleModelClient"},
                )
            msg = f"Model '{self._config.model}' returned no choices."
            raise ValueError(msg)
        choice = response.choices[0]
        tool_calls = self._assistant_tool_calls(choice.message)
        usage = self._token_usage(response.usage)
        assistant_content = self._assistant_content(choice.message)
        reasoning_content = self._assistant_reasoning_content(choice.message)
        if self._profile.parse_embedded_thinking and reasoning_content is None:
            reasoning_content, assistant_content = self._split_embedded_thinking(assistant_content)
        finish_reason = choice.finish_reason
        assistant_message = ConversationMessage.assistant(assistant_content, reasoning_content=reasoning_content, tool_calls=tool_calls)
        now = time.perf_counter()
        final_request_elapsed_ms = (now - request_start) * 1000.0
        client_elapsed_ms_excluding_request_logging = (now - model_call_start) * 1000.0
        if isinstance(raw_response, dict):
            raw_response["client_elapsed_ms_excluding_request_logging"] = client_elapsed_ms_excluding_request_logging
        if self.request_logger is not None and request_id is not None and request_started_at is not None:
            self.request_logger.log(
                request_id=request_id,
                started_at=request_started_at,
                elapsed_ms=final_request_elapsed_ms,
                request=request_payload,
                response=raw_response,
                metadata={"client": "OpenAICompatibleModelClient", "finish_reason": finish_reason},
            )
        return ModelResponse(
            message=assistant_message,
            finish_reason=finish_reason,
            usage=usage,
            raw_response=raw_response,
        )

    async def _create_chat_completion_with_endpoint_retries(
        self,
        client: Any,
        request_payload: dict[str, Any],
    ) -> tuple[Any, str | None, str | None, float, float]:
        model_call_start = time.perf_counter()
        transient_error_attempts = 0
        request_id, request_started_at, request_start = self._new_request_logging_context()
        while True:
            try:
                response = await client.chat.completions.create(**request_payload)
                return response, request_id, request_started_at, request_start, model_call_start
            except Exception as exc:
                self._log_request_error(
                    request_id=request_id,
                    request_started_at=request_started_at,
                    request_start=request_start,
                    request_payload=request_payload,
                    exc=exc,
                )
                status_code = self._extract_status_code(exc)
                if self._is_context_length_error(exc):
                    self._reset_endpoint_error_window()
                    raise ModelContextLimitError.from_exception(exc, status_code=status_code) from exc
                if self._is_endpoint_timeout_error(exc):
                    if not self._config.endpoint_retry_on_timeout:
                        self._reset_endpoint_error_window()
                        raise
                    retry_wait_seconds = self._transient_error_retry_wait_seconds(exc, attempt=transient_error_attempts, respect_retry_after=False)
                    transient_error_attempts += 1
                    await self._sleep_or_exit_on_transient_endpoint_error(
                        exc,
                        retry_wait_seconds=retry_wait_seconds,
                        error_kind="timeout",
                    )
                    request_id, request_started_at, request_start = self._new_request_logging_context()
                    continue
                if self._is_endpoint_connection_error(exc):
                    if not self._config.endpoint_retry_on_connection_error:
                        self._reset_endpoint_error_window()
                        raise
                    retry_wait_seconds = self._transient_error_retry_wait_seconds(exc, attempt=transient_error_attempts, respect_retry_after=False)
                    transient_error_attempts += 1
                    await self._sleep_or_exit_on_transient_endpoint_error(
                        exc,
                        retry_wait_seconds=retry_wait_seconds,
                        error_kind="connection",
                    )
                    request_id, request_started_at, request_start = self._new_request_logging_context()
                    continue
                if self._is_retryable_endpoint_status_code(status_code):
                    retry_wait_seconds = self._transient_error_retry_wait_seconds(
                        exc,
                        attempt=transient_error_attempts,
                        respect_retry_after=True,
                    )
                    transient_error_attempts += 1
                    await self._sleep_or_exit_on_transient_endpoint_error(
                        exc,
                        retry_wait_seconds=retry_wait_seconds,
                        error_kind=f"status {status_code}",
                    )
                    request_id, request_started_at, request_start = self._new_request_logging_context()
                    continue
                self._reset_endpoint_error_window()
                raise

    def _new_request_logging_context(self) -> tuple[str | None, str | None, float]:
        request_id = self.request_logger.next_id() if self.request_logger is not None else None
        request_started_at = self.request_logger.now() if self.request_logger is not None else None
        request_start = time.perf_counter()
        return request_id, request_started_at, request_start

    def _log_request_error(
        self,
        *,
        request_id: str | None,
        request_started_at: str | None,
        request_start: float,
        request_payload: dict[str, Any],
        exc: Exception,
    ) -> None:
        if self.request_logger is None or request_id is None or request_started_at is None:
            return
        self.request_logger.log(
            request_id=request_id,
            started_at=request_started_at,
            elapsed_ms=(time.perf_counter() - request_start) * 1000.0,
            request=request_payload,
            error={"type": type(exc).__name__, "message": str(exc)},
            metadata={"client": "OpenAICompatibleModelClient"},
        )

    def _merged_extra_body(self, extra_body_override: dict[str, Any] | None = None) -> dict[str, Any]:
        extra_body = dict(self._profile.request_extra_body)
        if self._config.extra_body:
            extra_body.update(self._config.extra_body)
        if extra_body_override:
            extra_body.update(extra_body_override)
        self._validate_extra_body(extra_body)
        return extra_body

    @staticmethod
    def _validate_extra_body(extra_body: dict[str, Any]) -> None:
        duplicated = sorted(_EXTRA_BODY_RESERVED_FIELDS.intersection(extra_body))
        if duplicated:
            joined = ", ".join(duplicated)
            msg = f"OpenAI-compatible standard request fields must be configured as top-level client parameters, not inside extra_body: {joined}"
            raise ValueError(msg)

    def _to_model_message(self, message: ConversationMessage) -> dict[str, Any]:
        return self._normalize_outbound_message(message.to_model_message())

    def estimate_tokens(self, messages: list[ConversationMessage]) -> int:
        payload = [self._to_model_message(message) for message in messages]
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        message_overhead = 4 * len(payload)
        return max(1, math.ceil(len(serialized) / self._config.token_estimation_chars_per_token) + message_overhead)

    def _normalize_outbound_message(self, message: dict[str, Any]) -> dict[str, Any]:
        payload = dict(message)
        reasoning = payload.pop("reasoning_content", None)
        if self._profile.preserve_reasoning_content and reasoning is not None:
            payload[self._profile.message_reasoning_field] = reasoning
        elif not self._profile.preserve_reasoning_content:
            payload.pop(self._profile.message_reasoning_field, None)
        return payload

    @staticmethod
    def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str) and arguments:
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def _assistant_tool_calls(self, message: Any) -> list[ToolCall] | None:
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return None
        return [
            ToolCall(
                id=tool_call.id,
                function=ToolCallSpec(name=tool_call.function.name, arguments=self._parse_tool_arguments(tool_call.function.arguments)),
            )
            for tool_call in tool_calls
        ]

    @staticmethod
    def _token_usage(usage: Any) -> TokenUsage | None:
        if usage is None:
            return None
        return TokenUsage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

    def _assistant_content(self, message: Any) -> str | None:
        content = getattr(message, "content", None)
        return content if isinstance(content, str) or content is None else str(content)

    def _assistant_reasoning_content(self, message: Any) -> str | None:
        dumped_message = message.model_dump() if hasattr(message, "model_dump") else None
        field_names = list(self._profile.response_reasoning_fields)
        for field_name in field_names:
            value = getattr(message, field_name, None)
            if value is None and isinstance(dumped_message, dict):
                value = dumped_message.get(field_name)
            if isinstance(value, str) and value.strip():
                return value
        return None

    @staticmethod
    def _split_embedded_thinking(content: str | None) -> tuple[str | None, str | None]:
        if not content:
            return None, content
        match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
        if match is None:
            return None, content
        reasoning = match.group(1).strip()
        visible = (content[: match.start()] + content[match.end() :]).lstrip("\n")
        return reasoning or None, visible or None

    def _reset_endpoint_error_window(self) -> None:
        self._endpoint_error_window_started_at = None

    async def _sleep_or_exit_on_transient_endpoint_error(
        self,
        exc: Exception,
        *,
        retry_wait_seconds: float,
        error_kind: str,
    ) -> None:
        duration_s = self._config.endpoint_error_exit_after_seconds

        now = self._monotonic_now()
        if self._endpoint_error_window_started_at is None:
            self._endpoint_error_window_started_at = now
        elapsed_s = now - self._endpoint_error_window_started_at
        if duration_s is not None and duration_s >= 0 and elapsed_s >= duration_s:
            message = f"Endpoint transient error persisted for {elapsed_s:.1f}s (threshold={duration_s:.1f}s); exiting process."
            logger.critical(message)
            raise SystemExit(message) from exc

        wait_s = max(0.0, retry_wait_seconds)
        if duration_s is not None and duration_s >= 0:
            remaining_s = max(0.0, duration_s - elapsed_s)
            sleep_s = min(wait_s, remaining_s) if wait_s > 0 else 0.0
            exit_after_label = f"{duration_s:.1f}s"
        else:
            sleep_s = wait_s
            exit_after_label = "disabled"
        logger.warning(
            "Endpoint transient error (%s; %s: %s); retrying in %.1fs (continuous_window_elapsed=%.1fs, exit_after=%s)",
            error_kind,
            type(exc).__name__,
            exc,
            sleep_s,
            elapsed_s,
            exit_after_label,
        )
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)

    def _is_retryable_endpoint_status_code(self, status_code: int | None) -> bool:
        if status_code is None:
            return False
        configured = set(self._config.endpoint_error_exit_status_codes or [])
        return status_code in configured

    def _transient_error_retry_wait_seconds(self, exc: Exception, *, attempt: int, respect_retry_after: bool) -> float:
        retry_after_s = self._retry_after_seconds(exc) if respect_retry_after and self._config.endpoint_error_respect_retry_after else None
        if retry_after_s is not None:
            return retry_after_s
        base_s = max(0.0, self._config.endpoint_connection_error_retry_wait_seconds)
        if base_s <= 0:
            return 0.0
        multiplier = max(1.0, self._config.endpoint_error_retry_backoff_multiplier)
        max_s = max(0.0, self._config.endpoint_error_retry_backoff_max_seconds)
        delay_s = min(base_s * (multiplier**attempt), max_s)
        if self._config.endpoint_error_retry_jitter and delay_s > 0:
            delay_s = delay_s * 0.5 + random.uniform(0.0, delay_s * 0.5)
        return delay_s

    @classmethod
    def _retry_after_seconds(cls, exc: Exception) -> float | None:
        headers = cls._exception_headers(exc)
        if not headers:
            return None
        value = None
        for key, candidate in headers.items():
            if key.lower() == "retry-after":
                value = candidate
                break
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return max(0.0, float(text))
        except ValueError:
            pass
        try:
            retry_at = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            return None
        return max(0.0, retry_at.timestamp() - time.time())

    @staticmethod
    def _exception_headers(exc: Exception) -> dict[str, Any] | None:
        headers = getattr(exc, "headers", None)
        if headers is not None:
            return dict(headers)
        response = getattr(exc, "response", None)
        response_headers = getattr(response, "headers", None)
        return dict(response_headers) if response_headers is not None else None

    @staticmethod
    def _is_endpoint_timeout_error(exc: Exception) -> bool:
        exc_type = type(exc).__name__
        if "timeout" in exc_type.lower():
            return True
        text = str(exc).lower()
        return exc_type == "APIError" and "timeout" in text

    @staticmethod
    def _is_endpoint_connection_error(exc: Exception) -> bool:
        exc_type = type(exc).__name__
        if "timeout" in exc_type.lower():
            return False
        if exc_type in {"APIConnectionError", "ConnectError"}:
            return True
        module = type(exc).__module__
        if module.startswith(("httpx", "httpcore")) and "connect" in exc_type.lower():
            return True
        text = str(exc).lower()
        return exc_type == "APIError" and any(marker in text for marker in ("connection error", "connect error"))

    @staticmethod
    def _monotonic_now() -> float:
        return time.monotonic()

    @staticmethod
    def _extract_status_code(exc: Exception) -> int | None:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        return response_status if isinstance(response_status, int) else None

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        text = str(exc).lower()
        context_markers = (
            "context length",
            "context_length",
            "maximum context",
            "max context",
            "token count",
            "too many tokens",
            "input tokens",
        )
        return any(marker in text for marker in context_markers)
