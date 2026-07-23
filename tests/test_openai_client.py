"""Tests for OpenAICompatibleModelClient response parsing.

Uses mock objects to simulate OpenAI API responses without making real HTTP calls.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest

from agentic.contracts import ConversationMessage, MessageRole, ModelResponse, ToolCall, ToolCallSpec
from agentic.model_clients.base import ModelClient
from agentic.model_clients.errors import ModelContextLimitError
from agentic.model_clients.openai_client import (
    DEFAULT_TRANSIENT_ENDPOINT_ERROR_STATUS_CODES,
    OpenAICompatibleModelClient,
    OpenAICompatibleModelClientConfig,
)
from agentic.model_clients.request_logger import ModelRequestLogger
from agentic.model_clients.retry_wrapper import RetryingModelClient

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Lightweight mock objects mimicking the openai SDK response shape
# ---------------------------------------------------------------------------


@dataclass
class _MockFunction:
    name: str
    arguments: str  # JSON string

    def model_dump(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class _MockToolCall:
    id: str
    type: str
    function: _MockFunction

    def model_dump(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "function": self.function.model_dump()}


@dataclass
class _MockMessage:
    content: str | None = None
    tool_calls: list[_MockToolCall] | None = None
    reasoning_content: str | None = None
    reasoning: str | None = None
    reasoning_details: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": self.tool_calls,
            "reasoning_content": self.reasoning_content,
            "reasoning": self.reasoning,
            "reasoning_details": self.reasoning_details,
        }


@dataclass
class _MockChoice:
    message: _MockMessage
    finish_reason: str = "stop"

    def model_dump(self) -> dict[str, Any]:
        return {"message": self.message.model_dump(), "finish_reason": self.finish_reason}


@dataclass
class _MockUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 5
    total_tokens: int = 15

    def model_dump(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class _MockResponse:
    choices: list[_MockChoice] = field(default_factory=list)
    usage: _MockUsage | None = None

    def model_dump(self) -> dict[str, Any]:
        return {"choices": self.choices, "usage": self.usage}


@dataclass
class _MockErrorResponse:
    status_code: int
    headers: dict[str, str] = field(default_factory=dict)


class _MockStatusError(Exception):
    def __init__(self, status_code: int, message: str = "endpoint error", headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = _MockErrorResponse(status_code=status_code, headers=headers or {})


class APIConnectionError(Exception):
    pass


class APITimeoutError(Exception):
    pass


class _LengthRetryCaptureClient(ModelClient):
    def __init__(self) -> None:
        super().__init__(context_window=100, max_output_tokens=50)
        self.max_tokens_seen: list[int] = []

    async def acomplete(
        self,
        messages: list[ConversationMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> ModelResponse:
        return await self.acomplete_raw([message.to_model_message() for message in messages], tools=tools, tool_choice=tool_choice)

    async def acomplete_raw(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        del messages, tools, tool_choice
        self.max_tokens_seen.append(kwargs["max_tokens_override"])
        finish_reason = "length" if len(self.max_tokens_seen) == 1 else "stop"
        return ModelResponse(message=ConversationMessage.assistant("ok"), finish_reason=finish_reason)

    def estimate_tokens(self, messages: list[ConversationMessage]) -> int:
        del messages
        return 90


def _build_config(**overrides: Any) -> OpenAICompatibleModelClientConfig:
    defaults = {
        "model": "test-model",
        "api_base": "http://localhost:1234",
        "context_window": 4096,
        "max_output_tokens": 512,
    }
    defaults.update(overrides)
    return OpenAICompatibleModelClientConfig(**defaults)


def _make_client(config: OpenAICompatibleModelClientConfig | None = None) -> OpenAICompatibleModelClient:
    return OpenAICompatibleModelClient(config or _build_config())


WEB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}, "num": {"type": "integer"}},
                "required": ["q"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_and_extract_info",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}, "info_to_extract": {"type": "string"}},
                "required": ["url", "info_to_extract"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_plain_text_response_parsing() -> None:
    """A plain text response should parse into an assistant message with no tool calls."""
    client = _make_client()
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content="Hello, world!"), finish_reason="stop")],
        usage=_MockUsage(prompt_tokens=8, completion_tokens=3, total_tokens=11),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    result = asyncio.run(client.acomplete([ConversationMessage.user("Hi")]))

    assert result.message.role == MessageRole.ASSISTANT
    assert result.message.content == "Hello, world!"
    assert result.message.tool_calls is None
    assert result.finish_reason == "stop"
    assert result.usage is not None
    assert result.usage.input_tokens == 8
    assert result.usage.output_tokens == 3
    assert result.usage.total_tokens == 11


def test_tool_call_response_parsing() -> None:
    """Tool calls should be parsed into ToolCall objects with deserialized arguments."""
    client = _make_client()
    mock_response = _MockResponse(
        choices=[
            _MockChoice(
                message=_MockMessage(
                    content=None,
                    tool_calls=[
                        _MockToolCall(
                            id="call_abc123",
                            type="function",
                            function=_MockFunction(name="get_weather", arguments='{"city": "Paris", "units": "celsius"}'),
                        ),
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=_MockUsage(),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    result = asyncio.run(client.acomplete([ConversationMessage.user("Weather?")]))

    assert result.message.content is None
    assert result.message.tool_calls is not None
    assert len(result.message.tool_calls) == 1
    tc = result.message.tool_calls[0]
    assert tc.id == "call_abc123"
    assert tc.function.name == "get_weather"
    assert tc.function.arguments == {"city": "Paris", "units": "celsius"}
    assert result.finish_reason == "tool_calls"


def test_tool_call_response_does_not_use_reasoning_content_by_default() -> None:
    """Provider-specific reasoning fields should stay out of visible content."""
    client = _make_client()
    mock_response = _MockResponse(
        choices=[
            _MockChoice(
                message=_MockMessage(
                    content=None,
                    reasoning_content="This should stay out of content by default.",
                    tool_calls=[
                        _MockToolCall(
                            id="call_abc123",
                            type="function",
                            function=_MockFunction(name="get_weather", arguments='{"city": "Paris"}'),
                        ),
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=_MockUsage(),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    result = asyncio.run(client.acomplete([ConversationMessage.user("Weather?")]))

    assert result.message.content is None
    assert result.message.reasoning_content == "This should stay out of content by default."
    assert result.message.tool_calls is not None


def test_tool_call_response_preserves_reasoning_content_separately() -> None:
    """Kimi-style reasoning_content is preserved without copying it into content."""
    client = _make_client()
    mock_response = _MockResponse(
        choices=[
            _MockChoice(
                message=_MockMessage(
                    content=None,
                    reasoning_content="Let me search for that.",
                    tool_calls=[
                        _MockToolCall(
                            id="call_abc123",
                            type="function",
                            function=_MockFunction(name="web_search", arguments='{"query": "test"}'),
                        ),
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=_MockUsage(),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    result = asyncio.run(client.acomplete([ConversationMessage.user("Search?")]))

    assert result.message.content is None
    assert result.message.reasoning_content == "Let me search for that."
    assert result.message.tool_calls is not None


def test_tool_call_response_preserves_configured_reasoning_alias() -> None:
    """GLM-style reasoning can be normalized into reasoning_content."""
    client = _make_client()
    mock_response = _MockResponse(
        choices=[
            _MockChoice(
                message=_MockMessage(
                    content=None,
                    reasoning="I need one source first.",
                    tool_calls=[
                        _MockToolCall(
                            id="call_abc123",
                            type="function",
                            function=_MockFunction(name="web_search", arguments='{"query": "test"}'),
                        ),
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=_MockUsage(),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    result = asyncio.run(client.acomplete([ConversationMessage.user("Search?")]))

    assert result.message.content is None
    assert result.message.reasoning_content == "I need one source first."
    assert result.message.tool_calls is not None


def test_default_reasoning_aliases_include_reasoning_details() -> None:
    client = _make_client()
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content="done", reasoning_details="Detailed hidden reasoning."), finish_reason="stop")],
        usage=_MockUsage(),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    result = asyncio.run(client.acomplete([ConversationMessage.user("Question")]))

    assert result.message.content == "done"
    assert result.message.reasoning_content == "Detailed hidden reasoning."


def test_embedded_thinking_is_not_split_when_reasoning_field_exists() -> None:
    config = _build_config(parse_embedded_thinking=True)
    client = _make_client(config)
    content = "<think>embedded should stay in content</think>\nvisible answer"
    mock_response = _MockResponse(
        choices=[
            _MockChoice(
                message=_MockMessage(
                    content=content,
                    reasoning_content="Provider reasoning field wins.",
                ),
                finish_reason="stop",
            )
        ],
        usage=_MockUsage(),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    result = asyncio.run(client.acomplete([ConversationMessage.user("Question")]))

    assert result.message.content == content
    assert result.message.reasoning_content == "Provider reasoning field wins."


def test_default_reasoning_alias_reads_reasoning() -> None:
    config = _build_config(model="glm-5.1", api_base="http://model.example/v1")
    client = _make_client(config)
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content="done", reasoning="I should answer."), finish_reason="stop")],
        usage=_MockUsage(),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    result = asyncio.run(client.acomplete([ConversationMessage.user("Question")]))

    assert result.message.content == "done"
    assert result.message.reasoning_content == "I should answer."


def test_deepseek_tool_call_content_stays_visible_without_reasoning_field() -> None:
    config = _build_config(model="deepseek-v4-pro", endpoint_profile="deepseek-v4-pro")
    client = _make_client(config)
    mock_response = _MockResponse(
        choices=[
            _MockChoice(
                message=_MockMessage(
                    content="I should search first.",
                    tool_calls=[
                        _MockToolCall(id="call_search", type="function", function=_MockFunction(name="web_search", arguments='{"query": "test"}')),
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=_MockUsage(),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    result = asyncio.run(client.acomplete([ConversationMessage.user("Search?")]))

    assert result.message.content == "I should search first."
    assert result.message.reasoning_content is None
    assert result.message.tool_calls is not None


def test_qwen_text_tool_call_markup_stays_visible_content() -> None:
    config = _build_config(model="exp19-swift-agent-hermes-ckpt600", endpoint_profile="qwen3-thinking")
    client = _make_client(config)
    content = (
        '<tool_call>\n{"name": "google_search", "arguments": {"q": "\\"discontent is a necessary\\" interview musician", "num": 10}}\n</tool_call>'
    )
    mock_response = _MockResponse(
        choices=[
            _MockChoice(
                message=_MockMessage(
                    content=content,
                    reasoning_content="Search the exact phrase.",
                ),
                finish_reason="stop",
            )
        ],
        usage=_MockUsage(),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    result = asyncio.run(client.acomplete([ConversationMessage.user("Search?")], tools=WEB_TOOLS, tool_choice="auto"))

    assert result.finish_reason == "stop"
    assert result.message.content == content
    assert result.message.reasoning_content == "Search the exact phrase."
    assert result.message.tool_calls is None


def test_deepseek_tool_call_content_is_replayed_as_content() -> None:
    config = _build_config(model="deepseek-v4-pro", endpoint_profile="deepseek-v4-pro")
    client = _make_client(config)
    first_response = _MockResponse(
        choices=[
            _MockChoice(
                message=_MockMessage(
                    content="Need evidence before answering.",
                    tool_calls=[
                        _MockToolCall(id="call_search", type="function", function=_MockFunction(name="web_search", arguments='{"query": "test"}')),
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=_MockUsage(),
    )
    second_response = _MockResponse(choices=[_MockChoice(message=_MockMessage(content="done"), finish_reason="stop")], usage=_MockUsage())

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(side_effect=[first_response, second_response])
    client._client = mock_openai

    user_message = ConversationMessage.user("Search?")
    first = asyncio.run(client.acomplete([user_message]))
    assert first.message.tool_calls is not None
    asyncio.run(
        client.acomplete(
            [
                user_message,
                first.message,
                ConversationMessage.tool("{}", tool_call_id=first.message.tool_calls[0].id, name="web_search"),
            ]
        )
    )

    second_call_kwargs = mock_openai.chat.completions.create.call_args_list[1].kwargs
    assistant_payload = second_call_kwargs["messages"][1]
    assert assistant_payload["content"] == "Need evidence before answering."
    assert "reasoning_content" not in assistant_payload
    assert assistant_payload["tool_calls"][0]["function"]["name"] == "web_search"


def test_kimi_tool_call_content_stays_visible_without_reasoning_field() -> None:
    config = _build_config(model="kimi-k2.6", endpoint_profile="kimi-k2.6")
    client = _make_client(config)
    mock_response = _MockResponse(
        choices=[
            _MockChoice(
                message=_MockMessage(
                    content="Visible tool preamble.",
                    tool_calls=[
                        _MockToolCall(id="call_search", type="function", function=_MockFunction(name="web_search", arguments='{"query": "test"}')),
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=_MockUsage(),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    result = asyncio.run(client.acomplete([ConversationMessage.user("Search?")]))

    assert result.message.content == "Visible tool preamble."
    assert result.message.reasoning_content is None
    assert result.message.tool_calls is not None


def test_deepseek_final_answer_content_stays_visible() -> None:
    config = _build_config(model="deepseek-v4-pro", endpoint_profile="deepseek-v4-pro")
    client = _make_client(config)
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content=r"\boxed{answer}"), finish_reason="stop")],
        usage=_MockUsage(),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    result = asyncio.run(client.acomplete([ConversationMessage.user("Question")]))

    assert result.message.content == r"\boxed{answer}"
    assert result.message.reasoning_content is None
    assert result.message.tool_calls is None


def test_reasoning_content_replayed_in_later_requests() -> None:
    client = _make_client()
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content="ok"))],
        usage=_MockUsage(),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    asyncio.run(
        client.acomplete(
            [
                ConversationMessage.user("Search"),
                ConversationMessage.assistant("Using a tool", reasoning_content="Need to search first."),
            ]
        )
    )

    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"][1]["reasoning_content"] == "Need to search first."


def test_token_estimator_counts_reasoning_and_tool_calls() -> None:
    client = _make_client(_build_config(token_estimation_chars_per_token=3.0))
    bare = client.estimate_tokens([ConversationMessage.assistant("x")])
    with_reasoning_and_tool = client.estimate_tokens(
        [
            ConversationMessage.assistant(
                "x",
                reasoning_content="r" * 300,
                tool_calls=[
                    ToolCall(id="call_search", function=ToolCallSpec(name="web_search", arguments={"query": "test"})),
                ],
            )
        ]
    )

    assert with_reasoning_and_tool > bare + 100


def test_configured_max_completion_tokens_field() -> None:
    config = _build_config(model="kimi-k2.6", api_base="https://api.moonshot.ai/v1", max_tokens_field="max_completion_tokens")
    client = _make_client(config)
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content="ok"))],
        usage=_MockUsage(),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    asyncio.run(client.acomplete([ConversationMessage.user("test")]))

    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert "max_completion_tokens" in call_kwargs
    assert "max_tokens" not in call_kwargs


def test_unknown_endpoint_profile_fails_fast() -> None:
    config = _build_config(endpoint_profile="qwen")

    with pytest.raises(ValueError, match="Unknown endpoint_profile"):
        _make_client(config)


def test_multiple_tool_calls_parsing() -> None:
    """Multiple tool calls in a single response should all be parsed."""
    client = _make_client()
    mock_response = _MockResponse(
        choices=[
            _MockChoice(
                message=_MockMessage(
                    content="Let me search for that.",
                    tool_calls=[
                        _MockToolCall(id="call_1", type="function", function=_MockFunction(name="search", arguments='{"query": "Python"}')),
                        _MockToolCall(id="call_2", type="function", function=_MockFunction(name="search", arguments='{"query": "Rust"}')),
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=_MockUsage(),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    result = asyncio.run(client.acomplete([ConversationMessage.user("Compare Python and Rust")]))

    assert result.message.content == "Let me search for that."
    assert result.message.tool_calls is not None
    assert len(result.message.tool_calls) == 2
    assert result.message.tool_calls[0].function.arguments == {"query": "Python"}
    assert result.message.tool_calls[1].function.arguments == {"query": "Rust"}


def test_no_usage_in_response() -> None:
    """When the API returns no usage info, usage should be None."""
    client = _make_client()
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content="ok"))],
        usage=None,
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    result = asyncio.run(client.acomplete([ConversationMessage.user("test")]))

    assert result.usage is None


def test_no_choices_raises_value_error() -> None:
    """An empty choices list should raise ValueError."""
    client = _make_client()
    mock_response = _MockResponse(choices=[], usage=None)

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    with pytest.raises(ValueError, match="returned no choices"):
        asyncio.run(client.acomplete([ConversationMessage.user("test")]))


def test_tool_calls_disabled_via_config() -> None:
    """When enable_tool_calls=False, tools and tool_choice should be passed as None to the API."""
    config = _build_config(enable_tool_calls=False)
    client = _make_client(config)
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content="ok"))],
        usage=_MockUsage(),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
    asyncio.run(client.acomplete([ConversationMessage.user("test")], tools=tools, tool_choice="auto"))

    call_kwargs = mock_openai.chat.completions.create.call_args
    assert call_kwargs.kwargs.get("tools") is None
    assert call_kwargs.kwargs.get("tool_choice") is None


def test_client_reused_across_calls() -> None:
    """The AsyncOpenAI client should be created once and reused."""
    client = _make_client()
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content="ok"))],
        usage=_MockUsage(),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

    with patch("agentic.model_clients.openai_client.OpenAICompatibleModelClient._get_client", return_value=mock_openai) as mock_get:
        asyncio.run(client.acomplete([ConversationMessage.user("call 1")]))
        asyncio.run(client.acomplete([ConversationMessage.user("call 2")]))
        assert mock_get.call_count == 2  # called each time, but returns the same cached instance


def test_extra_body_override_merged() -> None:
    """extra_body_override should be merged with config defaults."""
    config = _build_config(extra_body={"repetition_penalty": 1.05})
    client = _make_client(config)
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content="ok"))],
        usage=_MockUsage(),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    asyncio.run(client.acomplete_raw([{"role": "user", "content": "test"}], extra_body_override={"repeat_penalty": 1.1}))

    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    extra_body = call_kwargs.get("extra_body")
    assert extra_body == {"repetition_penalty": 1.05, "repeat_penalty": 1.1}


def test_standard_fields_are_rejected_inside_extra_body() -> None:
    config = _build_config(extra_body={"top_p": 0.9})
    client = _make_client(config)
    mock_openai = AsyncMock()
    client._client = mock_openai

    with pytest.raises(ValueError, match="top_p"):
        asyncio.run(client.acomplete_raw([{"role": "user", "content": "test"}]))


def test_request_logger_drains_payloads_on_close(tmp_path: Path) -> None:
    """Request logging should preserve payloads without requiring synchronous writes."""
    client = _make_client()
    request_logger = ModelRequestLogger(tmp_path, name="inference")
    client.set_request_logger(request_logger)
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content="ok"))],
        usage=_MockUsage(),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    result = asyncio.run(client.acomplete_raw([{"role": "user", "content": "test"}]))
    request_logger.close()

    assert result.message.content == "ok"
    part_path = tmp_path / "model_requests" / "inference" / "part-000001.jsonl"
    records = [json.loads(line) for line in part_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["request"]["messages"] == [{"role": "user", "content": "test"}]
    assert records[0]["response"]["choices"][0]["message"]["content"] == "ok"
    assert isinstance(records[0]["response"]["client_elapsed_ms_excluding_request_logging"], float)
    manifest = json.loads((tmp_path / "model_requests" / "inference" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["closed"] is True
    assert manifest["total_records"] == 1
    assert manifest["parts"] == [{"bytes": part_path.stat().st_size, "path": "part-000001.jsonl", "records": 1}]


def test_request_logger_preserves_provider_tool_call_arguments(tmp_path: Path) -> None:
    """Request logs keep the provider response after server-side tool parsing."""
    client = _make_client()
    request_logger = ModelRequestLogger(tmp_path, name="inference")
    client.set_request_logger(request_logger)
    mock_response = _MockResponse(
        choices=[
            _MockChoice(
                message=_MockMessage(
                    content=None,
                    tool_calls=[
                        _MockToolCall(
                            id="call_empty_python",
                            type="function",
                            function=_MockFunction(name="python_exec", arguments="{}"),
                        ),
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=_MockUsage(),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    result = asyncio.run(client.acomplete_raw([{"role": "user", "content": "test"}], tools=WEB_TOOLS, tool_choice="auto"))
    request_logger.close()

    assert result.message.tool_calls is not None
    assert result.message.tool_calls[0].function.name == "python_exec"
    assert result.message.tool_calls[0].function.arguments == {}
    part_path = tmp_path / "model_requests" / "inference" / "part-000001.jsonl"
    [record] = [json.loads(line) for line in part_path.read_text(encoding="utf-8").splitlines()]
    logged_tool_call = record["response"]["choices"][0]["message"]["tool_calls"][0]
    assert logged_tool_call["function"] == {"name": "python_exec", "arguments": "{}"}
    assert record["metadata"]["finish_reason"] == "tool_calls"


def test_request_logger_rolls_by_record_count(tmp_path: Path) -> None:
    request_logger = ModelRequestLogger(tmp_path, name="summary_llm", max_records_per_file=1)
    for idx in range(2):
        request_logger.log(
            request_id=f"req-{idx}",
            started_at=request_logger.now(),
            elapsed_ms=1.0,
            request={"idx": idx},
            response={"ok": True},
        )
    request_logger.close()

    request_dir = tmp_path / "model_requests" / "summary_llm"
    assert (request_dir / "part-000001.jsonl").exists()
    assert (request_dir / "part-000002.jsonl").exists()
    manifest = json.loads((request_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_records"] == 2
    assert [part["records"] for part in manifest["parts"]] == [1, 1]


def test_response_carries_elapsed_before_request_logging() -> None:
    """The orchestrator can use this elapsed value to exclude request logger overhead."""
    client = _make_client()
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content="ok"))],
        usage=_MockUsage(),
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai

    result = asyncio.run(client.acomplete_raw([{"role": "user", "content": "test"}]))

    assert isinstance(result.raw_response, dict)
    assert isinstance(result.raw_response["client_elapsed_ms_excluding_request_logging"], float)


def test_openai_sdk_internal_retries_are_disabled() -> None:
    client = _make_client()

    with patch("openai.AsyncOpenAI") as mock_openai_cls:
        client._get_client()

    assert mock_openai_cls.call_args.kwargs["max_retries"] == 0


def test_plain_400_fails_without_retry_or_endpoint_error_exit() -> None:
    config = _build_config(endpoint_error_exit_after_seconds=0.0, endpoint_connection_error_retry_wait_seconds=0.0)
    client = _make_client(config)
    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(side_effect=_MockStatusError(400, "Bad request"))
    client._client = mock_openai

    with pytest.raises(_MockStatusError, match="Bad request"):
        asyncio.run(client.acomplete([ConversationMessage.user("test")]))

    assert mock_openai.chat.completions.create.call_count == 1
    assert client._endpoint_error_window_started_at is None


def test_context_length_400_does_not_trigger_endpoint_error_exit() -> None:
    config = _build_config(endpoint_error_exit_after_seconds=0.0)
    client = _make_client(config)
    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(side_effect=_MockStatusError(400, "maximum context length exceeded"))
    client._client = mock_openai

    with pytest.raises(ModelContextLimitError, match="maximum context length"):
        asyncio.run(client.acomplete([ConversationMessage.user("test")]))

    assert client._endpoint_error_window_started_at is None


def test_transient_status_retries_inside_model_call() -> None:
    config = _build_config(endpoint_error_exit_after_seconds=300.0, endpoint_connection_error_retry_wait_seconds=0.0)
    client = _make_client(config)
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content="ok"), finish_reason="stop")],
        usage=_MockUsage(),
    )
    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(side_effect=[_MockStatusError(502, "Bad gateway"), mock_response])
    client._client = mock_openai

    result = asyncio.run(client.acomplete([ConversationMessage.user("test")]))

    assert result.message.content == "ok"
    assert mock_openai.chat.completions.create.call_count == 2
    assert client._endpoint_error_window_started_at is None


def test_429_retry_uses_retry_after_header() -> None:
    config = _build_config(endpoint_error_exit_after_seconds=300.0, endpoint_connection_error_retry_wait_seconds=60.0)
    client = _make_client(config)
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content="ok"), finish_reason="stop")],
        usage=_MockUsage(),
    )
    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[_MockStatusError(429, "Too many requests", headers={"Retry-After": "7"}), mock_response]
    )
    client._client = mock_openai

    sleep_mock = AsyncMock()
    with patch("agentic.model_clients.openai_client.asyncio.sleep", sleep_mock):
        result = asyncio.run(client.acomplete([ConversationMessage.user("test")]))

    assert result.message.content == "ok"
    sleep_mock.assert_awaited_once_with(7.0)


def test_connection_error_retries_inside_model_call() -> None:
    config = _build_config(endpoint_error_exit_after_seconds=300.0, endpoint_connection_error_retry_wait_seconds=0.0)
    client = _make_client(config)
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content="ok"), finish_reason="stop")],
        usage=_MockUsage(),
    )
    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(side_effect=[APIConnectionError("Connection error."), mock_response])
    client._client = mock_openai

    with patch("agentic.model_clients.openai_client.time.perf_counter", side_effect=[0.0, 0.0, 75.0, 80.0]):
        result = asyncio.run(client.acomplete([ConversationMessage.user("test")]))

    assert result.message.content == "ok"
    assert isinstance(result.raw_response, dict)
    assert result.raw_response["client_elapsed_ms_excluding_request_logging"] == pytest.approx(80_000.0)
    assert mock_openai.chat.completions.create.call_count == 2
    assert client._endpoint_error_window_started_at is None


def test_timeout_and_connection_retries_use_exponential_backoff() -> None:
    config = _build_config(
        endpoint_error_exit_after_seconds=300.0,
        endpoint_connection_error_retry_wait_seconds=5.0,
        endpoint_error_retry_backoff_multiplier=3.0,
        endpoint_error_retry_backoff_max_seconds=12.0,
        endpoint_error_retry_jitter=False,
    )
    client = _make_client(config)
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content="ok"), finish_reason="stop")],
        usage=_MockUsage(),
    )
    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[APITimeoutError("Request timed out."), APIConnectionError("Connection error."), mock_response]
    )
    client._client = mock_openai

    sleep_mock = AsyncMock()
    with patch("agentic.model_clients.openai_client.asyncio.sleep", sleep_mock):
        result = asyncio.run(client.acomplete([ConversationMessage.user("test")]))

    assert result.message.content == "ok"
    assert [call.args[0] for call in sleep_mock.await_args_list] == [5.0, 12.0]
    assert mock_openai.chat.completions.create.call_count == 3
    assert client._endpoint_error_window_started_at is None


def test_transient_status_retry_can_apply_equal_jitter() -> None:
    config = _build_config(
        endpoint_error_exit_after_seconds=300.0,
        endpoint_connection_error_retry_wait_seconds=10.0,
        endpoint_error_retry_backoff_multiplier=2.0,
        endpoint_error_retry_backoff_max_seconds=300.0,
        endpoint_error_retry_jitter=True,
    )
    client = _make_client(config)
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content="ok"), finish_reason="stop")],
        usage=_MockUsage(),
    )
    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(side_effect=[_MockStatusError(503, "Unavailable"), mock_response])
    client._client = mock_openai

    sleep_mock = AsyncMock()
    with (
        patch("agentic.model_clients.openai_client.random.uniform", return_value=2.5),
        patch("agentic.model_clients.openai_client.asyncio.sleep", sleep_mock),
    ):
        result = asyncio.run(client.acomplete([ConversationMessage.user("test")]))

    assert result.message.content == "ok"
    sleep_mock.assert_awaited_once_with(7.5)


def test_continuous_transient_endpoint_error_exits_process() -> None:
    config = _build_config(endpoint_error_exit_after_seconds=0.0, endpoint_connection_error_retry_wait_seconds=0.0)
    client = _make_client(config)
    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(side_effect=APIConnectionError("Connection error."))
    client._client = mock_openai

    with pytest.raises(SystemExit, match="Endpoint transient error persisted"):
        asyncio.run(client.acomplete([ConversationMessage.user("test")]))


def test_transient_status_exits_after_no_success_window() -> None:
    config = _build_config(endpoint_error_exit_after_seconds=300.0, endpoint_connection_error_retry_wait_seconds=0.0)
    client = _make_client(config)
    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(side_effect=_MockStatusError(503, "Unavailable"))
    client._client = mock_openai

    with (
        patch.object(client, "_monotonic_now", side_effect=[0.0, 301.0]),
        pytest.raises(SystemExit, match="Endpoint transient error persisted"),
    ):
        asyncio.run(client.acomplete([ConversationMessage.user("test")]))

    assert mock_openai.chat.completions.create.call_count == 2


def test_non_retryable_error_after_transient_resets_endpoint_error_window() -> None:
    config = _build_config(endpoint_error_exit_after_seconds=300.0, endpoint_connection_error_retry_wait_seconds=0.0)
    client = _make_client(config)
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content="ok"), finish_reason="stop")],
        usage=_MockUsage(),
    )
    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[_MockStatusError(502, "Bad gateway"), _MockStatusError(400, "Bad request"), _MockStatusError(502, "Bad gateway"), mock_response]
    )
    client._client = mock_openai

    with patch.object(client, "_monotonic_now", side_effect=[0.0, 301.0]):
        with pytest.raises(_MockStatusError, match="Bad request"):
            asyncio.run(client.acomplete([ConversationMessage.user("first")]))
        result = asyncio.run(client.acomplete([ConversationMessage.user("second")]))

    assert result.message.content == "ok"
    assert mock_openai.chat.completions.create.call_count == 4
    assert client._endpoint_error_window_started_at is None


def test_connection_timeout_does_not_count_request_wait_toward_exit_window() -> None:
    config = _build_config(endpoint_error_exit_after_seconds=300.0, endpoint_connection_error_retry_wait_seconds=0.0)
    client = _make_client(config)
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content="ok"), finish_reason="stop")],
        usage=_MockUsage(),
    )
    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(side_effect=[APIConnectionError("Connection error."), mock_response])
    client._client = mock_openai

    with patch.object(client, "_monotonic_now", return_value=1_800.0):
        result = asyncio.run(client.acomplete([ConversationMessage.user("test")]))

    assert result.message.content == "ok"
    assert mock_openai.chat.completions.create.call_count == 2
    assert client._endpoint_error_window_started_at is None


def test_success_resets_transient_endpoint_error_window() -> None:
    config = _build_config(endpoint_error_exit_after_seconds=300.0, endpoint_connection_error_retry_wait_seconds=0.0)
    client = _make_client(config)
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content="ok"), finish_reason="stop")],
        usage=_MockUsage(),
    )
    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[_MockStatusError(502, "Bad gateway"), mock_response, _MockStatusError(502, "Bad gateway"), mock_response]
    )
    client._client = mock_openai

    with patch.object(client, "_monotonic_now", side_effect=[0.0, 301.0]):
        first = asyncio.run(client.acomplete([ConversationMessage.user("first")]))
        second = asyncio.run(client.acomplete([ConversationMessage.user("second")]))

    assert first.message.content == "ok"
    assert second.message.content == "ok"
    assert client._endpoint_error_window_started_at is None


def test_default_transient_status_codes_exclude_400() -> None:
    config = _build_config()

    assert 400 not in config.endpoint_error_exit_status_codes
    assert tuple(config.endpoint_error_exit_status_codes) == DEFAULT_TRANSIENT_ENDPOINT_ERROR_STATUS_CODES


def test_custom_transient_status_code_retries_inside_model_call() -> None:
    config = _build_config(
        endpoint_error_exit_status_codes=[520],
        endpoint_error_exit_after_seconds=300.0,
        endpoint_connection_error_retry_wait_seconds=0.0,
    )
    client = _make_client(config)
    mock_response = _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content="ok"), finish_reason="stop")],
        usage=_MockUsage(),
    )
    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(side_effect=[_MockStatusError(520, "Provider transient"), mock_response])
    client._client = mock_openai

    result = asyncio.run(client.acomplete([ConversationMessage.user("test")]))

    assert result.message.content == "ok"
    assert mock_openai.chat.completions.create.call_count == 2
    assert client._endpoint_error_window_started_at is None


def test_context_length_error_parses_provider_token_counts() -> None:
    error = ModelContextLimitError.from_exception(
        _MockStatusError(400, "BadRequest: input 245854 + output 16384 = 262238 > 262144"),
        status_code=400,
    )

    assert error.status_code == 400
    assert error.input_tokens == 245854
    assert error.requested_output_tokens == 16384
    assert error.total_tokens == 262238
    assert error.context_window == 262144


def test_length_retry_caps_max_tokens_by_remaining_context() -> None:
    inner = _LengthRetryCaptureClient()
    client = RetryingModelClient(inner, max_retries=2, retry_wait_seconds=0, context_safety_margin=5)

    asyncio.run(client.acomplete([ConversationMessage.user("test")]))

    assert inner.max_tokens_seen == [5, 5]


def test_length_retry_uses_injected_context_token_estimator() -> None:
    inner = _LengthRetryCaptureClient()
    client = RetryingModelClient(inner, max_retries=2, retry_wait_seconds=0, context_safety_margin=5)
    client.set_context_token_estimator(lambda _messages: 80)

    asyncio.run(client.acomplete([ConversationMessage.user("test")]))

    assert inner.max_tokens_seen == [15, 15]


def test_length_retry_can_skip_local_context_token_cap() -> None:
    inner = _LengthRetryCaptureClient()
    client = RetryingModelClient(
        inner,
        max_retries=2,
        retry_wait_seconds=0,
        context_safety_margin=5,
        cap_max_tokens_for_context=False,
    )

    asyncio.run(client.acomplete([ConversationMessage.user("test")]))

    assert inner.max_tokens_seen == [50, 55]
