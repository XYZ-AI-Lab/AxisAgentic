# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Generic rollout model client for training-integration use cases.

This module provides the extension point used by external RL training loops to
plug into the agentic runtime.
The key pieces:

* :class:`TokenRecorder` — ``Protocol`` describing how generated and appended tokens are reported back to the external training loop.
* :class:`GenerationResult` — the minimal, framework-agnostic shape of a rollout worker's response (ids, logprobs, text, finish reason).
* :class:`RolloutModelClient` — ``ModelClient`` subclass that
  asks a subclass-provided ``_tokenize_and_extend`` hook for the newly-appended prefix tokens,
  calls ``_generate`` for the next turn's tokens,
  and routes both deltas through a :class:`TokenRecorder` so an external trainer can reconstruct the loss mask.

``agentic/`` imposes no dependency on any training framework; concrete integrations
live outside this package and supply the tokenizer/worker hooks.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from agentic.contracts import ConversationMessage, ModelResponse, TokenUsage
from agentic.model_clients.base import ModelClient

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agentic.contracts.messages import ToolCall


@runtime_checkable
class TokenRecorder(Protocol):
    """Sink for tokens produced during an agentic rollout.

    Implementations must be cheap and synchronous.
    The runtime calls :meth:`record_appended` for every prefix/tool-result extension
    and :meth:`record_generation` for every assistant generation
    so the caller can reconstruct a loss mask that marks only generated tokens as trainable.
    """

    def record_appended(self, token_ids: Sequence[int]) -> None:
        """Record non-generated tokens (system / user / tool) appended to the prefix."""

    def record_generation(self, token_ids: Sequence[int], logprobs: Sequence[float]) -> None:
        """Record tokens produced by the model for the most recent turn."""


@dataclass(slots=True)
class GenerationResult:
    """Framework-agnostic shape of a single rollout-worker response.

    Mirrors the fields a GRPO-style training pipeline needs:
    generated token ids, matching per-token logprobs, decoded text, and the provider's finish reason.
    ``tool_calls`` carries any structured tool calls the rollout worker has already parsed out of the raw model output
    (e.g. SGLang's :class:`FunctionCallParser`); the base class attaches them to the assistant :class:`ConversationMessage`
    so the orchestrator can dispatch them. Leave as ``None`` when the subclass has no parsed calls to offer.
    ``extra`` holds anything the subclass wants to round-trip for observability (e.g. latency, cached tokens, raw provider response).
    """

    output_ids: list[int]
    output_logprobs: list[float]
    output_text: str
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    tool_calls: list[ToolCall] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class RolloutModelClient(ModelClient):
    """Abstract ``ModelClient`` that reports every tokenization step to a recorder.

    Subclasses implement two hooks:

    * :meth:`_tokenize_and_extend` — tokenize any messages the runtime has appended since the previous turn,
      extend the subclass's internal prefix with those ids, and return just the delta so the recorder can mark them as "not in loss".
    * :meth:`_generate` — call the underlying rollout worker on the subclass-owned prefix and return a :class:`GenerationResult`.
      The subclass is responsible for extending its own prefix with the generated output ids before returning.

    The implementation of :meth:`acomplete` never touches token state directly;
    it just wires the hooks together and dispatches to the recorder.
    """

    def __init__(
        self,
        *,
        context_window: int,
        max_output_tokens: int,
        enable_tool_calls: bool = True,
        recorder: TokenRecorder | None = None,
    ) -> None:
        super().__init__(
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            enable_tool_calls=enable_tool_calls,
        )
        self._recorder = recorder

    @property
    def recorder(self) -> TokenRecorder | None:
        return self._recorder

    def bind_recorder(self, recorder: TokenRecorder | None) -> None:
        """Attach or detach a recorder.

        Useful when the client is constructed once per rollout loop but the recorder is per-episode (as in the GRPO ``TokenTrace`` case).
        """
        self._recorder = recorder

    @abstractmethod
    async def _tokenize_and_extend(self, messages: list[ConversationMessage]) -> list[int]:
        """Tokenize newly-appended messages, extend internal prefix, and return the delta."""

    @abstractmethod
    async def _generate(self) -> GenerationResult:
        """Run the rollout worker on the subclass-owned prefix and return its result."""

    async def acomplete(
        self,
        messages: list[ConversationMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> ModelResponse:
        del tools, tool_choice  # Rollout workers carry tool wiring via their own channels.

        appended_ids = await self._tokenize_and_extend(messages)
        if appended_ids and self._recorder is not None:
            self._recorder.record_appended(appended_ids)

        result = await self._generate()

        if self._recorder is not None:
            self._recorder.record_generation(result.output_ids, result.output_logprobs)

        return self._build_model_response(result)

    def _build_model_response(self, result: GenerationResult) -> ModelResponse:
        """Wrap a :class:`GenerationResult` as a :class:`ModelResponse` for the runtime."""
        message = ConversationMessage.assistant(result.output_text or None, tool_calls=result.tool_calls)
        return ModelResponse(
            message=message,
            finish_reason=result.finish_reason,
            usage=result.usage,
            raw_response=result.extra or None,
        )
