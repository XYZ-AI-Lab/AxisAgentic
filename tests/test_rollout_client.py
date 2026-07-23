"""Unit tests for :mod:`agentic.rl.rollout_client` using local stubs.

Exercises the recorder protocol, prefix extension, and recorder dispatch
order across multiple turns using an in-process stub subclass.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from agentic.contracts import ConversationMessage, TokenUsage
from agentic.rl import GenerationResult, RolloutModelClient

if TYPE_CHECKING:
    from collections.abc import Sequence


class ListRecorder:
    """``TokenRecorder`` backed by two parallel lists for verification."""

    def __init__(self) -> None:
        self.events: list[tuple[str, list[int], list[float]]] = []

    def record_appended(self, token_ids: Sequence[int]) -> None:
        self.events.append(("appended", list(token_ids), []))

    def record_generation(self, token_ids: Sequence[int], logprobs: Sequence[float]) -> None:
        self.events.append(("generation", list(token_ids), list(logprobs)))


class StubRolloutClient(RolloutModelClient):
    """Deterministic stub for recorder dispatch tests.

    Each appended message contributes a fixed id per character, and each generation
    returns a canned token sequence keyed by turn index.
    """

    def __init__(self, gen_outputs: list[GenerationResult]) -> None:
        super().__init__(context_window=2048, max_output_tokens=128)
        self._gen_outputs = gen_outputs
        self._turn = 0
        self._prefix_ids: list[int] = []
        self._known_messages: list[ConversationMessage] = []

    async def _tokenize_and_extend(self, messages: list[ConversationMessage]) -> list[int]:
        new = messages[len(self._known_messages) :]
        self._known_messages = list(messages)
        delta: list[int] = []
        for msg in new:
            text = msg.content or ""
            delta.extend(ord(c) for c in text)
        self._prefix_ids.extend(delta)
        return delta

    async def _generate(self) -> GenerationResult:
        result = self._gen_outputs[self._turn]
        self._turn += 1
        self._prefix_ids.extend(result.output_ids)
        return result


def test_rollout_client_records_appended_and_generation_in_order() -> None:
    recorder = ListRecorder()
    client = StubRolloutClient(
        gen_outputs=[
            GenerationResult(output_ids=[100, 101], output_logprobs=[-0.1, -0.2], output_text="ok"),
            GenerationResult(output_ids=[200], output_logprobs=[-0.3], output_text="done", finish_reason="stop"),
        ],
    )
    client.bind_recorder(recorder)

    # Turn 1: system + user already built by the runtime.
    messages_turn1 = [
        ConversationMessage.system("ab"),
        ConversationMessage.user("c"),
    ]
    response1 = asyncio.run(client.acomplete(messages_turn1))
    assert response1.message.content == "ok"
    assert response1.finish_reason is None

    # Turn 2: runtime appends an assistant + tool result, then asks for the next generation.
    messages_turn2 = [
        *messages_turn1,
        ConversationMessage.assistant("ok"),
        ConversationMessage.tool("d", tool_call_id="call_1", name="echo"),
    ]
    response2 = asyncio.run(client.acomplete(messages_turn2))
    assert response2.finish_reason == "stop"

    # Event ordering check: recorder saw the delta append BEFORE the generation
    # for each turn, and the appended lengths match the per-character tokenizer.
    kinds = [event[0] for event in recorder.events]
    assert kinds == ["appended", "generation", "appended", "generation"]

    appended_turn1 = recorder.events[0][1]
    gen_turn1 = recorder.events[1][1]
    appended_turn2 = recorder.events[2][1]
    gen_turn2 = recorder.events[3][1]

    assert appended_turn1 == [ord("a"), ord("b"), ord("c")]
    assert gen_turn1 == [100, 101]
    # Turn 2 appends the assistant "ok" (2 tokens) + tool "d" (1 token) = 3 new ids.
    assert appended_turn2 == [ord("o"), ord("k"), ord("d")]
    assert gen_turn2 == [200]


def test_rollout_client_skips_recorder_when_unbound() -> None:
    client = StubRolloutClient(
        gen_outputs=[GenerationResult(output_ids=[1], output_logprobs=[0.0], output_text="x")],
    )
    # No recorder bound — must not raise.
    asyncio.run(client.acomplete([ConversationMessage.user("hi")]))


def test_generation_result_extra_passed_through_on_model_response() -> None:
    recorder = ListRecorder()
    extras = {"e2e_seconds": 0.5, "cached_tokens": 3}
    client = StubRolloutClient(
        gen_outputs=[
            GenerationResult(
                output_ids=[1],
                output_logprobs=[-0.1],
                output_text="hi",
                usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                extra=extras,
            ),
        ],
    )
    client.bind_recorder(recorder)
    response = asyncio.run(client.acomplete([ConversationMessage.user("y")]))
    assert response.raw_response == extras
    assert response.usage is not None
    assert response.usage.input_tokens == 1


def test_rollout_client_ignores_empty_append_on_first_turn_when_messages_reused() -> None:
    """Verify idempotency of empty-response retry loops.

    If the runtime calls acomplete with identical messages twice, the second
    call records no appended tokens.
    """
    recorder = ListRecorder()
    client = StubRolloutClient(
        gen_outputs=[
            GenerationResult(output_ids=[1], output_logprobs=[0.0], output_text="a"),
            GenerationResult(output_ids=[2], output_logprobs=[0.0], output_text="b"),
        ],
    )
    client.bind_recorder(recorder)
    messages = [ConversationMessage.user("hi")]
    asyncio.run(client.acomplete(messages))
    asyncio.run(client.acomplete(messages))  # No new messages appended.
    kinds = [event[0] for event in recorder.events]
    # Only the first call produced a non-empty delta, so the second skips record_appended.
    assert kinds == ["appended", "generation", "generation"]
