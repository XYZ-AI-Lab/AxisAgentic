# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Discard-all context-management policy.

The discard-all strategy (DeepSeek-V3.2 / Kimi K2.5 / Qwen3.5 BrowseComp): when
the observed prompt-token usage exceeds a fraction of the context window, reset
the conversation by discarding *all* previous tool-call history and reopening
the task from a clean context (system instructions + the initial question).

Unlike Summary-style ``context_compression`` this performs no summarization and
injects no replacement message; it appends a ``discard_all`` CONVERSATION_RUNTIME
marker that, on replay, truncates the model-visible conversation to its leading
prefix. ``_full_conversation`` stays append-only, so persisted traces still
reconstruct the exact model-visible prompt (and SFT export stays byte-identical).

This manager is a stateless policy: it owns *when* to reset (``should_trigger``)
and *what* prefix to keep (``build_marker``). All mutable counters — last trigger
turn, reset count, and the global tool-call count — live on the orchestrator so
attempt resets and attempt-budget branch snapshots remain consistent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agentic.contracts import ConversationMessage, MessageRole, build_discard_all_marker

logger = logging.getLogger(__name__)

DEFAULT_DISCARD_ALL_TRIGGER_RATIO = 0.80
DEFAULT_DISCARD_ALL_MIN_TURNS_BETWEEN = 3
DEFAULT_DISCARD_ALL_MAX_TOOL_CALLS = 1800


@dataclass
class DiscardAllConfig:
    enabled: bool = False
    # Reset when observed prompt tokens exceed ``trigger_ratio`` of the context
    # window. The token usage signal is the provider-reported prompt tokens of
    # the most recent model call (the literal "token usage").
    trigger_ratio: float = DEFAULT_DISCARD_ALL_TRIGGER_RATIO
    # Minimum assistant turns between two resets; guards against a prefix that
    # alone approaches the threshold thrashing into a reset every turn.
    min_turns_between: int = DEFAULT_DISCARD_ALL_MIN_TURNS_BETWEEN
    # Global tool-call cap across the whole attempt; it does NOT reset on a
    # discard. Supersedes ``max_turns`` when discard-all is enabled, and on hit
    # forces a graceful final answer rather than a hard termination.
    max_tool_calls: int = DEFAULT_DISCARD_ALL_MAX_TOOL_CALLS


class DiscardAllManager:
    """Stateless discard-all policy (trigger decision + prefix selection)."""

    def __init__(self, cfg: DiscardAllConfig) -> None:
        if not 0.0 < cfg.trigger_ratio <= 1.0:
            msg = "discard_all trigger_ratio must be in (0, 1]"
            raise ValueError(msg)
        if cfg.min_turns_between < 0:
            msg = "discard_all min_turns_between must be >= 0"
            raise ValueError(msg)
        if cfg.max_tool_calls < 1:
            msg = "discard_all max_tool_calls must be >= 1"
            raise ValueError(msg)
        self.cfg = cfg

    @property
    def max_tool_calls(self) -> int:
        return self.cfg.max_tool_calls

    def should_trigger(
        self,
        *,
        observed_prompt_tokens: int | None,
        context_window: int | None,
        turn_count: int,
        last_trigger_turn: int | None,
    ) -> bool:
        """Return True when token usage warrants a discard-all reset.

        ``observed_prompt_tokens`` is the provider-reported prompt-token count of
        the most recent model call; ``context_window`` is the model's context
        limit. The reset fires when usage exceeds ``trigger_ratio * window``.

        ``last_trigger_turn`` is the turn of the previous reset, or ``None`` when
        no reset has happened yet. The ``min_turns_between`` cooldown only applies
        *between* resets, so the very first reset can fire as soon as usage
        crosses the threshold (an early oversized prompt must not be blocked
        until the cooldown elapses).
        """
        if not self.cfg.enabled:
            return False
        if context_window is None or context_window <= 0:
            return False
        if observed_prompt_tokens is None or observed_prompt_tokens <= 0:
            return False
        if last_trigger_turn is not None and turn_count - last_trigger_turn < self.cfg.min_turns_between:
            return False
        return observed_prompt_tokens > self.cfg.trigger_ratio * context_window

    def build_marker(self, visible_conversation: list[ConversationMessage]) -> ConversationMessage | None:
        """Build a discard-all marker keeping the system+task prefix.

        Returns ``None`` when there is nothing to discard (the visible
        conversation is already just the prefix, i.e. no tool-call turns have
        accumulated since the last reset), so a near-threshold prefix cannot
        force a no-progress reset every turn.
        """
        prefix_len = self._prefix_len(visible_conversation)
        if prefix_len >= len(visible_conversation):
            return None
        return build_discard_all_marker(prefix_len=prefix_len)

    @staticmethod
    def _prefix_len(visible_conversation: list[ConversationMessage]) -> int:
        """Leading system messages plus the first user message (the task)."""
        n = len(visible_conversation)
        prefix_len = 0
        while prefix_len < n and visible_conversation[prefix_len].role == MessageRole.SYSTEM:
            prefix_len += 1
        if prefix_len < n and visible_conversation[prefix_len].role == MessageRole.USER:
            prefix_len += 1
        return prefix_len
