# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Context compression manager.

Owns when to fire compression and how to rebuild the conversation history with
the summary inserted in place of compressed earlier turns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agentic.contracts import ConversationMessage, MessageRole, build_compaction_marker
from recipe.web_search.agent.context_compression_tool import (
    ContextCompressionTool,
    format_messages_for_compression,
)

logger = logging.getLogger(__name__)


_INTERNAL_SUMMARY_MARKER = "[context_compression_summary]"


@dataclass
class ContextCompressionConfig:
    enabled: bool = False
    interval: int = 10  # fire every N assistant turns
    recent_window: int = 10  # keep the last N turns intact (1 turn ~ 2 messages)


class ContextCompressionManager:
    """Periodically replaces older messages with a single summary block."""

    def __init__(
        self,
        cfg: ContextCompressionConfig,
        *,
        base_url: str,
        model_name: str,
        api_key: str,
    ) -> None:
        if cfg.interval < 1:
            msg = "context_compression interval must be >= 1"
            raise ValueError(msg)
        if cfg.recent_window < 1:
            msg = "context_compression recent_window must be >= 1"
            raise ValueError(msg)
        self.cfg = cfg
        self.tool = ContextCompressionTool(base_url=base_url, model_name=model_name, api_key=api_key)
        self._previous_summary: str = "(none yet)"
        self._last_trigger_turn: int = 0

    def should_trigger(self, turn_count: int) -> bool:
        if not self.cfg.enabled:
            return False
        if turn_count < self.cfg.interval:
            return False
        if turn_count == self._last_trigger_turn:
            return False
        return (turn_count % self.cfg.interval) == 0

    async def maybe_compress(
        self,
        turn_count: int,
        visible_conversation: list[ConversationMessage],
        task_text: str,
    ) -> tuple[ConversationMessage | None, str | None]:
        """Compress earlier visible messages into an append-only compaction marker.

        If a trigger fires, return the marker plus the new summary text;
        otherwise return ``(None, None)``.

        ``visible_conversation`` is the current model-visible conversation (the
        prompt), not the append-only ``_full_conversation`` log. The marker is a
        CONVERSATION_RUNTIME message recording the visible range it replaces; the
        runtime appends it and derives the compacted prompt by replaying it, so
        the full conversation only ever grows.
        """
        if not self.should_trigger(turn_count):
            return None, None

        self._last_trigger_turn = turn_count

        prefix, earlier, recent = self._split_history(visible_conversation)
        # Drop manager-injected summary messages from new_context: V5 contract is
        # that old summaries flow only through previous_summary, not as a duplicate
        # delta that re-feeds the model what it already produced.
        earlier_for_compression = [msg for msg in earlier if not _is_internal_summary_message(msg)]
        if not earlier_for_compression:
            logger.info(
                "[ContextCompression] turn=%d skip: no earlier messages to compress (total=%d)",
                turn_count,
                len(visible_conversation),
            )
            return None, None

        new_context_text = format_messages_for_compression(earlier_for_compression)
        task_context_text = task_text.strip() or "(none)"

        logger.info(
            "[ContextCompression] turn=%d compressing %d earlier messages (prefix=%d, recent=%d, previous_summary_chars=%d, new_context_chars=%d)",
            turn_count,
            len(earlier_for_compression),
            len(prefix),
            len(recent),
            len(self._previous_summary),
            len(new_context_text),
        )
        result = await self.tool.call(
            previous_summary=self._previous_summary,
            task_context=task_context_text,
            new_context=new_context_text,
        )
        if not result.get("success"):
            logger.info(
                "[ContextCompression] turn=%d compression failed: %s",
                turn_count,
                result.get("error", "unknown"),
            )
            return None, None

        summary_text = (result.get("summary") or "").strip()
        if not summary_text:
            logger.info("[ContextCompression] turn=%d empty summary", turn_count)
            return None, None

        self._previous_summary = summary_text
        # Append-only contract: record the compaction as a CONVERSATION_RUNTIME
        # marker instead of rewriting history. Replaying the marker replaces the
        # visible range [prefix_len : prefix_len + compressed_count] with a single
        # summary message; every original message stays in _full_conversation, so
        # the persisted trace can reconstruct the exact model-visible prompt.
        marker = build_compaction_marker(
            summary=f"{_INTERNAL_SUMMARY_MARKER}\n{summary_text}",
            prefix_len=len(prefix),
            compressed_count=len(earlier),
        )

        logger.info(
            "[ContextCompression] turn=%d compaction marker: prefix=%d compressed=%d recent=%d",
            turn_count,
            len(prefix),
            len(earlier),
            len(recent),
        )
        return marker, summary_text

    def _split_history(
        self,
        full_conversation: list[ConversationMessage],
    ) -> tuple[list[ConversationMessage], list[ConversationMessage], list[ConversationMessage]]:
        """Return ``(prefix, earlier, recent)``.

        ``prefix`` is the leading system messages plus the first user message
        (the task description) and is never compressed. ``earlier`` is the
        middle slice that will be summarized. ``recent`` is the trailing
        ``recent_window`` turns (~ 2 messages each) that are kept intact.
        """
        if not full_conversation:
            return [], [], []

        prefix: list[ConversationMessage] = []
        idx = 0
        while idx < len(full_conversation) and full_conversation[idx].role == MessageRole.SYSTEM:
            prefix.append(full_conversation[idx])
            idx += 1
        if idx < len(full_conversation) and full_conversation[idx].role == MessageRole.USER:
            prefix.append(full_conversation[idx])
            idx += 1

        window = self.cfg.recent_window * 2
        remaining = full_conversation[idx:]
        if len(remaining) <= window:
            return prefix, [], remaining

        split_point = len(remaining) - window
        earlier = remaining[:split_point]
        recent = remaining[split_point:]
        return prefix, earlier, recent


def _is_internal_summary_message(message: ConversationMessage) -> bool:
    """Return True for manager-injected ``[context_compression_summary]`` blocks."""
    if message.role != MessageRole.ASSISTANT:
        return False
    if message.tool_calls:
        return False
    content = (message.content or "").lstrip()
    return content.startswith(_INTERNAL_SUMMARY_MARKER)
