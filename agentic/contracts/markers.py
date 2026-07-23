# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Shared schema for CONVERSATION_RUNTIME context-management markers.

A compaction marker is the append-only record of a Summary-style context event:
the visible range ``[prefix_len : prefix_len + compressed_count]`` was replaced
by a single assistant message whose content is ``summary``.

A discard-all marker is the append-only record of a discard-all reset: the
visible conversation was truncated to its leading ``prefix_len`` messages
(system instructions + the initial task), discarding every later tool-call
turn so the model reopens the task with a clean context. Unlike compaction it
injects no replacement message.

This module is the single owner of both wire formats — the producers (the
context-compression / discard-all managers), the live replay
(ConversationRuntime), the offline replay (task orchestrator, trace tooling),
and tests must all build/parse/splice through these helpers so the live prompt
derivation and persisted-trace reconstruction can never drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agentic.contracts.messages import ConversationMessage, MessageRole

COMPACTION_MARKER_NAME = "compaction"
DISCARD_ALL_MARKER_NAME = "discard_all"


@dataclass(frozen=True, slots=True)
class CompactionMarker:
    """Parsed compaction payload describing a visible-range splice."""

    summary: str
    prefix_len: int
    compressed_count: int


def build_compaction_marker(*, summary: str, prefix_len: int, compressed_count: int) -> ConversationMessage:
    """Build the CONVERSATION_RUNTIME message recording a compaction splice."""
    payload = {"summary": summary, "prefix_len": prefix_len, "compressed_count": compressed_count}
    return ConversationMessage.runtime(json.dumps(payload), name=COMPACTION_MARKER_NAME)


def _parse_compaction_payload(content: str | None) -> CompactionMarker | None:
    try:
        payload = json.loads(content) if content is not None else None
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    prefix_len = payload.get("prefix_len")
    compressed_count = payload.get("compressed_count")
    # bool is an int subclass; reject it explicitly so corrupted payloads
    # cannot silently splice at a position the producer never wrote.
    if not isinstance(summary, str):
        return None
    if not isinstance(prefix_len, int) or isinstance(prefix_len, bool):
        return None
    if not isinstance(compressed_count, int) or isinstance(compressed_count, bool):
        return None
    if prefix_len < 0 or compressed_count <= 0:
        return None
    return CompactionMarker(summary=summary, prefix_len=prefix_len, compressed_count=compressed_count)


def parse_compaction_marker(message: ConversationMessage) -> CompactionMarker | None:
    """Parse a compaction marker message; ``None`` for non-compaction or malformed messages."""
    if message.role != MessageRole.CONVERSATION_RUNTIME or message.name != COMPACTION_MARKER_NAME:
        return None
    return _parse_compaction_payload(message.content)


def parse_compaction_marker_dict(message: dict[str, Any]) -> CompactionMarker | None:
    """Parse a compaction marker from trace-dict form (``to_model_message`` output)."""
    if message.get("role") != MessageRole.CONVERSATION_RUNTIME.value or message.get("name") != COMPACTION_MARKER_NAME:
        return None
    content = message.get("content")
    return _parse_compaction_payload(content if isinstance(content, str) else None)


def splice_compaction(
    visible: list[ConversationMessage],
    compaction: CompactionMarker,
) -> list[ConversationMessage] | None:
    """Apply a compaction splice to a visible conversation.

    Returns the new visible list, or ``None`` when the recorded range does not
    fit *visible* (a malformed or stale marker) — callers treat that as a no-op.
    """
    boundary = compaction.prefix_len + compaction.compressed_count
    if boundary > len(visible):
        return None
    summary_message = ConversationMessage.assistant(content=compaction.summary)
    return [*visible[: compaction.prefix_len], summary_message, *visible[boundary:]]


@dataclass(frozen=True, slots=True)
class DiscardAllMarker:
    """Parsed discard-all payload describing a truncate-to-prefix reset."""

    prefix_len: int


def build_discard_all_marker(*, prefix_len: int) -> ConversationMessage:
    """Build the CONVERSATION_RUNTIME message recording a discard-all reset."""
    payload = {"prefix_len": prefix_len}
    return ConversationMessage.runtime(json.dumps(payload), name=DISCARD_ALL_MARKER_NAME)


def _parse_discard_all_payload(content: str | None) -> DiscardAllMarker | None:
    try:
        payload = json.loads(content) if content is not None else None
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    prefix_len = payload.get("prefix_len")
    # bool is an int subclass; reject it explicitly so corrupted payloads
    # cannot silently truncate at a position the producer never wrote.
    if not isinstance(prefix_len, int) or isinstance(prefix_len, bool):
        return None
    if prefix_len < 0:
        return None
    return DiscardAllMarker(prefix_len=prefix_len)


def parse_discard_all_marker(message: ConversationMessage) -> DiscardAllMarker | None:
    """Parse a discard-all marker message; ``None`` for non-discard or malformed messages."""
    if message.role != MessageRole.CONVERSATION_RUNTIME or message.name != DISCARD_ALL_MARKER_NAME:
        return None
    return _parse_discard_all_payload(message.content)


def parse_discard_all_marker_dict(message: dict[str, Any]) -> DiscardAllMarker | None:
    """Parse a discard-all marker from trace-dict form (``to_model_message`` output)."""
    if message.get("role") != MessageRole.CONVERSATION_RUNTIME.value or message.get("name") != DISCARD_ALL_MARKER_NAME:
        return None
    content = message.get("content")
    return _parse_discard_all_payload(content if isinstance(content, str) else None)


def splice_discard_all(
    visible: list[ConversationMessage],
    discard: DiscardAllMarker,
) -> list[ConversationMessage] | None:
    """Apply a discard-all reset to a visible conversation.

    Returns ``visible[:prefix_len]`` (the retained prefix), or ``None`` when the
    recorded prefix does not fit *visible* (a malformed or stale marker) — callers
    treat that as a no-op. A ``prefix_len`` equal to ``len(visible)`` is a valid
    no-op reset (nothing left to discard) and returns the unchanged list.
    """
    if discard.prefix_len > len(visible):
        return None
    return list(visible[: discard.prefix_len])
