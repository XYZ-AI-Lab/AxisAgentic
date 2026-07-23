# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentic.contracts import ConversationMessage, MessageRole

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(slots=True)
class _TrackedSegment:
    messages: list[ConversationMessage]
    estimated_tokens: int


class ContextLengthTracker:
    """Maintain an incremental token estimate for the current visible conversation.

    The tracker is intentionally chat-template agnostic. Callers should provide
    only additive estimators, where token counts for appended message diffs can
    be summed safely. Chat-template renderers are usually non-additive because
    each standalone render may add prompt/template overhead. Pass ``None`` when
    only visibility tracking is needed; callers can still use a full-estimate
    fallback at their boundary.
    """

    def __init__(self, estimate_tokens: Callable[[list[ConversationMessage]], int] | None = None) -> None:
        self._estimate_tokens = estimate_tokens
        self._visible: list[_TrackedSegment] = []
        self._estimated_tokens = 0
        self._message_count = 0
        self._calibration_ratio = 1.0
        self._calibration_alpha = 0.2
        self._calibration_ratio_min = 0.25
        self._calibration_ratio_max = 4.0
        self._anchor_message_count: int | None = None
        self._anchor_raw_tokens = 0
        self._anchor_observed_tokens = 0

    @property
    def estimated_tokens(self) -> int:
        return self._calibrate_tokens(self._estimated_tokens)

    @property
    def raw_estimated_tokens(self) -> int:
        return self._estimated_tokens

    @property
    def calibration_ratio(self) -> float:
        return self._calibration_ratio

    def clone(self, *, estimate_tokens: Callable[[list[ConversationMessage]], int] | None = None) -> ContextLengthTracker:
        cloned = ContextLengthTracker(self._estimate_tokens if estimate_tokens is None else estimate_tokens)
        cloned._visible = [_TrackedSegment(messages=list(segment.messages), estimated_tokens=segment.estimated_tokens) for segment in self._visible]
        cloned._estimated_tokens = self._estimated_tokens
        cloned._message_count = self._message_count
        cloned._calibration_ratio = self._calibration_ratio
        cloned._calibration_alpha = self._calibration_alpha
        cloned._calibration_ratio_min = self._calibration_ratio_min
        cloned._calibration_ratio_max = self._calibration_ratio_max
        cloned._anchor_message_count = self._anchor_message_count
        cloned._anchor_raw_tokens = self._anchor_raw_tokens
        cloned._anchor_observed_tokens = self._anchor_observed_tokens
        return cloned

    def reset(self) -> None:
        self._visible.clear()
        self._estimated_tokens = 0
        self._message_count = 0
        self._calibration_ratio = 1.0
        self._clear_anchor()

    def visible_messages(self) -> list[ConversationMessage]:
        return [message for segment in self._visible for message in segment.messages]

    def append(self, message: ConversationMessage) -> None:
        self.append_many([message])

    def append_many(self, messages: list[ConversationMessage]) -> None:
        if not messages:
            return
        estimated_tokens = self._estimate_messages(messages)
        self._visible.append(_TrackedSegment(messages=list(messages), estimated_tokens=estimated_tokens))
        self._estimated_tokens += estimated_tokens
        self._message_count += len(messages)

    def rollback_last(self, count: int, *, preserve_system: bool = True) -> int:
        """Remove up to *count* trailing visible messages and return the number removed."""
        removed = 0
        remaining = max(0, count)
        while remaining and self._visible:
            segment = self._visible[-1]
            removable = self._count_removable_tail_messages(segment.messages, remaining, preserve_system=preserve_system)
            if removable == 0:
                break
            if removable == len(segment.messages):
                self._visible.pop()
                self._estimated_tokens -= segment.estimated_tokens
            else:
                removed_messages = segment.messages[-removable:]
                segment.messages[:] = segment.messages[:-removable]
                removed_tokens = self._estimate_messages(removed_messages)
                segment.estimated_tokens = max(0, segment.estimated_tokens - removed_tokens)
                self._estimated_tokens -= removed_tokens
            removed += removable
            remaining -= removable
            self._message_count -= removable
        self._estimated_tokens = max(0, self._estimated_tokens)
        self._message_count = max(0, self._message_count)
        if self._anchor_message_count is not None and self._message_count < self._anchor_message_count:
            self._clear_anchor()
        return removed

    def replace_visible(self, messages: list[ConversationMessage]) -> None:
        """Replace the tracked visible conversation wholesale (e.g. a compaction splice).

        Keeps the learned ``_calibration_ratio`` — it captures estimator-vs-provider
        scale, which is independent of which messages are visible — but drops the
        observed-token anchor because the anchored prefix no longer exists.
        """
        self._visible.clear()
        self._estimated_tokens = 0
        self._message_count = 0
        self._clear_anchor()
        self.append_many(messages)

    def estimate_with(self, additional_messages: list[ConversationMessage] | None = None) -> int:
        raw_tokens = self.raw_estimate_with(additional_messages)
        anchored = self._estimate_from_anchor(raw_tokens)
        if anchored is not None:
            return anchored
        return self._calibrate_tokens(raw_tokens)

    def raw_estimate_with(self, additional_messages: list[ConversationMessage] | None = None) -> int:
        if not additional_messages:
            return self._estimated_tokens
        return self._estimated_tokens + self._estimate_messages(additional_messages)

    def calibrate(self, *, observed_tokens: int, raw_estimated_tokens: int | None = None) -> None:
        """Update token calibration from provider usage.

        Provider prompt-token counts are the best available signal for the
        active serving stack. Let them correct both cheap-estimator undercounts
        and overcounts; the broad bounds only guard against malformed usage.
        """
        raw_tokens = self._estimated_tokens if raw_estimated_tokens is None else raw_estimated_tokens
        if observed_tokens <= 0 or raw_tokens <= 0:
            return
        observed_ratio = observed_tokens / raw_tokens
        bounded_ratio = min(
            self._calibration_ratio_max,
            max(self._calibration_ratio_min, observed_ratio),
        )
        self._calibration_ratio = (1.0 - self._calibration_alpha) * self._calibration_ratio + self._calibration_alpha * bounded_ratio

    def anchor_observed_tokens(self, *, observed_tokens: int, raw_estimated_tokens: int | None = None) -> None:
        """Anchor the current visible prefix to provider-reported prompt tokens.

        Future estimates reuse the observed token count for the already-sent
        prefix and only apply the cheap estimator to appended diffs.
        """
        raw_tokens = self._estimated_tokens if raw_estimated_tokens is None else raw_estimated_tokens
        if observed_tokens <= 0 or raw_tokens <= 0:
            return
        self.calibrate(observed_tokens=observed_tokens, raw_estimated_tokens=raw_tokens)
        self._anchor_message_count = self._message_count
        self._anchor_raw_tokens = raw_tokens
        self._anchor_observed_tokens = observed_tokens

    def matches(self, messages: list[ConversationMessage]) -> bool:
        if len(messages) != self._message_count:
            return False
        index = 0
        for segment in self._visible:
            for right in segment.messages:
                if messages[index] is not right:
                    return False
                index += 1
        return True

    @staticmethod
    def _count_removable_tail_messages(
        messages: list[ConversationMessage],
        limit: int,
        *,
        preserve_system: bool,
    ) -> int:
        removable = 0
        for message in reversed(messages):
            if preserve_system and message.role == MessageRole.SYSTEM:
                break
            removable += 1
            if removable == limit:
                break
        return removable

    def _estimate_messages(self, messages: list[ConversationMessage]) -> int:
        if self._estimate_tokens is None:
            return 0
        return max(0, int(self._estimate_tokens(messages)))

    def _estimate_from_anchor(self, raw_tokens: int) -> int | None:
        if self._anchor_message_count is None:
            return None
        if self._message_count < self._anchor_message_count:
            self._clear_anchor()
            return None
        if raw_tokens < self._anchor_raw_tokens:
            return None
        raw_delta = raw_tokens - self._anchor_raw_tokens
        return max(0, self._anchor_observed_tokens + self._calibrate_tokens(raw_delta))

    def _clear_anchor(self) -> None:
        self._anchor_message_count = None
        self._anchor_raw_tokens = 0
        self._anchor_observed_tokens = 0

    def _calibrate_tokens(self, raw_tokens: int) -> int:
        return max(0, int(raw_tokens * self._calibration_ratio))
