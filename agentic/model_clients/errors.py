# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass


def _first_int(patterns: tuple[str, ...], text: str) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is not None:
            return int(match.group(1))
    return None


@dataclass
class ModelContextLimitError(RuntimeError):
    """Typed model error raised when a provider rejects a request for context length."""

    message: str
    status_code: int | None = None
    input_tokens: int | None = None
    requested_output_tokens: int | None = None
    total_tokens: int | None = None
    context_window: int | None = None
    original_error_type: str | None = None

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)

    @classmethod
    def from_exception(cls, exc: Exception, *, status_code: int | None = None) -> ModelContextLimitError:
        text = str(exc)
        total_tokens = _first_int(
            (
                r"=\s*(\d+)\s*>",
                r"total(?:\s+tokens?)?\D+(\d+)",
            ),
            text,
        )
        return cls(
            message=text,
            status_code=status_code,
            input_tokens=_first_int(
                (
                    r"input(?:\s+tokens?)?\D+(\d+)",
                    r"prompt(?:\s+tokens?)?\D+(\d+)",
                ),
                text,
            ),
            requested_output_tokens=_first_int(
                (
                    r"output(?:\s+tokens?)?\D+(\d+)",
                    r"max(?:imum)?(?:\s+completion)?(?:\s+tokens?)?\D+(\d+)",
                ),
                text,
            ),
            total_tokens=total_tokens,
            context_window=_first_int(
                (
                    r">\s*(\d+)",
                    r"context(?:\s+(?:length|window))?\D+(\d+)",
                    r"limit\D+(\d+)",
                ),
                text,
            ),
            original_error_type=type(exc).__name__,
        )

    def to_info(self) -> dict[str, object]:
        return {
            "message": self.message,
            "status_code": self.status_code,
            "input_tokens": self.input_tokens,
            "requested_output_tokens": self.requested_output_tokens,
            "total_tokens": self.total_tokens,
            "context_window": self.context_window,
            "original_error_type": self.original_error_type,
        }
