# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agentic.contracts import ConversationMessage

if TYPE_CHECKING:
    from collections.abc import Callable


type TokenEstimator = Callable[[list[ConversationMessage]], int]
type AdditionalTokenEstimator = Callable[[list[ConversationMessage], list[ConversationMessage]], int]


@dataclass(frozen=True)
class ContextBudgetEstimator:
    """Estimate remaining context budget for ConversationRuntime decisions."""

    context_window: int | None
    safety_margin: int
    min_tokens_for_generation: int
    warning_threshold: int
    max_output_tokens: int
    tools: list[dict[str, Any]]
    token_estimator_includes_tools: bool
    estimate_tokens: TokenEstimator
    estimate_tokens_with_additional: AdditionalTokenEstimator | None = None

    def estimate_generation_budget(self, visible_conversation: list[ConversationMessage]) -> dict[str, object]:
        if self.context_window is None:
            return {
                "estimated_prompt_tokens": 0,
                "reserved_output_tokens": self.max_output_tokens,
                "context_safety_margin": self.safety_margin,
                "estimated_total_tokens": 0,
                "remaining_tokens": None,
                "min_tokens_for_generation": self.min_tokens_for_generation,
                "context_warning_threshold": self.warning_threshold,
                "context_window": None,
                "hard_stop": False,
                "needs_final_warning": False,
                "exceeds_context": False,
            }

        estimated_prompt_tokens = self._estimate_prompt_tokens(visible_conversation)
        remaining_tokens = self.context_window - estimated_prompt_tokens - self.safety_margin
        hard_stop = self.min_tokens_for_generation > 0 and remaining_tokens < self.min_tokens_for_generation
        needs_final_warning = not hard_stop and remaining_tokens < self.warning_threshold
        reserved_output_tokens = self.warning_threshold
        estimated_total_tokens = estimated_prompt_tokens + reserved_output_tokens + self.safety_margin
        return {
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "reserved_output_tokens": reserved_output_tokens,
            "context_safety_margin": self.safety_margin,
            "estimated_total_tokens": estimated_total_tokens,
            "remaining_tokens": remaining_tokens,
            "min_tokens_for_generation": self.min_tokens_for_generation,
            "context_warning_threshold": self.warning_threshold,
            "context_window": self.context_window,
            "hard_stop": hard_stop,
            "needs_final_warning": needs_final_warning,
            "exceeds_context": hard_stop or needs_final_warning,
        }

    def estimate_force_finalization_budget(
        self,
        visible_conversation: list[ConversationMessage],
        *,
        force_final_prompt: str | None,
    ) -> dict[str, object]:
        context_info = self.estimate_generation_budget(visible_conversation)
        warning_fit_info = self.estimate_force_final_prompt_fit(
            visible_conversation,
            force_final_prompt=force_final_prompt,
        )
        context_info.update(warning_fit_info)

        prompt_with_warning_tokens = int(warning_fit_info["estimated_prompt_with_warning_tokens"])
        context_info["estimated_total_tokens"] = max(
            int(context_info["estimated_total_tokens"]),
            prompt_with_warning_tokens + self.warning_threshold + self.safety_margin,
        )
        warning_remaining_tokens = warning_fit_info["warning_remaining_tokens"]
        if warning_remaining_tokens is not None and int(warning_remaining_tokens) < self.warning_threshold:
            context_info["needs_final_warning"] = True
        context_info["exceeds_context"] = bool(context_info["hard_stop"] or context_info["needs_final_warning"])
        return context_info

    def estimate_force_final_prompt_fit(
        self,
        visible_conversation: list[ConversationMessage],
        *,
        force_final_prompt: str | None,
    ) -> dict[str, object]:
        if self.context_window is None or force_final_prompt is None:
            return {
                "estimated_prompt_with_warning_tokens": 0,
                "warning_remaining_tokens": None,
                "warning_prompt_fits": True,
            }

        estimated_prompt_with_warning_tokens = self._estimate_prompt_tokens(
            visible_conversation,
            additional_messages=[ConversationMessage.user(force_final_prompt)],
        )
        warning_remaining_tokens = self.context_window - estimated_prompt_with_warning_tokens - self.safety_margin
        return {
            "estimated_prompt_with_warning_tokens": estimated_prompt_with_warning_tokens,
            "warning_remaining_tokens": warning_remaining_tokens,
            "warning_prompt_fits": warning_remaining_tokens > 0,
        }

    def _estimate_prompt_tokens(
        self,
        visible_conversation: list[ConversationMessage],
        *,
        additional_messages: list[ConversationMessage] | None = None,
    ) -> int:
        additional = list(additional_messages or [])
        if self.tools and not self.token_estimator_includes_tools:
            additional.append(ConversationMessage.user(str(self.tools)))
        if self.estimate_tokens_with_additional is not None:
            return self.estimate_tokens_with_additional(visible_conversation, additional)
        return self.estimate_tokens([*visible_conversation, *additional])
