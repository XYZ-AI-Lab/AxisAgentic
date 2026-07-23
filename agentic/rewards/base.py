# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentic.contracts import ConversationStage
from agentic.contracts.messages import ToolResultStatus

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from agentic.contracts import ConversationMessage
    from agentic.tools.manager import ToolExecutionOutcome


@dataclass(slots=True)
class RewardContext:
    conversation: list[ConversationMessage]
    assistant_message: ConversationMessage | None = None
    tool_outcomes: Sequence[ToolExecutionOutcome] = field(default_factory=tuple)
    stage: ConversationStage | None = None
    info: Mapping[str, Any] = field(default_factory=dict)


class RewardEvaluator(ABC):
    @abstractmethod
    def evaluate(self, context: RewardContext) -> float:
        """Return the reward for the current orchestration transition."""


class ZeroRewardEvaluator(RewardEvaluator):
    def evaluate(self, context: RewardContext) -> float:
        del context
        return 0.0


class ToolCallRewardEvaluator(RewardEvaluator):
    def __init__(
        self,
        *,
        reward_per_success_tool_call: float,
        reward_per_failed_tool_call: float,
        reward_per_rejected_tool_call: float,
        reward_for_wrong_tool_call_format: float,
        max_success_reward: float = 1.0,
        min_failure_penalty: float = -0.5,
        min_rejection_penalty: float = -0.5,
    ) -> None:
        self.reward_per_success_tool_call = reward_per_success_tool_call
        self.reward_per_failed_tool_call = reward_per_failed_tool_call
        self.reward_per_rejected_tool_call = reward_per_rejected_tool_call
        self.reward_for_wrong_tool_call_format = reward_for_wrong_tool_call_format
        self.max_success_reward = max_success_reward
        self.min_failure_penalty = min_failure_penalty
        self.min_rejection_penalty = min_rejection_penalty

    def evaluate(self, context: RewardContext) -> float:
        if context.stage == ConversationStage.TOOL_CALL_FORMAT_ERROR_PROMPT_AWAIT_ASSISTANT:
            return self.reward_for_wrong_tool_call_format

        reward = 0.0
        num_success = sum(1 for outcome in context.tool_outcomes if outcome.result.status == ToolResultStatus.SUCCESS)
        reward += min(self.max_success_reward, num_success * self.reward_per_success_tool_call)
        num_failed = sum(1 for outcome in context.tool_outcomes if outcome.result.status == ToolResultStatus.FAILED)
        reward += max(self.min_failure_penalty, num_failed * self.reward_per_failed_tool_call)
        num_rejected = sum(1 for outcome in context.tool_outcomes if outcome.result.status == ToolResultStatus.REJECTED)
        reward += max(self.min_rejection_penalty, num_rejected * self.reward_per_rejected_tool_call)
        return reward
