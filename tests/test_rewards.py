import pytest

from agentic.contracts import ConversationMessage, ConversationStage, ToolRequest
from agentic.contracts.messages import ToolResultReason, ToolResultStatus
from agentic.rewards import RewardContext, ToolCallRewardEvaluator, ZeroRewardEvaluator
from agentic.tools import ToolResult
from agentic.tools.manager import ToolExecutionOutcome


def _outcome(
    status: ToolResultStatus = ToolResultStatus.SUCCESS,
    reason: ToolResultReason | None = None,
    call_id: str = "call_1",
) -> ToolExecutionOutcome:
    return ToolExecutionOutcome(
        request=ToolRequest(tool_name="echo", arguments={"text": "ok"}, call_id=call_id),
        result=ToolResult(content="ok", status=status, reason=reason),
        message=ConversationMessage.tool("ok", tool_call_id=call_id, name="echo"),
    )


def test_zero_reward_evaluator_returns_zero() -> None:
    evaluator = ZeroRewardEvaluator()
    reward = evaluator.evaluate(RewardContext(conversation=[ConversationMessage.user("hello")]))
    assert reward == 0.0


def test_tool_call_reward_per_success_clamped() -> None:
    evaluator = ToolCallRewardEvaluator(
        reward_per_success_tool_call=2.0,
        reward_per_failed_tool_call=-1.0,
        reward_per_rejected_tool_call=0.0,
        reward_for_wrong_tool_call_format=0.0,
    )
    # Single success: min(1.0, 1 * 2.0) = 1.0
    reward = evaluator.evaluate(RewardContext(conversation=[], tool_outcomes=[_outcome()]))
    assert reward == 1.0


def test_tool_call_reward_per_failed_clamped() -> None:
    evaluator = ToolCallRewardEvaluator(
        reward_per_success_tool_call=2.0,
        reward_per_failed_tool_call=-1.0,
        reward_per_rejected_tool_call=0.0,
        reward_for_wrong_tool_call_format=0.0,
    )
    # Single failure: max(-0.5, 1 * -1.0) = -0.5
    reward = evaluator.evaluate(RewardContext(conversation=[], tool_outcomes=[_outcome(ToolResultStatus.FAILED, ToolResultReason.EXECUTION_ERROR)]))
    assert reward == -0.5


def test_tool_call_reward_per_rejected_clamped() -> None:
    evaluator = ToolCallRewardEvaluator(
        reward_per_success_tool_call=2.0,
        reward_per_failed_tool_call=0.0,
        reward_per_rejected_tool_call=-0.5,
        reward_for_wrong_tool_call_format=0.0,
    )
    # 3 rejected: max(-0.5, 3 * -0.5) = max(-0.5, -1.5) = -0.5
    rejected = [_outcome(ToolResultStatus.REJECTED, ToolResultReason.PER_STEP_LIMIT_EXCEEDED, call_id=f"c{i}") for i in range(3)]
    reward = evaluator.evaluate(RewardContext(conversation=[], tool_outcomes=rejected))
    assert reward == -0.5


def test_tool_call_reward_mixed_outcomes() -> None:
    evaluator = ToolCallRewardEvaluator(
        reward_per_success_tool_call=0.5,
        reward_per_failed_tool_call=-0.2,
        reward_per_rejected_tool_call=-0.1,
        reward_for_wrong_tool_call_format=0.0,
    )
    outcomes = [
        _outcome(ToolResultStatus.SUCCESS, call_id="c1"),
        _outcome(ToolResultStatus.FAILED, ToolResultReason.EXECUTION_ERROR, call_id="c2"),
        _outcome(ToolResultStatus.REJECTED, ToolResultReason.PER_STEP_LIMIT_EXCEEDED, call_id="c3"),
    ]
    # success: min(1.0, 1 * 0.5) = 0.5
    # failed: max(-0.5, 1 * -0.2) = -0.2
    # rejected: max(-0.5, 1 * -0.1) = -0.1
    reward = evaluator.evaluate(RewardContext(conversation=[], tool_outcomes=outcomes))
    assert reward == pytest.approx(0.2)


def test_tool_call_reward_wrong_tool_call_format() -> None:
    evaluator = ToolCallRewardEvaluator(
        reward_per_success_tool_call=2.0,
        reward_per_failed_tool_call=-1.0,
        reward_per_rejected_tool_call=-0.5,
        reward_for_wrong_tool_call_format=-3.0,
    )
    reward = evaluator.evaluate(
        RewardContext(
            conversation=[],
            stage=ConversationStage.TOOL_CALL_FORMAT_ERROR_PROMPT_AWAIT_ASSISTANT,
        )
    )
    assert reward == -3.0


def test_tool_call_reward_no_outcomes_returns_zero() -> None:
    evaluator = ToolCallRewardEvaluator(
        reward_per_success_tool_call=2.0,
        reward_per_failed_tool_call=-1.0,
        reward_per_rejected_tool_call=-0.5,
        reward_for_wrong_tool_call_format=0.0,
    )
    reward = evaluator.evaluate(RewardContext(conversation=[]))
    assert reward == 0.0
