from agentic.contracts import ConversationMessage, MessageRole
from agentic.conversations import ContextLengthTracker


def _estimate_tokens(messages: list[ConversationMessage]) -> int:
    return sum(len(message.content or "") for message in messages)


def test_context_length_tracker_estimates_appended_messages() -> None:
    tracker = ContextLengthTracker(_estimate_tokens)
    system = ConversationMessage.system("sys")
    user = ConversationMessage.user("hello")

    tracker.append_many([system, user])

    assert tracker.estimated_tokens == len("syshello")
    assert tracker.estimate_with([ConversationMessage.user("next")]) == len("syshellonext")
    assert tracker.visible_messages() == [system, user]


def test_context_length_tracker_subtracts_rolled_back_messages() -> None:
    tracker = ContextLengthTracker(_estimate_tokens)
    system = ConversationMessage.system("s")
    user = ConversationMessage.user("u")
    assistant = ConversationMessage.assistant("aaa")
    tool = ConversationMessage.tool("tool-result", tool_call_id="tc", name="search")
    tracker.append_many([system, user, assistant, tool])

    assert tracker.rollback_last(2) == 2

    visible = tracker.visible_messages()
    assert [message.role for message in visible] == [MessageRole.SYSTEM, MessageRole.USER]
    assert tracker.estimated_tokens == len("su")

    assert tracker.rollback_last(10) == 1
    assert [message.role for message in tracker.visible_messages()] == [MessageRole.SYSTEM]
    assert tracker.estimated_tokens == len("s")


def test_context_length_tracker_calibrates_upward_from_observed_tokens() -> None:
    tracker = ContextLengthTracker(_estimate_tokens)
    tracker.append_many([ConversationMessage.user("1234567890")])

    assert tracker.raw_estimated_tokens == 10
    assert tracker.estimated_tokens == 10

    tracker.calibrate(observed_tokens=30, raw_estimated_tokens=10)

    assert tracker.calibration_ratio > 1.0
    assert tracker.estimated_tokens > tracker.raw_estimated_tokens


def test_context_length_tracker_calibrates_downward_from_observed_tokens() -> None:
    tracker = ContextLengthTracker(_estimate_tokens)
    tracker.append_many([ConversationMessage.user("1234567890")])

    assert tracker.raw_estimated_tokens == 10
    assert tracker.estimated_tokens == 10

    tracker.calibrate(observed_tokens=5, raw_estimated_tokens=10)

    assert tracker.calibration_ratio < 1.0
    assert tracker.estimated_tokens < tracker.raw_estimated_tokens


def test_context_length_tracker_anchors_provider_tokens_for_visible_prefix() -> None:
    tracker = ContextLengthTracker(_estimate_tokens)
    tracker.append_many([ConversationMessage.user("1234567890")])

    tracker.anchor_observed_tokens(observed_tokens=5, raw_estimated_tokens=10)
    tracker.append_many([ConversationMessage.assistant("abcd")])

    assert tracker.estimate_with() < tracker.raw_estimate_with()
    assert tracker.estimate_with() >= 5


def test_context_length_tracker_anchor_can_include_static_probe_tokens() -> None:
    tracker = ContextLengthTracker(_estimate_tokens)
    probe = ConversationMessage.user("tools")
    tracker.append_many([ConversationMessage.user("1234567890")])

    tracker.anchor_observed_tokens(observed_tokens=12, raw_estimated_tokens=tracker.raw_estimate_with([probe]))

    assert tracker.estimate_with([probe]) == 12
    tracker.append_many([ConversationMessage.assistant("abcd")])

    assert tracker.estimate_with([probe]) > 12
    assert tracker.estimate_with([probe]) < tracker.raw_estimate_with([probe])
