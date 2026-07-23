from __future__ import annotations

import pytest

from agentic.contracts import ConversationMessage, MessageRole
from recipe.web_search.agent.prompts import FORMAT_ERROR_MESSAGE
from recipe.wide_search.agent.orchestrator import WideSearchTaskOrchestrator


class _StubOrchestrator(WideSearchTaskOrchestrator):
    """Bypasses the heavy base __init__ — we exercise extract_final_output only."""

    def __init__(self, extractor_mode: str, agent_prompt_profile: str = "project") -> None:  # type: ignore[no-untyped-def]
        self._extractor_mode = extractor_mode  # type: ignore[assignment]
        self._agent_prompt_profile = agent_prompt_profile
        self._intermediate_boxed_answers: list[str] = []


def _assistant(content: str) -> ConversationMessage:
    return ConversationMessage(role=MessageRole.ASSISTANT, content=content)


def _user(content: str) -> ConversationMessage:
    return ConversationMessage(role=MessageRole.USER, content=content)


def test_extract_final_output_returns_boxed_when_present() -> None:
    orchestrator = _StubOrchestrator("boxed_first")
    boxed_table = "\\boxed{\n|name|score|\n|---|---|\n|alpha|1|\n|beta|2|\n}"
    result = orchestrator.extract_final_output([_user("ignore"), _assistant(boxed_table)])
    assert "alpha" in result
    assert "beta" in result


def test_extract_final_output_returns_message_content_when_boxed_pointer_and_table_in_same_message() -> None:
    """``\\boxed{See table above}`` plus the table earlier in the same message
    must yield the table, not the pointer text. Regression for the case where
    the agent boxes a textual reference instead of the table itself.
    """
    orchestrator = _StubOrchestrator("boxed_first")
    message = "Here is the requested table:\n\n|name|score|\n|---|---|\n|alpha|1|\n|beta|2|\n\n\\boxed{See table above}"
    result = orchestrator.extract_final_output([_user("ignore"), _assistant(message)])
    assert result == message
    assert "alpha" in result
    assert result != FORMAT_ERROR_MESSAGE


def test_extract_final_output_walks_back_when_boxed_is_pure_pointer() -> None:
    """When the latest assistant message is *only* ``\\boxed{See table above}``
    with no table, the table really is on a prior assistant turn — walk back.
    """
    orchestrator = _StubOrchestrator("boxed_first")
    earlier = _assistant("```markdown\n|name|score|\n|---|---|\n|alpha|1|\n|beta|2|\n```")
    later = _assistant("\\boxed{See table above}")
    result = orchestrator.extract_final_output([_user("ignore"), earlier, later])
    assert "alpha" in result
    assert result != FORMAT_ERROR_MESSAGE


def test_extract_final_output_format_error_when_boxed_pointer_and_no_table_anywhere() -> None:
    """``\\boxed{nope}`` on the latest turn with no table on any prior assistant
    turn must surface ``FORMAT_ERROR_MESSAGE`` rather than returning ``"nope"``.
    """
    orchestrator = _StubOrchestrator("boxed_first")
    earlier = _assistant("Still searching, no table yet.")
    later = _assistant("\\boxed{See table above}")
    result = orchestrator.extract_final_output([_user("ignore"), earlier, later])
    assert result == FORMAT_ERROR_MESSAGE


def test_extract_final_output_format_error_when_boxed_is_abandonment_message() -> None:
    """``\\boxed{I could not complete it}`` after an earlier draft table must
    NOT inherit that draft. A non-pointer boxed payload is a give-up signal —
    walking back would treat the abandoned draft as a valid final and skip
    retries. Regression for the pointer-gating fix.
    """
    orchestrator = _StubOrchestrator("boxed_first")
    earlier = _assistant("Draft so far:\n\n```markdown\n|name|score|\n|---|---|\n|alpha|1|\n```")
    later = _assistant("\\boxed{I could not complete it}")
    result = orchestrator.extract_final_output([_user("ignore"), earlier, later])
    assert result == FORMAT_ERROR_MESSAGE


def test_extract_final_output_format_error_when_boxed_mentions_previous_but_is_not_pointer() -> None:
    """Mentioning a previous draft is not enough to inherit it as final."""
    orchestrator = _StubOrchestrator("boxed_first")
    earlier = _assistant("Draft so far:\n\n```markdown\n|name|score|\n|---|---|\n|alpha|1|\n```")
    later = _assistant("\\boxed{I could not verify the previous draft}")
    result = orchestrator.extract_final_output([_user("ignore"), earlier, later])
    assert result == FORMAT_ERROR_MESSAGE


@pytest.mark.parametrize(
    "pointer_text",
    [
        "See table above",
        "as shown above",
        "see the previous table",
        "table from earlier",
        "见上表",
        "上面的表格",
    ],
)
def test_extract_final_output_walks_back_for_recognized_pointer_phrases(pointer_text: str) -> None:
    """Common English and Chinese pointer phrases should all gate the walk-back."""
    orchestrator = _StubOrchestrator("boxed_first")
    earlier = _assistant("```markdown\n|name|score|\n|---|---|\n|alpha|1|\n```")
    later = _assistant(f"\\boxed{{{pointer_text}}}")
    result = orchestrator.extract_final_output([_user("ignore"), earlier, later])
    assert "alpha" in result
    assert result != FORMAT_ERROR_MESSAGE


def test_extract_final_output_accepts_unboxed_markdown_table() -> None:
    orchestrator = _StubOrchestrator("official_only")
    response = "Here is the requested table:\n\n```markdown\n|name|score|\n|---|---|\n|alpha|1|\n|beta|2|\n```\n"
    result = orchestrator.extract_final_output([_user("ignore"), _assistant(response)])
    assert result == response
    assert result != FORMAT_ERROR_MESSAGE


def test_extract_final_output_rejects_message_without_table() -> None:
    orchestrator = _StubOrchestrator("official_only")
    response = "I tried but could not find the answer."
    result = orchestrator.extract_final_output([_user("ignore"), _assistant(response)])
    assert result == FORMAT_ERROR_MESSAGE


def test_extract_final_output_rejects_empty_assistant_content() -> None:
    orchestrator = _StubOrchestrator("official_only")
    result = orchestrator.extract_final_output([_user("ignore"), _assistant("")])
    assert result == FORMAT_ERROR_MESSAGE


def test_extract_final_output_rejects_when_no_assistant_messages() -> None:
    orchestrator = _StubOrchestrator("official_only")
    result = orchestrator.extract_final_output([_user("only user")])
    assert result == FORMAT_ERROR_MESSAGE


def test_extract_final_output_uses_latest_assistant_message() -> None:
    orchestrator = _StubOrchestrator("official_only")
    earlier = _assistant("```markdown\n|x|y|\n|-|-|\n|1|2|\n```")
    later = _assistant("oops, no table this time")
    result = orchestrator.extract_final_output([_user("ignore"), earlier, later])
    assert result == FORMAT_ERROR_MESSAGE


@pytest.mark.parametrize("mode", ["boxed_first", "official_only"])
def test_extract_final_output_extractor_mode_round_trips(mode: str) -> None:
    orchestrator = _StubOrchestrator(mode)
    response = "```markdown\n|c|d|\n|-|-|\n|3|4|\n```"
    result = orchestrator.extract_final_output([_assistant(response)])
    assert result == response


def test_latest_intermediate_boxed_answer_returns_latest_table() -> None:
    """Cache walks back to the latest entry that parses as a non-empty table."""
    orchestrator = _StubOrchestrator("boxed_first")
    orchestrator._intermediate_boxed_answers = [
        "|name|score|\n|---|---|\n|alpha|1|",
        "|name|score|\n|---|---|\n|beta|2|",
    ]
    assert orchestrator._latest_intermediate_boxed_answer() == ("|name|score|\n|---|---|\n|beta|2|")


def test_latest_intermediate_boxed_answer_skips_pointer_text() -> None:
    """Boxed pointer/abandonment strings stored in the cache must not be
    surfaced as fallback finals — the base ``_return_final_attempt_result``
    fallback would otherwise return ``"See table above"`` verbatim as the
    final answer. Walk back to the latest valid-table entry.
    """
    orchestrator = _StubOrchestrator("boxed_first")
    orchestrator._intermediate_boxed_answers = [
        "|name|score|\n|---|---|\n|alpha|1|",
        "See table above",
        "I could not complete it",
    ]
    assert orchestrator._latest_intermediate_boxed_answer() == ("|name|score|\n|---|---|\n|alpha|1|")


def test_latest_intermediate_boxed_answer_returns_empty_when_no_valid_table() -> None:
    orchestrator = _StubOrchestrator("boxed_first")
    orchestrator._intermediate_boxed_answers = [
        "See table above",
        "I could not complete it",
    ]
    assert orchestrator._latest_intermediate_boxed_answer() == ""


def test_latest_intermediate_boxed_answer_returns_empty_when_cache_empty() -> None:
    orchestrator = _StubOrchestrator("boxed_first")
    assert orchestrator._latest_intermediate_boxed_answer() == ""


def _build_runtime_capture_orchestrator(profile: str) -> tuple[_StubOrchestrator, dict]:
    """Construct a stub orchestrator wired to capture the runtime config.

    Bypasses the base __init__ entirely and stubs out everything
    ``build_conversation_runtime`` touches, so we can assert that the
    forced-final ``final_response_prompt`` is the WideSearch summary prompt
    (markdown table) rather than the BrowseComp summary prompt (boxed scalar).
    """
    captured: dict = {}

    class _StubRuntimeConfig:
        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self.__dict__.update(kwargs)

        def model_copy(self, *, update):  # type: ignore[no-untyped-def]
            merged = {**self.__dict__, **update}
            return _StubRuntimeConfig(**merged)

    class _StubModelClient:
        max_output_tokens = 4096

        def estimate_tokens(self, *_args, **_kwargs) -> int:  # type: ignore[no-untyped-def]
            return 0

    class _StubRuntime:
        def __init__(self, *, config, **kwargs) -> None:  # type: ignore[no-untyped-def]
            captured["config"] = config
            captured["kwargs"] = kwargs

    class _OuterConfig:
        conversation = _StubRuntimeConfig(early_stop_announcement_prompt="x", final_response_prompt="x")

    orchestrator = _StubOrchestrator("boxed_first", agent_prompt_profile=profile)
    orchestrator._current_task = "list me the rows"  # type: ignore[attr-defined]
    orchestrator.config = _OuterConfig()  # type: ignore[attr-defined]
    orchestrator._conversation_runtime_class = _StubRuntime  # type: ignore[attr-defined]
    orchestrator.model_client = _StubModelClient()  # type: ignore[attr-defined]
    orchestrator.context_token_estimator = None  # type: ignore[attr-defined]
    orchestrator.context_limit_preflight_enabled = False  # type: ignore[attr-defined]
    orchestrator._skip_turn_limit_final_response_this_attempt = False  # type: ignore[attr-defined]
    return orchestrator, captured


def test_build_conversation_runtime_uses_widesearch_summary_for_project_profile() -> None:
    orchestrator, captured = _build_runtime_capture_orchestrator("project")
    orchestrator.build_conversation_runtime(tools=None)
    final_prompt = captured["config"].final_response_prompt
    assert "Markdown" in final_prompt
    assert "list me the rows" in final_prompt
    assert "\\boxed" in final_prompt
    assert "comma-separated list of numbers" not in final_prompt


def test_build_conversation_runtime_uses_widesearch_summary_for_official_profile() -> None:
    orchestrator, captured = _build_runtime_capture_orchestrator("official")
    orchestrator.build_conversation_runtime(tools=None)
    final_prompt = captured["config"].final_response_prompt
    assert "Markdown table" in final_prompt
    assert "\\boxed" not in final_prompt
    assert "comma-separated list of numbers" not in final_prompt
