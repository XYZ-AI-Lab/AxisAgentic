from __future__ import annotations

from agentic.contracts import ConversationMessage, MessageRole
from recipe.wide_search.agent.runtime import WideSearchConversationRuntime


def _runtime() -> WideSearchConversationRuntime:
    """Bypass the heavy base ``__init__``; ``_extract_direct_final_answer``
    does not read instance state beyond what super already provides.
    """
    return object.__new__(WideSearchConversationRuntime)


def _assistant(content: str) -> ConversationMessage:
    return ConversationMessage(role=MessageRole.ASSISTANT, content=content)


def test_extract_direct_final_answer_prefers_boxed() -> None:
    """When ``\\boxed{...}`` is present, the runtime must return the boxed
    payload exactly as the base class does — the table fallback only kicks in
    on un-boxed responses.
    """
    runtime = _runtime()
    message = _assistant("Final: \\boxed{42}")
    assert runtime._extract_direct_final_answer(message) == "42"


def test_extract_direct_final_answer_accepts_unboxed_fenced_markdown_table() -> None:
    """Official-profile responses lack ``\\boxed{}``; a bare fenced markdown
    table must count as a final answer so the runtime stops re-prompting.
    """
    runtime = _runtime()
    table = "Here is the answer:\n\n```markdown\n|name|score|\n|---|---|\n|alpha|1|\n|beta|2|\n```\n"
    result = runtime._extract_direct_final_answer(_assistant(table))
    assert result == table


def test_extract_direct_final_answer_rejects_unboxed_pipe_table() -> None:
    """A bare pipe-region table is NOT enough to terminate the agent loop —
    "draft so far: |a|b|..." mid-thought messages would otherwise prematurely
    terminate. The orchestrator's ``extract_final_output`` keeps the lenient
    pipe-region salvage path for end-of-conversation extraction.
    """
    runtime = _runtime()
    content = "Here is the answer:\n|name|score|\n|---|---|\n|alpha|1|\n|beta|2|\n"
    assert runtime._extract_direct_final_answer(_assistant(content)) is None


def test_extract_direct_final_answer_rejects_intermediate_draft_with_continuation() -> None:
    """A fenced block followed by "I'll verify next" prose still parses to a
    non-empty DataFrame, but the runtime should accept it as final because the
    explicit ```markdown fence is a stronger commit signal than a bare pipe
    region. Documents the asymmetry.
    """
    runtime = _runtime()
    content = "Draft so far:\n```markdown\n|name|score|\n|---|---|\n|alpha|1|\n```\nI'll verify next."
    # Intentional: a complete fenced block IS treated as a final. Models
    # following the official prompt only emit the fence once they're done.
    assert runtime._extract_direct_final_answer(_assistant(content)) == content


def test_extract_direct_final_answer_returns_none_when_no_table_and_no_boxed() -> None:
    """Prose without a table must NOT be treated as a final answer — the
    runtime keeps the agent loop going.
    """
    runtime = _runtime()
    result = runtime._extract_direct_final_answer(_assistant("Still searching."))
    assert result is None


def test_extract_direct_final_answer_returns_none_on_empty_content() -> None:
    runtime = _runtime()
    assert runtime._extract_direct_final_answer(_assistant("")) is None
