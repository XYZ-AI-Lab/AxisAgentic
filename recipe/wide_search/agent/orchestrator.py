# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from agentic.contracts import ConversationMessage, MessageRole
from recipe.web_search.agent.orchestrator import WebSearchTaskOrchestrator
from recipe.web_search.agent.prompts import FORMAT_ERROR_MESSAGE, extract_boxed_content
from recipe.wide_search.agent.prompts import widesearch_summary_prompt
from recipe.wide_search.eval.answer_extractor import ExtractorMode, extract_dataframe

if TYPE_CHECKING:
    from agentic.conversations.conversation_runtime import ConversationRuntime


_BOXED_TABLE_POINTER_RE = re.compile(
    r"(?i)^("
    r"(?:see|refer to|check|use)\s+(?:the\s+)?"
    r"(?:(?:above|previous|prior|earlier|preceding)\s+)?"
    r"(?:markdown\s+)?table(?:\s+(?:above|previously|earlier|below))?"
    r"|(?:the\s+)?(?:markdown\s+)?"
    r"table\s+(?:above|previously|earlier|below)"
    r"|as\s+shown\s+above"
    r"|(?:same\s+as|as\s+shown\s+above|answer\s+is\s+in)\s+(?:the\s+)?"
    r"(?:(?:above|previous|prior|earlier|preceding)\s+)?"
    r"(?:markdown\s+)?table"
    r"|table\s+from\s+(?:above|previously|earlier|before)"
    r"|(?:见|参考|参见|查看)(?:上|下|前)(?:方|面|述)?(?:的)?(?:markdown)?表(?:格)?"
    r"|(?:上|下|前)(?:方|面|述)?(?:的)?(?:markdown)?表(?:格)?"
    r"|(?:如上表|见上表|参考上表|参见上表|上表)"
    r")[\s。.!?]*$"
)


def _boxed_payload_is_pointer(text: str) -> bool:
    """Return whether a non-table boxed payload points to a table elsewhere.

    Give-up messages should not inherit stale draft tables from prior turns.
    """
    normalized = " ".join(text.strip().split())
    if not normalized:
        return False
    return bool(_BOXED_TABLE_POINTER_RE.match(normalized))


class WideSearchTaskOrchestrator(WebSearchTaskOrchestrator):
    r"""Orchestrator that also accepts unboxed Markdown-table answers.

    The base orchestrator rejects any final assistant message that lacks a
    ``\\boxed{...}`` payload, returning ``FORMAT_ERROR_MESSAGE``. That works for
    the project preset (`agent_prompt.profile="project"`), where the system
    prompt instructs the agent to wrap its table in ``\\boxed{}``. It misfires
    on the official preset (`agent_prompt.profile="official"`): the upstream
    WideSearch system prompt asks for a plain ```` ```markdown ``` ```` table,
    so every otherwise-valid final answer is rejected and the orchestrator
    burns ``max_task_retries`` attempts before giving up.

    Final-output extraction follows three rules in order:

    1. **Boxed payload is itself a table** → return the boxed payload.
    2. **Boxed payload exists but is not a table** (e.g. ``\\boxed{See table
       above}``) → fall back to extracting the table from the full assistant
       message; if not found AND the boxed text reads like an explicit
       pointer ("above", "previous", "prior", ...) walk back to the prior
       assistant message. Non-pointer boxed text (``\\boxed{I could not
       complete it}``) instead returns ``FORMAT_ERROR_MESSAGE`` so retries
       fire — without this gating a stale earlier draft table would be
       returned as a valid final.
    3. **No boxed payload** → try the unboxed table extractor on the latest
       assistant message; do not walk back.

    Rules 2 and 3 are necessary because the official preset emits raw
    ```` ```markdown ``` ```` blocks with no ``\\boxed`` wrapper, and because
    Qwen-style models sometimes box a textual reference to a table they wrote
    earlier in the same (or a prior) message.

    It also overrides :meth:`build_conversation_runtime` so the forced-final
    recovery prompt (used on turn/context/tool exhaustion) is the WideSearch
    markdown-table summary prompt instead of the BrowseComp boxed-scalar one
    the base class injects via ``generate_summary_prompt``.
    """

    def __init__(
        self,
        *args: Any,
        extractor_mode: ExtractorMode,
        agent_prompt_profile: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._extractor_mode: ExtractorMode = extractor_mode
        self._agent_prompt_profile: str = agent_prompt_profile

    def extract_final_output(self, conversation: list[ConversationMessage]) -> Any:
        for message in reversed(conversation):
            if message.role != MessageRole.ASSISTANT:
                continue
            if not message.content:
                return FORMAT_ERROR_MESSAGE
            content = str(message.content)
            boxed = extract_boxed_content(content)
            if boxed:
                df = extract_dataframe(boxed, mode=self._extractor_mode)
                if df is not None and not df.empty:
                    return boxed
                df = extract_dataframe(content, mode=self._extractor_mode)
                if df is not None and not df.empty:
                    return content
                if _boxed_payload_is_pointer(boxed):
                    # Boxed payload is an explicit pointer (e.g. "See table
                    # above"); the table is on a prior assistant turn —
                    # walk back. Other non-table boxed payloads (e.g.
                    # abandonment messages) fall through to FORMAT_ERROR_MESSAGE.
                    continue
                return FORMAT_ERROR_MESSAGE
            df = extract_dataframe(content, mode=self._extractor_mode)
            if df is not None and not df.empty:
                return content
            return FORMAT_ERROR_MESSAGE
        return FORMAT_ERROR_MESSAGE

    def _latest_intermediate_boxed_answer(self) -> str:
        r"""Skip cached boxed strings that aren't valid tables.

        The base orchestrator collects every boxed payload it sees into
        ``_intermediate_boxed_answers`` and ``_return_final_attempt_result``
        uses the latest one as a fallback final answer. For widesearch a
        non-table boxed payload (``\\boxed{See table above}``,
        ``\\boxed{I could not complete it}``) must NOT be returned as the
        final — the override here walks back to the latest entry that parses
        as a non-empty table and returns ``""`` if none qualifies.
        """
        for boxed in reversed(self._intermediate_boxed_answers):
            df = extract_dataframe(boxed, mode=self._extractor_mode)
            if df is not None and not df.empty:
                return boxed
        return ""

    def build_conversation_runtime(self, tools: list[dict[str, Any]] | None) -> ConversationRuntime:
        summary_prompt = widesearch_summary_prompt(self._current_task, profile=self._agent_prompt_profile)
        config = self.config.conversation.model_copy(
            update={
                "early_stop_announcement_prompt": None,
                "final_response_prompt": summary_prompt,
            }
        )
        runtime = self._conversation_runtime_class(
            config=config,
            tools=tools,
            max_output_tokens=self.model_client.max_output_tokens,
            token_estimator=self.context_token_estimator or self.model_client.estimate_tokens,
            token_estimator_includes_tools=bool(getattr(self.context_token_estimator, "token_estimator_includes_tools", False)),
            token_estimator_is_additive=bool(getattr(self.context_token_estimator, "token_estimator_is_additive", False)),
            context_limit_preflight_enabled=self.context_limit_preflight_enabled,
        )
        if hasattr(runtime, "skip_turn_limit_final_response"):
            runtime_with_skip: Any = runtime
            runtime_with_skip.skip_turn_limit_final_response = self._skip_turn_limit_final_response_this_attempt
        if hasattr(runtime, "skip_prompted_final_response"):
            runtime_with_prompt_skip: Any = runtime
            runtime_with_prompt_skip.skip_prompted_final_response = self._skip_turn_limit_final_response_this_attempt
        return runtime


__all__ = ["WideSearchTaskOrchestrator"]
