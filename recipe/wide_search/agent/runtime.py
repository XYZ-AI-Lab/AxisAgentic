# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

r"""Conversation runtime that recognizes Markdown tables as final answers.

The base :class:`WebSearchConversationRuntime._extract_direct_final_answer` only
treats ``\\boxed{...}`` payloads as direct final answers. WideSearch's official
agent prompt (``agent_prompt.profile="official"``) asks the model for a bare
```` ```markdown ``` ```` table with no ``\\boxed`` wrapper. Without this
override the runtime keeps prompting the model for a final response even after
it has already produced a valid table, wasting turns and (under turn-limit
caps) discarding otherwise correct outputs.

We require an explicit ```` ```markdown ``` ```` fence here (not the looser
pipe-region heuristic ``extract_dataframe`` falls back on at end-of-conversation)
because the runtime fires per-message: a "draft so far: |a|b|..." pipe block
mid-thought would otherwise terminate the loop while the model is still
working. The orchestrator's ``extract_final_output`` keeps the lenient
pipe-region salvage path for the end-of-conversation extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from recipe.web_search.agent.runtime import WebSearchConversationRuntime
from recipe.wide_search.eval.answer_extractor import has_fenced_markdown_table

if TYPE_CHECKING:
    from agentic.contracts import ConversationMessage


class WideSearchConversationRuntime(WebSearchConversationRuntime):
    def _extract_direct_final_answer(self, message: ConversationMessage) -> str | None:
        direct = super()._extract_direct_final_answer(message)
        if direct:
            return direct
        if not message.content:
            return None
        content = str(message.content)
        if has_fenced_markdown_table(content):
            return content
        return None


__all__ = ["WideSearchConversationRuntime"]
