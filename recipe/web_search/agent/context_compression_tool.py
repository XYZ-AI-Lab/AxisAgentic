# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Context compression summary tool (V5 prompt, Qwen3 path).

Produces a single ``[context_summary]`` JSON block that downstream code injects
back into the conversation in place of compressed earlier turns.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agentic.contracts import ConversationMessage, MessageRole

logger = logging.getLogger(__name__)


REQUIRED_SUMMARY_KEYS = (
    "confirmed_facts",
    "current_hypothesis",
    "rejected_paths",
    "dismissed_candidates",
    "open_questions",
    "answer_attempts",
    "conflict_candidates",
    "search_state",
    "retry_policy",
)


SUMMARY_MARKER_OPEN = "[context_summary]"
SUMMARY_MARKER_CLOSE = "[/context_summary]"


SYSTEM_PROMPT = """You are a context compression module for a long-horizon tool-using search agent.

Your job is to compress or update earlier agent context into a compact state summary that will be inserted back into the agent conversation.

Do not solve the task.
Do not continue the investigation.
Do not add external knowledge.
Do not invent sources, URLs, citations, or tool results.
Do not turn guesses into confirmed facts.
Do not mark a search path as rejected only because a small number of queries failed.
Do not copy full reasoning traces.
Preserve uncertainty and source status.
Group repeated searches into search strategies instead of listing every raw query.

Output budget rules:
- Be concise enough to finish in one completion without dropping decision-critical information.
- Prefer merging repeated evidence and searches over adding many separate rows, but preserve every item that changes candidate selection, confidence, blockers, or next checks.
- Use short phrases inside arrays; do not copy long snippets, long query lists, long table rows, or full URLs unless they are decision-critical.
- Keep repeated or low-value items merged into compact ledger entries. Do not merge away hard-case details that are needed to distinguish plausible candidates or answer slots.
- Spend space first on active answer_attempts, conflict_candidates, unresolved blockers, discriminating facts, and next checks. Compress old search history after those are preserved.
- If the state is large, preserve the active candidates, blockers, conflicts, and next checks first; compress old search history into grouped ledgers.
- Never run out of tokens mid-JSON. If space is tight, shorten item text and merge entries, then always close the JSON object and the [/context_summary] marker.

Source-reference rules:
- In source fields and evidence notes, prefer compact source handles: tool name plus host, source title, citation id, or source family when available. Examples of acceptable handles are "serper:example.com", "jina:example.org", "official site", "newspaper archive", "source missing", and "tool result omitted".
- Do not paste raw snippets, search result dumps, or long URL lists into any summary field. Keep an exact full URL only when it is the sole identifier needed for the next targeted scrape, citation, or disambiguation; otherwise keep the host/source family and the verified fact, candidate, blocker, or next check.
- When many URLs or snippets from the same source family were checked, merge them into one compact coverage note naming the source family, useful lead or gap, and remaining verification target.

Confirmed-fact source rules:
- Use confirmed_facts only for decision-relevant world facts attributed to tool, search, scrape, or source evidence in previous_summary or new_context. Do not put original user-task wording, task_context constraints, task progress, search plans, or candidate-to-task matching claims in confirmed_facts.
- Never use task_context, user question, requested constraint, or requested as the source or evidence_status for a confirmed_facts item. A requested condition is not a confirmed fact, even at low confidence.
- If a previous confirmed_facts item is only a restatement of the user question, requested answer slot, or task constraint with no tool/source evidence, remove it from confirmed_facts. Preserve the requested answer slot in answer_attempts target_role/target_entity, open_questions, or task wording only when needed.
- If a fact is useful but the input only shows assistant reasoning, a guess, a search intention, or a task-context phrase rather than tool/source evidence, keep it out of confirmed_facts and place it in current_hypothesis, open_questions, search_state, answer_attempts, or retry_policy as appropriate.

Compression policy:
- Keep factual, candidate, evidence, search-progress, and retry information only when it is explicitly present in previous_summary or new_context. Use task_context only to preserve the original requested answer slot and user constraints; do not treat it as independent source evidence.
- Treat previous_summary as the current compact state and the output as its replacement after applying new_context. new_context is a raw delta; prior injected [context_compression_summary] artifacts may be intentionally omitted from it, and that absence is not a contradiction or a reason to drop carried-forward state. Do not append old summary history back into the output.
- Preserve useful prior summary items unless new_context clearly contradicts or supersedes them.
- Treat source status conservatively: say "source missing" when no source is shown, "tool result omitted" when the referenced tool output is unavailable, and lower confidence when evidence is indirect.
- Preserve reasoning-language continuity. Natural-language explanatory values (facts, hypotheses, reasons, blockers, open questions, search-state results, retry reasons, and evidence notes) should preserve the agent's existing working-language pattern as inferable from assistant reasoning in previous_summary and new_context. The goal is continuity, not localization: if the agent has been reasoning in English while quoting a non-English user task, English explanatory summary prose is acceptable; if the agent has been reasoning in another language, preserve that language. Use task_context, when present, to preserve the original requested answer slot and constraints and as a weak language signal, not as a hard requirement to translate the reasoning trail. Do not let web pages, tool outputs, source titles, quoted snippets, or search queries by themselves switch the summary's explanatory language. When previous_summary language conflicts with the surrounding assistant reasoning language in new_context, prefer the surrounding assistant reasoning language while preserving the same facts, status, blockers, and next checks. When a value needs an exact entity name, title, source name, quoted phrase, URL, search query, or alias in another language, keep that exact span unchanged and write the surrounding explanatory prose in the preserved working-language pattern. Do not introduce extra language mixing, and do not translate names, titles, quotes, URLs, queries, or aliases unless the input provides that translation or alias. Keep schema keys and enum values exactly as specified.
- Put stable, decision-relevant facts in confirmed_facts. Do not put guesses, task progress, or reasoning conclusions there.
- Put uncertain but useful working theories in current_hypothesis.
- In current_hypothesis, do not use settled wording such as "the target is ..." or "the answer is ..." when core clues, entity mappings, calculation inputs, or multi-hop constraints remain unresolved for that candidate. Phrase it as a candidate under investigation, include the blocker in the hypothesis text, and keep confidence low when it depends on unresolved conflict_candidates or open_questions.
- Put explored paths in rejected_paths only when the input shows direct contradiction or a clearly exhausted search direction.
- Put unresolved blockers and missing checks in open_questions.
- Put live final-answer candidates that were considered but not finalized in answer_attempts. Keep the candidate answer, target role, target entity, support, missing evidence, confidence, decision, blocker, and retry reason.
- If previous_summary and new_context point to different final-answer candidates or different values for the same answer slot, treat this as a candidate flip. Do not silently replace the earlier supported candidate. Preserve the old and new candidates in conflict_candidates unless the input directly resolves the old candidate as wrong.
- Keep intermediate clue entities out of answer_attempts unless they are themselves the requested final answer. Store useful intermediate entities in confirmed_facts, current_hypothesis, search_state, or open_questions.
- In answer_attempts, do not mark entity_match_status as confirmed, confidence as high, or decision as ready_if_verified when any core task constraint for that candidate remains missing, contradicted, or listed in missing_evidence or blocking_conflict. Treat unresolved multi-hop clue chains as blockers, not minor final checks.
- If multiple values or entities remain plausible for the same answer slot, preserve them in conflict_candidates rather than choosing silently.
- If a clue phrase can map to multiple entities, preserve the competing entity mappings in conflict_candidates or open_questions. Do not promote one mapping to a high-confidence hypothesis unless direct tool evidence rules out the others.
- Preserve calculation assumptions and derived numeric targets as assumptions with their required next check when any required input, entity mapping, formula, or constraint remains ambiguous.
- Use dismissed_candidates only for named final-answer candidates that the input shows were considered and then set aside without full verification.

Candidate convergence rules:
- Treat answer_attempts as the compact live-candidate state for the requested final answer slot, not as a dump of every named lead. Keep a source-backed previous answer_attempt available unless new_context directly contradicts it or proves it is the wrong target.
- Do not copy a large slate of weak same-slot alternatives into answer_attempts. If several candidates share the same final slot and none is clearly answer-ready, put the competing values in conflict_candidates and keep answer_attempts focused on the strongest source-backed live candidate or an explicit not_answered blocker.
- Move candidates with direct contradictory evidence, wrong-target evidence, or only weak/failed clue support to rejected_paths or dismissed_candidates instead of keeping them as active answer_attempts only because they were mentioned. Keep such a candidate in answer_attempts only when the input shows it still anchors a live conflict the next agent must resolve.
- For a source-backed live candidate, preserve what is settled, the concrete blocker that could change the ranking, and the next discriminating check. Do not let generic "search more alternatives" pressure or repeated null searches displace that candidate.
- If the same candidate remains the best source-backed candidate across previous_summary and new_context and no named contradiction appears, preserve it as the lead answer_attempt and close stale broad searches around it as mostly_exhausted or dead_end search_state rather than new open_questions.

- Group repeated searches in search_state. Preserve the strategy, useful result, dead end, or retry need, not every raw query.
- In search_state, use one row per search strategy or source family. Include compact coverage such as checked ranges, key named leads, and remaining gaps; do not repeat every query, failed URL, or evidence already captured in confirmed_facts or answer_attempts.
- Rebuild search_state on each update instead of appending a historical ledger. Carry forward an old search_state item only if it still changes the next search action, source-verification plan, or candidate disambiguation; otherwise merge it into a broader strategy row or drop it when its evidence/blocker is already captured elsewhere.
- Keep search_state action-oriented: what strategy/source family was tried, what useful lead or gap remains, and what exact retry or verification is still needed. Do not let stale broad searches, repeated misses, or completed checks dominate the summary over active candidates and blockers.
- Use retry_policy for paths that should be retried, rephrased, source-verified, softly rejected, or hard rejected.
- Treat search_state, open_questions, and retry_policy as a forward plan, not an instruction to keep searching forever. When a broad strategy has been tried and no concrete named lead, source, or discriminator remains, record it once as mostly_exhausted or dead_end and do not also add open_questions or retry_policy items that ask for more broad variants of the same search.
- When new_context addresses an old open question or retry item, remove it or turn it into a closed search_state/rejected_paths note instead of carrying it forward as a live blocker.
- When an answer_attempt already has direct evidence for the requested final slot and no named conflict remains, do not add broad alternative-search questions just to challenge it. Add an open question only for a specific unresolved core clue, named competing candidate, conflicting value, missing source, or answer-format check.
- For each active conflict, prefer the next discriminating check that would actually change the candidate ranking. Avoid duplicating the same next check across search_state, open_questions, and retry_policy.

Before writing the summary, silently check each item:
1. Is it explicitly supported by previous_summary or new_context, except for original answer-slot or user-constraint wording preserved outside confirmed_facts from task_context?
2. Is it in the right top-level section?
3. Is the source/evidence status overstated?
4. Is the confidence too strong?
5. Does it help the next agent continue the task without repeating old work?
6. If it is an answer attempt, does it describe a candidate for the requested answer slot rather than only an intermediate clue?
7. If it is a confirmed_facts item, is it attributed to tool/search/scrape/source evidence rather than task_context, original task wording, a plan, or a candidate-to-task matching claim?
8. If it is a confirmed_facts item, have you avoided source or evidence_status values such as task_context, user question, requested constraint, or requested?

Only include items that pass this check. Do not write the check itself.

Output format rules:
- Output exactly one summary block.
- The block MUST begin with the literal line "[context_summary]" (square brackets, no angle brackets).
- Immediately after that line, output a single JSON object (one `{ ... }` document).
- Immediately after that JSON object, output the literal line "[/context_summary]".
- Do not include explanations, reasoning traces, hidden chain-of-thought, or analysis before or after the block.
- Do not include any other XML-style tags, Markdown headings, or code fences.
- All nine required keys must appear as top-level keys in the JSON object, each bound to a JSON array (possibly empty).

Use this exact schema (JSON, top-level object with nine required keys):

[context_summary]
{
  "confirmed_facts": [
    {
      "fact": "...",
      "source": "...",
      "evidence_status": "...",
      "confidence": "high|medium|low"
    }
  ],
  "current_hypothesis": [
    {
      "hypothesis": "...",
      "confidence": "high|medium|low"
    }
  ],
  "rejected_paths": [
    {
      "path": "...",
      "reason": "...",
      "confidence": "high|medium|low"
    }
  ],
  "dismissed_candidates": [
    {
      "candidate": "...",
      "target_role": "...",
      "dismissal_reason": "...",
      "reconsideration_hint": "..."
    }
  ],
  "open_questions": [
    "..."
  ],
  "answer_attempts": [
    {
      "candidate_answer": "...",
      "target_role": "...",
      "target_entity": "...",
      "entity_match_status": "confirmed|ambiguous|conflicting|wrong_target",
      "supporting_evidence": ["..."],
      "missing_evidence": ["..."],
      "confidence": "high|medium|low",
      "decision": "not_answered|rejected|needs_verification|ready_if_verified",
      "blocking_conflict": "...",
      "retry_reason": "..."
    }
  ],
  "conflict_candidates": [
    {
      "answer_slot": "...",
      "candidates": [
        {
          "value": "...",
          "evidence": "...",
          "source": "...",
          "confidence": "high|medium|low"
        }
      ],
      "conflict_reason": "...",
      "next_check": "..."
    }
  ],
  "search_state": [
    {
      "query_or_strategy": "...",
      "result": "...",
      "status": "useful|useful_needs_verification|weak_result|mostly_exhausted|dead_end|not_tried"
    }
  ],
  "retry_policy": [
    {
      "item": "...",
      "policy": "hard_reject|soft_reject|needs_rephrase|needs_source_verification",
      "reason": "..."
    }
  ]
}
[/context_summary]"""


USER_PROMPT_TEMPLATE = """Compress and update the search-agent state.

Inputs:
- task_context: original user task/instructions, or "(none)".
- new_context: raw conversation/tool delta after previous_summary; prior injected [context_compression_summary] artifacts are intentionally omitted.
- previous_summary: prior compact state, shown last as the state anchor to update, or "(none yet)".

Task:
- Rewrite one current state summary: start from previous_summary, apply new_context updates, and output the replacement state rather than appended history.
- Follow the system prompt's schema, item-placement rules, evidence/confidence policy, compact ledger budgets, and output format exactly.
- Carry forward useful previous_summary state unless new_context directly contradicts, resolves, or supersedes it; absence from new_context is not a reason to drop it.
- Use only the provided input; preserve missing sources, omitted tool results, uncertainty, and unresolved blockers instead of inventing facts.
- Use task_context to preserve the requested answer slot and original user constraints; do not treat it as independent source evidence or as a hard language target.
- Preserve the agent's existing working-language pattern in explanatory summary values as inferable from assistant reasoning in previous_summary and new_context; do not translate exact names, titles, quotes, URLs, queries, aliases, schema keys, or enum values.
- Output only the updated [context_summary] block.

[task_context]
{task_context}
[/task_context]

[new_context]
{new_context}
[/new_context]

[previous_summary]
{previous_summary}
[/previous_summary]"""


DEFAULT_MAX_TOKENS = 10_240
DEFAULT_RETRY_MAX_TOKENS = 16_384


class ContextCompressionTool:
    """Compress earlier messages into a single ``[context_summary]`` JSON block.

    The model is assumed to be Qwen3 family: the system prompt gets a trailing
    ``/no_think`` and the request rides on Qwen-style ``extra_body`` fields.
    """

    def __init__(self, *, base_url: str, model_name: str, api_key: str) -> None:
        if not base_url or not model_name:
            msg = "ContextCompressionTool requires base_url and model_name"
            raise ValueError(msg)
        self.base_url = base_url
        self.model_name = model_name
        self._api_key = api_key or "dummy_key"
        self._client: Any | None = None
        logger.info(
            "ContextCompressionTool initialized: model=%s, base_url=%s",
            self.model_name,
            self.base_url,
        )

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key, base_url=self.base_url, timeout=600)
        return self._client

    async def call(
        self,
        previous_summary: str,
        task_context: str,
        new_context: str,
        *,
        temperature: float = 0.6,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        top_p: float = 0.95,
        top_k: int = 20,
    ) -> dict[str, Any]:
        system_prompt = SYSTEM_PROMPT + "\n\n/no_think"
        user_prompt = USER_PROMPT_TEMPLATE.format(
            previous_summary=previous_summary,
            task_context=task_context,
            new_context=new_context,
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
                "top_k": top_k,
            },
        }

        client = self._get_client()
        try:
            response = await client.chat.completions.create(**request_kwargs)
            raw = response.choices[0].message.content or ""
            summary = self._sanitize_output(raw)
            is_valid, validation_error = self._validate_summary(summary)

            if not is_valid and self._should_retry_invalid_summary(
                validation_error=validation_error,
                raw=raw,
                max_tokens=max_tokens,
            ):
                retry_max_tokens = max(DEFAULT_RETRY_MAX_TOKENS, max_tokens + 1024)
                retry_kwargs = dict(request_kwargs)
                retry_kwargs["messages"] = self._with_retry_instruction(messages, validation_error=validation_error)
                retry_kwargs["max_tokens"] = retry_max_tokens
                logger.warning(
                    "Retrying invalid compression summary: %s; max_tokens %d -> %d",
                    validation_error,
                    max_tokens,
                    retry_max_tokens,
                )
                response = await client.chat.completions.create(**retry_kwargs)
                raw = response.choices[0].message.content or ""
                summary = self._sanitize_output(raw)
                is_valid, validation_error = self._validate_summary(summary)

            if not is_valid:
                logger.error("Invalid compression summary skipped: %s", validation_error)
                return {"success": False, "summary": "", "raw": raw, "error": f"invalid summary: {validation_error}"}
            return {"success": True, "summary": summary, "raw": raw}
        except Exception as exc:
            logger.exception("ContextCompressionTool error")
            return {"success": False, "summary": "", "error": str(exc)}

    def _should_retry_invalid_summary(
        self,
        *,
        validation_error: str,
        raw: str,
        max_tokens: int,
    ) -> bool:
        retry_max_tokens = max(DEFAULT_RETRY_MAX_TOKENS, max_tokens + 1024)
        if max_tokens >= retry_max_tokens:
            return False
        if self._looks_like_truncated_summary(raw):
            return True
        if validation_error in {"empty summary", "empty summary body"}:
            return True
        if not raw.strip():
            return False
        retryable_prefixes = (
            "invalid JSON body",
            "missing opening marker",
            "missing closing marker",
            "missing top-level key",
        )
        if any(validation_error.startswith(prefix) for prefix in retryable_prefixes):
            return True
        return validation_error == "summary body must be a JSON object" or validation_error.endswith(" must be a JSON array")

    @staticmethod
    def _looks_like_truncated_summary(raw: str) -> bool:
        if not raw:
            return False
        has_open = SUMMARY_MARKER_OPEN in raw
        has_close = SUMMARY_MARKER_CLOSE in raw
        if has_open and not has_close:
            return True
        stripped = raw.strip()
        return stripped.startswith(SUMMARY_MARKER_OPEN) and not stripped.endswith(SUMMARY_MARKER_CLOSE)

    @staticmethod
    def _with_retry_instruction(
        messages: list[dict[str, str]],
        *,
        validation_error: str,
    ) -> list[dict[str, str]]:
        retry_instruction = (
            "\n\n[compression_retry_instruction]\n"
            f"Your previous response was not accepted: {validation_error}. "
            "Rewrite the summary much more compactly as strict JSON. Merge repeated "
            "searches, drop low-value details, keep only decision-critical state, "
            "escape all internal quotes and backslashes inside JSON strings, do not "
            "use trailing commas or comments, include all nine required top-level "
            "array keys, make each top-level key value a JSON array even when it has "
            "one item, and output one complete [context_summary] JSON block with "
            "the closing [/context_summary] marker.\n"
            "[/compression_retry_instruction]"
        )
        retry_messages = [dict(message) for message in messages]
        retry_messages[-1]["content"] = retry_messages[-1]["content"] + retry_instruction
        return retry_messages

    @staticmethod
    def _sanitize_output(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"<\|[^|]*\|>", "", text)
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

        open_escape = re.escape(SUMMARY_MARKER_OPEN)
        close_escape = re.escape(SUMMARY_MARKER_CLOSE)
        pattern = rf"{open_escape}\s*(\{{.*?\}})\s*{close_escape}"
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return f"{SUMMARY_MARKER_OPEN}\n{match.group(1).strip()}\n{SUMMARY_MARKER_CLOSE}"
        try:
            parsed = json.loads(text.strip())
        except json.JSONDecodeError:
            return ""
        if isinstance(parsed, dict):
            return f"{SUMMARY_MARKER_OPEN}\n{json.dumps(parsed, ensure_ascii=False, indent=2)}\n{SUMMARY_MARKER_CLOSE}"
        return ""

    @staticmethod
    def _validate_summary(summary: str) -> tuple[bool, str]:
        if not summary.strip():
            return False, "empty summary"
        stripped = summary.strip()
        if not stripped.startswith(SUMMARY_MARKER_OPEN):
            return False, f"missing opening marker {SUMMARY_MARKER_OPEN}"
        if not stripped.endswith(SUMMARY_MARKER_CLOSE):
            return False, f"missing closing marker {SUMMARY_MARKER_CLOSE}"

        inner = stripped[len(SUMMARY_MARKER_OPEN) : -len(SUMMARY_MARKER_CLOSE)].strip()
        if not inner:
            return False, "empty summary body"

        try:
            parsed = json.loads(inner)
        except json.JSONDecodeError as exc:
            return False, f"invalid JSON body: {exc}"

        if not isinstance(parsed, dict):
            return False, "summary body must be a JSON object"

        for key in REQUIRED_SUMMARY_KEYS:
            if key not in parsed:
                return False, f"missing top-level key: {key}"
            if not isinstance(parsed[key], list):
                return False, f"{key} must be a JSON array"

        return True, ""


def format_messages_for_compression(messages: list[ConversationMessage]) -> str:
    """Flatten conversation messages into a plain-text block for the user prompt.

    Existing message renderers target the chat-template/OpenAI-API path rather
    than embedding conversation history inside another prompt as plain text.
    """
    parts: list[str] = []
    for idx, msg in enumerate(messages, 1):
        role_value = msg.role.value if isinstance(msg.role, MessageRole) else str(msg.role)
        rendered = _render_message_text(msg)
        if len(rendered) > 4000:
            rendered = rendered[:4000] + "...[truncated]"
        parts.append(f"[#{idx}] {role_value.upper()}:\n{rendered}")
    return "\n\n".join(parts) if parts else "[no earlier messages]"


def _render_message_text(message: ConversationMessage) -> str:
    if message.role == MessageRole.TOOL:
        body = (message.content or "")[:1000]
        return f"[tool_result] {body}"
    fragments: list[str] = []
    if message.content:
        fragments.append(message.content)
    if message.tool_calls:
        for tool_call in message.tool_calls:
            arguments_repr = str(tool_call.function.arguments)[:500]
            fragments.append(f"[tool_use name={tool_call.function.name} input={arguments_repr}]")
    return "\n".join(fragments)
