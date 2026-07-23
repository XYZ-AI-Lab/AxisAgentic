# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from recipe.wide_search.eval.judge_client import WideSearchJudgeClient, parse_markdown_json
from recipe.wide_search.eval.prompts import primary_key_prompt


async def aprimary_key_preprocess(
    response: list[str],
    reference: list[str],
    *,
    client: WideSearchJudgeClient,
    prompt_profile: str = "official",
) -> tuple[dict[str, str], str | None]:
    """Ask the judge LLM to align ``response`` strings to ``reference`` semantics.

    Returns ``(mapping, error)`` where ``mapping[origin] = transform``. On any
    failure, ``mapping`` is empty and ``error`` describes what went wrong; the
    caller continues evaluation with original names (matching official semantics).
    """
    template = primary_key_prompt(prompt_profile)
    prompt = template.format(response=response, reference=reference)
    content = await client.chat_completion(prompt)
    if content is None:
        return {}, "primary_key_preprocess: empty completion"
    parsed = parse_markdown_json(content)
    if parsed is None:
        return {}, "primary_key_preprocess: parse failure"
    cleaned = {k: v for k, v in parsed.items() if isinstance(k, str) and isinstance(v, str)}
    if not cleaned:
        return {}, "primary_key_preprocess: empty mapping"
    return cleaned, None
