# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from recipe.wide_search.eval.judge_client import WideSearchJudgeClient, parse_markdown_json
from recipe.wide_search.eval.prompts import eval_column_prompt


async def allm_judge_column(
    response: list[str],
    target: list[str],
    criterion: object,
    *,
    client: WideSearchJudgeClient,
    prompt_profile: str = "official",
) -> tuple[list[float], list[str]]:
    """Batch-judge a column of (response, target) pairs.

    Sends a single LLM call asking for ``{idx_n: 0|1}`` JSON. On parse failure
    retries once at the completion level (per the deleted in-tree
    ``_llm_judge_column``); on second failure substitutes ``[0.0] * len(pairs)``
    for the whole column. Length is always equal to ``len(response)``.
    """
    if len(response) != len(target):
        msg = f"len(response)={len(response)} != len(target)={len(target)}"
        raise ValueError(msg)

    n = len(response)
    if n == 0:
        return [], []

    response_dict = {f"idx_{i}": {"response": response[i], "target": target[i]} for i in range(n)}
    template = eval_column_prompt(prompt_profile)
    prompt = template.format(criterion=criterion, response=response_dict)

    score_dict: dict[str, object] | None = None
    raw_content: str | None = None
    last_error: str | None = None
    attempts = 1 + max(0, client.config.max_retries)
    for _ in range(attempts):
        content = await client.chat_completion(prompt)
        if content is None:
            last_error = "llm judge column: empty completion"
            continue
        parsed = parse_markdown_json(content)
        if parsed is None:
            last_error = "llm judge column: parse failure"
            raw_content = content
            continue
        score_dict = parsed
        raw_content = content
        last_error = None
        break

    if score_dict is None:
        return [0.0] * n, [last_error or "llm judge column: failed"] * n

    scores: list[float] = []
    for i in range(n):
        raw = score_dict.get(f"idx_{i}", 0)
        try:
            scores.append(float(int(raw)))
        except (TypeError, ValueError):
            scores.append(0.0)
    if len(scores) != n:
        return [0.0] * n, ["llm judge column: length mismatch"] * n

    return scores, [raw_content or ""] * n
