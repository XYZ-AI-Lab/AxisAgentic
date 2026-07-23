# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""LLM judge prompt profiles and verifier construction for web-search recipes.

References:
- BrowseComp-ZH and GAIA base wording: Tongyi DeepResearch
  https://github.com/Alibaba-NLP/DeepResearch
- DeepSearchQA judge prompt: official starter notebook
  https://www.kaggle.com/code/andrewmingwang/deepsearchqa-starter-code
"""

from __future__ import annotations

import json
import re

from recipe.web_search.eval.llm_judge_prompt import BROWSECOMP_JUDGE_PROMPT
from recipe.web_search.eval.llm_verifier import LLMVerifier, parse_ab_judge_choice

DEEPSEARCHQA_DEFAULT_JUDGE_MAX_TOKENS = 1024
DEFAULT_JUDGE_MAX_TOKENS = 2

BROWSECOMP_ZH_JUDGE_PROMPT = "\n".join(  # noqa: FLY002
    [
        "请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。您的任务是将结果评定为：【正确】、【错误】。",  # noqa: RUF001
        "",
        "首先，我们将列出每个评定类别的示例，然后请您对新问题的预测答案进行评定。",  # noqa: RUF001
        "以下是【正确】的答复示例：",  # noqa: RUF001
        "```",
        "问题：贝拉克·奥巴马的孩子叫什么名字？",  # noqa: RUF001
        "标准答案：玛丽亚·奥巴马和萨莎·奥巴马",  # noqa: RUF001
        "模型预测1：Malia Obama and Sasha Obama",  # noqa: RUF001
        "模型预测2：玛丽亚和萨沙",  # noqa: RUF001
        "模型预测3：大多数人会说是玛丽亚和萨莎，但我不确定，需要再确认",  # noqa: RUF001
        "模型预测4：巴拉克·奥巴马有两个女儿，她们分别是玛丽亚·安和娜塔莎·玛丽安，但通常称作玛丽亚·奥巴马和萨莎·奥巴马。",  # noqa: RUF001
        "```",
        "这些答复均为【正确】，因为：",  # noqa: RUF001
        "    - 完整地包含了标准答案中的重要信息。",
        "    - 不包含任何与标准答案矛盾的信息。",
        "    - 只关注语义内容，中英文，大小写、标点、语法和顺序不重要。",  # noqa: RUF001
        "    - 答复中出现模糊语句或猜测是可以接受的，前提是包含了标准答案且不含有不正确信息或矛盾。",  # noqa: RUF001
        "",
        "以下是【错误】的答复示例：",  # noqa: RUF001
        "```",
        "问题：巴拉克·奥巴马的孩子叫什么名字？",  # noqa: RUF001
        "标准答案：玛丽亚·奥巴马和萨莎·奥巴马",  # noqa: RUF001
        "模型预测1：玛丽亚",  # noqa: RUF001
        "模型预测2：玛丽亚、萨莎和苏珊和萨莎·奥巴马或玛丽亚·奥巴马，或娜塔莎·玛丽安，或爱因斯坦",  # noqa: RUF001
        "模型预测3：虽然我不知道他们的确切名字，但能说出巴拉克·奥巴马有两个孩子。",  # noqa: RUF001
        "模型预测4：你可能是想说贝茜和奥利维亚。不过您应通过最新的参考资料确认详细信息。那是正确的答案吗？",  # noqa: RUF001
        "模型预测5：巴拉克·奥巴马的孩子",  # noqa: RUF001
        "```",
        "这些答复均为【错误】，因为：",  # noqa: RUF001
        "    - 答复中包含与标准答案矛盾的事实陈述。",
        "    - 答案为空、重复表述问题。",
        "    - 答案枚举了多个答案，重复表述答案。",  # noqa: RUF001
        "",
        "需要格外注意的是：",  # noqa: RUF001
        '- 标准答案中包含对于问题中多个方面的回答，并且在同一个方面的答案中可能会有多种不同的描述，这些描述均是正确的，并且在同一个括号中给出，通过逗号连接。例如，考虑问题"抖音自己的人工智能大模型叫什么名字？"，标准答案为"【【豆包，云雀】】"：',  # noqa: RUF001
        '    - 预测答案"豆包"、"豆包、云雀"、"云雀"等均为【正确】。',
        '- 对于标准答案中包含的不同方面的回答，模型需要同时给出所有方面的回答才可以算是正确，否则直接判断为【错误】，不存在【部分正确】这种输出方式，这些答案会在不同的括号中给出。例如，考虑问题"TFBOYS组合中的成员有哪些？"，标准答案为"【【王俊凯】【王源】【易洋千玺】】"：',  # noqa: RUF001
        '    - 预测答案"王俊凯、王源、易洋千玺"等同时包含所有答案，才可以算为【正确】。',  # noqa: RUF001
        '    - 预测答案为"王俊凯、易洋千玺"等没有同时包含所有答案，会被算为【错误】。',  # noqa: RUF001
        "",
        "另外注意以下几点：",  # noqa: RUF001
        '- 对于标准答案为数字的问题，预测答案应和标准答案一致。例如，考虑问题"金山铁路黄浦江特大桥的全长是多少米？"，标准答案为"3518.17"：',  # noqa: RUF001
        '    - 预测答案"3518"、"3518.1"、"3518.17"均为【正确】。',
        '    - 预测答案"3520"和"3600"均为【错误】。',
        "- 如果模型预测并没有直接回答问题，模型试图绕过或未能直接给出标准答案视为【错误】答案。",  # noqa: RUF001
        '    - 例如：问题"林宥嘉的老婆是谁"，标准答案为"丁文琪"。模型预测"林宥嘉的老婆"、"林宥嘉的老婆应该很优秀"、"林宥嘉的老婆可能是某个公众人物"均为【错误】。',  # noqa: RUF001
        "- 如果标准答案包含比问题更多的信息，预测答案只需包含问题中提到的信息。",  # noqa: RUF001
        '    - 例如，考虑问题"菱镁矿的主要化学成分是什么？"标准答案为"碳酸镁（MgCO3）"。"碳酸镁"或"MgCO3"均视为【正确】答案。',  # noqa: RUF001
        "- 如果从问题中明显可以推断出预测答案省略的信息，那么算作正确。",  # noqa: RUF001
        '    - 例如，问题"巴鲁米尼的努拉吉遗迹在1997年被联合国教科文组织列为世界文化遗产，那么这遗址在哪个地区？"标准答案为"意大利撒丁岛"，预测答案"撒丁岛"被视为【正确】。',  # noqa: RUF001
        "- 如果能明显看出名字翻译版本不同但是是同一个人也认为正确。",
        '    - 例如，如果标准答案是"Robinson"，那么回答鲁滨逊或者鲁滨孙均正确。',  # noqa: RUF001
        "- 你应该更关注标准答案和模型预测的匹配度，而不是关心标准答案是否是正确的。",  # noqa: RUF001
        "",
        "下面是一个新的问题示例。请只回复【正确】、【错误】之一，不要道歉或纠正自己的错误，只需要评估该回答。",  # noqa: RUF001
        "```",
        "问题: {question}",
        "标准答案: {correct_answer}",
        "预测答案: {response}",
        "```",
        "",
        "将此新问题的预测答案评定为以下之一：",  # noqa: RUF001
        "A.【正确】",
        "B.【错误】",
        "",
        "只返回【正确】、【错误】所代表的选项即可，即仅返回A或B即可，无须添加任何其他的文本。",  # noqa: RUF001
    ],
)

GAIA_JUDGE_PROMPT = """
Based on the given question, standard answer, and model-predicted answer, evaluate whether the model's response is correct. \
Your task is to classify the result as: [CORRECT] or [INCORRECT].

First, we'll list examples for each category, then you'll evaluate a new question's predicted answer.
Here are examples of [CORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia Obama and Sasha Obama
Model Prediction 2: Malia and Sasha
Model Prediction 3: Most would say Malia and Sasha, but I'm not sure, I should verify
Model Prediction 4: Barack Obama has two daughters, Malia Ann and Natasha Marian, commonly known as Malia Obama and Sasha Obama.
```
These responses are all [CORRECT] because they:
    - Fully include the important information from the standard answer.
    - Don't contain any information that contradicts the standard answer.
    - Focus only on semantic content; language, capitalization, punctuation, grammar, and order aren't important.
    - Vague statements or guesses are acceptable as long as they include the standard answer and \
don't contain incorrect information or contradictions.

Here are examples of [INCORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia
Model Prediction 2: Malia, Sasha and Susan or Sasha Obama or Malia Obama, or Natasha Marian, or Einstein
Model Prediction 3: While I don't know their exact names, I can tell you Barack Obama has two children.
Model Prediction 4: You might be thinking of Betsy and Olivia. But you should verify the details with the latest references. \
Is that the correct answer?
Model Prediction 5: Barack Obama's children
```
These responses are all [INCORRECT] because they:
    - Contain factual statements that contradict the standard answer.
    - Are empty or merely repeat the question.
    - Enumerate multiple answers or repeat the answer.

Pay special attention to the following:
- The standard answer may contain responses to multiple aspects of the question, and within the same aspect, \
there might be different descriptions, all of which are correct and are given in the same bracket, connected by commas. \
For example, for the question "What is the name of ByteDance's AI model?", the standard answer is "[[Doubao, Skylark]]":
    - Predicted answers "Doubao", "Doubao, Skylark", "Skylark", etc. are all [CORRECT].
- For standard answers containing responses to different aspects, the model needs to provide answers to all aspects to be considered correct; \
otherwise, it's directly judged as [INCORRECT]. There is no [PARTIALLY CORRECT] output option. These answers will be given in different brackets. \
For example, for the question "Who are the members of TFBOYS?", the standard answer is "[[Wang Junkai][Wang Yuan][Yi Yangqianxi]]":
    - Predicted answers like "Wang Junkai, Wang Yuan, Yi Yangqianxi" that include all answers are [CORRECT].
    - Predicted answers like "Wang Junkai, Yi Yangqianxi" that don't include all answers are [INCORRECT].

Also note the following points:
- For questions with numerical standard answers, the predicted answer should match the standard answer. \
For example, for the question "What is the total length in meters of the Huangpu River Bridge on the Jinshan Railway?", \
the standard answer is "3518.17":
    - Predicted answers "3518", "3518.1", "3518.17" are all [CORRECT].
    - Predicted answers "3520" and "3600" are [INCORRECT].
- If the model prediction doesn't directly answer the question, attempts to circumvent or fails to directly provide the standard answer, \
it's considered an [INCORRECT] answer.
    - For example, for the question "Who is JJ Lin's wife?", with the standard answer "Ding Wenqi", \
model predictions like "JJ Lin's wife", "JJ Lin's wife should be excellent", "JJ Lin's wife might be a public figure" are all [INCORRECT].
- If the standard answer contains more information than the question asks for, \
the predicted answer only needs to include the information mentioned in the question.
    - For example, for the question "What is the main chemical component of magnesite?", \
with the standard answer "Magnesium carbonate (MgCO3)", "Magnesium carbonate" or "MgCO3" are both considered [CORRECT] answers.
- If information omitted in the predicted answer can be clearly inferred from the question, it's considered correct.
    - For example, for the question "The Nuragic ruins of Barumini were listed as a World Cultural Heritage by UNESCO in 1997, \
so where is this site located?", with the standard answer "Sardinia, Italy", the predicted answer "Sardinia" is considered [CORRECT].
- If it's clear that different translations of a name refer to the same person, it's considered correct.
    - For example, if the standard answer is "Robinson", answers like "Lubinson" or "Lubinsun" are both correct.
- You should focus more on the match between the standard answer and the model prediction, rather than whether the standard answer itself is correct.
- Accept common abbreviation/full-form equivalences, e.g. “St.” = “Saint”.
- Ignore differences in capitalization, punctuation, currency symbols, and harmless extra context.
- Accept format variations as long as the core entity, value, or fact matches the gold answer.
- Do not mark an answer as correct solely because it appears to be a spelling mistake.
- If the gold answer contains a typo, fix the data instead of loosening the judge prompt.
- Numerical Expressions: Words representing numbers (e.g., "Five Hundred") and their Arabic numeral counterparts (e.g., "500") are strictly equivalent.
- Punctuation and Spacing: Ignore differences in punctuation (colons, dashes, quotes, commas) and extra/missing spaces (e.g., "Too Late:" vs. "Too Late").

Below is a new question example. \
Please reply with only [CORRECT] or [INCORRECT], without apologies or corrections to your own errors, just evaluate the answer.
```
Question: {question}
Standard Answer: {correct_answer}
Predicted Answer: {response}
```

Reply with only CORRECT if they are equivalent, or INCORRECT if they are not equivalent.
""".strip()


HLE_JUDGE_PROMPT = """You are a strict but equivalence-aware answer verifier.

Judge whether the final answer in the Response should be accepted as correct for the Question, using the Reference correct answer.

Question:
{question}

Response:
{response}

Reference correct answer:
{correct_answer}

First, internally normalize the Response final answer and the Reference answer before judging:
- Simplify algebraic expressions, signs, fractions, powers, and constants.
- Expand or factor expressions when this proves exact equivalence.
- Substitute definitions given in the question or reference answer, such as Hc = 2Jc b/pi.
- Normalize units and scientific notation, including micro/milli prefixes.
- Normalize harmless formatting: LaTeX, spacing, punctuation, capitalization, commas, labels, and delimiters.
- For multiple-choice questions, map option letters to option text. Accept either the correct option letter, the correct option text, or both, if they point to the same choice.
- For unordered sets/lists, ignore order unless the question explicitly requires order.
- Treat infinity notation such as infinity, infinite, and \\infty as equivalent.

ACCEPT if, after normalization, the final answer satisfies any of the following:
1. The final answer exactly matches the reference.
2. It is mathematically equivalent to the reference answer by standard simplification, expansion, factorization, sign rearrangement, or fraction simplification.
3. It is an equivalent floor/ceil/piecewise form of the reference expression.
4. It gives the same formula after substituting definitions supplied by the question or reference.
5. It differs only by reasonable rounding, requested precision, or gives a more precise value consistent with the reference.
6. It uses equivalent units or scientific notation with the same physical value.
7. It gives the same structured answer with harmless delimiter or label differences, such as `a = ...` vs `a: ...`.
8. It gives the same entity/work/object under a synonymous or historically equivalent name, and extra context does not contradict the reference.
9. It gives the reference answer plus extra non-contradictory context, without adding wrong requested items.

HARD REJECT: Mark INCORRECT if, after normalization, any of the following applies:
1. For multiple-choice questions, the option letter and option text point to different choices, or the chosen option differs from the reference.
2. A required number, formula, exponent, sign, coefficient, variable, matrix, vector, set/list item, sequence item, entity, structure, or action is materially different.
3. The response gives only part of a multi-part answer, omits a required item, or answers only one sub-question.
4. The response adds extra wrong items, groups, conditions, or explanations that change the requested answer.
5. A required structured field is missing or has the wrong value.
6. A numerical difference exceeds the precision or tolerance implied by the question.
7. For minimum/optimal/tight complexity questions, the response adds extra asymptotic terms, gives a looser bound, or fails to simplify to the reference tight expression.
8. The final answer is missing, malformed, ambiguous, or cannot be confidently read.
9. The response is merely related, plausible, or partially correct, but not equivalent to the reference.

Be conservative about materially different answers, but do not reject exact mathematical equivalence, unit equivalence, option-text equivalence, or harmless formatting differences.

Reply with only CORRECT or INCORRECT."""

DEEPSEARCHQA_JUDGE_PROMPT = """Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question.
            It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers.
            The order might not matter unless specified otherwise. The response might include more answers than the list.
            Determine the correctness *only* based on the list first and then check if the response includes answers
            not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing
        specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean
        indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer".
            Each key will be a string indicating the expected answer part, and the value will be a boolean indicating
            whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response
        provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers.
        Return an empty list when there's no excessive answers in the response.

**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys:
`"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (a string),
`"Correctness Details"` (a dictionary where each key is the expected correct answer, and the value is a boolean
indicating whether the response contains the correct answer), and `"Excessive Answers"` (a list of strings
indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON
string. Make sure to escape all special characters and quotes in the JSON string.

**Example (Partial):**

"```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true
    }},
    "Excessive Answers": [ "Italy" ]
  }}
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{question}
</prompt>
--------------------

**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{correct_answer}
</answer>
--------------------

AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

--------------------
Rating:"""


def _json_from_judge_response(content: str) -> dict[str, object] | None:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    raw_json = fenced.group(1) if fenced else None
    if raw_json is None:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        raw_json = match.group(0) if match else None
    if raw_json is None:
        return None
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_deepsearchqa_correctness(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def parse_deepsearchqa_judge_choice(content: str | None) -> str | None:
    """Parse the official DeepSearchQA JSON judge response into ``A`` or ``B``."""
    ab_choice = parse_ab_judge_choice(content)
    if ab_choice is not None:
        return ab_choice
    if content is None:
        return None

    parsed = _json_from_judge_response(content)
    if parsed is None:
        return None
    answer_correctness = parsed.get("Answer Correctness")
    if not isinstance(answer_correctness, dict):
        return None
    correctness_details = answer_correctness.get("Correctness Details")
    if not isinstance(correctness_details, dict) or not correctness_details:
        return None
    excessive_answers = answer_correctness.get("Excessive Answers", [])
    if not isinstance(excessive_answers, list):
        return None
    correctness_values = [_coerce_deepsearchqa_correctness(value) for value in correctness_details.values()]
    if any(value is None for value in correctness_values):
        return None
    return "A" if all(correctness_values) and not excessive_answers else "B"


def default_judge_max_tokens_for_benchmark(benchmark_name: str) -> int:
    if benchmark_name == "deepsearchqa":
        return DEEPSEARCHQA_DEFAULT_JUDGE_MAX_TOKENS
    return DEFAULT_JUDGE_MAX_TOKENS


def resolve_judge_max_tokens_for_benchmark(benchmark_name: str, judge_max_tokens: int | None) -> int:
    if judge_max_tokens is None:
        return default_judge_max_tokens_for_benchmark(benchmark_name)
    return max(1, int(judge_max_tokens))


def judge_prompt_template_for_benchmark(benchmark_name: str) -> str:
    if benchmark_name == "browsecomp":
        return BROWSECOMP_JUDGE_PROMPT
    if benchmark_name == "browsecomp_zh":
        return BROWSECOMP_ZH_JUDGE_PROMPT
    if benchmark_name == "gaia":
        return GAIA_JUDGE_PROMPT
    if benchmark_name == "hle":
        return HLE_JUDGE_PROMPT
    if benchmark_name == "deepsearchqa":
        return DEEPSEARCHQA_JUDGE_PROMPT
    if benchmark_name == "livebrowsecomp":
        # LiveBrowseComp answers share BrowseComp's short-factual schema/style, so
        # reuse the official BrowseComp judge template (and its A/B parser) for a
        # single, consistent grading rubric.
        return BROWSECOMP_JUDGE_PROMPT
    msg = f"unsupported web_search benchmark for judge: {benchmark_name!r}"
    raise ValueError(msg)


def create_web_search_llm_verifier(
    *,
    benchmark_name: str,
    judge_model: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key_env: str = "JUDGE_API_KEY",
    max_retries: int = 3,
    judge_times: int = 5,
    judge_max_tokens: int | None = None,
    judge_empty_length_retry_max_tokens: int = 1024,
) -> LLMVerifier:
    return LLMVerifier(
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        judge_api_key_env=judge_api_key_env,
        max_retries=max_retries,
        judge_times=judge_times,
        judge_max_tokens=resolve_judge_max_tokens_for_benchmark(benchmark_name, judge_max_tokens),
        judge_empty_length_retry_max_tokens=judge_empty_length_retry_max_tokens,
        judge_prompt_template=judge_prompt_template_for_benchmark(benchmark_name),
        judge_prompt_profile=benchmark_name,
        judge_response_parser=parse_deepsearchqa_judge_choice if benchmark_name == "deepsearchqa" else None,
    )
