# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

FORMAT_ERROR_MESSAGE = "No \\boxed{} content found in the final answer."

BOXED_ANSWER_BLACKLIST: list[str | None] = [
    "?",
    "??",
    "???",
    "\uff1f",
    "\u2026\u2026",
    "\u2026",
    "...",
    "unknown",
    None,
]

_BOXED_RE = re.compile(r"\\boxed\b", re.DOTALL)

CODE_EXEC_TOOL_RULES = """\
- Use python_exec for Python source code. It runs in a shared stateful interpreter for this task attempt.
  Variables, imports, functions, files, and installed packages can persist across Python calls.
- Use shell_exec only for shell-native work such as ls/cat/touch, make, gcc/g++, curl/wget, pip install,
  or probing system programs. Do not pass Python source code to shell_exec.
- python_exec and shell_exec share the same sandbox filesystem and package environment:
  packages installed with shell_exec can be imported from python_exec, and files written by either tool can be read by the other.
"""

SYSTEM_PROMPT_TEMPLATE = """\
You are a task-solving research agent with access to native structured tools.
Use the tools step-by-step to answer the user's question accurately and completely.
Today is: {date}

# Tool-Use Rules

- Use the function-calling interface to invoke tools. Do not write tool calls as plain text.
- Call at most ONE tool per assistant turn. The next message will contain that tool's result; use it to decide your next step.
- Only use tools declared in the provided tools schema.
- Use web_search to discover relevant sources, verify facts, or find promising URLs.
- Use scrape_and_extract_info to read a specific URL or extract focused facts from a page.
{code_exec_tool_rules}\
- Prefer focused searches and focused extraction requests. Avoid repeating the same query or URL unless the previous result was unusable.
- For source-grounded questions, do not rely only on search snippets when a page needs to be inspected.

# Answering

- When you have sufficient evidence, stop calling tools and provide the final answer wrapped in \\boxed{}.
- Follow the user's requested output format exactly.
- Keep the boxed answer concise: a number, short phrase, or comma-separated list when appropriate.
- Do not output a final prose response without \\boxed{}.
"""

SUMMARY_PROMPT_TEMPLATE = """\
Summarize the above conversation, and output the FINAL ANSWER to the original question.

If a clear answer has already been found, extract that answer and reformat it to match the required format below.
If a definitive answer could not be determined, make a well-informed educated guess based on the conversation.

The original question is repeated here for reference:

"{task_description}"

Wrap your final answer in \\boxed{{}}.
Your final answer should be:
- a number, OR
- as few words as possible, OR
- a comma-separated list of numbers and/or strings.

Strictly follow any formatting instructions in the original question. If asked for a number, express it numerically,
do not use commas, and do not include units unless specified. Do not output final sentence punctuation.
Do not call tools while answering this final prompt.
"""

TOOL_PROFILE_SUMMARY_PROMPT_TEMPLATE = "\n".join(  # noqa: FLY002
    [
        "Summarize the above conversation, and output the FINAL ANSWER to the original question.",
        "",
        (
            "If a clear answer has already been provided earlier in the conversation, do not rethink or recalculate it \u2014 "
            "simply extract that answer and reformat it to match the required format below."
        ),
        "If a definitive answer could not be determined, make a well-informed educated guess based on the conversation.",
        "",
        "The original question is repeated here for reference:",
        "",
        '"{task_description}"',
        "",
        "Wrap your final answer in \\boxed{{}}.",
        "Your final answer should be:",
        "- a number, OR",
        "- as few words as possible, OR",
        "- a comma-separated list of numbers and/or strings.",
        "",
        (
            "ADDITIONALLY, your final answer MUST strictly follow any formatting instructions in the original question \u2014 "
            "such as alphabetization, sequencing, units, rounding, decimal places, etc."
        ),
        (
            "If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, "
            "and DO NOT INCLUDE UNITS such as $ or USD or percent signs unless specified otherwise."
        ),
        (
            "If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. "
            "Don't output any final sentence punctuation such as '.', '!', or '?'."
        ),
        "If you are asked for a comma-separated list, apply the above rules depending on whether the elements are numbers or strings.",
        "Do NOT include any punctuation such as '.', '!', or '?' at the end of the answer.",
        "Do NOT include any invisible or non-printable characters in the answer output.",
        "",
        "You must absolutely not perform any tool call, tool invocation, search, scrape, code execution, or similar actions.",
        "You can only answer the original question based on the information already retrieved and your own internal knowledge.",
        "If you attempt to call any tool, it will be considered a mistake.",
    ]
)

FAILURE_SUMMARY_PROMPT = """\
The task was not completed successfully. Do NOT call any tools. Provide a summary:

Failure type: [incomplete / blocked / misdirected / format_missed]
What happened: [describe the approach taken and why a final answer was not reached]
Useful findings: [list facts, URLs, intermediate results, or conclusions discovered that should be reused]"""

FAILURE_SUMMARY_ASSISTANT_PREFIX = """\
We need to write a structured post-mortem summary without calling tools.

"""

_FAILURE_EXPERIENCE_ITEM_TEMPLATE = """\
[Attempt {attempt_number}]
{failure_summary}
"""

_ENHANCED_FAILURE_EXPERIENCE_TEMPLATE = """\
{original_task}

=== Previous Attempts Analysis ===
The following summarizes what was tried before and why it did not work. Use this to guide a new approach.

{failure_items}
=== End of Analysis ===

Based on the above, try a different strategy this time.
"""


_LIVEBROWSECOMP_TOOL_RULE_LINE = (
    "You only have access to the tools declared in the provided tools schema. Use the agent tool-call interface step-by-step; "
    "call at most ONE tool per assistant turn, then use the next tool response to decide the next step."
)

LIVEBROWSECOMP_SYSTEM_PROMPT_TEMPLATE = "\n".join(  # noqa: FLY002
    [
        (
            "You are a deep search assistant. Your primary role is to perform rigorous, multi-step, "
            "multi-source investigations on any topic--covering both broad, open-domain questions and "
            "highly specialized academic inquiries. For each user request, you must actively seek out and "
            "cross-check information from credible and diverse sources, then integrate the findings into a "
            "response that is comprehensive, accurate, well-structured, and objective."
        ),
        "Operating principles:",
        (
            "1. Plan and execute research: Break complex questions into sub-questions, gather evidence across "
            "multiple sources, and prioritize primary sources and authoritative references when available."
        ),
        (
            "2. Evaluate source quality: Prefer reputable institutions, peer-reviewed research, official "
            "documentation, and high-quality journalism. Note uncertainty, conflicts, and limitations when "
            "sources disagree."
        ),
        (
            "3. Synthesize, don't just list: Combine evidence into a coherent narrative or structured output, "
            "highlighting key takeaways and nuanced trade-offs."
        ),
        ("4. Maintain neutrality: Present competing viewpoints fairly when relevant, and avoid unsupported speculation."),
        "",
        _LIVEBROWSECOMP_TOOL_RULE_LINE,
        "Today is: {date}",
        "",
        (
            "When you have collected sufficient information and are ready to deliver the definitive response, "
            "you must wrap the entire final answer in \\boxed{}."
        ),
    ]
)


# Same as livebrowsecomp but without the tool-call rule line — for the no-tools baseline.
LIVEBROWSECOMP_NOTOOLS_SYSTEM_PROMPT_TEMPLATE = "\n".join(
    line for line in LIVEBROWSECOMP_SYSTEM_PROMPT_TEMPLATE.split("\n") if line != _LIVEBROWSECOMP_TOOL_RULE_LINE
)


# Sentinels (case-insensitive) that resolve system_prompt_date to the current
# system date at runtime instead of a pinned literal.
_AUTO_DATE_SENTINELS = frozenset({"today", "auto", "now"})


def _resolve_prompt_date(date: str | None) -> str:
    """Resolve the system-prompt date, auto-detecting today when unset/sentinel.

    ``None`` or one of the ``today``/``auto``/``now`` sentinels resolves to the
    current date (Asia/Shanghai, UTC+8). Any other value is used verbatim so
    runs can pin a fixed date for reproducibility.
    """
    if date is not None and date.strip().lower() not in _AUTO_DATE_SENTINELS:
        return date
    return datetime.now(tz=timezone(timedelta(hours=8))).strftime("%Y-%m-%d")


# System prompt for the "deepsearchqa" profile.
# Selected by generate_system_prompt() when prompt_profile == "deepsearchqa";
# wire it up from an eval config via:  agent.prompt_profile: deepsearchqa
# "{date}" is substituted with the run date (see _resolve_prompt_date / system_prompt_date).
# Section map:
#   intro + tool rules : one tool-call per turn, no plain-text MCP XML.
#   # Research Method   : decompose multi-constraint questions, cross-check across sources, track uncertainty.
#   # single-answer     : stop once the unique answer is corroborated.
#   # set / list        : balance COMPLETENESS (enumerate all, don't stop early) vs PRECISION (only verified
#                         items, don't pad) — targets DSQA's found-but-not-submitted and over-padding failures.
#   # Output            : follow the requested format; final answer in \boxed{}; no tool calls while finalizing.
DEEPSEARCHQA_SYSTEM_PROMPT_TEMPLATE = """You are a deep research agent that answers questions through rigorous, multi-step, multi-source investigation. You work iteratively, breaking each task into clear sub-questions and resolving them methodically with tools.

You only have access to the tools declared in the provided tools schema. Use the agent tool-call interface step-by-step; call at most ONE tool per assistant turn, then use the next tool response to decide the next step.
Today is: {date}

# Research Method
- Decompose: break complex / multi-constraint questions into sub-questions; apply the constraints one at a time to narrow down candidates (e.g. "born same day as X" -> "debuted at 22" -> "role name contains a number").
- Cross-check: corroborate every key fact across MULTIPLE credible sources; prefer primary and authoritative references. Never commit to a candidate on a single weak source.
- Track uncertainty: note conflicts and gaps; discard any candidate you cannot verify against EVERY stated condition.

# Answering -- single-answer questions
- Once you have corroborated evidence for the unique answer, stop and report it.

# Answering -- set / list questions (the question asks for ALL items meeting conditions)
Treat COMPLETENESS and PRECISION as equally important:
- COMPLETENESS (avoid missing items): systematically enumerate candidates -- scan by time, region, category, or an authoritative list -- and find ALL qualifying items. Do NOT stop after finding a few. Before finalizing, explicitly ask yourself "Are there other items I might have missed?" and search again to close the gaps.
- PRECISION (avoid extra items): include ONLY items you have verified satisfy EVERY stated condition. Exclude anything you are unsure about -- do not pad the list with plausible-but-unverified guesses.
- Before answering, re-check the set BOTH ways: nothing missing, nothing extra.

# Output
- Follow the user's requested output format exactly.
- When you have sufficient evidence, stop calling tools and wrap the final answer in \\boxed{}.
- Do not call tools while producing the final answer."""


def generate_system_prompt(date: str | None = None, *, prompt_profile: str = "default", code_exec_enabled: bool = False) -> str:
    date = _resolve_prompt_date(date)
    if prompt_profile == "deepsearchqa":
        return DEEPSEARCHQA_SYSTEM_PROMPT_TEMPLATE.replace("{date}", date)
    if prompt_profile == "livebrowsecomp_notools":
        return LIVEBROWSECOMP_NOTOOLS_SYSTEM_PROMPT_TEMPLATE.replace("{date}", date)
    if prompt_profile == "livebrowsecomp":
        return LIVEBROWSECOMP_SYSTEM_PROMPT_TEMPLATE.replace("{date}", date)
    code_exec_tool_rules = CODE_EXEC_TOOL_RULES if code_exec_enabled else ""
    return SYSTEM_PROMPT_TEMPLATE.replace("{date}", date).replace("{code_exec_tool_rules}", code_exec_tool_rules)


def generate_user_prompt_template(*, prompt_profile: str = "default") -> str:
    if prompt_profile in ("deepsearchqa", "livebrowsecomp", "livebrowsecomp_notools"):
        return "{task}"
    return "{task}\nFollow the request's format instructions strictly and wrap the final answer in \\boxed{{}}."


def generate_summary_prompt(task_description: str, *, prompt_profile: str = "default") -> str:
    if prompt_profile in ("deepsearchqa", "livebrowsecomp", "livebrowsecomp_notools"):
        return TOOL_PROFILE_SUMMARY_PROMPT_TEMPLATE.format(task_description=task_description)
    return SUMMARY_PROMPT_TEMPLATE.format(task_description=task_description)


def build_failure_enhanced_task(original_task: str, failure_summaries: list[str]) -> str:
    if not failure_summaries:
        return original_task
    failure_items = "\n".join(
        _FAILURE_EXPERIENCE_ITEM_TEMPLATE.format(attempt_number=idx, failure_summary=summary)
        for idx, summary in enumerate(failure_summaries, start=1)
    )
    return _ENHANCED_FAILURE_EXPERIENCE_TEMPLATE.format(original_task=original_task, failure_items=failure_items)


def extract_boxed_content(text: str) -> str:
    if not text:
        return ""

    last_result: str | None = None
    i = 0
    n = len(text)
    while True:
        match = _BOXED_RE.search(text, i)
        if not match:
            break
        j = match.end()
        while j < n and text[j].isspace():
            j += 1
        if j >= n or text[j] != "{":
            i = j
            continue

        depth = 0
        k = j
        escaped = False
        found_closing = False
        while k < n:
            ch = text[k]
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_result = text[j + 1 : k]
                    i = k + 1
                    found_closing = True
                    break
            k += 1

        if not found_closing and depth > 0:
            last_result = text[j + 1 : n]
            i = k
        elif not found_closing:
            i = j + 1

    if last_result in BOXED_ANSWER_BLACKLIST:
        return ""
    return last_result.strip() if last_result is not None else ""
