# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""DeepSearchQA macro-F1 verifier.

Follow https://www.kaggle.com/code/andrewmingwang/deepsearchqa-starter-code.
Contains both the pure scoring primitives (dataclasses, prompt builder, JSON
parsers, metric aggregation) and the pipeline entry ``run_dsqa_f1_verify`` that
stitches them to an OpenAI-compatible judge endpoint. ``llm_judge.py`` stays
untouched for byte-identical reproduction of the standard A/B judge — this
module is the single owner of the starter path.
"""

from __future__ import annotations

import collections
import concurrent.futures
import csv
import dataclasses
import hashlib
import json
import logging
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


PROMPT_SOURCE = "kaggle:andrewmingwang/deepsearchqa-starter-code scriptVersionId=285323691"


DSQA_STARTER_TASK_PROMPT = """\
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.


**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (a string), `"Correctness Details"` (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and `"Excessive Answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.


"""


DSQA_STARTER_OUTPUT_EXAMPLE = """**Example (Partial):**

"```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true,
    }},
    "Excessive Answers": [ "Italy" ]
  }}
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

--------------------
Rating:"""


@dataclasses.dataclass
class ItemRating:
    original_index: int | None
    task_id: str
    query: str
    response: str
    category_type: str | None = None
    answer_type: str | None = None
    expected_correct_answer: str | None = None
    answer_correctness_explanation: str | None = None
    expected_correct_answer_list: list[str] | None = None
    response_wrong_answers_list: list[str] | None = None
    grader_ratings_list: list[bool] | None = None
    empty_model_response: bool = False
    empty_auto_rater_response: bool = False
    invalid_auto_rater_response: bool = False
    rating_response: str = ""
    rating_prompt_file: str = ""
    rating_response_file: str = ""
    parsed_response_file: str = ""
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ProjectRating:
    num_total_ratings: int = 0
    num_empty_model_response: int = 0
    num_invalid_auto_rater_response: int = 0
    num_empty_auto_rater_response: int = 0
    num_valid_ratings: int = 0
    num_answer_correctness_evaluated: int = 0
    pct_w_ci_all_answers_correct: str = ""
    pct_w_ci_fully_incorrect_items: str = ""
    pct_w_ci_correct_with_excessive_answers: str = ""
    pct_empty_model_response: float = 0.0
    pct_invalid_auto_rater_response: float = 0.0
    pct_empty_auto_rater_response: float = 0.0
    precision: str = ""
    recall: str = ""
    f1_score: str = ""
    precision_value: float | None = None
    recall_value: float | None = None
    f1_score_value: float | None = None
    num_all_answers_correct: int = 0
    num_fully_incorrect_items: int = 0
    num_correct_with_excessive_answers: int = 0
    micro_true_positives: int = 0
    micro_false_positives: int = 0
    micro_false_negatives: int = 0
    micro_precision: float | None = None
    micro_recall: float | None = None
    micro_f1_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def stable_hash(data: Any) -> str:
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_starter_grader_prompt(*, problem: str, prompt_type: str, answer: str, response: str) -> str:
    return DSQA_STARTER_TASK_PROMPT + DSQA_STARTER_OUTPUT_EXAMPLE.format(
        prompt=str(problem).strip(),
        prompt_type=str(prompt_type).strip(),
        answer=str(answer).strip(),
        response=str(response).strip(),
    )


def parse_json_response(text: str) -> Any:
    stripped = text.strip()
    start_marker = "```json"
    start_idx = stripped.find(start_marker)
    if start_idx != -1:
        stripped = stripped[start_idx + len(start_marker) :].strip()
        end_idx = stripped.rfind("```")
        if end_idx != -1:
            stripped = stripped[:end_idx].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def get_answer_correctness_details(json_response: Any) -> dict[str, bool] | None:
    try:
        details = json_response["Answer Correctness"]["Correctness Details"]
    except (KeyError, TypeError):
        return None
    if not isinstance(details, dict):
        return None
    if not all(isinstance(key, str) for key in details):
        return None
    if not all(isinstance(value, bool) for value in details.values()):
        return None
    return details


def get_excessive_answers(json_response: Any) -> list[str] | None:
    try:
        excessive_answers = json_response["Answer Correctness"]["Excessive Answers"]
    except (KeyError, TypeError):
        return []
    if not isinstance(excessive_answers, list):
        return None
    if not all(isinstance(item, str) for item in excessive_answers):
        return None
    return excessive_answers


def reduce_response(item_rating: ItemRating, grader_response_text: str) -> tuple[ItemRating, Any]:
    """Populate ``item_rating`` from a judge's raw response text.

    Returns the (possibly-mutated) item_rating alongside the parsed JSON dict on
    success (or ``None`` when parsing failed). Callers own persistence of the
    parsed payload — this function performs no disk I/O.
    """
    item_rating.rating_response = grader_response_text
    if not item_rating.response:
        item_rating.empty_model_response = True
        item_rating.error_message = "AI response was empty."
        return item_rating, None
    if not grader_response_text:
        item_rating.empty_auto_rater_response = True
        item_rating.error_message = "Auto-rater response was empty."
        return item_rating, None
    try:
        parsed_json_response = parse_json_response(grader_response_text)
    except Exception as exc:
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = f"Invalid JSON response from auto-rater: {exc!r}"
        return item_rating, None
    if not isinstance(parsed_json_response, dict):
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = "Auto-rater JSON was not an object."
        return item_rating, parsed_json_response
    answer_correctness_node = parsed_json_response.get("Answer Correctness")
    if not isinstance(answer_correctness_node, dict):
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = "Missing or malformed 'Answer Correctness' node."
        return item_rating, parsed_json_response
    explanation = answer_correctness_node.get("Explanation")
    if not isinstance(explanation, str):
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = "Missing or malformed 'Explanation'."
        return item_rating, parsed_json_response
    item_rating.answer_correctness_explanation = explanation
    details = get_answer_correctness_details(parsed_json_response)
    if details is None:
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = "Invalid 'Correctness Details'."
        return item_rating, parsed_json_response
    excessive_answers = get_excessive_answers(parsed_json_response)
    if excessive_answers is None:
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = "Invalid 'Excessive Answers'."
        return item_rating, parsed_json_response
    item_rating.expected_correct_answer_list = list(details.keys())
    item_rating.grader_ratings_list = list(details.values())
    if excessive_answers:
        item_rating.response_wrong_answers_list = excessive_answers
    return item_rating, parsed_json_response


def calculate_ci_str(count: int, total: int, z: float = 1.96) -> str:
    if total == 0:
        return f"N/A ({count}/{total})"
    count = min(max(count, 0), total)
    p = count / total
    margin = z * math.sqrt((p * (1.0 - p)) / total)
    result = f"{p * 100.0:.2f} ± {margin * 100.0:.2f} ({count}/{total})"
    if total <= 5:
        result += " (CI not robust for n<=5)"
    return result


def calculate_metric(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1_score": f1}


def aggregate_ratings(item_ratings: list[ItemRating]) -> tuple[ProjectRating, list[dict[str, Any]]]:
    project = ProjectRating(num_total_ratings=len(item_ratings))
    evaluated = 0
    all_correct = 0
    fully_incorrect = 0
    correct_with_excessive = 0
    per_item_metrics: dict[str, list[float]] = collections.defaultdict(list)
    row_metrics: list[dict[str, Any]] = []

    for rating in item_ratings:
        if rating.invalid_auto_rater_response:
            project.num_invalid_auto_rater_response += 1
            continue
        if rating.empty_auto_rater_response:
            project.num_empty_auto_rater_response += 1
            continue
        if rating.empty_model_response:
            project.num_empty_model_response += 1
            continue
        project.num_valid_ratings += 1
        if rating.grader_ratings_list is None:
            continue

        evaluated += 1
        ratings = rating.grader_ratings_list
        num_correct = sum(1 for value in ratings if value)
        tp = num_correct
        fn = len(ratings) - num_correct
        has_expected = bool(ratings)
        all_expected_answers_correct = False
        if has_expected:
            all_expected_answers_correct = num_correct == len(ratings)
            if num_correct == 0:
                fully_incorrect += 1
        excessive_answers = rating.response_wrong_answers_list or []
        fp = len(excessive_answers)
        if excessive_answers and (all_expected_answers_correct or not has_expected):
            correct_with_excessive += 1
        is_all_correct = (all_expected_answers_correct or not has_expected) and not excessive_answers
        if is_all_correct:
            all_correct += 1

        metric = calculate_metric(tp, fp, fn)
        for key, value in metric.items():
            per_item_metrics[key].append(value)
        project.micro_true_positives += tp
        project.micro_false_positives += fp
        project.micro_false_negatives += fn
        row_metrics.append(
            {
                "task_id": rating.task_id,
                "task_index": rating.original_index,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1_score": metric["f1_score"],
                "fully_correct": is_all_correct,
                "fully_incorrect": has_expected and num_correct == 0,
                "correct_with_excessive_answers": bool(excessive_answers and (all_expected_answers_correct or not has_expected)),
                "expected_answer_count": len(ratings),
                "excessive_answer_count": fp,
            }
        )

    total = len(item_ratings)
    if total:
        project.pct_empty_model_response = round(project.num_empty_model_response * 100.0 / total, 2)
        project.pct_invalid_auto_rater_response = round(project.num_invalid_auto_rater_response * 100.0 / total, 2)
        project.pct_empty_auto_rater_response = round(project.num_empty_auto_rater_response * 100.0 / total, 2)
    if evaluated:
        project.num_answer_correctness_evaluated = evaluated
        project.num_all_answers_correct = all_correct
        project.num_fully_incorrect_items = fully_incorrect
        project.num_correct_with_excessive_answers = correct_with_excessive
        project.pct_w_ci_all_answers_correct = calculate_ci_str(all_correct, evaluated)
        project.pct_w_ci_fully_incorrect_items = calculate_ci_str(fully_incorrect, evaluated)
        project.pct_w_ci_correct_with_excessive_answers = calculate_ci_str(correct_with_excessive, evaluated)
        project.precision_value = sum(per_item_metrics["precision"]) / len(per_item_metrics["precision"])
        project.recall_value = sum(per_item_metrics["recall"]) / len(per_item_metrics["recall"])
        project.f1_score_value = sum(per_item_metrics["f1_score"]) / len(per_item_metrics["f1_score"])
        project.precision = f"{project.precision_value:.2%}"
        project.recall = f"{project.recall_value:.2%}"
        project.f1_score = f"{project.f1_score_value:.2%}"
        micro = calculate_metric(
            project.micro_true_positives,
            project.micro_false_positives,
            project.micro_false_negatives,
        )
        project.micro_precision = micro["precision"]
        project.micro_recall = micro["recall"]
        project.micro_f1_score = micro["f1_score"]
    return project, row_metrics


# ---------------------------------------------------------------------------
# Pipeline entry: stitch the primitives above to an OpenAI-compatible judge
# ---------------------------------------------------------------------------
#
# Kept in this module so the standalone verifier is one artifact. The runner
# wires it in via ``recipe.web_search.runners.run_eval_config`` and gates it
# behind ``judge.dsqa_f1_verify.enabled``; leaving the flag off skips this
# entry-point entirely so old runs stay byte-identical.

_DSQA_TASK_INDEX_RE = re.compile(r"deepsearchqa__(\d+)")

_DSQA_OUT_SOURCE_ROWS = "dsqa_f1_source_rows.json"
_DSQA_OUT_ITEM_RATINGS = "dsqa_f1_item_ratings.jsonl"
_DSQA_OUT_ROW_METRICS = "dsqa_f1_row_metrics.jsonl"
_DSQA_OUT_SUMMARY = "dsqa_f1_verify_summary.json"
_DSQA_OUT_REPORT = "dsqa_f1_verify_report.md"
_DSQA_RAW_DIR = "dsqa_f1_raw"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_dataset(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {index: dict(row) for index, row in enumerate(reader)}


def _parse_task_index(task_id: str) -> int | None:
    match = _DSQA_TASK_INDEX_RE.fullmatch(task_id)
    return int(match.group(1)) if match else None


def _source_rows(run_dir: Path, dataset_rows: dict[int, dict[str, str]]) -> tuple[list[dict[str, Any]], Path]:
    result_path = run_dir / "llm_judge_results.jsonl"
    if not result_path.exists():
        result_path = run_dir / "benchmark_results.jsonl"
    if not result_path.exists():
        raise FileNotFoundError(f"No llm_judge_results.jsonl or benchmark_results.jsonl found under {run_dir}")
    rows = _load_jsonl(result_path)
    merged: list[dict[str, Any]] = []
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        task_index = _parse_task_index(task_id)
        dataset_row = dataset_rows.get(task_index) if task_index is not None else None
        if dataset_row is None:
            raise ValueError(f"Could not map task_id={task_id!r} to dataset row from {result_path}")
        prediction = row.get("prediction", row.get("output", ""))
        merged.append(
            {
                "task_id": task_id,
                "task_index": task_index,
                "problem": dataset_row["problem"],
                "problem_category": dataset_row.get("problem_category", ""),
                "answer": dataset_row["answer"],
                "answer_type": dataset_row["answer_type"],
                "response": "" if prediction is None else str(prediction),
                "source_prediction": prediction,
                "source_ground_truth": row.get("ground_truth"),
                "source_file": str(result_path),
            }
        )
    merged.sort(key=lambda item: item["task_index"])
    return merged, result_path


def _chat_completion(
    *,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    max_tokens: int,
    request_timeout: int,
    temperature: float | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    request = urllib.request.Request(  # noqa: S310 -- URL comes from the configured HTTP(S) judge endpoint.
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=request_timeout) as response:  # noqa: S310 -- Request URL was configured above.
        return json.loads(response.read().decode("utf-8"))


def _extract_content(raw: dict[str, Any]) -> str:
    try:
        return str(raw["choices"][0]["message"].get("content") or "")
    except (KeyError, IndexError, TypeError):
        return ""


def _process_row(
    row: dict[str, Any],
    *,
    raw_dir: Path,
    base_url: str,
    model: str,
    api_key: str,
    request_timeout: int,
    max_tokens: int,
    temperature: float | None,
    retries: int,
    retry_sleep_cap: float,
    resume: bool,
) -> ItemRating:
    task_id = str(row["task_id"])
    item_rating = ItemRating(
        original_index=row["task_index"],
        task_id=task_id,
        query=str(row["problem"]).strip(),
        response=str(row["response"]).strip(),
        category_type=str(row.get("problem_category", "")).strip(),
        answer_type=str(row.get("answer_type", "")).strip(),
        expected_correct_answer=str(row.get("answer", "")).strip(),
    )
    prompt = build_starter_grader_prompt(
        problem=row["problem"],
        prompt_type=row["answer_type"],
        answer=row["answer"],
        response=row["response"],
    )
    stem = f"{task_id}.{stable_hash({'prompt_source': PROMPT_SOURCE, 'prompt': prompt})}"
    prompt_file = raw_dir / "prompts" / f"{stem}.prompt.txt"
    response_file = raw_dir / "responses" / f"{stem}.response.json"
    parsed_file = raw_dir / "parsed" / f"{stem}.parsed.json"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    response_file.parent.mkdir(parents=True, exist_ok=True)
    parsed_file.parent.mkdir(parents=True, exist_ok=True)
    item_rating.rating_prompt_file = str(prompt_file)
    item_rating.rating_response_file = str(response_file)
    if not prompt_file.exists():
        prompt_file.write_text(prompt, encoding="utf-8")

    def _finalize(rating: ItemRating, parsed_payload: Any) -> ItemRating:
        if parsed_payload is not None:
            _write_json(parsed_file, parsed_payload)
            rating.parsed_response_file = str(parsed_file)
        return rating

    if resume and response_file.exists():
        try:
            raw = json.loads(response_file.read_text(encoding="utf-8"))
            content = _extract_content(raw)
            cached_rating, parsed_payload = reduce_response(item_rating, content)
            if not (cached_rating.invalid_auto_rater_response or cached_rating.empty_auto_rater_response):
                return _finalize(cached_rating, parsed_payload)
            item_rating.error_message = f"Cached response invalid, retrying: {cached_rating.error_message}"
            item_rating.invalid_auto_rater_response = False
            item_rating.empty_auto_rater_response = False
            item_rating.rating_response = ""
        except Exception as exc:
            item_rating.error_message = f"Cached response invalid, retrying: {exc!r}"

    last_error: str | None = None
    last_invalid_rating: ItemRating | None = None
    last_invalid_parsed: Any = None
    for attempt in range(1, retries + 1):
        try:
            raw = _chat_completion(
                base_url=base_url,
                model=model,
                api_key=api_key,
                prompt=prompt,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
                temperature=temperature,
            )
            raw["_verifier_attempt"] = attempt
            _write_json(response_file, raw)
            content = _extract_content(raw)
            attempted_rating, parsed_payload = reduce_response(item_rating, content)
            if not (attempted_rating.invalid_auto_rater_response or attempted_rating.empty_auto_rater_response):
                return _finalize(attempted_rating, parsed_payload)
            last_invalid_rating = attempted_rating
            last_invalid_parsed = parsed_payload
            last_error = attempted_rating.error_message
            if attempt < retries:
                time.sleep(min(retry_sleep_cap, 1 + (2 ** (attempt + random.random()))))
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            last_error = repr(exc)
            if attempt < retries:
                time.sleep(min(retry_sleep_cap, 1 + (2 ** (attempt + random.random()))))
    if last_invalid_rating is not None:
        return _finalize(last_invalid_rating, last_invalid_parsed)
    item_rating.empty_auto_rater_response = True
    item_rating.error_message = f"LLM call failed after {retries} attempts: {last_error}"
    return item_rating


def _build_report(experiment: str, base_url: str, model: str, summary: dict[str, Any]) -> str:
    project = summary["project_rating"]
    lines = [
        f"# DeepSearchQA F1 Verify - {experiment}",
        "",
        "## Scope",
        "",
        f"- Experiment: `{experiment}`",
        f"- Rows selected: `{summary['num_source_rows']}`",
        f"- Valid ratings: `{project['num_valid_ratings']}`",
        f"- Evaluated ratings: `{project['num_answer_correctness_evaluated']}`",
        f"- Autorater: `{base_url}` / `{model}`",
        f"- Prompt source: `{PROMPT_SOURCE}`",
        "",
        "## Starter Metrics",
        "",
        f"- F1: `{project['f1_score']}` (`{project['f1_score_value']}`)",
        f"- Precision: `{project['precision']}` (`{project['precision_value']}`)",
        f"- Recall: `{project['recall']}` (`{project['recall_value']}`)",
        f"- Fully Correct: `{project['pct_w_ci_all_answers_correct']}`",
        f"- Fully Incorrect: `{project['pct_w_ci_fully_incorrect_items']}`",
        f"- Correct with Extraneous Answers: `{project['pct_w_ci_correct_with_excessive_answers']}`",
        f"- Invalid autorater responses: `{project['num_invalid_auto_rater_response']}`",
        f"- Empty autorater responses: `{project['num_empty_auto_rater_response']}`",
        "",
    ]
    return "\n".join(lines)


def run_dsqa_f1_verify(
    *,
    run_dir: Path,
    dataset_csv: Path,
    judge_model: str,
    judge_base_url: str,
    api_key_env: str,
    experiment: str | None = None,
    workers: int = 5,
    retries: int = 5,
    request_timeout: int = 180,
    max_tokens: int = 8192,
    temperature: float | None = None,
    retry_sleep_cap: float = 30.0,
    progress_every: int = 50,
    resume: bool = True,
    task_ids: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Score a benchmark run dir with the Kaggle DeepSearchQA starter macro-F1 judge.

    Reads predictions from ``<run_dir>/llm_judge_results.jsonl`` (falls back to
    ``<run_dir>/benchmark_results.jsonl``), maps ``deepsearchqa__<idx>`` task
    ids back to rows in ``dataset_csv``, then dispatches one judge call per row
    via a ThreadPoolExecutor. Emits the ``dsqa_f1_*`` artifact family at the
    run dir root plus a per-task response cache under ``dsqa_f1_raw/``. Returns
    the same summary dict written to ``dsqa_f1_verify_summary.json``.

    ``dataset_csv`` must be the same CSV the generation used — task index
    mapping is only meaningful against the source dataset.
    """
    run_dir = Path(run_dir)
    dataset_csv = Path(dataset_csv)
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(f"run_dsqa_f1_verify: environment variable {api_key_env!r} is empty; cannot authenticate to the judge endpoint")
    dataset_rows = _load_dataset(dataset_csv)
    source_rows, source_result_path = _source_rows(run_dir, dataset_rows)
    if task_ids:
        selected = set(task_ids)
        source_rows = [row for row in source_rows if row["task_id"] in selected]
    if limit is not None:
        source_rows = source_rows[:limit]

    raw_dir = run_dir / _DSQA_RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / _DSQA_OUT_SOURCE_ROWS, source_rows)

    total = len(source_rows)
    exp_name = experiment or run_dir.name
    logger.info(
        "dsqa_f1_verify: %d rows from %s -> judge=%s@%s",
        total,
        source_result_path,
        judge_model,
        judge_base_url,
    )

    def _maybe_print_progress(done: int, task_id: str) -> None:
        step = max(1, progress_every)
        if done == total or done <= 5 or done % step == 0:
            logger.info("dsqa_f1_verify progress %d/%d task=%s", done, total, task_id)

    item_ratings: list[ItemRating] = []
    if workers <= 1:
        for done, row in enumerate(source_rows, start=1):
            item_ratings.append(
                _process_row(
                    row,
                    raw_dir=raw_dir,
                    base_url=judge_base_url,
                    model=judge_model,
                    api_key=api_key,
                    request_timeout=request_timeout,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    retries=retries,
                    retry_sleep_cap=retry_sleep_cap,
                    resume=resume,
                )
            )
            _maybe_print_progress(done, row["task_id"])
    else:
        ratings_by_index: list[ItemRating | None] = [None] * total
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(
                    _process_row,
                    row,
                    raw_dir=raw_dir,
                    base_url=judge_base_url,
                    model=judge_model,
                    api_key=api_key,
                    request_timeout=request_timeout,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    retries=retries,
                    retry_sleep_cap=retry_sleep_cap,
                    resume=resume,
                ): index
                for index, row in enumerate(source_rows)
            }
            for done, future in enumerate(concurrent.futures.as_completed(future_to_index), start=1):
                index = future_to_index[future]
                ratings_by_index[index] = future.result()
                _maybe_print_progress(done, source_rows[index]["task_id"])
        item_ratings = [rating for rating in ratings_by_index if rating is not None]

    rating_rows = [rating.to_dict() for rating in item_ratings]
    _write_jsonl(run_dir / _DSQA_OUT_ITEM_RATINGS, rating_rows)
    project_rating, row_metrics = aggregate_ratings(item_ratings)
    _write_jsonl(run_dir / _DSQA_OUT_ROW_METRICS, row_metrics)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_type": "deepsearchqa_kaggle_starter_llm_judge_f1",
        "prompt_source": PROMPT_SOURCE,
        "experiment": exp_name,
        "run_dir": str(run_dir),
        "source_result_file": str(source_result_path),
        "dataset_csv": str(dataset_csv),
        "base_url": judge_base_url,
        "model": judge_model,
        "num_source_rows": total,
        "project_rating": project_rating.to_dict(),
    }
    _write_json(run_dir / _DSQA_OUT_SUMMARY, summary)
    (run_dir / _DSQA_OUT_REPORT).write_text(
        _build_report(exp_name, judge_base_url, judge_model, summary),
        encoding="utf-8",
    )
    logger.info(
        "dsqa_f1_verify complete: F1=%s precision=%s recall=%s valid=%d/%d",
        project_rating.f1_score,
        project_rating.precision,
        project_rating.recall,
        project_rating.num_valid_ratings,
        project_rating.num_total_ratings,
    )
    return summary
