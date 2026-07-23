# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""QA exact-match verifier: extract an answer, normalize it, and compare."""

from __future__ import annotations

import json
import re
import string

from agentic.evaluation.verifier import BaseVerifier

_DEFAULT_ANSWER_PATTERN = r"<answer>(.*?)</answer>"


def normalize_answer(s: str) -> str:
    """Normalize answer string: lowercase, remove articles/punctuation, fix whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def em_check(prediction: str, golden_answers: str | list[str]) -> bool:
    """Check if prediction matches any golden answer after normalization."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    return any(normalize_answer(ga) == normalized_prediction for ga in golden_answers)


class QAExactMatchVerifier(BaseVerifier):
    """Verifier for factoid QA using normalized exact match.

    Extracts the answer from the output text using a configurable regex pattern and compares against golden answers.
    The label can be a JSON-encoded list of acceptable answers or a single string.

    Args:
        answer_pattern: Regex with one capture group for the answer text. Defaults to ``<answer>(.*?)</answer>``.
    """

    def __init__(self, *, answer_pattern: str = _DEFAULT_ANSWER_PATTERN) -> None:
        self._answer_pattern = re.compile(answer_pattern, re.DOTALL)

    def extract_answer(self, output_text: str) -> str | None:
        """Extract the last match of the answer pattern from the output text."""
        matches = list(self._answer_pattern.finditer(output_text))
        if not matches:
            return None
        return matches[-1].group(1).strip()

    def score(self, label: str, extracted_answer: str | None) -> float:
        if extracted_answer is None:
            return 0.0
        golden_answers = self._parse_label(label)
        return 1.0 if em_check(extracted_answer, golden_answers) else 0.0

    @staticmethod
    def _parse_label(label: str) -> list[str]:
        """Parse label into a list of golden answers.

        Accepts a JSON-encoded list (e.g. ``'["Paris", "paris"]'``) or a plain string.
        """
        try:
            parsed = json.loads(label)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (json.JSONDecodeError, TypeError):
            pass
        return [label]
