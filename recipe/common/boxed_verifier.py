# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Shared boxed-answer exact-match verifier."""

from __future__ import annotations

import re
import unicodedata

from agentic.evaluation.verifier import BaseVerifier

_DASH_CHARS_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212\ufe58\ufe63\uff0d]")
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_RE = re.compile(r"[\"'`.,:;!?/\\\[\]{}()]+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_boxed_answer(answer: str) -> str:
    """Normalize boxed answers for conservative exact-match scoring."""
    normalized = unicodedata.normalize("NFKC", str(answer))
    normalized = "".join(char for char in unicodedata.normalize("NFKD", normalized) if not unicodedata.combining(char))
    normalized = normalized.translate(
        str.maketrans(
            {
                "\u2019": "'",
                "\u2018": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u00a0": " ",
            }
        )
    )
    normalized = _DASH_CHARS_RE.sub("-", normalized)
    normalized = re.sub(r"\\([&%#_$])", r"\1", normalized)
    normalized = re.sub(r"\bvs\.?\b|\bversus\b", " v ", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("&", " and ")
    normalized = normalized.lower()
    normalized = _ARTICLE_RE.sub(" ", normalized)
    normalized = _PUNCT_RE.sub(" ", normalized)
    normalized = normalized.replace("-", " ")
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def extract_boxed_content(text: str) -> str:
    r"""Extract content from the last ``\boxed{...}`` expression in *text*."""
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return ""

    content_start = start + len(marker)
    depth = 1
    chars: list[str] = []
    for char in text[content_start:]:
        if char == "{":
            depth += 1
            chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
            chars.append(char)
        else:
            chars.append(char)
    return ""


class BoxedAnswerVerifier(BaseVerifier):
    r"""Verifier that extracts ``\boxed{}`` answers and scores by normalized exact match."""

    def extract_answer(self, output_text: str) -> str | None:
        return extract_boxed_content(output_text) or None

    def score(self, label: str, extracted_answer: str | None) -> float:
        if not extracted_answer:
            return 0.0
        return 1.0 if normalize_boxed_answer(extracted_answer) == normalize_boxed_answer(label) else 0.0
