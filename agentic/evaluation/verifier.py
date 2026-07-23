# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseVerifier(ABC):
    """Verify model output against a ground-truth label.

    Subclasses must override ``score`` to compare the extracted answer against the label.
    Optionally override ``extract_answer`` for task-specific answer extraction.

    The ``verify`` method orchestrates both steps: extract then score.
    """

    def verify(self, label: str, output_text: str) -> float:
        """Extract the answer from *output_text* and score it against *label*."""
        answer = self.extract_answer(output_text)
        return self.score(label, answer)

    def extract_answer(self, output_text: str) -> str | None:
        r"""Extract the predicted answer from the model output.

        Defaults to returning the full *output_text* unchanged.
        Subclasses override this to apply task-specific extraction (e.g. regex for ``####``, ``<answer>`` tags, ``\boxed{}``, etc.).
        """
        return output_text

    @abstractmethod
    def score(self, label: str, extracted_answer: str | None) -> float:
        """Compare *extracted_answer* against *label* and return a reward.

        Subclasses must implement task-specific comparison logic.
        """


class ExactMatchVerifier(BaseVerifier):
    """Simple verifier that returns 1.0 on exact string match, 0.0 otherwise."""

    def score(self, label: str, extracted_answer: str | None) -> float:
        if extracted_answer is None:
            return 0.0
        return 1.0 if extracted_answer == label else 0.0
