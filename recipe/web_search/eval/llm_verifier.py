# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Configurable LLM-as-judge verifier shared by web-search benchmarks."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from agentic.evaluation.verifier import BaseVerifier
from recipe.common.boxed_verifier import normalize_boxed_answer
from recipe.web_search.agent.prompts import extract_boxed_content
from recipe.web_search.eval.llm_judge_prompt import BROWSECOMP_JUDGE_PROMPT

if TYPE_CHECKING:
    from agentic.model_clients.request_logger import ModelRequestLogger

logger = logging.getLogger(__name__)

_LENGTH_EMPTY_RETRY_MIN_MAX_TOKENS = 128
_DEFAULT_LENGTH_EMPTY_RETRY_MAX_TOKENS = 1024
_CHOICE_BY_LABEL = {"CORRECT": "A", "INCORRECT": "B", "正确": "A", "错误": "B"}
_DECISION_PREFIX = r"(?:(?:final\s+answer|answer|choice|option|judg(?:e)?ment|result|decision)\s*(?:is\s*)?(?:[:=\-]\s*)?)?"
_VERDICT_LABEL_RE = re.compile(r"\b(CORRECT|INCORRECT)\b", flags=re.IGNORECASE)
_LETTER_DECISION_RE = re.compile(rf"^{_DECISION_PREFIX}\[?\s*(?P<letter>[AB])\s*\]?(?:[\.)])?$", flags=re.IGNORECASE)
_JUDGE_LABEL_PATTERN = r"CORRECT|INCORRECT|正确|错误"
_LABEL_DECISION_RE = re.compile(rf"^{_DECISION_PREFIX}\[?\s*(?P<label>{_JUDGE_LABEL_PATTERN})\s*\]?(?:[\.)])?$", flags=re.IGNORECASE)
_LETTER_LABEL_DECISION_RE = re.compile(
    rf"^{_DECISION_PREFIX}"
    r"(?P<letter>[AB])\s*(?:[\.\):\-]\s*)?"
    rf"\[?\s*(?P<label>{_JUDGE_LABEL_PATTERN})\s*\]?(?:[\.)])?$",
    flags=re.IGNORECASE,
)
_TRAILING_PREFIXED_DECISION_RE = re.compile(
    r"((?:final\s+answer|answer|choice|option|judg(?:e)?ment|result|decision)\s*(?:is\s*)?(?:[:=\-]\s*)?"
    rf"(?:\[?\s*[AB]\s*\]?(?:[\.)])?|\[?\s*(?:{_JUDGE_LABEL_PATTERN})\s*\]?"
    rf"|[AB]\s*(?:[\.\):\-]\s*)?\[?\s*(?:{_JUDGE_LABEL_PATTERN})\s*\]?))\s*[\.)]?\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)
JudgeResponseParser = Callable[[str | None], str | None]


class LLMJudgeError(RuntimeError):
    """Raised when the LLM judge cannot produce a usable decision."""


def normalize_openai_base_url(base_url: str | None) -> str | None:
    """Return an OpenAI SDK-compatible base URL.

    Some CLI tools, including curl, accept host-only URLs such as
    ``inferhub.example.com:8007/v1`` by assuming HTTP. The OpenAI SDK/httpx
    requires an explicit scheme, otherwise the hostname can be misread as the
    protocol.
    """
    if base_url is None:
        return None

    stripped = base_url.strip()
    if not stripped:
        return None

    parsed = urlparse(stripped)
    if parsed.scheme in {"http", "https"}:
        return stripped
    if "://" in stripped:
        return stripped
    return f"http://{stripped}"


def _next_length_retry_max_tokens(current_max_tokens: int, retry_max_tokens: int) -> int:
    if current_max_tokens < _LENGTH_EMPTY_RETRY_MIN_MAX_TOKENS:
        return _LENGTH_EMPTY_RETRY_MIN_MAX_TOKENS
    # Output was truncated (finish_reason == "length"): grow the cap 4x per retry
    # (clamped to the configured ceiling). Quadrupling reaches a large ceiling in
    # a couple of retries while still degrading gracefully when the ceiling is
    # configured above what the judge model can actually emit (a smaller model
    # may succeed at an intermediate cap before the over-large ceiling errors).
    # max_tokens is only a cap, so normal (non-truncating) cases never pay for it.
    return min(retry_max_tokens, current_max_tokens * 4)


def _first_choice(response: Any) -> Any | None:
    try:
        return response.choices[0]
    except (AttributeError, IndexError, TypeError):
        return None


def _choice_message(choice: Any | None) -> Any | None:
    if choice is None:
        return None
    return getattr(choice, "message", None)


def _message_text(message: Any | None, field_name: str) -> str:
    value = getattr(message, field_name, None)
    return value if isinstance(value, str) else ""


def _normalize_decision_fragment(text: str) -> str:
    fragment = text.strip()
    fragment = re.sub(r"^```(?:\w+)?\s*", "", fragment)
    fragment = re.sub(r"\s*```$", "", fragment)
    fragment = re.sub(r"^[-*•]\s+", "", fragment.strip())
    fragment = fragment.translate(str.maketrans({"【": "[", "】": "]", "。": ".", "：": ":", "）": ")", "（": "("}))  # noqa: RUF001
    return fragment.strip().strip("`*_").strip()


def _choice_from_label(label: str) -> str:
    return _CHOICE_BY_LABEL[label.upper()]


def _parse_decision_fragment(fragment: str) -> str | None:
    normalized = _normalize_decision_fragment(fragment)
    if not normalized:
        return None

    match = _LETTER_LABEL_DECISION_RE.fullmatch(normalized)
    if match:
        letter_choice = match.group("letter").upper()
        label_choice = _choice_from_label(match.group("label"))
        return letter_choice if letter_choice == label_choice else None

    match = _LABEL_DECISION_RE.fullmatch(normalized)
    if match:
        return _choice_from_label(match.group("label"))

    match = _LETTER_DECISION_RE.fullmatch(normalized)
    if match:
        return match.group("letter").upper()

    return None


def parse_ab_judge_choice(content: str | None) -> str | None:
    """Parse an A/B judge response when its decision is unambiguous."""
    if content is None:
        return None
    text = content.strip()
    if not text:
        return None

    labels = {match.group(1).upper() for match in _VERDICT_LABEL_RE.finditer(text)}
    if len(labels) > 1:
        return None
    label_choice = _choice_from_label(next(iter(labels))) if labels else None

    fragments = [text]
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if nonempty_lines:
        fragments.extend(nonempty_lines)
    trailing_match = _TRAILING_PREFIXED_DECISION_RE.search(text)
    if trailing_match:
        fragments.append(trailing_match.group(1))

    choices = {_parse_decision_fragment(fragment) for fragment in fragments}
    choices.discard(None)
    if len(choices) != 1:
        return None
    choice = next(iter(choices))
    if label_choice is not None and choice != label_choice:
        return None
    return choice


class LLMVerifier(BaseVerifier):
    """LLM-based semantic verifier configurable for each benchmark.

    Uses an OpenAI-compatible judge model to evaluate
    whether a predicted answer semantically matches the ground truth, following
    a benchmark-specific judge prompt.

    Args:
        judge_model: Model name for the judge.
        judge_base_url: API base URL. Falls back to the main provider endpoint env value.
        judge_api_key_env: Env var name holding the judge API key.
        max_retries: Maximum retry attempts for failed/unparseable judge responses.
        judge_times: Number of independent judge decisions to average per QA pair.
        judge_max_tokens: Maximum tokens requested for each judge decision.
        judge_empty_length_retry_max_tokens: Maximum retry cap used when an unparseable judge
            response has ``finish_reason="length"``.
    """

    def __init__(
        self,
        *,
        judge_model: str | None = None,
        judge_base_url: str | None = None,
        judge_api_key_env: str = "JUDGE_API_KEY",
        max_retries: int = 3,
        judge_times: int = 5,
        judge_max_tokens: int = 2,
        judge_empty_length_retry_max_tokens: int = _DEFAULT_LENGTH_EMPTY_RETRY_MAX_TOKENS,
        judge_prompt_template: str | None = None,
        judge_prompt_profile: str = "browsecomp",
        judge_response_parser: JudgeResponseParser | None = None,
    ) -> None:
        self.judge_model = judge_model or "gpt-4.1-mini"
        self.judge_base_url = normalize_openai_base_url(judge_base_url)
        self.judge_api_key_env = judge_api_key_env
        self.max_retries = max_retries
        self.judge_times = max(1, int(judge_times))
        self.judge_max_tokens = max(1, int(judge_max_tokens))
        self.judge_empty_length_retry_max_tokens = max(1, int(judge_empty_length_retry_max_tokens))
        self.judge_prompt_template = judge_prompt_template or BROWSECOMP_JUDGE_PROMPT
        self.judge_prompt_profile = judge_prompt_profile
        self.judge_response_parser = judge_response_parser or parse_ab_judge_choice
        self._client: Any | None = None
        self.request_logger: ModelRequestLogger | None = None

    def set_request_logger(self, request_logger: ModelRequestLogger | None) -> None:
        self.request_logger = request_logger

    def _get_client(self) -> Any:
        """Return a reusable OpenAI-compatible async client.

        Creating one ``AsyncOpenAI`` per judged answer prevents connection reuse
        and can add seconds of per-request overhead after the server has already
        generated its response.
        """
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=os.environ.get(self.judge_api_key_env) or os.environ.get("OPENAI_API_KEY"),
                base_url=self.judge_base_url or normalize_openai_base_url(os.environ.get("OPENAI_BASE_URL")),
            )
        return self._client

    async def aclose(self) -> None:
        """Close the reusable judge client."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    def extract_answer(self, output_text: str) -> str | None:
        return extract_boxed_content(output_text) or None

    def score(self, label: str, extracted_answer: str | None) -> float:
        """Synchronous exact-match scoring.

        For the full LLM-based judge, use :meth:`ascore` with ``await``.
        """
        if not extracted_answer:
            return 0.0
        return 1.0 if normalize_boxed_answer(extracted_answer) == normalize_boxed_answer(label) else 0.0

    def _log_judge_request(
        self,
        *,
        request_id: str | None,
        request_started_at: str | None,
        request_start: float,
        request_payload: dict[str, Any],
        judge_index: int,
        attempt: int,
        request_max_tokens: int,
        response: Any | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        if self.request_logger is None or request_id is None or request_started_at is None:
            return
        self.request_logger.log(
            request_id=request_id,
            started_at=request_started_at,
            elapsed_ms=(self.request_logger.perf_counter() - request_start) * 1000.0,
            request=request_payload,
            response=response,
            error=error,
            metadata={
                "client": "LLMVerifier",
                "judge_index": judge_index,
                "attempt": attempt,
                "configured_max_tokens": self.judge_max_tokens,
                "request_max_tokens": request_max_tokens,
                "empty_length_retry_max_tokens": self.judge_empty_length_retry_max_tokens,
            },
        )

    def _record_unparseable_attempt(
        self,
        *,
        unparseable_attempts: list[dict[str, Any]],
        judge_index: int,
        attempt: int,
        content: str,
        finish_reason: Any,
        reasoning_content: str,
        current_max_tokens: int,
    ) -> int:
        length_response = finish_reason == "length"
        next_max_tokens = (
            _next_length_retry_max_tokens(current_max_tokens, self.judge_empty_length_retry_max_tokens) if length_response else current_max_tokens
        )
        unparseable_attempts.append(
            {
                "judge_index": judge_index,
                "attempt": attempt,
                "finish_reason": finish_reason,
                "content_empty": not content,
                "reasoning_content_chars": len(reasoning_content),
                "request_max_tokens": current_max_tokens,
                "next_request_max_tokens": next_max_tokens if next_max_tokens > current_max_tokens else None,
                "empty_length_retry_max_tokens": self.judge_empty_length_retry_max_tokens,
            }
        )
        # Only record + compute the next max_tokens here. The retry/give-up
        # decision is owned by the caller loop (which counts endpoint errors and
        # parse failures separately). next_max_tokens > current signals a
        # length/empty (output-truncation) failure to be retried with more room.
        if next_max_tokens > current_max_tokens:
            logger.warning(
                "Judge output truncated by length (judge %d, max_tokens=%d, content_empty=%s); will retry with max_tokens=%d",
                judge_index,
                current_max_tokens,
                not content,
                next_max_tokens,
            )
        else:
            logger.warning(
                "Judge response unparseable (judge %d, finish_reason=%s, max_tokens=%d): %s",
                judge_index,
                finish_reason,
                current_max_tokens,
                content,
            )
        return next_max_tokens

    async def ajudge(  # noqa: PLR0915
        self,
        label: str,
        extracted_answer: str | None,
        *,
        question: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        r"""Return LLM judge score plus the request/response payload used."""
        if not extracted_answer or not str(extracted_answer).strip():
            return {
                "llm_judge_score": 0.0,
                "llm_judge_result": "INCORRECT",
                "llm_judge_skipped_reason": "missing_prediction",
                "llm_judge_times": self.judge_times,
                "llm_judge_scores": [0.0] * self.judge_times,
            }
        if self.score(label, extracted_answer) > 0.5:
            logger.debug("LLM judge skipped because prediction exactly matches label")
            return {
                "llm_judge_score": 1.0,
                "llm_judge_result": "CORRECT",
                "llm_judge_skipped_reason": "exact_match",
                "llm_judge_times": self.judge_times,
                "llm_judge_scores": [1.0] * self.judge_times,
            }

        prompt_fields = {
            "question": question,
            "correct_answer": label,
            "response": extracted_answer,
            "prompt_type": "Single Answer",
            "evaluation": "",
        }
        if metadata:
            for key, value in metadata.items():
                prompt_fields.setdefault(key, value)
            if metadata.get("answer_type"):
                prompt_fields["prompt_type"] = metadata["answer_type"]
        prompt = self.judge_prompt_template.format(**prompt_fields)
        base_request_payload: dict[str, Any] = {
            "model": self.judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.judge_max_tokens,
        }
        try:
            client = self._get_client()
        except Exception as exc:
            logger.exception("Failed to initialize LLM judge client")
            raise LLMJudgeError("Failed to initialize LLM judge client") from exc

        decisions: list[dict[str, Any]] = []
        unparseable_attempts: list[dict[str, Any]] = []
        for judge_index in range(self.judge_times):
            last_response_content = ""
            current_max_tokens = self.judge_max_tokens
            # Separate failure budgets so the three failure modes don't steal
            # each other's retries:
            #   - endpoint_errors: transient endpoint hiccups (timeout/5xx) -> retry
            #   - parse_failures:  call ok but JSON unparseable (format error) -> retry
            #   - length/empty truncation: grow max_tokens (does NOT spend a budget)
            endpoint_errors = 0
            parse_failures = 0
            call_index = 0
            decided = False
            while True:
                call_index += 1
                request_payload = {**base_request_payload, "max_tokens": current_max_tokens}
                request_id = self.request_logger.next_id() if self.request_logger is not None else None
                request_started_at = self.request_logger.now() if self.request_logger is not None else None
                request_start = self.request_logger.perf_counter() if self.request_logger is not None else 0.0
                try:
                    response = await client.chat.completions.create(**request_payload, timeout=900)
                    choice = _first_choice(response)
                    finish_reason = getattr(choice, "finish_reason", None)
                    message = _choice_message(choice)
                    content = _message_text(message, "content")
                    reasoning_content = _message_text(message, "reasoning_content")
                    raw_response = response.model_dump(mode="json") if hasattr(response, "model_dump") else None
                    self._log_judge_request(
                        request_id=request_id,
                        request_started_at=request_started_at,
                        request_start=request_start,
                        request_payload=request_payload,
                        judge_index=judge_index + 1,
                        attempt=call_index,
                        request_max_tokens=current_max_tokens,
                        response=raw_response,
                    )
                    last_response_content = content or (
                        f"<empty content; finish_reason={finish_reason!r}; reasoning_content_chars={len(reasoning_content)}>"
                    )
                    parsed_choice = self.judge_response_parser(content)
                    if parsed_choice:
                        score = 1.0 if parsed_choice == "A" else 0.0
                        decisions.append(
                            {
                                "content": content,
                                "parsed_choice": parsed_choice,
                                "score": score,
                                "judge_index": judge_index + 1,
                                "attempt": call_index,
                                "finish_reason": finish_reason,
                                "request_max_tokens": current_max_tokens,
                                "configured_max_tokens": self.judge_max_tokens,
                                "empty_length_retry_max_tokens": self.judge_empty_length_retry_max_tokens,
                                "raw": raw_response,
                            }
                        )
                        decided = True
                        break
                    next_max_tokens = self._record_unparseable_attempt(
                        unparseable_attempts=unparseable_attempts,
                        judge_index=judge_index + 1,
                        attempt=call_index,
                        content=content,
                        finish_reason=finish_reason,
                        reasoning_content=reasoning_content,
                        current_max_tokens=current_max_tokens,
                    )
                    if next_max_tokens > current_max_tokens:
                        # (2) output truncated/empty: grow max_tokens and retry.
                        # Does NOT consume the parse/endpoint budgets.
                        current_max_tokens = next_max_tokens
                        continue
                    # (3) JSON format error (or already at the output ceiling).
                    parse_failures += 1
                    if parse_failures >= self.max_retries:
                        logger.error(
                            "LLM judge unparseable %d times (judge %d/%d); giving up on this judge",
                            parse_failures,
                            judge_index + 1,
                            self.judge_times,
                        )
                        break
                    continue
                except Exception as exc:
                    self._log_judge_request(
                        request_id=request_id,
                        request_started_at=request_started_at,
                        request_start=request_start,
                        request_payload=request_payload,
                        judge_index=judge_index + 1,
                        attempt=call_index,
                        request_max_tokens=current_max_tokens,
                        error={"type": type(exc).__name__, "message": str(exc)},
                    )
                    # (1) transient endpoint failure: retry on its own budget.
                    endpoint_errors += 1
                    if endpoint_errors >= self.max_retries:
                        logger.exception(
                            "Judge call failed %d times (judge %d/%d); giving up on this judge",
                            endpoint_errors,
                            judge_index + 1,
                            self.judge_times,
                        )
                        break
                    logger.warning(
                        "Judge call failed; retrying (judge %d/%d endpoint_error %d/%d): %s: %s",
                        judge_index + 1,
                        self.judge_times,
                        endpoint_errors,
                        self.max_retries,
                        type(exc).__name__,
                        exc,
                    )
                    continue
            if not decided:
                raise LLMJudgeError(
                    f"LLM judge failed for judge {judge_index + 1}/{self.judge_times} "
                    f"(endpoint_errors={endpoint_errors}, parse_failures={parse_failures}); "
                    f"last_response={last_response_content!r}"
                )

        scores = [float(decision["score"]) for decision in decisions]
        average_score = sum(scores) / len(scores)
        effective_max_tokens = max(int(decision["request_max_tokens"]) for decision in decisions)
        return {
            "llm_judge_score": average_score,
            "llm_judge_result": "CORRECT" if average_score > 0.5 else "INCORRECT",
            "llm_judge_request": base_request_payload,
            "llm_judge_effective_max_tokens": effective_max_tokens,
            "llm_judge_empty_length_retry_max_tokens": self.judge_empty_length_retry_max_tokens,
            "llm_judge_response": {
                "content": decisions[-1]["content"],
                "parsed_choice": decisions[-1]["parsed_choice"],
                "raw": decisions[-1]["raw"],
                "decisions": decisions,
                "unparseable_attempts": unparseable_attempts,
                "empty_length_retry_max_tokens": self.judge_empty_length_retry_max_tokens,
            },
            "llm_judge_times": self.judge_times,
            "llm_judge_scores": scores,
        }

    async def ascore(
        self,
        label: str,
        extracted_answer: str | None,
        *,
        question: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> float:
        r"""Async LLM-based scoring via the configured benchmark judge prompt.

        Args:
            label: Ground-truth answer.
            extracted_answer: Model prediction extracted from ``\boxed{}``.
            question: Original question for judge prompt context.
            metadata: Optional benchmark metadata for prompt formatting.

        Returns ``1.0`` for CORRECT, ``0.0`` for INCORRECT.
        """
        judgement = await self.ajudge(label, extracted_answer, question=question, metadata=metadata)
        return float(judgement["llm_judge_score"])
