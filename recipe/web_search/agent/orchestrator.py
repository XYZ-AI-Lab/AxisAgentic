# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from agentic.contracts import (
    ConversationMessage,
    ConversationStage,
    ConversationStepResult,
    FinalizationTrigger,
    MessageRole,
    ModelResponse,
    RollbackReason,
    StepAction,
    TokenUsage,
)
from agentic.contracts.messages import ToolResultStatus
from agentic.model_clients.errors import ModelContextLimitError
from agentic.orchestration.task_orchestrator import OrchestrationResult, TaskOrchestrator
from recipe.web_search.agent.prompts import (
    FAILURE_SUMMARY_ASSISTANT_PREFIX,
    FAILURE_SUMMARY_PROMPT,
    FORMAT_ERROR_MESSAGE,
    build_failure_enhanced_task,
    extract_boxed_content,
    generate_summary_prompt,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentic.conversations.conversation_runtime import ConversationRuntime
    from recipe.web_search.agent.context_compression_manager import ContextCompressionManager
    from recipe.web_search.agent.discard_all_manager import DiscardAllManager

logger = logging.getLogger(__name__)

_QUERY_EXTRACTION_MAP: dict[str, tuple[str, list[str]]] = {
    "google_search": ("google_search", ["q"]),
    "web_search": ("web_search", ["query"]),
    "scrape_and_extract_info": ("scrape_and_extract_info", ["url", "info_to_extract"]),
}
_SEMANTIC_QUERY_BUDGET_TOOL_NAMES = frozenset({"google_search", "web_search"})
_SEMANTIC_QUERY_BUDGET_REASON = "semantic_query_budget"
_ROLLBACK_STORM_SEARCH_TOOL_NAMES = frozenset({"google_search", "web_search"})
_ROLLBACK_STORM_SCRAPE_TOOL_NAMES = frozenset({"scrape_and_extract_info"})
_ROLLBACK_STORM_PREVIEW_CHARS = 120
_ROLLBACK_STORM_SECRET_PARAM_RE = re.compile(
    r"(?i)(^|[?&;\s])"
    r"((?:token|secret|key|api_key|password|auth|credential)\s*[:=]\s*)"
    r"([^&#;\s]+)"
)
_ROLLBACK_STORM_AUTHORIZATION_RE = re.compile(r"(?i)\b(authorization\s*[:=]\s*)([^,;&#\s]+(?:\s+[^,;&#\s]+)?)")
_ROLLBACK_STORM_BEARER_RE = re.compile(r"(?i)\b(bearer\s+)([^,;&#\s]+)")
_NORMALIZED_QUERY_TOKEN_RE = re.compile(r"[\W_]+", re.UNICODE)
_SELF_VERIFICATION_TRACE_DIR = "self_verification"
_SELF_VERIFICATION_SYSTEM_PROMPT = """\
You are a careful answer verifier with access to structured web-search tools.
Your job is to determine whether a candidate answer to the original question is correct.

Break the original question into its required conditions and verify those conditions one by one.
The candidate answer should be judged against the exact conditions in the question, not against
a looser or different question. Actively look for contradictions and alternative answers, but
use the verdict "incorrect" only when a required condition is clearly not satisfied, clearly
contradicted by reliable evidence, or the candidate answers a different entity/value. Do not
mark the candidate incorrect merely because a detail is hard to find, evidence is incomplete,
or another answer is possible but not clearly better. Avoid relying on search-result snippets,
social-media pages, SEO/trivia/crossword pages, or future-dated pages as decisive evidence for
either verdict.

Use tools when external evidence is needed. Call at most ONE tool per assistant turn.
When you are done verifying, output exactly one JSON object and no other text:
{"rationale":"...","verdict":"correct"|"incorrect"}

Do not wrap the JSON in \\boxed{} and do not include markdown fences.
"""
_SELF_VERIFICATION_VERDICT_PROMPT = """\
You cannot call tools now. Based only on the verification work above, decide whether the
candidate answer clearly fails any required condition in the original question. Use
"incorrect" only for an explicit condition failure, contradiction, or different entity/value.
If the verification is incomplete or uncertain but no obvious mismatch was found, use
"correct". Output exactly one JSON object:
{"rationale":"...","verdict":"correct"|"incorrect"}
"""
_SELF_VERIFICATION_RESAMPLE_PROMPT = """\
Your previous verifier verdict was not valid JSON in the required schema.
Do not call tools. Preserve the same condition-by-condition verification standard: use
"incorrect" only when the candidate clearly fails a required condition, is contradicted by
reliable evidence, or answers a different entity/value.
Output exactly one JSON object and no other text:
{"rationale":"...","verdict":"correct"|"incorrect"}
"""


def get_query_str_from_tool_call(tool_name: str, arguments: dict[str, Any]) -> str | None:
    entry = _QUERY_EXTRACTION_MAP.get(tool_name)
    if entry is None:
        return None
    prefix, fields = entry
    return f"{prefix}_{'_'.join(str(arguments.get(field, '')) for field in fields)}"


def get_semantic_query_key_from_tool_call(tool_name: str, arguments: dict[str, Any]) -> str | None:
    entry = _QUERY_EXTRACTION_MAP.get(tool_name)
    if entry is None:
        return None
    prefix, fields = entry
    parts = [_normalize_semantic_query_part(arguments.get(field, "")) for field in fields]
    return f"{prefix}:{'|'.join(parts)}"


def get_semantic_budget_key_from_tool_call(tool_name: str, arguments: dict[str, Any]) -> str | None:
    if tool_name not in _SEMANTIC_QUERY_BUDGET_TOOL_NAMES:
        return None
    return get_semantic_query_key_from_tool_call(tool_name, arguments)


def _normalize_semantic_query_part(value: Any) -> str:
    text = str(value or "").casefold()
    text = _NORMALIZED_QUERY_TOKEN_RE.sub(" ", text)
    return " ".join(text.split())


@dataclass(slots=True)
class AttemptBudgetResult:
    budget: int
    result: OrchestrationResult
    reused_from_budget: int | None = None


@dataclass(slots=True)
class AttemptBudgetBranchCheckpoint:
    source_task_id: str
    branch_task_id: str
    tools: list[dict[str, Any]] | None
    runtime: ConversationRuntime
    step_result: ConversationStepResult
    turn_count: int
    total_attempts: int
    max_turns: int | None
    max_attempts: int
    total_reward: float
    run_info: dict[str, Any]
    attempt_state: dict[str, Any]
    extra_trace_metadata: dict[str, Any] | None


@dataclass(slots=True)
class SelfVerificationVerdict:
    verdict: str
    raw_content: str
    parsed: dict[str, Any] | None
    parse_error: str | None
    resample_attempts: int = 0


class WebSearchTaskOrchestrator(TaskOrchestrator):
    """Benchmark-oriented orchestrator for native structured tool calls."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        max_task_retries: int = 0,
        include_failure_summary_in_retry: bool = False,
        max_final_answer_attempts: int = 1,
        prompt_profile: str = "default",
        context_token_estimator: Callable[[list[ConversationMessage]], int] | None = None,
        context_limit_preflight_enabled: bool = True,
        semantic_query_budget_enabled: bool = False,
        semantic_query_budget_max_unique: int | None = None,
        retry_attempt_provenance_enabled: bool = False,
        retry_no_box_turn_limit_cap_enabled: bool = False,
        retry_no_box_turn_limit_cap: int = 3,
        generation_limit_recovery_non_final_attempt: str = "retry",
        generation_limit_recovery_final_attempt: str = "rollback",
        self_verification_enabled: bool = False,
        self_verification_max_reanswer_attempts: int = 1,
        self_verification_max_turns: int | None = None,
        self_verification_verdict_resample_max_attempts: int = 3,
        rollback_storm_shadow_enabled: bool = False,
        rollback_storm_duplicate_threshold: int = 20,
        rollback_storm_tool_error_threshold: int = 10,
        rollback_storm_late_turn_threshold: int = 250,
        rollback_storm_preview_max_items: int = 5,
        context_compression_manager: ContextCompressionManager | None = None,
        discard_all_manager: DiscardAllManager | None = None,
        discard_all_last_attempt_max_turns: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_task_retries = max_task_retries
        self.include_failure_summary_in_retry = include_failure_summary_in_retry
        self.max_final_answer_attempts = max_final_answer_attempts
        self.prompt_profile = prompt_profile
        self.context_token_estimator = context_token_estimator
        self.context_limit_preflight_enabled = context_limit_preflight_enabled
        self.semantic_query_budget_enabled = semantic_query_budget_enabled
        self.semantic_query_budget_max_unique = semantic_query_budget_max_unique
        self.retry_attempt_provenance_enabled = retry_attempt_provenance_enabled
        self.retry_no_box_turn_limit_cap_enabled = retry_no_box_turn_limit_cap_enabled
        self.retry_no_box_turn_limit_cap = retry_no_box_turn_limit_cap
        self.self_verification_enabled = bool(self_verification_enabled)
        self.self_verification_max_reanswer_attempts = max(0, int(self_verification_max_reanswer_attempts))
        self.self_verification_max_turns = self_verification_max_turns
        self.self_verification_verdict_resample_max_attempts = max(1, int(self_verification_verdict_resample_max_attempts))
        if generation_limit_recovery_non_final_attempt not in {"retry", "rollback"}:
            msg = "generation_limit_recovery_non_final_attempt must be 'retry' or 'rollback'."
            raise ValueError(msg)
        if generation_limit_recovery_final_attempt not in {"rollback", "terminate"}:
            msg = "generation_limit_recovery_final_attempt must be 'rollback' or 'terminate'."
            raise ValueError(msg)
        self.generation_limit_recovery_non_final_attempt = generation_limit_recovery_non_final_attempt
        self.generation_limit_recovery_final_attempt = generation_limit_recovery_final_attempt
        self.rollback_storm_shadow_enabled = rollback_storm_shadow_enabled
        self.rollback_storm_duplicate_threshold = rollback_storm_duplicate_threshold
        self.rollback_storm_tool_error_threshold = rollback_storm_tool_error_threshold
        self.rollback_storm_late_turn_threshold = rollback_storm_late_turn_threshold
        self.rollback_storm_preview_max_items = rollback_storm_preview_max_items
        self._context_compression_manager = context_compression_manager
        self._discard_all_manager = discard_all_manager
        self._discard_all_last_attempt_max_turns = discard_all_last_attempt_max_turns

        self._used_queries: dict[str, int] = {}
        self._semantic_query_keys: dict[str, int] = {}
        self._intermediate_boxed_answers: list[str] = []
        self._current_task = ""
        self._skip_turn_limit_final_response_this_attempt = False
        self._recover_generation_limits_this_attempt = False
        # Discard-all mutable state (per attempt; snapshotted for attempt-budget
        # branches). ``max_tool_calls`` is a boundary for entering the final
        # no-discard attempt, not a hard ToolManager rejection cap.
        self._last_observed_prompt_tokens: int | None = None
        self._discard_all_last_trigger_turn: int | None = None
        self._discard_all_reset_count = 0
        self._discard_all_last_attempt_mode = False
        self._reset_rollback_storm_shadow_state()

    async def run(
        self,
        task: str | dict[str, Any],
        task_id: str | None = None,
        *,
        extra_trace_metadata: dict[str, Any] | None = None,
    ) -> OrchestrationResult:
        self._current_task = self.normalize_task(task) if isinstance(task, str) else str(task)
        max_attempts = self.max_task_retries + 1
        failure_summaries: list[str] = []
        enhanced_task = self._current_task
        last_result: OrchestrationResult | None = None
        attempt_provenance: list[dict[str, Any]] = []
        consecutive_no_box_turn_limit_attempts = 0
        previous_terminal_reason: str | None = None

        for attempt in range(max_attempts):
            self._reset_attempt_state()
            attempt_id = f"{task_id}_attempt-{attempt + 1}" if task_id else None
            is_final = attempt == max_attempts - 1
            self._skip_turn_limit_final_response_this_attempt = self.max_task_retries > 0 and not is_final
            self._recover_generation_limits_this_attempt = self._should_recover_generation_limits_this_attempt(is_final=is_final)

            try:
                await self.tool_manager.begin_task(task_id=attempt_id)
                result = await self._run_single_attempt(enhanced_task, task_id=attempt_id, extra_trace_metadata=extra_trace_metadata)
            finally:
                await self.tool_manager.end_task(task_id=attempt_id)
                self._skip_turn_limit_final_response_this_attempt = False
                self._recover_generation_limits_this_attempt = False
            last_result = result
            result, output, reached_turn_limit = await self._prepare_retry_attempt_output(
                result=result,
                attempt=attempt + 1,
                max_attempts=max_attempts,
                is_final=is_final,
            )
            last_result = result

            output_status = self._retry_output_status(output)

            consecutive_no_box_turn_limit_attempts = self._next_no_box_turn_limit_attempt_count(
                current_count=consecutive_no_box_turn_limit_attempts,
                reached_turn_limit=reached_turn_limit,
                output_status=output_status,
            )

            cap_blocks_retry = self._should_block_retry_for_no_box_turn_limit(
                consecutive_no_box_turn_limit_attempts=consecutive_no_box_turn_limit_attempts,
                is_final=is_final,
            )

            if output and output != FORMAT_ERROR_MESSAGE:
                result = self._attach_rollback_storm_shadow(result=result, output_status=output_status, attempt_id=attempt_id)
                last_result = result
                self._append_retry_attempt_provenance(
                    attempt_provenance,
                    result=result,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    attempt_id=attempt_id,
                    is_final=is_final,
                    output=output,
                    output_status=output_status,
                    previous_terminal_reason=previous_terminal_reason,
                    consecutive_no_box_turn_limit_attempts=consecutive_no_box_turn_limit_attempts,
                    retry_decision="return_success",
                    retry_decision_reason="valid_output",
                    next_attempt_launched=False,
                    next_attempt_launch_reason=None,
                    next_attempt_blocked_reason=None,
                    would_block_by_no_box_turn_limit_cap=False,
                )
                success_result = self._copy_result_with_retry_attempt_provenance(result, attempt_provenance)
                return await self._maybe_apply_self_verification(
                    result=success_result,
                    canonical_task_id=attempt_id,
                    original_task=self._current_task,
                )

            if is_final:
                final_result = self._return_final_attempt_result(
                    result=result,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    attempt_id=attempt_id,
                    output=output,
                    output_status=output_status,
                    previous_terminal_reason=previous_terminal_reason,
                    consecutive_no_box_turn_limit_attempts=consecutive_no_box_turn_limit_attempts,
                    attempt_provenance=attempt_provenance,
                )
                return await self._maybe_apply_self_verification(
                    result=final_result,
                    canonical_task_id=attempt_id,
                    original_task=self._current_task,
                )

            if cap_blocks_retry:
                return self._return_no_box_turn_limit_cap_result(
                    result=result,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    attempt_id=attempt_id,
                    output=output,
                    output_status=output_status,
                    previous_terminal_reason=previous_terminal_reason,
                    consecutive_no_box_turn_limit_attempts=consecutive_no_box_turn_limit_attempts,
                    attempt_provenance=attempt_provenance,
                )

            retry_reason = self._retry_reason_for_failed_attempt(result=result, reached_turn_limit=reached_turn_limit)
            result = self._attach_rollback_storm_shadow(result=result, output_status=output_status, attempt_id=attempt_id)
            last_result = result
            self._append_retry_attempt_provenance(
                attempt_provenance,
                result=result,
                attempt=attempt + 1,
                max_attempts=max_attempts,
                attempt_id=attempt_id,
                is_final=is_final,
                output=output,
                output_status=output_status,
                previous_terminal_reason=previous_terminal_reason,
                consecutive_no_box_turn_limit_attempts=consecutive_no_box_turn_limit_attempts,
                retry_decision="retry",
                retry_decision_reason=retry_reason,
                next_attempt_launched=True,
                next_attempt_launch_reason=retry_reason,
                next_attempt_blocked_reason=None,
                would_block_by_no_box_turn_limit_cap=False,
            )
            previous_terminal_reason = result.reason
            if self.include_failure_summary_in_retry:
                failure_summary = await self._generate_failure_summary(result)
                if failure_summary:
                    failure_summaries.append(failure_summary)
                enhanced_task = build_failure_enhanced_task(self._current_task, failure_summaries)
                logger.info("Retrying with failure experience (attempt %d/%d)", attempt + 2, max_attempts)
            else:
                logger.info("Retrying without failure summary (attempt %d/%d)", attempt + 2, max_attempts)

        result = last_result or OrchestrationResult(output=FORMAT_ERROR_MESSAGE, reason="no_attempts", done=True)
        return self._copy_result_with_retry_attempt_provenance(result, attempt_provenance)

    async def run_attempt_budget_sweep(  # noqa: PLR0915
        self,
        task: str | dict[str, Any],
        task_id: str | None = None,
        *,
        extra_trace_metadata: dict[str, Any] | None = None,
    ) -> dict[int, AttemptBudgetResult]:
        self._current_task = self.normalize_task(task) if isinstance(task, str) else str(task)
        task_id = task_id or uuid4().hex
        # Match the non-sweep ``run`` path: ``max_task_retries`` counts retries, so
        # the total attempt budget (and the largest sweep budget) is retries + 1.
        # The max budget then reproduces exactly what a full non-sweep run does.
        max_attempts = self.max_task_retries + 1
        failure_summaries: list[str] = []
        enhanced_task = self._current_task
        attempt_provenance: list[dict[str, Any]] = []
        consecutive_no_box_turn_limit_attempts = 0
        previous_terminal_reason: str | None = None
        budget_results: dict[int, AttemptBudgetResult] = {}
        saved_context_limit_preflight_enabled = self.context_limit_preflight_enabled

        for attempt in range(max_attempts):
            self._reset_attempt_state()
            attempt_no = attempt + 1
            attempt_id = f"{task_id}_attempt-{attempt_no}"
            is_final = attempt_no == max_attempts
            self._skip_turn_limit_final_response_this_attempt = not is_final
            self._recover_generation_limits_this_attempt = self._should_recover_generation_limits_this_attempt(is_final=is_final)
            self.context_limit_preflight_enabled = saved_context_limit_preflight_enabled
            attempt_metadata = {
                **(extra_trace_metadata or {}),
                "attempt_budget_sweep": True,
                "attempt_budget_role": "retry_path" if not is_final else "max_budget_final_path",
                "attempt_budget": None if not is_final else max_attempts,
                "attempt_number": attempt_no,
                "max_attempt_budget": max_attempts,
            }
            branch_attempt_id = f"{task_id}_budget-{attempt_no}_attempt-{attempt_no}" if not is_final else None
            branch_metadata = {
                **(extra_trace_metadata or {}),
                "attempt_budget_sweep": True,
                "attempt_budget_role": "budget_final_branch",
                "attempt_budget": attempt_no,
                "attempt_number": attempt_no,
                "max_attempt_budget": max_attempts,
            }
            try:
                await self.tool_manager.begin_task(task_id=attempt_id)
                raw_result = await self._run_single_attempt(
                    enhanced_task,
                    task_id=attempt_id,
                    extra_trace_metadata=attempt_metadata,
                    final_branch_task_id=branch_attempt_id,
                    final_branch_trace_metadata=branch_metadata,
                )
            finally:
                await self.tool_manager.end_task(task_id=attempt_id)
                self._skip_turn_limit_final_response_this_attempt = False
                self._recover_generation_limits_this_attempt = False
                self.context_limit_preflight_enabled = saved_context_limit_preflight_enabled

            branch_result: OrchestrationResult | None = None
            if isinstance(raw_result, tuple):
                result, branch_result = raw_result
            else:
                result = raw_result
            result, output, reached_turn_limit = await self._prepare_retry_attempt_output(
                result=result,
                attempt=attempt_no,
                max_attempts=max_attempts,
                is_final=is_final,
            )
            output_status = self._retry_output_status(output)
            consecutive_no_box_turn_limit_attempts = self._next_no_box_turn_limit_attempt_count(
                current_count=consecutive_no_box_turn_limit_attempts,
                reached_turn_limit=reached_turn_limit,
                output_status=output_status,
            )
            cap_blocks_retry = self._should_block_retry_for_no_box_turn_limit(
                consecutive_no_box_turn_limit_attempts=consecutive_no_box_turn_limit_attempts,
                is_final=is_final,
            )

            if output and output != FORMAT_ERROR_MESSAGE:
                result = self._attach_rollback_storm_shadow(result=result, output_status=output_status, attempt_id=attempt_id)
                self._append_retry_attempt_provenance(
                    attempt_provenance,
                    result=result,
                    attempt=attempt_no,
                    max_attempts=max_attempts,
                    attempt_id=attempt_id,
                    is_final=is_final,
                    output=output,
                    output_status=output_status,
                    previous_terminal_reason=previous_terminal_reason,
                    consecutive_no_box_turn_limit_attempts=consecutive_no_box_turn_limit_attempts,
                    retry_decision="return_success",
                    retry_decision_reason="valid_output",
                    next_attempt_launched=False,
                    next_attempt_launch_reason=None,
                    next_attempt_blocked_reason=None,
                    would_block_by_no_box_turn_limit_cap=False,
                )
                success_result = self._copy_result_with_retry_attempt_provenance(result, attempt_provenance)
                success_result = await self._maybe_apply_self_verification(
                    result=success_result,
                    canonical_task_id=attempt_id,
                    original_task=self._current_task,
                )
                for budget in range(attempt_no, max_attempts + 1):
                    reused_from = attempt_no if budget != attempt_no else None
                    budget_results[budget] = AttemptBudgetResult(
                        budget=budget,
                        result=self._copy_result_with_attempt_budget_metadata(
                            success_result,
                            budget=budget,
                            actual_attempts=attempt_no,
                            reused_from_budget=reused_from,
                        ),
                        reused_from_budget=reused_from,
                    )
                return budget_results

            budget_final_source = branch_result if branch_result is not None else result
            budget_final_attempt_id = branch_attempt_id if branch_result is not None and branch_attempt_id is not None else attempt_id
            budget_final_result = await self._finalize_attempt_budget_result(
                result=budget_final_source,
                attempt=attempt_no,
                max_attempts=max_attempts,
                attempt_id=budget_final_attempt_id,
                previous_terminal_reason=previous_terminal_reason,
                consecutive_no_box_turn_limit_attempts=consecutive_no_box_turn_limit_attempts,
                attempt_provenance=attempt_provenance,
            )
            budget_results[attempt_no] = AttemptBudgetResult(
                budget=attempt_no,
                result=self._copy_result_with_attempt_budget_metadata(
                    budget_final_result,
                    budget=attempt_no,
                    actual_attempts=attempt_no,
                    reused_from_budget=None,
                ),
            )

            if is_final:
                return budget_results

            if cap_blocks_retry:
                blocked_result = self._return_no_box_turn_limit_cap_result(
                    result=result,
                    attempt=attempt_no,
                    max_attempts=max_attempts,
                    attempt_id=attempt_id,
                    output=output,
                    output_status=output_status,
                    previous_terminal_reason=previous_terminal_reason,
                    consecutive_no_box_turn_limit_attempts=consecutive_no_box_turn_limit_attempts,
                    attempt_provenance=attempt_provenance,
                )
                for budget in range(attempt_no + 1, max_attempts + 1):
                    budget_results[budget] = AttemptBudgetResult(
                        budget=budget,
                        result=self._copy_result_with_attempt_budget_metadata(
                            blocked_result,
                            budget=budget,
                            actual_attempts=attempt_no,
                            reused_from_budget=attempt_no,
                        ),
                        reused_from_budget=attempt_no,
                    )
                return budget_results

            retry_reason = self._retry_reason_for_failed_attempt(result=result, reached_turn_limit=reached_turn_limit)
            result = self._attach_rollback_storm_shadow(result=result, output_status=output_status, attempt_id=attempt_id)
            self._append_retry_attempt_provenance(
                attempt_provenance,
                result=result,
                attempt=attempt_no,
                max_attempts=max_attempts,
                attempt_id=attempt_id,
                is_final=False,
                output=output,
                output_status=output_status,
                previous_terminal_reason=previous_terminal_reason,
                consecutive_no_box_turn_limit_attempts=consecutive_no_box_turn_limit_attempts,
                retry_decision="retry",
                retry_decision_reason=retry_reason,
                next_attempt_launched=True,
                next_attempt_launch_reason=retry_reason,
                next_attempt_blocked_reason=None,
                would_block_by_no_box_turn_limit_cap=False,
            )
            previous_terminal_reason = result.reason
            if self.include_failure_summary_in_retry:
                failure_summary = await self._generate_failure_summary(result)
                if failure_summary:
                    failure_summaries.append(failure_summary)
                enhanced_task = build_failure_enhanced_task(self._current_task, failure_summaries)
                logger.info("Sweep retrying with failure experience (attempt %d/%d)", attempt_no + 1, max_attempts)
            else:
                logger.info("Sweep retrying without failure summary (attempt %d/%d)", attempt_no + 1, max_attempts)

        return budget_results

    async def _finalize_attempt_budget_result(
        self,
        *,
        result: OrchestrationResult,
        attempt: int,
        max_attempts: int,
        attempt_id: str | None,
        previous_terminal_reason: str | None,
        consecutive_no_box_turn_limit_attempts: int,
        attempt_provenance: list[dict[str, Any]],
    ) -> OrchestrationResult:
        budget_provenance = [dict(item) for item in attempt_provenance]
        result, output, reached_turn_limit = await self._prepare_retry_attempt_output(
            result=result,
            attempt=attempt,
            max_attempts=max_attempts,
            is_final=True,
        )
        output_status = self._retry_output_status(output)
        if output and output != FORMAT_ERROR_MESSAGE:
            result = self._attach_rollback_storm_shadow(result=result, output_status=output_status, attempt_id=attempt_id)
            self._append_retry_attempt_provenance(
                budget_provenance,
                result=result,
                attempt=attempt,
                max_attempts=max_attempts,
                attempt_id=attempt_id,
                is_final=True,
                output=output,
                output_status=output_status,
                previous_terminal_reason=previous_terminal_reason,
                consecutive_no_box_turn_limit_attempts=consecutive_no_box_turn_limit_attempts,
                retry_decision="return_success",
                retry_decision_reason="valid_output",
                next_attempt_launched=False,
                next_attempt_launch_reason=None,
                next_attempt_blocked_reason=None,
                would_block_by_no_box_turn_limit_cap=False,
            )
            success_result = self._copy_result_with_retry_attempt_provenance(result, budget_provenance)
            return await self._maybe_apply_self_verification(
                result=success_result,
                canonical_task_id=attempt_id,
                original_task=self._current_task,
            )
        final_result = self._return_final_attempt_result(
            result=result,
            attempt=attempt,
            max_attempts=max_attempts,
            attempt_id=attempt_id,
            output=output,
            output_status=output_status,
            previous_terminal_reason=previous_terminal_reason,
            consecutive_no_box_turn_limit_attempts=consecutive_no_box_turn_limit_attempts
            if not reached_turn_limit
            else max(1, consecutive_no_box_turn_limit_attempts),
            attempt_provenance=budget_provenance,
        )
        return await self._maybe_apply_self_verification(
            result=final_result,
            canonical_task_id=attempt_id,
            original_task=self._current_task,
        )

    def _should_recover_generation_limits_this_attempt(self, *, is_final: bool) -> bool:
        if is_final:
            return self.generation_limit_recovery_final_attempt == "rollback"
        return self.generation_limit_recovery_non_final_attempt == "rollback"

    async def _prepare_retry_attempt_output(
        self,
        *,
        result: OrchestrationResult,
        attempt: int,
        max_attempts: int,
        is_final: bool,
    ) -> tuple[OrchestrationResult, str, bool]:
        output = str(result.output or "")
        reached_turn_limit = result.reason == "terminated_turn_limit"
        prompted_final_answer = self._is_prompted_final_answer_result(result)
        if self.max_task_retries > 0 and reached_turn_limit and not is_final:
            logger.info("Reached turn limit on non-final attempt %d/%d; retrying instead of accepting final guess", attempt, max_attempts)
            return result, FORMAT_ERROR_MESSAGE, reached_turn_limit
        if self.max_task_retries > 0 and prompted_final_answer and not is_final and not self._is_generation_limit_recovery_result(result):
            logger.info("Rejecting prompted final answer on non-final attempt %d/%d; retrying", attempt, max_attempts)
            return result, FORMAT_ERROR_MESSAGE, reached_turn_limit
        if is_final and (output == "" or output == FORMAT_ERROR_MESSAGE) and self.max_final_answer_attempts > 1:
            retried = await self._retry_final_answer(result)
            if retried is not None:
                return retried, str(retried.output or ""), retried.reason == "terminated_turn_limit"
        return result, output, reached_turn_limit

    @staticmethod
    def _is_prompted_final_answer_result(result: OrchestrationResult) -> bool:
        return result.reason in {
            "terminated_context_limit",
            "terminated_turn_limit",
            "terminated_tools_exhausted",
            "assistant_tool_calls_rejected_due_to_force_finalization",
        }

    @staticmethod
    def _is_generation_limit_recovery_result(result: OrchestrationResult) -> bool:
        info = result.info if isinstance(result.info, dict) else {}
        return bool(info.get("force_finalization_recovery") or info.get("token_exhaustion_recovery") or info.get("context_limit_error_recovery"))

    @staticmethod
    def _retry_output_status(output: str) -> str:
        if output == "":
            return "empty"
        if output == FORMAT_ERROR_MESSAGE:
            return "format_error"
        return "valid"

    def _retry_reason_for_failed_attempt(self, *, result: OrchestrationResult, reached_turn_limit: bool) -> str:
        if reached_turn_limit:
            return "non_final_turn_limit"
        if self._is_prompted_final_answer_result(result):
            return "non_final_prompted_final_answer"
        if result.reason == "terminated_token_exceed":
            return "non_final_token_exhaustion"
        return "format_error_retry_available"

    @staticmethod
    def _next_no_box_turn_limit_attempt_count(
        *,
        current_count: int,
        reached_turn_limit: bool,
        output_status: str,
    ) -> int:
        if reached_turn_limit and output_status in {"empty", "format_error"}:
            return current_count + 1
        return 0

    def _should_block_retry_for_no_box_turn_limit(
        self,
        *,
        consecutive_no_box_turn_limit_attempts: int,
        is_final: bool,
    ) -> bool:
        if is_final or not self.retry_no_box_turn_limit_cap_enabled:
            return False
        if self.retry_no_box_turn_limit_cap <= 0:
            return False
        return consecutive_no_box_turn_limit_attempts >= self.retry_no_box_turn_limit_cap

    def _return_final_attempt_result(
        self,
        *,
        result: OrchestrationResult,
        attempt: int,
        max_attempts: int,
        attempt_id: str | None,
        output: str,
        output_status: str,
        previous_terminal_reason: str | None,
        consecutive_no_box_turn_limit_attempts: int,
        attempt_provenance: list[dict[str, Any]],
    ) -> OrchestrationResult:
        retry_decision = "return_final_failure"
        retry_decision_reason = "final_attempt"
        next_attempt_blocked_reason = "final_attempt"
        fallback = self._latest_intermediate_boxed_answer()
        fallback_reason = "intermediate_boxed_fallback"
        if not fallback:
            fallback = self._fallback_answer_from_compression_state()
            fallback_reason = "compression_state_answer_attempt"
        if fallback:
            logger.info("Using %s fallback: %s", fallback_reason, fallback[:100])
            result = self._copy_result_with_output(result, fallback)
            output = fallback
            output_status = self._retry_output_status(output)
            retry_decision = "return_final_fallback"
            retry_decision_reason = fallback_reason
            next_attempt_blocked_reason = None
        result = self._attach_rollback_storm_shadow(result=result, output_status=output_status, attempt_id=attempt_id)
        self._append_retry_attempt_provenance(
            attempt_provenance,
            result=result,
            attempt=attempt,
            max_attempts=max_attempts,
            attempt_id=attempt_id,
            is_final=True,
            output=output,
            output_status=output_status,
            previous_terminal_reason=previous_terminal_reason,
            consecutive_no_box_turn_limit_attempts=consecutive_no_box_turn_limit_attempts,
            retry_decision=retry_decision,
            retry_decision_reason=retry_decision_reason,
            next_attempt_launched=False,
            next_attempt_launch_reason=None,
            next_attempt_blocked_reason=next_attempt_blocked_reason,
            would_block_by_no_box_turn_limit_cap=False,
        )
        return self._copy_result_with_retry_attempt_provenance(result, attempt_provenance)

    def _return_no_box_turn_limit_cap_result(
        self,
        *,
        result: OrchestrationResult,
        attempt: int,
        max_attempts: int,
        attempt_id: str | None,
        output: str,
        output_status: str,
        previous_terminal_reason: str | None,
        consecutive_no_box_turn_limit_attempts: int,
        attempt_provenance: list[dict[str, Any]],
    ) -> OrchestrationResult:
        logger.info(
            "Stopping retries after %d consecutive no-box turn-limit attempt(s)",
            consecutive_no_box_turn_limit_attempts,
        )
        result = self._attach_rollback_storm_shadow(result=result, output_status=output_status, attempt_id=attempt_id)
        self._append_retry_attempt_provenance(
            attempt_provenance,
            result=result,
            attempt=attempt,
            max_attempts=max_attempts,
            attempt_id=attempt_id,
            is_final=False,
            output=output,
            output_status=output_status,
            previous_terminal_reason=previous_terminal_reason,
            consecutive_no_box_turn_limit_attempts=consecutive_no_box_turn_limit_attempts,
            retry_decision="block_no_box_turn_limit_cap",
            retry_decision_reason="no_box_turn_limit_cap",
            next_attempt_launched=False,
            next_attempt_launch_reason=None,
            next_attempt_blocked_reason="no_box_turn_limit_cap",
            would_block_by_no_box_turn_limit_cap=True,
        )
        cap_metadata = self._no_box_turn_limit_cap_metadata(
            blocked_after_attempt=attempt,
            consecutive_no_box_turn_limit_attempts=consecutive_no_box_turn_limit_attempts,
            saved_remaining_attempts_estimate=max_attempts - attempt,
        )
        blocked_result = self._copy_result_with_no_box_turn_limit_cap_metadata(result, cap_metadata=cap_metadata)
        self._write_no_box_turn_limit_cap_trace_metadata(
            attempt_id=attempt_id,
            result=blocked_result,
            cap_metadata=cap_metadata,
            attempt_provenance=attempt_provenance,
        )
        return self._copy_result_with_retry_attempt_provenance(blocked_result, attempt_provenance)

    def _append_retry_attempt_provenance(
        self,
        attempt_provenance: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        if not self.retry_attempt_provenance_enabled:
            return
        attempt_provenance.append(self._summarize_retry_attempt(**kwargs))

    def _attach_rollback_storm_shadow(
        self,
        *,
        result: OrchestrationResult,
        output_status: str,
        attempt_id: str | None,
    ) -> OrchestrationResult:
        if not self.rollback_storm_shadow_enabled:
            return result
        shadow_metadata = self._rollback_storm_shadow_metadata(result=result, output_status=output_status)
        result = self._copy_result_with_rollback_storm_shadow(result, shadow_metadata=shadow_metadata)
        self._write_rollback_storm_shadow_trace_metadata(
            attempt_id=attempt_id,
            result=result,
            shadow_metadata=shadow_metadata,
        )
        return result

    def _reset_attempt_state(self) -> None:
        self._used_queries = {}
        self._semantic_query_keys = {}
        self._intermediate_boxed_answers = []
        self._last_observed_prompt_tokens = None
        self._discard_all_last_trigger_turn = None
        self._discard_all_reset_count = 0
        self._discard_all_last_attempt_mode = False
        self._reset_rollback_storm_shadow_state()

    def _reset_rollback_storm_shadow_state(self) -> None:
        self._rollback_storm_events: list[dict[str, Any]] = []
        self._rollback_storm_tool_call_count = 0
        self._rollback_storm_search_call_count = 0
        self._rollback_storm_scrape_call_count = 0

    def _record_rollback_storm_tool_requests(self, tool_requests: list[Any]) -> None:
        if not self.rollback_storm_shadow_enabled:
            return
        self._rollback_storm_tool_call_count += len(tool_requests)
        for request in tool_requests:
            tool_name = str(getattr(request, "tool_name", ""))
            if tool_name in _ROLLBACK_STORM_SEARCH_TOOL_NAMES:
                self._rollback_storm_search_call_count += 1
            elif tool_name in _ROLLBACK_STORM_SCRAPE_TOOL_NAMES:
                self._rollback_storm_scrape_call_count += 1

    def _record_rollback_storm_event(
        self,
        *,
        reason: str,
        turn_idx: int,
        tool_requests: list[Any] | None = None,
        tool_outcomes: list[Any] | None = None,
    ) -> None:
        if not self.rollback_storm_shadow_enabled:
            return
        event: dict[str, Any] = {
            "reason": reason,
            "turn": turn_idx,
        }
        if tool_requests is not None:
            event["preview"] = self._rollback_storm_request_preview(tool_requests, reason=reason)
        if tool_outcomes is not None:
            event["preview"] = self._rollback_storm_outcome_preview(tool_outcomes, reason=reason)
        self._rollback_storm_events.append(event)

    def _rollback_storm_request_preview(self, tool_requests: list[Any], *, reason: str) -> list[dict[str, Any]]:
        limit = max(0, self.rollback_storm_preview_max_items)
        preview: list[dict[str, Any]] = []
        for request in tool_requests[:limit]:
            tool_name = str(getattr(request, "tool_name", ""))
            arguments = self._safe_tool_arguments(getattr(request, "arguments", {}))
            preview.append(self._rollback_storm_preview_item(tool_name=tool_name, arguments=arguments, reason=reason))
        return preview

    def _rollback_storm_outcome_preview(self, tool_outcomes: list[Any], *, reason: str) -> list[dict[str, Any]]:
        limit = max(0, self.rollback_storm_preview_max_items)
        preview: list[dict[str, Any]] = []
        for outcome in tool_outcomes[:limit]:
            request = getattr(outcome, "request", None)
            tool_name = str(getattr(request, "tool_name", ""))
            arguments = self._safe_tool_arguments(getattr(request, "arguments", {}))
            item = self._rollback_storm_preview_item(tool_name=tool_name, arguments=arguments, reason=reason)
            result = getattr(outcome, "result", None)
            status = getattr(result, "status", None)
            item["result_status"] = getattr(status, "value", str(status)) if status is not None else None
            preview.append(item)
        return preview

    @staticmethod
    def _safe_tool_arguments(arguments: Any) -> dict[str, Any]:
        return arguments if isinstance(arguments, dict) else {}

    @staticmethod
    def _truncate_shadow_preview(text: str) -> str:
        return text[:_ROLLBACK_STORM_PREVIEW_CHARS]

    def _rollback_storm_preview_item(self, *, tool_name: str, arguments: dict[str, Any], reason: str) -> dict[str, Any]:
        redacted_arguments = self._redacted_rollback_storm_arguments(tool_name=tool_name, arguments=arguments)
        preview_value = self._rollback_storm_query_or_url_preview(tool_name=tool_name, arguments=redacted_arguments)
        normalized_key = get_semantic_query_key_from_tool_call(tool_name, redacted_arguments)
        return {
            "reason": reason,
            "tool_name": tool_name,
            "normalized_key": self._truncate_shadow_preview(normalized_key or ""),
            "query_or_url_preview": self._truncate_shadow_preview(preview_value),
        }

    @staticmethod
    def _redacted_rollback_storm_arguments(*, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        redacted = dict(arguments)
        if tool_name == "google_search" and "q" in redacted:
            redacted["q"] = _redact_rollback_storm_secret_params(str(redacted["q"]))
        elif tool_name == "web_search" and "query" in redacted:
            redacted["query"] = _redact_rollback_storm_secret_params(str(redacted["query"]))
        elif tool_name == "scrape_and_extract_info":
            if "url" in redacted:
                redacted["url"] = _redact_rollback_storm_secret_params(str(redacted["url"]))
            if "info_to_extract" in redacted:
                redacted["info_to_extract"] = _redact_rollback_storm_secret_params(str(redacted["info_to_extract"]))
        return redacted

    @staticmethod
    def _rollback_storm_query_or_url_preview(*, tool_name: str, arguments: dict[str, Any]) -> str:
        if tool_name == "google_search":
            return str(arguments.get("q", ""))
        if tool_name == "web_search":
            return str(arguments.get("query", ""))
        if tool_name == "scrape_and_extract_info":
            url = str(arguments.get("url", ""))
            info_to_extract = str(arguments.get("info_to_extract", ""))
            return f"{url} {info_to_extract}".strip()
        return ""

    async def _run_single_attempt(  # noqa: PLR0915
        self,
        task: str | dict[str, Any],
        task_id: str | None = None,
        *,
        extra_trace_metadata: dict[str, Any] | None = None,
        final_branch_task_id: str | None = None,
        final_branch_trace_metadata: dict[str, Any] | None = None,
        runtime_config_overrides: dict[str, Any] | None = None,
        trace_tool_path_suffix: list[str] | None = None,
    ) -> OrchestrationResult | tuple[OrchestrationResult, OrchestrationResult | None]:
        task_id = task_id or uuid4().hex
        normalized_task = self.normalize_task(task)
        self._maybe_save_run_config()
        self._start_trace(task_id, normalized_task, extra_metadata=extra_trace_metadata, extra_tool_path=trace_tool_path_suffix)
        self.tool_manager.reset_task_state()
        tools = self.tool_manager.list_tool_definitions()
        conv_runtime = self.build_conversation_runtime(tools=tools, config_overrides=runtime_config_overrides)

        effective_turn_count = 0
        total_attempts = 0
        max_turns = self.config.conversation.max_turns
        max_attempts = (max_turns or 10_000) + 200

        t0 = time.perf_counter()
        step_result = conv_runtime.initialize_conversation(normalized_task)
        runtime_elapsed_ms = (time.perf_counter() - t0) * 1000
        self._log_runtime_step(task_id, effective_turn_count, step_result, elapsed_ms=runtime_elapsed_ms)
        self._sync_trace(task_id, step_result.appended_messages)

        done = False
        reason = None
        total_reward = 0.0
        run_info: dict[str, Any] = self._update_run_info({}, dict(step_result.info))
        final_branch_result: OrchestrationResult | None = None

        while not done:
            if (
                final_branch_task_id is not None
                and final_branch_result is None
                and self._should_fork_attempt_budget_final_branch(runtime=conv_runtime, step_result=step_result)
            ):
                final_branch_result = await self._run_attempt_budget_final_branch_from_state(
                    source_task_id=task_id,
                    branch_task_id=final_branch_task_id,
                    tools=tools,
                    runtime=conv_runtime,
                    step_result=step_result,
                    turn_count=effective_turn_count,
                    total_attempts=total_attempts,
                    max_turns=max_turns,
                    max_attempts=max_attempts,
                    total_reward=total_reward,
                    run_info=run_info,
                    extra_trace_metadata=final_branch_trace_metadata,
                )

            if total_attempts >= max_attempts:
                self._log_step(
                    task_id,
                    effective_turn_count,
                    "orchestrator.error",
                    f"Safety limit reached ({total_attempts}/{max_attempts} total attempts)",
                    emoji="❌",
                )
                reason = "terminated_loop_safety_limit"
                done = True
                break

            loop_before = {
                "turn_count": effective_turn_count,
                "total_attempts": total_attempts,
                "max_turns": max_turns,
                "max_attempts": max_attempts,
            }
            counts_as_model_attempt = not (
                step_result.stage == ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT
                and step_result.info.get("finalization_trigger")
                in {FinalizationTrigger.TURN_LIMIT.value, FinalizationTrigger.CONTEXT_LIMIT.value, FinalizationTrigger.TOOLS_EXHAUSTED.value}
                and getattr(conv_runtime, "skip_turn_limit_final_response", False)
            )
            if counts_as_model_attempt:
                total_attempts += 1
            turn_idx = effective_turn_count + 1

            branch_result_sink: list[OrchestrationResult] | None = None
            branch_checkpoint_factory: Callable[[], AttemptBudgetBranchCheckpoint] | None = None
            if final_branch_task_id is not None and final_branch_result is None:
                branch_result_sink = []

                def branch_checkpoint_factory(
                    step_result_snapshot: ConversationStepResult = step_result,
                    effective_turn_count_snapshot: int = effective_turn_count,
                    total_attempts_snapshot: int = total_attempts,
                    max_turns_snapshot: int = max_turns,
                    max_attempts_snapshot: int = max_attempts,
                    total_reward_snapshot: float = total_reward,
                    run_info_snapshot: dict[str, Any] = run_info,
                ) -> AttemptBudgetBranchCheckpoint:
                    return AttemptBudgetBranchCheckpoint(
                        source_task_id=task_id,
                        branch_task_id=final_branch_task_id,
                        tools=tools,
                        runtime=conv_runtime.clone(),
                        step_result=step_result_snapshot.model_copy(deep=True),
                        turn_count=effective_turn_count_snapshot,
                        total_attempts=total_attempts_snapshot,
                        max_turns=max_turns_snapshot,
                        max_attempts=max_attempts_snapshot,
                        total_reward=total_reward_snapshot,
                        run_info=copy.deepcopy(run_info_snapshot),
                        attempt_state=self._snapshot_attempt_mutable_state(),
                        extra_trace_metadata=final_branch_trace_metadata,
                    )

            step_result, step_reward, done, reason, step_info = await self._run_turn(
                runtime=conv_runtime,
                tools=tools,
                step_result=step_result,
                task_id=task_id,
                turn_idx=turn_idx,
                attempt_budget_branch_checkpoint_factory=branch_checkpoint_factory,
                attempt_budget_branch_result_sink=branch_result_sink,
            )
            if branch_result_sink and final_branch_result is None:
                final_branch_result = branch_result_sink[-1]
            total_reward += step_reward

            effective_turn_count = int(step_result.info.get("assistant_turn_idx", effective_turn_count))
            step_info = dict(step_info)
            compression_info = await self._maybe_apply_context_compression(
                runtime=conv_runtime,
                task_id=task_id,
                turn_count=effective_turn_count,
                done=done,
            )
            if compression_info is not None:
                step_info["context_compression"] = compression_info
                # The next model call builds its prompt from this snapshot; refresh
                # it so compaction reaches the model on the very next request
                # instead of one call late (and so the provider-token observation
                # for that request still matches the tracker for calibration).
                step_result = step_result.model_copy(
                    update={
                        "visible_conversation": conv_runtime._build_visible_conversation(),
                        "full_conversation": list(conv_runtime._full_conversation),
                    }
                )
            if not done:
                last_attempt_info = self._maybe_enter_discard_all_last_attempt(
                    runtime=conv_runtime,
                    task_id=task_id,
                    turn_count=effective_turn_count,
                    reason="max_tool_calls_reached",
                )
                if last_attempt_info is not None:
                    step_info["discard_all_last_attempt"] = last_attempt_info
            if step_info.get("discard_all_last_attempt") is not None:
                max_turns, max_attempts = self._sync_loop_limits_for_discard_all_last_attempt(
                    runtime=conv_runtime,
                    max_turns=max_turns,
                    max_attempts=max_attempts,
                )
            discard_all_info = await self._maybe_apply_discard_all(
                runtime=conv_runtime,
                task_id=task_id,
                turn_count=effective_turn_count,
                done=done,
            )
            if discard_all_info is not None:
                step_info["discard_all"] = discard_all_info
                # Refresh the snapshot so the post-reset prompt reaches the model
                # on the very next request (mirror of the compaction sync above).
                step_result = step_result.model_copy(
                    update={
                        "visible_conversation": conv_runtime._build_visible_conversation(),
                        "full_conversation": list(conv_runtime._full_conversation),
                    }
                )
            loop_after = {
                "turn_count": effective_turn_count,
                "total_attempts": total_attempts,
                "max_turns": max_turns,
                "max_attempts": max_attempts,
            }
            step_info["loop_accounting_step"] = {
                "before": loop_before,
                "after": loop_after,
                "counts_as_model_attempt": counts_as_model_attempt,
            }
            run_info = self._update_run_info(run_info, step_info)

        if reason is None:
            msg = "Web-search orchestration loop ended without a termination reason."
            raise RuntimeError(msg)

        result = self._finalize_single_attempt_result(
            task_id=task_id,
            step_result=step_result,
            conv_runtime=conv_runtime,
            tools=tools,
            reason=reason,
            total_reward=total_reward,
            run_info=run_info,
            turn_count=effective_turn_count,
            total_attempts=total_attempts,
            max_turns=max_turns,
            max_attempts=max_attempts,
        )
        if final_branch_task_id is not None:
            return result, final_branch_result
        return result

    def _finalize_single_attempt_result(
        self,
        *,
        task_id: str,
        step_result: ConversationStepResult,
        conv_runtime: ConversationRuntime,
        tools: list[dict[str, Any]] | None,
        reason: str,
        total_reward: float,
        run_info: dict[str, Any],
        turn_count: int,
        total_attempts: int,
        max_turns: int | None,
        max_attempts: int,
    ) -> OrchestrationResult:
        runtime_timing = run_info.setdefault(
            "inference_timing",
            {"model_client_elapsed_ms_sum": 0.0, "engine_e2e_elapsed_s_sum": 0.0, "num_inference_calls": 0},
        )
        runtime_timing["tokenizer_elapsed_ms_sum"] = float(conv_runtime.tokenizer_elapsed_ms)
        runtime_timing["runtime_step_elapsed_ms_sum"] = float(conv_runtime.runtime_step_elapsed_ms)
        run_info["loop_accounting"] = {
            "turn_count": turn_count,
            "total_attempts": total_attempts,
            "max_turns": max_turns,
            "max_attempts": max_attempts,
        }

        output = self.extract_final_output(step_result.visible_conversation)
        if reason == "terminated_semantic_query_budget":
            output = FORMAT_ERROR_MESSAGE
            self._intermediate_boxed_answers = []
        else:
            self._intermediate_boxed_answers = self._boxed_answers_from_visible_conversation(step_result.visible_conversation)
        metadata = {
            "task_id": task_id,
            "orchestrator_name": self.config.name,
            "output": output,
            "max_turns": max_turns,
            "tool_names": self.tool_manager.list_tool_names(),
            "tools": tools,
            "turn_used": turn_count,
            "total_attempts": total_attempts,
            "done_reason": reason,
            "run_info": run_info,
            "unknown_tool_names": self.tool_manager.unknown_tool_names_snapshot(),
        }
        if "semantic_query_budget" in run_info:
            metadata["semantic_query_budget"] = run_info["semantic_query_budget"]
        if self.task_logger is not None:
            task_logger_timing = self.task_logger.finish_task(task_id, status=reason, metadata=metadata, level=self.level)
            metadata.update(task_logger_timing)
            run_info["task_logger_timing"] = task_logger_timing

        return OrchestrationResult(
            output=output,
            reason=reason,
            conversation=step_result.full_conversation,
            visible_conversation=step_result.visible_conversation,
            num_turns=turn_count,
            reward=total_reward,
            done=True,
            info=run_info,
            metadata=metadata,
        )

    def build_conversation_runtime(
        self,
        tools: list[dict[str, Any]] | None,
        *,
        config_overrides: dict[str, Any] | None = None,
    ) -> ConversationRuntime:
        summary_prompt = generate_summary_prompt(self._current_task, prompt_profile=self.prompt_profile)
        update = {
            "early_stop_announcement_prompt": None,
            "final_response_prompt": summary_prompt,
        }
        if config_overrides:
            update.update(config_overrides)
        config = self.config.conversation.model_copy(
            update=update,
        )
        runtime = self._conversation_runtime_class(
            config=config,
            tools=tools,
            max_output_tokens=self.model_client.max_output_tokens,
            token_estimator=self.context_token_estimator or self.model_client.estimate_tokens,
            token_estimator_includes_tools=bool(getattr(self.context_token_estimator, "token_estimator_includes_tools", False)),
            token_estimator_is_additive=bool(getattr(self.context_token_estimator, "token_estimator_is_additive", False)),
            context_limit_preflight_enabled=self.context_limit_preflight_enabled,
        )
        if hasattr(runtime, "skip_turn_limit_final_response"):
            runtime_with_skip: Any = runtime
            runtime_with_skip.skip_turn_limit_final_response = self._skip_turn_limit_final_response_this_attempt
        if hasattr(runtime, "skip_prompted_final_response"):
            runtime_with_prompt_skip: Any = runtime
            runtime_with_prompt_skip.skip_prompted_final_response = self._skip_turn_limit_final_response_this_attempt
        return runtime

    @staticmethod
    def _should_fork_attempt_budget_final_branch(
        *,
        runtime: ConversationRuntime,
        step_result: ConversationStepResult,
    ) -> bool:
        return (
            step_result.stage == ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT
            and bool(getattr(runtime, "skip_turn_limit_final_response", False))
            and bool(step_result.info.get("skipped_budget_limit_final_response"))
        )

    def _snapshot_attempt_mutable_state(self) -> dict[str, Any]:
        return {
            "used_queries": dict(self._used_queries),
            "semantic_query_keys": dict(self._semantic_query_keys),
            "intermediate_boxed_answers": list(self._intermediate_boxed_answers),
            "rollback_storm_events": [dict(item) for item in self._rollback_storm_events],
            "rollback_storm_tool_call_count": self._rollback_storm_tool_call_count,
            "rollback_storm_search_call_count": self._rollback_storm_search_call_count,
            "rollback_storm_scrape_call_count": self._rollback_storm_scrape_call_count,
            "last_observed_prompt_tokens": self._last_observed_prompt_tokens,
            "discard_all_last_trigger_turn": self._discard_all_last_trigger_turn,
            "discard_all_reset_count": self._discard_all_reset_count,
            "discard_all_last_attempt_mode": self._discard_all_last_attempt_mode,
            "tool_manager": self.tool_manager.snapshot_task_state(),
        }

    def _restore_attempt_mutable_state(self, snapshot: dict[str, Any]) -> None:
        self._used_queries = dict(snapshot.get("used_queries", {}))
        self._semantic_query_keys = dict(snapshot.get("semantic_query_keys", {}))
        self._intermediate_boxed_answers = list(snapshot.get("intermediate_boxed_answers", []))
        self._rollback_storm_events = [dict(item) for item in snapshot.get("rollback_storm_events", [])]
        self._rollback_storm_tool_call_count = int(snapshot.get("rollback_storm_tool_call_count", 0) or 0)
        self._rollback_storm_search_call_count = int(snapshot.get("rollback_storm_search_call_count", 0) or 0)
        self._rollback_storm_scrape_call_count = int(snapshot.get("rollback_storm_scrape_call_count", 0) or 0)
        self._last_observed_prompt_tokens = snapshot.get("last_observed_prompt_tokens")
        self._discard_all_last_trigger_turn = snapshot.get("discard_all_last_trigger_turn")
        self._discard_all_reset_count = int(snapshot.get("discard_all_reset_count", 0) or 0)
        self._discard_all_last_attempt_mode = bool(snapshot.get("discard_all_last_attempt_mode", False))
        tool_state = snapshot.get("tool_manager")
        if isinstance(tool_state, dict):
            self.tool_manager.restore_task_state(tool_state)

    def _copy_attempt_mutable_state(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "used_queries": dict(snapshot.get("used_queries", {})),
            "semantic_query_keys": dict(snapshot.get("semantic_query_keys", {})),
            "intermediate_boxed_answers": list(snapshot.get("intermediate_boxed_answers", [])),
            "rollback_storm_events": [dict(item) for item in snapshot.get("rollback_storm_events", [])],
            "rollback_storm_tool_call_count": int(snapshot.get("rollback_storm_tool_call_count", 0) or 0),
            "rollback_storm_search_call_count": int(snapshot.get("rollback_storm_search_call_count", 0) or 0),
            "rollback_storm_scrape_call_count": int(snapshot.get("rollback_storm_scrape_call_count", 0) or 0),
            "last_observed_prompt_tokens": snapshot.get("last_observed_prompt_tokens"),
            "discard_all_last_trigger_turn": snapshot.get("discard_all_last_trigger_turn"),
            "discard_all_reset_count": int(snapshot.get("discard_all_reset_count", 0) or 0),
            "discard_all_last_attempt_mode": bool(snapshot.get("discard_all_last_attempt_mode", False)),
            "tool_manager": dict(snapshot.get("tool_manager", {})),
        }

    async def _run_attempt_budget_final_branch_from_state(
        self,
        *,
        source_task_id: str,
        branch_task_id: str,
        tools: list[dict[str, Any]] | None,
        runtime: ConversationRuntime,
        step_result: ConversationStepResult,
        turn_count: int,
        total_attempts: int,
        max_turns: int | None,
        max_attempts: int,
        total_reward: float,
        run_info: dict[str, Any],
        extra_trace_metadata: dict[str, Any] | None,
        attempt_state: dict[str, Any] | None = None,
    ) -> OrchestrationResult:
        saved_state = self._snapshot_attempt_mutable_state()
        branch_state = self._copy_attempt_mutable_state(attempt_state or saved_state)
        saved_recover_generation_limits = self._recover_generation_limits_this_attempt
        saved_skip_turn_limit_final_response = self._skip_turn_limit_final_response_this_attempt
        saved_context_limit_preflight_enabled = self.context_limit_preflight_enabled
        try:
            self._restore_attempt_mutable_state(branch_state)
            branch_runtime = runtime.clone()
            branch_runtime.context_limit_preflight_enabled = False
            branch_runtime_any: Any = branch_runtime
            if hasattr(branch_runtime, "skip_turn_limit_final_response"):
                branch_runtime_any.skip_turn_limit_final_response = False
            if hasattr(branch_runtime, "skip_prompted_final_response"):
                branch_runtime_any.skip_prompted_final_response = False
            self._recover_generation_limits_this_attempt = self._should_recover_generation_limits_this_attempt(is_final=True)
            self._skip_turn_limit_final_response_this_attempt = False
            self.context_limit_preflight_enabled = False

            if self.task_logger is not None:
                self.task_logger.fork_live_trace(
                    source_task_id,
                    branch_task_id,
                    metadata=extra_trace_metadata,
                    level=self.level,
                )
            branch_step_result = self._prepare_silent_final_branch_step_result(
                task_id=branch_task_id,
                runtime=branch_runtime,
                step_result=step_result,
                turn_count=turn_count,
            )
            return await self._continue_single_attempt_from_state(
                task_id=branch_task_id,
                tools=tools,
                runtime=branch_runtime,
                step_result=branch_step_result,
                turn_count=turn_count,
                total_attempts=total_attempts,
                max_turns=max_turns,
                max_attempts=max_attempts,
                total_reward=total_reward,
                run_info=copy.deepcopy(run_info),
            )
        finally:
            self.context_limit_preflight_enabled = saved_context_limit_preflight_enabled
            self._recover_generation_limits_this_attempt = saved_recover_generation_limits
            self._skip_turn_limit_final_response_this_attempt = saved_skip_turn_limit_final_response
            self._restore_attempt_mutable_state(saved_state)

    async def _run_attempt_budget_final_branch_from_checkpoint(
        self,
        checkpoint: AttemptBudgetBranchCheckpoint,
    ) -> OrchestrationResult:
        return await self._run_attempt_budget_final_branch_from_state(
            source_task_id=checkpoint.source_task_id,
            branch_task_id=checkpoint.branch_task_id,
            tools=checkpoint.tools,
            runtime=checkpoint.runtime,
            step_result=checkpoint.step_result,
            turn_count=checkpoint.turn_count,
            total_attempts=checkpoint.total_attempts,
            max_turns=checkpoint.max_turns,
            max_attempts=checkpoint.max_attempts,
            total_reward=checkpoint.total_reward,
            run_info=checkpoint.run_info,
            extra_trace_metadata=checkpoint.extra_trace_metadata,
            attempt_state=checkpoint.attempt_state,
        )

    def _prepare_silent_final_branch_step_result(
        self,
        *,
        task_id: str,
        runtime: ConversationRuntime,
        step_result: ConversationStepResult,
        turn_count: int,
    ) -> ConversationStepResult:
        if step_result.stage != ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT:
            return step_result.model_copy(
                update={
                    "visible_conversation": runtime._build_visible_conversation(),
                    "full_conversation": list(runtime._full_conversation),
                }
            )
        prompt = runtime._build_early_stop_prompt()
        if not prompt:
            return step_result.model_copy(
                update={
                    "visible_conversation": runtime._build_visible_conversation(),
                    "full_conversation": list(runtime._full_conversation),
                }
            )
        prompt_message = ConversationMessage.user(prompt)
        runtime._append_messages([prompt_message])
        runtime._force_final_has_prompt = True
        self._sync_trace(task_id, [prompt_message])
        self._log_step(
            task_id,
            turn_count,
            "sweep.final_branch_prompt",
            "inserted final prompt for no-retry attempt-budget branch",
            metadata={"attempt_budget_final_branch": True},
        )
        info = dict(step_result.info)
        info["attempt_budget_final_branch"] = True
        info["has_early_stop_prompt"] = True
        info.pop("skipped_budget_limit_final_response", None)
        info.pop("skipped_turn_limit_final_response", None)
        info.pop("skipped_context_limit_final_response", None)
        info.pop("skipped_tools_exhausted_final_response", None)
        return step_result.model_copy(
            update={
                "appended_messages": [prompt_message],
                "visible_conversation": runtime._build_visible_conversation(),
                "full_conversation": list(runtime._full_conversation),
                "info": info,
            }
        )

    async def _continue_single_attempt_from_state(
        self,
        *,
        task_id: str,
        tools: list[dict[str, Any]] | None,
        runtime: ConversationRuntime,
        step_result: ConversationStepResult,
        turn_count: int,
        total_attempts: int,
        max_turns: int | None,
        max_attempts: int,
        total_reward: float,
        run_info: dict[str, Any],
    ) -> OrchestrationResult:
        done = False
        reason: str | None = None
        effective_turn_count = turn_count
        while not done:
            if total_attempts >= max_attempts:
                self._log_step(
                    task_id,
                    effective_turn_count,
                    "orchestrator.error",
                    f"Safety limit reached ({total_attempts}/{max_attempts} total attempts)",
                    emoji="❌",
                )
                reason = "terminated_loop_safety_limit"
                done = True
                break

            loop_before = {
                "turn_count": effective_turn_count,
                "total_attempts": total_attempts,
                "max_turns": max_turns,
                "max_attempts": max_attempts,
            }
            counts_as_model_attempt = not (
                step_result.stage == ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT
                and step_result.info.get("finalization_trigger")
                in {FinalizationTrigger.TURN_LIMIT.value, FinalizationTrigger.CONTEXT_LIMIT.value, FinalizationTrigger.TOOLS_EXHAUSTED.value}
                and getattr(runtime, "skip_turn_limit_final_response", False)
            )
            if counts_as_model_attempt:
                total_attempts += 1
            turn_idx = effective_turn_count + 1
            step_result, step_reward, done, reason, step_info = await self._run_turn(
                runtime=runtime,
                tools=tools,
                step_result=step_result,
                task_id=task_id,
                turn_idx=turn_idx,
            )
            total_reward += step_reward
            effective_turn_count = int(step_result.info.get("assistant_turn_idx", effective_turn_count))
            step_info = dict(step_info)
            compression_info = await self._maybe_apply_context_compression(
                runtime=runtime,
                task_id=task_id,
                turn_count=effective_turn_count,
                done=done,
            )
            if compression_info is not None:
                step_info["context_compression"] = compression_info
                step_result = step_result.model_copy(
                    update={
                        "visible_conversation": runtime._build_visible_conversation(),
                        "full_conversation": list(runtime._full_conversation),
                    }
                )
            if not done:
                last_attempt_info = self._maybe_enter_discard_all_last_attempt(
                    runtime=runtime,
                    task_id=task_id,
                    turn_count=effective_turn_count,
                    reason="max_tool_calls_reached",
                )
                if last_attempt_info is not None:
                    step_info["discard_all_last_attempt"] = last_attempt_info
            if step_info.get("discard_all_last_attempt") is not None:
                max_turns, max_attempts = self._sync_loop_limits_for_discard_all_last_attempt(
                    runtime=runtime,
                    max_turns=max_turns,
                    max_attempts=max_attempts,
                )
            discard_all_info = await self._maybe_apply_discard_all(
                runtime=runtime,
                task_id=task_id,
                turn_count=effective_turn_count,
                done=done,
            )
            if discard_all_info is not None:
                step_info["discard_all"] = discard_all_info
                step_result = step_result.model_copy(
                    update={
                        "visible_conversation": runtime._build_visible_conversation(),
                        "full_conversation": list(runtime._full_conversation),
                    }
                )
            loop_after = {
                "turn_count": effective_turn_count,
                "total_attempts": total_attempts,
                "max_turns": max_turns,
                "max_attempts": max_attempts,
            }
            step_info["loop_accounting_step"] = {
                "before": loop_before,
                "after": loop_after,
                "counts_as_model_attempt": counts_as_model_attempt,
            }
            run_info = self._update_run_info(run_info, step_info)

        if reason is None:
            msg = "Web-search branch loop ended without a termination reason."
            raise RuntimeError(msg)

        return self._finalize_single_attempt_result(
            task_id=task_id,
            step_result=step_result,
            conv_runtime=runtime,
            tools=tools,
            reason=reason,
            total_reward=total_reward,
            run_info=run_info,
            turn_count=effective_turn_count,
            total_attempts=total_attempts,
            max_turns=max_turns,
            max_attempts=max_attempts,
        )

    async def _maybe_apply_context_compression(
        self,
        *,
        runtime: ConversationRuntime,
        task_id: str,
        turn_count: int,
        done: bool,
    ) -> dict[str, Any] | None:
        manager = self._context_compression_manager
        if manager is None or done or not manager.should_trigger(turn_count):
            return None
        # Compress the *untruncated* visible conversation; _full_conversation
        # stays append-only. The per-request rendering (keep_tool_result) would
        # feed the summarizer "omitted" placeholders instead of real tool
        # evidence; the splice indices are identical for both views (truncation
        # replaces content 1:1, never changes message count or positions).
        visible_before = runtime.untruncated_visible_conversation()
        visible_len_before = len(visible_before)
        marker, summary_text = await manager.maybe_compress(
            turn_count=turn_count,
            visible_conversation=visible_before,
            task_text=self._current_task,
        )
        if marker is None:
            return None
        # Appending the marker both records the event in _full_conversation and
        # compacts the visible prompt (the runtime replays markers on append).
        runtime._append_messages([marker])
        # The marker is appended between conversation steps, so it never reaches
        # the persisted trace via step_result.appended_messages the way rollback
        # markers do; sync it explicitly to keep the saved trace a faithful
        # append-only log that can reconstruct the model-visible prompt.
        self._sync_trace(task_id, [marker])
        return {
            "turn": turn_count,
            "summary_chars": len(summary_text or ""),
            "visible_len_before": visible_len_before,
            "visible_len_after": len(runtime.untruncated_visible_conversation()),
        }

    async def _maybe_apply_discard_all(
        self,
        *,
        runtime: ConversationRuntime,
        task_id: str,
        turn_count: int,
        done: bool,
    ) -> dict[str, Any] | None:
        manager = self._discard_all_manager
        if manager is None or done or self._discard_all_last_attempt_mode:
            return None
        if self.tool_manager.task_total_calls >= manager.max_tool_calls:
            return None
        force_reason = self._preemptable_force_final_reason(runtime)
        context_window = getattr(runtime, "context_window", None)
        should_trigger = force_reason is not None or manager.should_trigger(
            observed_prompt_tokens=self._last_observed_prompt_tokens,
            context_window=context_window,
            turn_count=turn_count,
            last_trigger_turn=self._discard_all_last_trigger_turn,
        )
        if not should_trigger:
            return None
        return self._apply_discard_all_reset(
            runtime=runtime,
            task_id=task_id,
            turn_count=turn_count,
            reason=force_reason or "threshold",
        )

    def _preemptable_force_final_reason(self, runtime: ConversationRuntime) -> str | None:
        if runtime.stage != ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT:
            return None
        trigger = getattr(runtime, "_force_final_trigger", None)
        if trigger in {FinalizationTrigger.CONTEXT_LIMIT, FinalizationTrigger.TURN_LIMIT}:
            return f"preempt_{trigger.value}"
        return None

    def _apply_discard_all_reset(
        self,
        *,
        runtime: ConversationRuntime,
        task_id: str,
        turn_count: int,
        reason: str,
    ) -> dict[str, Any] | None:
        manager = self._discard_all_manager
        if manager is None or self._discard_all_last_attempt_mode:
            return None
        if self.tool_manager.task_total_calls >= manager.max_tool_calls:
            return None
        # Discard against the *untruncated* tracked view; keep_tool_result
        # truncation is a per-request rendering concern and the marker prefix
        # addresses this list (same view compaction splices against).
        visible_before = runtime.untruncated_visible_conversation()
        visible_len_before = len(visible_before)
        marker = manager.build_marker(visible_before)
        if marker is None:
            return None
        # Append-only: appending the marker records the reset in
        # _full_conversation and truncates the visible prompt to the system+task
        # prefix (the runtime replays markers on append). It is appended between
        # conversation steps, so sync it to the trace explicitly the way
        # compaction does.
        runtime._append_messages([marker])
        runtime.reopen_after_context_reset()
        self._sync_trace(task_id, [marker])
        self._discard_all_last_trigger_turn = turn_count
        self._discard_all_reset_count += 1
        visible_len_after = len(runtime.untruncated_visible_conversation())
        self._log_step(
            task_id,
            turn_count,
            "orchestrator.discard_all",
            f"discard-all reset #{self._discard_all_reset_count}: visible {visible_len_before} -> {visible_len_after} "
            f"(reason={reason}, observed_prompt_tokens={self._last_observed_prompt_tokens}, "
            f"context_window={getattr(runtime, 'context_window', None)})",
            emoji="🧹",
        )
        return {
            "turn": turn_count,
            "reset_index": self._discard_all_reset_count,
            "reason": reason,
            "observed_prompt_tokens": self._last_observed_prompt_tokens,
            "context_window": getattr(runtime, "context_window", None),
            "trigger_ratio": manager.cfg.trigger_ratio,
            "prefix_len": visible_len_after,
            "visible_len_before": visible_len_before,
            "visible_len_after": visible_len_after,
            "tool_call_count": self.tool_manager.task_total_calls,
        }

    def _maybe_enter_discard_all_last_attempt(
        self,
        *,
        runtime: ConversationRuntime,
        task_id: str,
        turn_count: int,
        reason: str,
    ) -> dict[str, Any] | None:
        manager = self._discard_all_manager
        if manager is None or self._discard_all_last_attempt_mode:
            return None
        if self.tool_manager.task_total_calls < manager.max_tool_calls:
            return None
        self._discard_all_last_attempt_mode = True
        current_assistant_turn = int(getattr(runtime, "_assistant_turn_count", turn_count) or turn_count)
        runtime_max_turns_before = runtime.max_turns
        if self._discard_all_last_attempt_max_turns is not None:
            runtime.max_turns = current_assistant_turn + self._discard_all_last_attempt_max_turns
        else:
            runtime.max_turns = None
        info = {
            "turn": turn_count,
            "reason": reason,
            "tool_call_count": self.tool_manager.task_total_calls,
            "discard_all_max_tool_calls": manager.max_tool_calls,
            "last_attempt_max_turns": self._discard_all_last_attempt_max_turns,
            "runtime_max_turns_before": runtime_max_turns_before,
            "runtime_max_turns_after": runtime.max_turns,
        }
        self._log_step(
            task_id,
            turn_count,
            "orchestrator.discard_all_last_attempt",
            "discard-all budget reached; disabling further resets and continuing with no-discard final-attempt limits",
            metadata=info,
            emoji="🧹",
        )
        return info

    def _sync_loop_limits_for_discard_all_last_attempt(
        self,
        *,
        runtime: ConversationRuntime,
        max_turns: int | None,
        max_attempts: int,
    ) -> tuple[int | None, int]:
        if not self._discard_all_last_attempt_mode:
            return max_turns, max_attempts
        runtime_max_turns = runtime.max_turns
        safety_turn_limit = runtime_max_turns if runtime_max_turns is not None else 10_000
        return runtime_max_turns, max(max_attempts, safety_turn_limit + 200)

    def _normalize_skipped_budget_final_response_info(
        self,
        step_result: ConversationStepResult,
    ) -> tuple[str | None, dict[str, Any] | None]:
        finalization_trigger = step_result.info.get("finalization_trigger")
        if finalization_trigger == FinalizationTrigger.TURN_LIMIT.value:
            done_reason = "terminated_turn_limit"
        elif finalization_trigger == FinalizationTrigger.CONTEXT_LIMIT.value:
            done_reason = "terminated_context_limit"
        elif finalization_trigger == FinalizationTrigger.TOOLS_EXHAUSTED.value:
            done_reason = "terminated_tools_exhausted"
        else:
            return None, None

        info = dict(step_result.info)
        info["skipped_budget_limit_final_response"] = True
        if finalization_trigger == FinalizationTrigger.TURN_LIMIT.value:
            info["skipped_turn_limit_final_response"] = True
            info.pop("skipped_context_limit_final_response", None)
            info.pop("skipped_tools_exhausted_final_response", None)
        elif finalization_trigger == FinalizationTrigger.CONTEXT_LIMIT.value:
            info["skipped_context_limit_final_response"] = True
            info.pop("skipped_turn_limit_final_response", None)
            info.pop("skipped_tools_exhausted_final_response", None)
        else:
            info["skipped_tools_exhausted_final_response"] = True
            info.pop("skipped_turn_limit_final_response", None)
            info.pop("skipped_context_limit_final_response", None)
        return done_reason, info

    def _handle_model_error(
        self,
        *,
        runtime: ConversationRuntime,
        tools: list[dict[str, Any]] | None,
        step_result: ConversationStepResult,
        task_id: str,
        turn_idx: int,
        error: Exception,
    ) -> tuple[ConversationStepResult, float, bool, str | None, dict[str, Any]]:
        if isinstance(error, ModelContextLimitError):
            return self._handle_web_search_context_limit_error(
                runtime=runtime,
                step_result=step_result,
                task_id=task_id,
                turn_idx=turn_idx,
                error=error,
            )
        return super()._handle_model_error(
            runtime=runtime,
            tools=tools,
            step_result=step_result,
            task_id=task_id,
            turn_idx=turn_idx,
            error=error,
        )

    def _handle_web_search_context_limit_error(
        self,
        *,
        runtime: ConversationRuntime,
        step_result: ConversationStepResult,
        task_id: str,
        turn_idx: int,
        error: ModelContextLimitError,
    ) -> tuple[ConversationStepResult, float, bool, str | None, dict[str, Any]]:
        self._log_step(task_id, turn_idx, "orchestrator.context_limit_error", str(error), emoji="🧱")
        error_info = error.to_info()
        current_turn = int(getattr(runtime, "_assistant_turn_count", max(0, turn_idx - 1)) or max(0, turn_idx - 1))
        last_attempt_info = self._maybe_enter_discard_all_last_attempt(
            runtime=runtime,
            task_id=task_id,
            turn_count=current_turn,
            reason="max_tool_calls_reached_before_context_error",
        )
        discard_all_info = self._apply_discard_all_reset(
            runtime=runtime,
            task_id=task_id,
            turn_count=current_turn,
            reason="provider_context_limit_error",
        )
        if discard_all_info is not None:
            info = {
                "model_context_limit_error": error_info,
                "discarded_context_limit_error_with_discard_all": True,
                "discard_all": discard_all_info,
            }
            return (
                step_result.model_copy(
                    update={
                        "stage": runtime.stage,
                        "action": StepAction.CALL_MODEL,
                        "visible_conversation": runtime._build_visible_conversation(),
                        "full_conversation": list(runtime._full_conversation),
                        "appended_messages": [],
                        "info": info,
                    }
                ),
                0.0,
                False,
                None,
                info,
            )
        if not getattr(self, "_recover_generation_limits_this_attempt", False):
            info = {
                "model_context_limit_error": error_info,
                "skipped_context_limit_recovery_for_retry": True,
            }
            if last_attempt_info is not None:
                info["discard_all_last_attempt"] = last_attempt_info
            return step_result, 0.0, True, "terminated_context_limit", info

        recovered = self._recover_generation_limit_by_rollback(
            runtime=runtime,
            task_id=task_id,
            turn_idx=turn_idx,
            rollback_reason=RollbackReason.CONTEXT_LIMIT,
            extra_info={
                "model_context_limit_error": error_info,
                "discarded_context_limit_error_for_recovery": True,
            },
        )
        if recovered is not None:
            if last_attempt_info is not None:
                recovered[4]["discard_all_last_attempt"] = last_attempt_info
            return recovered

        info = {
            "model_context_limit_error": error_info,
            "context_limit_recovery_failed": True,
            "context_limit_recovery_failed_reason": "no_tool_exchange_to_rollback",
        }
        if last_attempt_info is not None:
            info["discard_all_last_attempt"] = last_attempt_info
        return step_result, 0.0, True, "terminated_context_limit", info

    async def _run_turn(  # noqa: PLR0915
        self,
        *,
        runtime: ConversationRuntime,
        tools: list[dict[str, Any]] | None,
        step_result: ConversationStepResult,
        task_id: str,
        turn_idx: int,
        attempt_budget_branch_checkpoint_factory: Callable[[], AttemptBudgetBranchCheckpoint] | None = None,
        attempt_budget_branch_result_sink: list[OrchestrationResult] | None = None,
    ) -> tuple[ConversationStepResult, float, bool, str | None, dict[str, Any]]:
        if step_result.stage == ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT and getattr(runtime, "skip_turn_limit_final_response", False):
            done_reason, info = self._normalize_skipped_budget_final_response_info(step_result)
            if done_reason is not None and info is not None:
                finalization_trigger = step_result.info.get("finalization_trigger")
                logger.info("Skipping %s final response on non-final web-search retry attempt", finalization_trigger)
                return step_result, 0.0, True, done_reason, info

        try:
            t0 = time.perf_counter()
            model_response = await self.model_client.acomplete(
                messages=self._model_visible_conversation(step_result.visible_conversation),
                tools=tools,
                tool_choice="auto" if tools else None,
            )
            model_elapsed_ms = (time.perf_counter() - t0) * 1000
        except (ValueError, RuntimeError) as exc:
            if isinstance(exc, ModelContextLimitError):
                branch_result = await self._maybe_run_attempt_budget_generation_limit_final_branch(
                    attempt_budget_branch_checkpoint_factory,
                )
                if branch_result is not None and attempt_budget_branch_result_sink is not None:
                    attempt_budget_branch_result_sink.append(branch_result)
            return self._handle_model_error(runtime=runtime, tools=tools, step_result=step_result, task_id=task_id, turn_idx=turn_idx, error=exc)

        model_extra: dict[str, Any] | None = None
        raw = getattr(model_response, "raw_response", None)
        if isinstance(raw, dict):
            logged_elapsed = raw.get("client_elapsed_ms_excluding_request_logging")
            if isinstance(logged_elapsed, int | float):
                model_elapsed_ms = float(logged_elapsed)
            model_extra = {key: raw[key] for key in ("e2e_elapsed_seconds", "cached_tokens", "retry", "stop_reason") if key in raw}
        inference_timing: dict[str, float] = {"model_client_elapsed_ms": model_elapsed_ms}
        if model_extra and "e2e_elapsed_seconds" in model_extra:
            inference_timing["engine_e2e_elapsed_s"] = float(model_extra["e2e_elapsed_seconds"])

        prompt_visible_conversation = step_result.visible_conversation
        if model_response.finish_reason == "length":
            branch_result = await self._maybe_run_attempt_budget_generation_limit_final_branch(
                attempt_budget_branch_checkpoint_factory,
            )
            if branch_result is not None and attempt_budget_branch_result_sink is not None:
                attempt_budget_branch_result_sink.append(branch_result)
        recovery_result = self._maybe_recover_length_generation(
            runtime=runtime,
            model_response=model_response,
            prompt_visible_conversation=prompt_visible_conversation,
            task_id=task_id,
            turn_idx=turn_idx,
            model_elapsed_ms=model_elapsed_ms,
            model_extra=model_extra,
            inference_timing=inference_timing,
        )
        if recovery_result is not None:
            return recovery_result

        if not model_response.message.content and not model_response.message.tool_calls and model_response.finish_reason != "length":
            logger.warning("No valid response from LLM, retrying")
            self._log_token_usage(task_id, turn_idx, model_response.usage, elapsed_ms=model_elapsed_ms, extra=model_extra)
            if model_response.usage is not None:
                runtime.record_observed_prompt_tokens(prompt_visible_conversation, input_tokens=model_response.usage.input_tokens)
                self._last_observed_prompt_tokens = model_response.usage.input_tokens
            info, reward = self._build_step_info(
                step_result=step_result,
                assistant_message=None,
                tool_outcomes=[],
                usage=model_response.usage,
                inference_timing=inference_timing,
            )
            return step_result, reward, False, None, info

        t0 = time.perf_counter()
        step_result = runtime.apply_assistant_message(model_response.message, finish_reason=model_response.finish_reason)
        runtime_elapsed_ms = (time.perf_counter() - t0) * 1000
        self._sync_trace(task_id, step_result.appended_messages)
        self._log_runtime_step(task_id, turn_idx, step_result, elapsed_ms=runtime_elapsed_ms)
        self._log_token_usage(task_id, turn_idx, model_response.usage, elapsed_ms=model_elapsed_ms, extra=model_extra)
        if model_response.usage is not None:
            runtime.record_observed_prompt_tokens(prompt_visible_conversation, input_tokens=model_response.usage.input_tokens)
            self._last_observed_prompt_tokens = model_response.usage.input_tokens

        for msg in step_result.appended_messages:
            if msg.role == MessageRole.ASSISTANT and msg.content:
                boxed = extract_boxed_content(msg.content)
                if boxed:
                    self._intermediate_boxed_answers.append(boxed)

        if step_result.action == StepAction.EXECUTE_TOOLS:
            return await self._run_tool_phase(
                runtime=runtime,
                step_result=step_result,
                task_id=task_id,
                turn_idx=turn_idx,
                assistant_message=model_response.message,
                usage=model_response.usage,
                inference_timing=inference_timing,
            )

        if step_result.action == StepAction.DONE:
            done_reason = self._done_reason_for_stage(step_result.stage, info=dict(step_result.info))
            info, reward = self._build_step_info(
                step_result=step_result,
                assistant_message=model_response.message,
                tool_outcomes=[],
                usage=model_response.usage,
                inference_timing=inference_timing,
            )
            return step_result, reward, True, done_reason, info

        info, reward = self._build_step_info(
            step_result=step_result,
            assistant_message=model_response.message,
            tool_outcomes=[],
            usage=model_response.usage,
            inference_timing=inference_timing,
        )
        return step_result, reward, False, None, info

    async def _maybe_run_attempt_budget_generation_limit_final_branch(
        self,
        checkpoint_factory: Callable[[], AttemptBudgetBranchCheckpoint] | None,
    ) -> OrchestrationResult | None:
        if checkpoint_factory is None:
            return None
        checkpoint = checkpoint_factory()
        return await self._run_attempt_budget_final_branch_from_checkpoint(checkpoint)

    def _maybe_recover_length_generation(
        self,
        *,
        runtime: ConversationRuntime,
        model_response: ModelResponse,
        prompt_visible_conversation: list[ConversationMessage],
        task_id: str,
        turn_idx: int,
        model_elapsed_ms: float,
        model_extra: dict[str, Any] | None,
        inference_timing: dict[str, float],
    ) -> tuple[ConversationStepResult, float, bool, str | None, dict[str, Any]] | None:
        if model_response.finish_reason != "length":
            return None
        if not getattr(self, "_recover_generation_limits_this_attempt", False):
            return self._terminate_length_generation_without_recovery(
                runtime=runtime,
                model_response=model_response,
                prompt_visible_conversation=prompt_visible_conversation,
                task_id=task_id,
                turn_idx=turn_idx,
                model_elapsed_ms=model_elapsed_ms,
                model_extra=model_extra,
                inference_timing=inference_timing,
                skipped_for_retry=True,
            )

        recovered = self._recover_generation_limit_by_rollback(
            runtime=runtime,
            task_id=task_id,
            turn_idx=turn_idx,
            rollback_reason=RollbackReason.TOKEN_EXHAUSTION,
            extra_info={"discarded_length_response_for_recovery": True},
            usage=model_response.usage,
            model_elapsed_ms=model_elapsed_ms,
            model_extra=model_extra,
            prompt_visible_conversation=prompt_visible_conversation,
            inference_timing=inference_timing,
        )
        if recovered is not None:
            return recovered

        return self._terminate_length_generation_without_recovery(
            runtime=runtime,
            model_response=model_response,
            prompt_visible_conversation=prompt_visible_conversation,
            task_id=task_id,
            turn_idx=turn_idx,
            model_elapsed_ms=model_elapsed_ms,
            model_extra=model_extra,
            inference_timing=inference_timing,
            skipped_for_retry=False,
        )

    def _recover_generation_limit_by_rollback(
        self,
        *,
        runtime: ConversationRuntime,
        task_id: str,
        turn_idx: int,
        rollback_reason: RollbackReason,
        extra_info: dict[str, Any],
        usage: TokenUsage | None = None,
        model_elapsed_ms: float | None = None,
        model_extra: dict[str, Any] | None = None,
        prompt_visible_conversation: list[ConversationMessage] | None = None,
        inference_timing: dict[str, float] | None = None,
    ) -> tuple[ConversationStepResult, float, bool, str | None, dict[str, Any]] | None:
        t0 = time.perf_counter()
        restored_prompt_skip_flags = self._temporarily_enable_generation_limit_recovery_prompt(runtime)
        try:
            recovered = runtime.rollback_latest_tool_exchange_and_force_finalize(reason=rollback_reason)
        except Exception:
            for attr, value in restored_prompt_skip_flags:
                setattr(runtime, attr, value)
            raise
        runtime_elapsed_ms = (time.perf_counter() - t0) * 1000
        if recovered is None:
            for attr, value in restored_prompt_skip_flags:
                setattr(runtime, attr, value)
            return None

        self._sync_trace(task_id, recovered.appended_messages)
        self._log_runtime_step(task_id, turn_idx, recovered, elapsed_ms=runtime_elapsed_ms)
        if model_elapsed_ms is not None:
            self._log_token_usage(task_id, turn_idx, usage, elapsed_ms=model_elapsed_ms, extra=model_extra)
        if usage is not None and prompt_visible_conversation is not None:
            runtime.record_observed_prompt_tokens(prompt_visible_conversation, input_tokens=usage.input_tokens)
        info, reward = self._build_step_info(
            step_result=recovered,
            assistant_message=None,
            tool_outcomes=[],
            usage=usage,
            inference_timing=inference_timing,
        )
        info.update(extra_info)
        return recovered, reward, False, None, info

    @staticmethod
    def _temporarily_enable_generation_limit_recovery_prompt(runtime: ConversationRuntime) -> list[tuple[str, bool]]:
        restored: list[tuple[str, bool]] = []
        for attr in ("skip_prompted_final_response", "skip_turn_limit_final_response"):
            if hasattr(runtime, attr):
                old_value = bool(getattr(runtime, attr))
                restored.append((attr, old_value))
                setattr(runtime, attr, False)
        return restored

    def _terminate_length_generation_without_recovery(
        self,
        *,
        runtime: ConversationRuntime,
        model_response: ModelResponse,
        prompt_visible_conversation: list[ConversationMessage],
        task_id: str,
        turn_idx: int,
        model_elapsed_ms: float,
        model_extra: dict[str, Any] | None,
        inference_timing: dict[str, float],
        skipped_for_retry: bool,
    ) -> tuple[ConversationStepResult, float, bool, str | None, dict[str, Any]]:
        t0 = time.perf_counter()
        terminal = runtime.apply_assistant_message(ConversationMessage.assistant(""), finish_reason="length")
        runtime_elapsed_ms = (time.perf_counter() - t0) * 1000
        self._sync_trace(task_id, terminal.appended_messages)
        self._log_runtime_step(task_id, turn_idx, terminal, elapsed_ms=runtime_elapsed_ms)
        self._log_token_usage(task_id, turn_idx, model_response.usage, elapsed_ms=model_elapsed_ms, extra=model_extra)
        if model_response.usage is not None:
            runtime.record_observed_prompt_tokens(prompt_visible_conversation, input_tokens=model_response.usage.input_tokens)
        info, reward = self._build_step_info(
            step_result=terminal,
            assistant_message=None,
            tool_outcomes=[],
            usage=model_response.usage,
            inference_timing=inference_timing,
        )
        info["discarded_length_response_without_recovery"] = True
        if skipped_for_retry:
            info["skipped_length_recovery_for_retry"] = True
        return terminal, reward, True, "terminated_token_exceed", info

    async def _run_tool_phase(
        self,
        *,
        runtime: ConversationRuntime,
        step_result: ConversationStepResult,
        task_id: str,
        turn_idx: int,
        assistant_message: ConversationMessage,
        usage: TokenUsage | None = None,
        inference_timing: dict[str, float] | None = None,
    ) -> tuple[ConversationStepResult, float, bool, str | None, dict[str, Any]]:
        tool_requests = self._build_tool_requests(assistant_message)
        self._record_rollback_storm_tool_requests(tool_requests)

        duplicate_requests = self._duplicate_tool_requests(tool_requests)
        if duplicate_requests:
            duplicate_rollback = self._apply_duplicate_query_rollback(
                runtime=runtime,
                task_id=task_id,
                turn_idx=turn_idx,
                duplicate_requests=duplicate_requests,
                inference_timing=inference_timing,
            )
            if duplicate_rollback is not None:
                return duplicate_rollback

        semantic_budget_result = self._maybe_terminate_for_semantic_query_budget(
            runtime=runtime,
            step_result=step_result,
            tool_requests=tool_requests,
            task_id=task_id,
            turn_idx=turn_idx,
        )
        if semantic_budget_result is not None:
            self._discard_intermediate_boxed_answer_from_message(assistant_message)
            return semantic_budget_result

        t0 = time.perf_counter()
        tool_outcomes = await self.tool_manager.execute_tool_calls(tool_requests, task_id=task_id)
        tool_elapsed_ms = (time.perf_counter() - t0) * 1000
        self._log_step(
            task_id,
            turn_idx,
            "tool.execution",
            f"processed {len(tool_outcomes)} tool call(s)",
            metadata={"tool_names": [outcome.request.tool_name for outcome in tool_outcomes]},
            emoji="⚒️",
            elapsed_ms=tool_elapsed_ms,
        )

        assistant_turn_count = int(step_result.info.get("assistant_turn_idx", turn_idx) or turn_idx)
        last_attempt_info = self._maybe_enter_discard_all_last_attempt(
            runtime=runtime,
            task_id=task_id,
            turn_count=assistant_turn_count,
            reason="max_tool_calls_reached",
        )

        if self._should_rollback_tool_results(tool_outcomes):
            tool_error_rollback = self._apply_tool_error_rollback(
                runtime=runtime,
                task_id=task_id,
                turn_idx=turn_idx,
                tool_outcomes=tool_outcomes,
                inference_timing=inference_timing,
            )
            if tool_error_rollback is not None:
                if last_attempt_info is not None:
                    tool_error_rollback[4]["discard_all_last_attempt"] = last_attempt_info
                return tool_error_rollback

        tool_messages = [outcome.message for outcome in tool_outcomes]
        rejected_call_ids = frozenset(outcome.request.call_id for outcome in tool_outcomes if outcome.result.status == ToolResultStatus.REJECTED)
        tools_exhausted = self.tool_manager.all_tools_exhausted()
        t0 = time.perf_counter()
        new_step_result = runtime.apply_batch_tool_messages(tool_messages, tools_exhausted=tools_exhausted, rejected_call_ids=rejected_call_ids)
        runtime_elapsed_ms = (time.perf_counter() - t0) * 1000
        self._sync_trace(task_id, new_step_result.appended_messages)
        self._log_runtime_step(task_id, turn_idx, new_step_result, elapsed_ms=runtime_elapsed_ms)

        info, reward = self._build_step_info(
            step_result=new_step_result,
            assistant_message=assistant_message,
            tool_outcomes=tool_outcomes,
            usage=usage,
            inference_timing=inference_timing,
        )
        if last_attempt_info is not None:
            info["discard_all_last_attempt"] = last_attempt_info
        runtime.reset_rollback_counter()

        for req in tool_requests:
            query_str = get_query_str_from_tool_call(req.tool_name, req.arguments)
            if query_str is not None:
                self._used_queries[query_str] = self._used_queries.get(query_str, 0) + 1
            semantic_query_key = get_semantic_budget_key_from_tool_call(req.tool_name, req.arguments)
            if semantic_query_key is not None:
                self._semantic_query_keys[semantic_query_key] = self._semantic_query_keys.get(semantic_query_key, 0) + 1

        return new_step_result, reward, False, None, info

    def _duplicate_tool_requests(self, tool_requests: list[Any]) -> list[Any]:
        duplicate_requests: list[Any] = []
        for req in tool_requests:
            query_str = get_query_str_from_tool_call(req.tool_name, req.arguments)
            if query_str is not None and self._used_queries.get(query_str, 0) > 0:
                logger.info("Duplicate query detected for %s: %s", req.tool_name, query_str[:100])
                duplicate_requests.append(req)
                break
        return duplicate_requests

    def _apply_duplicate_query_rollback(
        self,
        *,
        runtime: ConversationRuntime,
        task_id: str,
        turn_idx: int,
        duplicate_requests: list[Any],
        inference_timing: dict[str, float] | None,
    ) -> tuple[ConversationStepResult, float, bool, str | None, dict[str, Any]] | None:
        rollback_result = runtime.rollback_assistant_tool_calls(reason=RollbackReason.DUPLICATE_QUERY)
        if rollback_result is None:
            logger.info("Rollback limit reached, allowing duplicate query")
            return None
        self._sync_trace(task_id, rollback_result.appended_messages)
        self._log_runtime_step(task_id, turn_idx, rollback_result)
        self._record_rollback_storm_event(
            reason=RollbackReason.DUPLICATE_QUERY.value,
            turn_idx=turn_idx,
            tool_requests=duplicate_requests,
        )
        info: dict[str, Any] = dict(rollback_result.info)
        info["duplicate_query_rollback"] = True
        if inference_timing:
            info["inference_timing"] = dict(inference_timing)
        return rollback_result, 0.0, False, None, info

    def _apply_tool_error_rollback(
        self,
        *,
        runtime: ConversationRuntime,
        task_id: str,
        turn_idx: int,
        tool_outcomes: list[Any],
        inference_timing: dict[str, float] | None,
    ) -> tuple[ConversationStepResult, float, bool, str | None, dict[str, Any]] | None:
        rollback_result = runtime.rollback_assistant_tool_calls(reason=RollbackReason.TOOL_ERROR)
        if rollback_result is None:
            logger.info("Rollback limit reached, proceeding with tool error results")
            return None
        self._sync_trace(task_id, rollback_result.appended_messages)
        self._log_runtime_step(task_id, turn_idx, rollback_result)
        self._record_rollback_storm_event(
            reason=RollbackReason.TOOL_ERROR.value,
            turn_idx=turn_idx,
            tool_outcomes=tool_outcomes,
        )
        info: dict[str, Any] = dict(rollback_result.info)
        info["tool_error_rollback"] = True
        if inference_timing:
            info["inference_timing"] = dict(inference_timing)
        return rollback_result, 0.0, False, None, info

    def _maybe_terminate_for_semantic_query_budget(
        self,
        *,
        runtime: ConversationRuntime,
        step_result: ConversationStepResult,
        tool_requests: list[Any],
        task_id: str,
        turn_idx: int,
    ) -> tuple[ConversationStepResult, float, bool, str | None, dict[str, Any]] | None:
        semantic_budget = self._semantic_query_budget_exceeded(tool_requests)
        if semantic_budget is None:
            return None

        rollback_result = runtime.rollback_assistant_tool_calls(reason=_SEMANTIC_QUERY_BUDGET_REASON)
        rollback_applied = rollback_result is not None
        semantic_budget["rollback_applied"] = rollback_applied
        if rollback_result is not None:
            self._sync_trace(task_id, rollback_result.appended_messages)
            self._log_runtime_step(task_id, turn_idx, rollback_result)
            info: dict[str, Any] = dict(rollback_result.info)
            terminal_step_result = rollback_result
        else:
            info = {"rollback_reason": _SEMANTIC_QUERY_BUDGET_REASON}
            terminal_step_result = step_result.model_copy(
                update={"visible_conversation": self._without_latest_assistant_tool_call(step_result.visible_conversation)}
            )
        semantic_budget["action"] = "rollback_terminate" if rollback_applied else "terminate"
        info["semantic_query_budget"] = semantic_budget
        logger.info(
            "Semantic query budget exceeded (%d > %d); terminating attempt before tool execution",
            semantic_budget["candidate_unique_count"],
            semantic_budget["max_unique"],
        )
        return terminal_step_result, 0.0, True, "terminated_semantic_query_budget", info

    def _semantic_query_budget_exceeded(self, tool_requests: list[Any]) -> dict[str, Any] | None:
        if not self.semantic_query_budget_enabled:
            return None
        max_unique = self.semantic_query_budget_max_unique
        if max_unique is None or max_unique < 0:
            return None
        requested_keys = [
            key for key in (get_semantic_budget_key_from_tool_call(req.tool_name, req.arguments) for req in tool_requests) if key is not None
        ]
        if not requested_keys:
            return None
        existing_keys = set(self._semantic_query_keys)
        candidate_keys = existing_keys | set(requested_keys)
        if len(candidate_keys) <= max_unique:
            return None
        overflow_keys = sorted(set(requested_keys) - existing_keys)
        return {
            "enabled": True,
            "exceeded": True,
            "reason": _SEMANTIC_QUERY_BUDGET_REASON,
            "scope": "search_only",
            "max_unique": max_unique,
            "unique_count_before": len(existing_keys),
            "candidate_unique_count": len(candidate_keys),
            "requested_key_count": len(requested_keys),
            "overflow_keys": overflow_keys,
        }

    def _discard_intermediate_boxed_answer_from_message(self, message: ConversationMessage) -> None:
        if not message.content:
            return
        boxed = extract_boxed_content(message.content)
        if boxed and self._intermediate_boxed_answers and self._intermediate_boxed_answers[-1] == boxed:
            self._intermediate_boxed_answers.pop()

    @staticmethod
    def _without_latest_assistant_tool_call(conversation: list[ConversationMessage]) -> list[ConversationMessage]:
        visible = list(conversation)
        for idx in range(len(visible) - 1, -1, -1):
            message = visible[idx]
            if message.role == MessageRole.ASSISTANT and message.tool_calls:
                del visible[idx]
                break
        return visible

    def _update_run_info(self, run_info: dict[str, Any], step_info: dict[str, Any]) -> dict[str, Any]:
        run_info = super()._update_run_info(run_info, step_info)
        semantic_query_budget = step_info.get("semantic_query_budget")
        if isinstance(semantic_query_budget, dict):
            run_info["semantic_query_budget"] = dict(semantic_query_budget)
        for key in (
            "force_finalization_recovery",
            "context_limit_error_recovery",
            "discarded_context_limit_error_for_recovery",
            "token_exhaustion_recovery",
            "discarded_length_response_for_recovery",
            "discarded_length_response_without_recovery",
            "skipped_length_recovery_for_retry",
        ):
            if key in step_info:
                run_info[key] = step_info[key]
        return run_info

    def extract_final_output(self, conversation: list[ConversationMessage]) -> Any:
        """Extract boxed content from the latest assistant message.

        Intermediate boxed fallback is intentionally handled in ``run()`` so the
        policy can remain attempt-local and only apply when no retry remains.
        """
        for message in reversed(conversation):
            if message.role == MessageRole.ASSISTANT:
                if message.content:
                    boxed = extract_boxed_content(message.content)
                    if boxed:
                        return boxed
                return FORMAT_ERROR_MESSAGE
        return FORMAT_ERROR_MESSAGE

    def _latest_intermediate_boxed_answer(self) -> str:
        if self._intermediate_boxed_answers:
            return self._intermediate_boxed_answers[-1]
        return ""

    def _fallback_answer_from_compression_state(self) -> str:
        """Insight-driven no-answer rescue for empty finalization.

        When an attempt would finalize EMPTY, commit the best source-backed candidate
        from the context-compression state's answer_attempts. This is not a blind force:
        only a vetted candidate (has supporting evidence or a real decision, not
        wrong_target/rejected) is used; otherwise the output stays empty.
        """
        mgr = self._context_compression_manager
        if mgr is None:
            return ""
        summary = getattr(mgr, "_previous_summary", "") or ""
        if not summary or summary.strip() == "(none yet)":
            return ""
        import json
        import re as _re

        match = _re.search(r"\[context_summary\]\s*(\{.*\})\s*\[/context_summary\]", summary, _re.DOTALL)
        raw = match.group(1) if match else summary
        try:
            data = json.loads(raw)
        except Exception:
            return ""
        attempts = data.get("answer_attempts") if isinstance(data, dict) else None
        if not isinstance(attempts, list):
            return ""
        conf_rank = {"high": 3, "medium": 2, "low": 1}
        dec_rank = {"ready_if_verified": 3, "needs_verification": 2, "not_answered": 0, "rejected": -1}
        best = ""
        best_key = None
        for a in attempts:
            if not isinstance(a, dict):
                continue
            cand = str(a.get("candidate_answer") or "").strip()
            if not cand:
                continue
            if str(a.get("entity_match_status") or "").strip() == "wrong_target":
                continue
            decision = str(a.get("decision") or "").strip()
            if decision == "rejected":
                continue
            support = a.get("supporting_evidence")
            n_support = len(support) if isinstance(support, list) else 0
            if n_support == 0 and decision not in ("ready_if_verified", "needs_verification"):
                continue
            key = (dec_rank.get(decision, 0), conf_rank.get(str(a.get("confidence") or "").strip(), 0), n_support)
            if best_key is None or key > best_key:
                best_key = key
                best = cand
        return best

    @staticmethod
    def _boxed_answers_from_visible_conversation(conversation: list[ConversationMessage]) -> list[str]:
        answers: list[str] = []
        for message in conversation:
            if message.role == MessageRole.ASSISTANT and message.content:
                boxed = extract_boxed_content(message.content)
                if boxed:
                    answers.append(boxed)
        return answers

    @staticmethod
    def _copy_result_with_output(result: OrchestrationResult, output: str) -> OrchestrationResult:
        metadata = dict(result.metadata)
        metadata["output"] = output
        info = dict(result.info)
        info["fallback_output_used"] = True
        return OrchestrationResult(
            output=output,
            reason=result.reason,
            conversation=result.conversation,
            visible_conversation=result.visible_conversation,
            num_turns=result.num_turns,
            reward=result.reward,
            done=result.done,
            info=info,
            metadata=metadata,
        )

    @staticmethod
    def _copy_result_with_attempt_budget_metadata(
        result: OrchestrationResult,
        *,
        budget: int,
        actual_attempts: int,
        reused_from_budget: int | None,
    ) -> OrchestrationResult:
        info = dict(result.info)
        metadata = dict(result.metadata)
        budget_metadata = {
            "attempt_budget_sweep": True,
            "attempt_budget": budget,
            "attempt_budget_actual_attempts": actual_attempts,
            "attempt_budget_reused_from_budget": reused_from_budget,
        }
        info.update(budget_metadata)
        metadata.update(budget_metadata)
        provenance_value = info.get("retry_attempt_provenance", metadata.get("retry_attempt_provenance"))
        if isinstance(provenance_value, list):
            provenance = WebSearchTaskOrchestrator._normalize_attempt_budget_provenance(provenance_value, budget=budget)
            info["retry_attempt_provenance"] = provenance
            metadata["retry_attempt_provenance"] = provenance
        return result.model_copy(update={"info": info, "metadata": metadata})

    @staticmethod
    def _normalize_attempt_budget_provenance(provenance: list[Any], *, budget: int) -> list[Any]:
        normalized: list[Any] = []
        for item in provenance:
            if not isinstance(item, dict):
                normalized.append(item)
                continue
            entry = dict(item)
            entry["max_attempts"] = budget
            try:
                attempt = int(entry.get("attempt"))
            except (TypeError, ValueError):
                attempt = None
            if attempt is not None and attempt <= budget:
                entry["is_final"] = attempt == budget
            normalized.append(entry)
        return normalized

    def _copy_result_with_retry_attempt_provenance(
        self,
        result: OrchestrationResult,
        attempt_provenance: list[dict[str, Any]],
    ) -> OrchestrationResult:
        if not self.retry_attempt_provenance_enabled:
            return result
        provenance = [dict(item) for item in attempt_provenance]
        info = dict(result.info)
        metadata = dict(result.metadata)
        info["retry_attempt_provenance"] = provenance
        metadata["retry_attempt_provenance"] = provenance
        return result.model_copy(update={"info": info, "metadata": metadata})

    def _copy_result_with_rollback_storm_shadow(
        self,
        result: OrchestrationResult,
        *,
        shadow_metadata: dict[str, Any],
    ) -> OrchestrationResult:
        info = dict(result.info)
        metadata = dict(result.metadata)
        info["rollback_storm_shadow"] = dict(shadow_metadata)
        metadata["rollback_storm_shadow"] = dict(shadow_metadata)
        return result.model_copy(update={"info": info, "metadata": metadata})

    def _rollback_storm_shadow_metadata(self, *, result: OrchestrationResult, output_status: str) -> dict[str, Any]:
        duplicate_events = [event for event in self._rollback_storm_events if event.get("reason") == RollbackReason.DUPLICATE_QUERY.value]
        tool_error_events = [event for event in self._rollback_storm_events if event.get("reason") == RollbackReason.TOOL_ERROR.value]
        duplicate_turns = [int(event["turn"]) for event in duplicate_events if isinstance(event.get("turn"), int)]
        tool_error_turns = [int(event["turn"]) for event in tool_error_events if isinstance(event.get("turn"), int)]
        duplicate_count = len(duplicate_events)
        tool_error_count = len(tool_error_events)
        total_count = duplicate_count + tool_error_count
        last_rollback_turn = max([*duplicate_turns, *tool_error_turns], default=None)
        late_duplicate = (
            duplicate_count >= self.rollback_storm_duplicate_threshold
            and bool(duplicate_turns)
            and max(duplicate_turns) >= self.rollback_storm_late_turn_threshold
        )
        late_tool_error = (
            tool_error_count >= self.rollback_storm_tool_error_threshold
            and bool(tool_error_turns)
            and max(tool_error_turns) >= self.rollback_storm_late_turn_threshold
        )
        late_mixed = (
            total_count >= self.rollback_storm_duplicate_threshold
            and last_rollback_turn is not None
            and last_rollback_turn >= self.rollback_storm_late_turn_threshold
        )
        candidate_types: list[str] = []
        if late_duplicate:
            candidate_types.append("late_duplicate_query")
        if late_tool_error:
            candidate_types.append("late_tool_error")
        if late_mixed and len(candidate_types) != 1:
            candidate_types.append("late_mixed_rollback")
        output = str(result.output or "")
        return {
            "enabled": True,
            "schema_version": 1,
            "rollback_counter_source": "runtime_step_metadata",
            "duplicate_query_rollback_count": duplicate_count,
            "tool_error_rollback_count": tool_error_count,
            "total_rollback_count": total_count,
            "first_duplicate_query_rollback_turn": min(duplicate_turns, default=None),
            "last_duplicate_query_rollback_turn": max(duplicate_turns, default=None),
            "first_tool_error_rollback_turn": min(tool_error_turns, default=None),
            "last_tool_error_rollback_turn": max(tool_error_turns, default=None),
            "duplicate_query_threshold": self.rollback_storm_duplicate_threshold,
            "tool_error_threshold": self.rollback_storm_tool_error_threshold,
            "late_turn_threshold": self.rollback_storm_late_turn_threshold,
            "preview_max_items": self.rollback_storm_preview_max_items,
            "late_duplicate_query_storm_shadow": late_duplicate,
            "late_tool_error_storm_shadow": late_tool_error,
            "late_mixed_rollback_storm_shadow": late_mixed,
            "shadow_candidate_type": candidate_types[0] if len(candidate_types) == 1 else ("multiple" if candidate_types else "none"),
            "shadow_candidate_types": candidate_types,
            "terminal_reason": result.reason,
            "output_status": output_status,
            "has_valid_output": bool(output and output != FORMAT_ERROR_MESSAGE),
            "turn_count": result.num_turns,
            "tool_call_count": self._rollback_storm_tool_call_count,
            "search_call_count": self._rollback_storm_search_call_count,
            "scrape_call_count": self._rollback_storm_scrape_call_count,
            "duplicate_query_preview": self._rollback_storm_flattened_preview(duplicate_events),
            "tool_error_preview": self._rollback_storm_flattened_preview(tool_error_events),
        }

    def _rollback_storm_flattened_preview(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        limit = max(0, self.rollback_storm_preview_max_items)
        preview: list[dict[str, Any]] = []
        for event in events:
            turn = event.get("turn")
            for item in event.get("preview") or []:
                if not isinstance(item, dict):
                    continue
                entry = dict(item)
                entry["turn"] = turn
                preview.append(entry)
                if len(preview) >= limit:
                    return preview
        return preview

    def _write_rollback_storm_shadow_trace_metadata(
        self,
        *,
        attempt_id: str | None,
        result: OrchestrationResult,
        shadow_metadata: dict[str, Any],
    ) -> None:
        if self.task_logger is None or attempt_id is None:
            return
        updated = self.task_logger.update_finished_trace_metadata(
            attempt_id,
            tool_path=[self.config.name],
            metadata={"rollback_storm_shadow": dict(shadow_metadata)},
            step_name="rollback_storm.shadow",
            step_message="recorded default-off rollback-storm shadow telemetry",
            step_metadata={"rollback_storm_shadow": dict(shadow_metadata)},
            turn_idx=result.num_turns,
        )
        if not updated:
            logger.warning("Failed to update rollback-storm shadow metadata for trace %s", attempt_id)

    def _copy_result_with_no_box_turn_limit_cap_metadata(
        self,
        result: OrchestrationResult,
        *,
        cap_metadata: dict[str, Any],
    ) -> OrchestrationResult:
        info = dict(result.info)
        metadata = dict(result.metadata)
        info.update(cap_metadata)
        metadata.update(cap_metadata)
        return result.model_copy(update={"info": info, "metadata": metadata})

    def _no_box_turn_limit_cap_metadata(
        self,
        *,
        blocked_after_attempt: int,
        consecutive_no_box_turn_limit_attempts: int,
        saved_remaining_attempts_estimate: int,
    ) -> dict[str, Any]:
        return {
            "retry_blocked_by_no_box_turn_limit_cap": True,
            "retry_blocked_after_attempt": blocked_after_attempt,
            "retry_blocked_consecutive_no_box_turn_limit_attempts": consecutive_no_box_turn_limit_attempts,
            "retry_blocked_saved_remaining_attempts_estimate": saved_remaining_attempts_estimate,
            "retry_no_box_turn_limit_cap_enabled": self.retry_no_box_turn_limit_cap_enabled,
            "retry_no_box_turn_limit_cap": self.retry_no_box_turn_limit_cap,
        }

    def _write_no_box_turn_limit_cap_trace_metadata(
        self,
        *,
        attempt_id: str | None,
        result: OrchestrationResult,
        cap_metadata: dict[str, Any],
        attempt_provenance: list[dict[str, Any]],
    ) -> None:
        if self.task_logger is None or attempt_id is None:
            return
        trace_metadata = dict(cap_metadata)
        if self.retry_attempt_provenance_enabled:
            trace_metadata["retry_attempt_provenance"] = [dict(item) for item in attempt_provenance]
        updated = self.task_logger.update_finished_trace_metadata(
            attempt_id,
            tool_path=[self.config.name],
            metadata=trace_metadata,
            step_name="retry.no_box_turn_limit_cap",
            step_message="blocked remaining retries after repeated no-box turn-limit attempts",
            step_metadata=trace_metadata,
            turn_idx=result.num_turns,
        )
        if not updated:
            logger.warning("Failed to update no-box turn-limit cap metadata for trace %s", attempt_id)

    @staticmethod
    def _summarize_retry_attempt(
        *,
        result: OrchestrationResult,
        attempt: int,
        max_attempts: int,
        attempt_id: str | None,
        is_final: bool,
        output: str | None = None,
        output_status: str | None = None,
        previous_terminal_reason: str | None = None,
        consecutive_no_box_turn_limit_attempts: int = 0,
        retry_decision: str | None = None,
        retry_decision_reason: str | None = None,
        next_attempt_launched: bool | None = None,
        next_attempt_launch_reason: str | None = None,
        next_attempt_blocked_reason: str | None = None,
        would_block_by_no_box_turn_limit_cap: bool = False,
    ) -> dict[str, Any]:
        output = str(result.output or "") if output is None else output
        info = result.info if isinstance(result.info, dict) else {}
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        summary: dict[str, Any] = {
            "attempt": attempt,
            "attempt_id": attempt_id,
            "max_attempts": max_attempts,
            "is_final": is_final,
            "reason": result.reason,
            "num_turns": result.num_turns,
            "turn_used": metadata.get("turn_used", result.num_turns),
            "total_attempts": metadata.get("total_attempts"),
            "output_present": bool(output and output != FORMAT_ERROR_MESSAGE),
            "output_is_format_error": output == FORMAT_ERROR_MESSAGE,
            "output_preview": output[:200],
            "output_status": output_status or WebSearchTaskOrchestrator._retry_output_status(output),
            "previous_terminal_reason": previous_terminal_reason,
            "consecutive_no_box_turn_limit_attempts": consecutive_no_box_turn_limit_attempts,
            "retry_decision": retry_decision,
            "retry_decision_reason": retry_decision_reason,
            "next_attempt_launched": next_attempt_launched,
            "next_attempt_launch_reason": next_attempt_launch_reason,
            "next_attempt_blocked_reason": next_attempt_blocked_reason,
            "would_block_by_no_box_turn_limit_cap": would_block_by_no_box_turn_limit_cap,
            "retry_after_turn_limit": result.reason == "terminated_turn_limit" and not is_final,
        }
        for key in ("loop_accounting", "rollback_stats", "semantic_query_budget", "rollback_storm_shadow"):
            value = info.get(key)
            if isinstance(value, dict):
                summary[key] = dict(value)
        return summary

    @staticmethod
    def _should_rollback_tool_results(tool_outcomes: list[Any]) -> bool:
        for outcome in tool_outcomes:
            content = outcome.message.content or ""
            if content.startswith(("Unknown tool:", "Error executing tool")):
                return True
            if outcome.result.status == ToolResultStatus.FAILED:
                return True
            if outcome.request.tool_name in {"google_search", "web_search"} and _is_empty_search(content):
                return True
        return False

    async def _generate_failure_summary(self, result: OrchestrationResult) -> str:
        messages = self._model_visible_conversation(list(result.visible_conversation or result.conversation))
        if messages and messages[-1].role == MessageRole.USER:
            messages = messages[:-1]
        messages.append(ConversationMessage.user(FAILURE_SUMMARY_PROMPT))
        try:
            model_response = await self.model_client.acomplete(messages)
            failure_text = model_response.message.content or ""
            if not failure_text.startswith("Failure type:"):
                failure_text = FAILURE_SUMMARY_ASSISTANT_PREFIX + failure_text
            return failure_text.strip()
        except Exception:
            logger.exception("Failed to generate failure summary")
            return "Failure type: incomplete\nWhat happened: Unable to generate failure summary.\nUseful findings: None."

    async def _maybe_apply_self_verification(
        self,
        *,
        result: OrchestrationResult,
        canonical_task_id: str | None,
        original_task: str,
    ) -> OrchestrationResult:
        if not self.self_verification_enabled:
            return result
        initial_answer = str(result.output or "")
        if not initial_answer or initial_answer == FORMAT_ERROR_MESSAGE:
            return result

        canonical_task_id = canonical_task_id or str(result.metadata.get("task_id") or "")
        settings = self._self_verification_settings()
        current_answer = initial_answer
        current_answer_source = "initial_answer"
        verification_records: list[dict[str, Any]] = []
        reanswer_records: list[dict[str, Any]] = []
        auxiliary_trace_refs: list[dict[str, Any]] = []
        reanswer_attempts_used = 0
        final_verdict = "not_run"
        final_answer_source = current_answer_source
        verification_passed = False

        while True:
            if reanswer_attempts_used > 0 and reanswer_attempts_used >= self.self_verification_max_reanswer_attempts:
                final_verdict = "not_run_final_reanswer"
                final_answer_source = f"reanswer_{reanswer_attempts_used}_unverified_final"
                break

            verification = await self._run_self_verification_round(
                original_task=original_task,
                candidate_answer=current_answer,
                candidate_source=current_answer_source,
                canonical_task_id=canonical_task_id,
                candidate_index=len(verification_records) + 1,
            )
            verification_records.append(verification["record"])
            auxiliary_trace_refs.append(verification["trace_ref"])
            verdict = verification["verdict"]
            final_verdict = verdict.verdict
            if verdict.verdict == "correct":
                verification_passed = True
                final_answer_source = f"{current_answer_source}_verified"
                break

            if reanswer_attempts_used >= self.self_verification_max_reanswer_attempts:
                final_answer_source = f"{current_answer_source}_verification_failed_no_reanswer_budget"
                break

            reanswer_attempts_used += 1
            reanswer = await self._run_self_verification_reanswer_attempt(
                original_task=original_task,
                canonical_task_id=canonical_task_id,
                reanswer_attempt=reanswer_attempts_used,
            )
            reanswer_records.append(reanswer["record"])
            auxiliary_trace_refs.append(reanswer["trace_ref"])
            reanswer_output = str(reanswer["result"].output or "")
            if reanswer_output and reanswer_output != FORMAT_ERROR_MESSAGE:
                current_answer = reanswer_output
                current_answer_source = f"reanswer_{reanswer_attempts_used}"
            elif reanswer_attempts_used >= self.self_verification_max_reanswer_attempts:
                final_answer_source = f"reanswer_{reanswer_attempts_used}_failed_kept_{current_answer_source}"
                break

        self_verification_metadata = {
            "schema_version": 1,
            "enabled": True,
            "settings": settings,
            "initial_answer": initial_answer,
            "final_answer": current_answer,
            "final_verdict": final_verdict,
            "verification_passed": verification_passed,
            "final_answer_source": final_answer_source,
            "reanswer_attempts_used": reanswer_attempts_used,
            "verification_attempts": verification_records,
            "reanswer_attempts": reanswer_records,
            "auxiliary_trace_refs": auxiliary_trace_refs,
        }
        verified_result = self._copy_result_with_self_verification(
            result,
            output=current_answer,
            self_verification_metadata=self_verification_metadata,
        )
        self._write_self_verification_metadata_to_trace(
            canonical_task_id=canonical_task_id,
            result=verified_result,
            self_verification_metadata=self_verification_metadata,
        )
        return verified_result

    async def _run_self_verification_round(
        self,
        *,
        original_task: str,
        candidate_answer: str,
        candidate_source: str,
        canonical_task_id: str | None,
        candidate_index: int,
    ) -> dict[str, Any]:
        task = self._build_self_verification_task(original_task=original_task, candidate_answer=candidate_answer)
        task_id = self._auxiliary_task_id(canonical_task_id, f"sv-{candidate_index}")
        metadata = {
            "self_verification_auxiliary": True,
            "self_verification_role": "verifier",
            "candidate_source": candidate_source,
            "candidate_index": candidate_index,
            "settings": self._self_verification_settings(),
        }
        result = await self._run_self_verification_auxiliary_attempt(
            task=task,
            task_id=task_id,
            current_task_for_summary=task,
            extra_trace_metadata=metadata,
            runtime_config_overrides={
                "system_prompt": _SELF_VERIFICATION_SYSTEM_PROMPT,
                "user_prompt_template": "{task}",
                "early_stop_announcement_prompt": None,
                "final_response_prompt": _SELF_VERIFICATION_VERDICT_PROMPT,
                "max_turns": self.self_verification_max_turns or self.config.conversation.max_turns,
            },
        )
        raw_content = self._last_assistant_content(result.visible_conversation or result.conversation)
        verdict = self._parse_self_verification_verdict(raw_content)
        if verdict.parse_error:
            verdict = await self._resample_self_verification_verdict(
                result=result,
                initial_verdict=verdict,
            )
        record = {
            "task_id": task_id,
            "candidate_source": candidate_source,
            "candidate_index": candidate_index,
            "candidate_answer_preview": candidate_answer[:200],
            "verdict": verdict.verdict,
            "parse_error": verdict.parse_error,
            "resample_attempts": verdict.resample_attempts,
            "raw_content_preview": verdict.raw_content[:500],
            "parsed": verdict.parsed,
            "reason": result.reason,
            "num_turns": result.num_turns,
        }
        return {
            "result": result,
            "verdict": verdict,
            "record": record,
            "trace_ref": self._auxiliary_trace_ref(task_id=task_id, role="verifier"),
        }

    async def _run_self_verification_reanswer_attempt(
        self,
        *,
        original_task: str,
        canonical_task_id: str | None,
        reanswer_attempt: int,
    ) -> dict[str, Any]:
        task = self._build_self_verification_reanswer_task(original_task=original_task)
        task_id = self._auxiliary_task_id(canonical_task_id, f"sv-reanswer-{reanswer_attempt}")
        metadata = {
            "self_verification_auxiliary": True,
            "self_verification_role": "reanswer",
            "reanswer_attempt": reanswer_attempt,
            "settings": self._self_verification_settings(),
        }
        result = await self._run_self_verification_auxiliary_attempt(
            task=task,
            task_id=task_id,
            current_task_for_summary=original_task,
            extra_trace_metadata=metadata,
            runtime_config_overrides=None,
        )
        output = str(result.output or "")
        record = {
            "task_id": task_id,
            "reanswer_attempt": reanswer_attempt,
            "output_status": self._retry_output_status(output),
            "output_preview": output[:200],
            "reason": result.reason,
            "num_turns": result.num_turns,
        }
        return {
            "result": result,
            "record": record,
            "trace_ref": self._auxiliary_trace_ref(task_id=task_id, role="reanswer"),
        }

    async def _run_self_verification_auxiliary_attempt(
        self,
        *,
        task: str,
        task_id: str,
        current_task_for_summary: str,
        extra_trace_metadata: dict[str, Any],
        runtime_config_overrides: dict[str, Any] | None,
    ) -> OrchestrationResult:
        saved_current_task = self._current_task
        saved_state = self._snapshot_attempt_mutable_state()
        try:
            self._current_task = current_task_for_summary
            await self.tool_manager.begin_task(task_id=task_id)
            try:
                raw_result = await self._run_single_attempt(
                    task,
                    task_id=task_id,
                    extra_trace_metadata=extra_trace_metadata,
                    runtime_config_overrides=runtime_config_overrides,
                    trace_tool_path_suffix=[_SELF_VERIFICATION_TRACE_DIR],
                )
            finally:
                await self.tool_manager.end_task(task_id=task_id)
            if isinstance(raw_result, tuple):
                return raw_result[0]
            return raw_result
        finally:
            self._current_task = saved_current_task
            self._restore_attempt_mutable_state(saved_state)

    async def _resample_self_verification_verdict(
        self,
        *,
        result: OrchestrationResult,
        initial_verdict: SelfVerificationVerdict,
    ) -> SelfVerificationVerdict:
        verdict = initial_verdict
        messages = self._model_visible_conversation(list(result.visible_conversation or result.conversation))
        if messages and messages[-1].role == MessageRole.ASSISTANT:
            messages = messages[:-1]
        messages.append(ConversationMessage.user(_SELF_VERIFICATION_RESAMPLE_PROMPT))
        for resample_idx in range(1, self.self_verification_verdict_resample_max_attempts + 1):
            try:
                model_response = await self.model_client.acomplete(messages, tools=None, tool_choice=None)
            except Exception as exc:
                verdict = SelfVerificationVerdict(
                    verdict="unparseable",
                    raw_content="",
                    parsed=None,
                    parse_error=f"{type(exc).__name__}: {exc}",
                    resample_attempts=resample_idx,
                )
                continue
            content = model_response.message.content or ""
            parsed = self._parse_self_verification_verdict(content)
            verdict = SelfVerificationVerdict(
                verdict=parsed.verdict,
                raw_content=parsed.raw_content,
                parsed=parsed.parsed,
                parse_error=parsed.parse_error,
                resample_attempts=resample_idx,
            )
            if verdict.parse_error is None:
                return verdict
        return verdict

    def _self_verification_settings(self) -> dict[str, object]:
        return {
            "enabled": bool(self.self_verification_enabled),
            "max_reanswer_attempts": self.self_verification_max_reanswer_attempts,
            "verification_max_turns": self.self_verification_max_turns,
            "verdict_resample_max_attempts": self.self_verification_verdict_resample_max_attempts,
        }

    @staticmethod
    def _build_self_verification_task(*, original_task: str, candidate_answer: str) -> str:
        return (
            "Verify whether the candidate answer is correct for the original question.\n\n"
            "Original question:\n"
            f"{original_task}\n\n"
            "Candidate answer:\n"
            f"{candidate_answer}\n\n"
            "Break the question into required conditions and verify, one by one, whether the candidate "
            "answer satisfies each condition. Do not only search for support for the candidate; look "
            'for contradictions and alternative answers. Use the verdict "incorrect" only if a '
            "required condition is clearly not satisfied, clearly contradicted by reliable evidence, "
            "or the candidate answers a different entity/value. If a detail is hard to find or remains "
            "uncertain but there is no obvious mismatch, do not use that uncertainty alone to reject "
            "the candidate.\n\n"
            "Use tools if needed to verify the answer. When finished, output exactly one JSON object with this schema:\n"
            '{"rationale":"...","verdict":"correct"|"incorrect"}'
        )

    @staticmethod
    def _build_self_verification_reanswer_task(*, original_task: str) -> str:
        return original_task

    @staticmethod
    def _parse_self_verification_verdict(raw_content: str) -> SelfVerificationVerdict:
        text = (raw_content or "").strip()
        if not text:
            return SelfVerificationVerdict(verdict="unparseable", raw_content=raw_content, parsed=None, parse_error="empty verdict")

        parsed, parse_error = WebSearchTaskOrchestrator._load_self_verification_json_verdict(text)
        if parsed is None:
            parsed = WebSearchTaskOrchestrator._extract_self_verification_json_verdict(text)
        if parsed is None:
            loose = WebSearchTaskOrchestrator._parse_loose_self_verification_verdict(text)
            if loose is not None:
                return SelfVerificationVerdict(verdict=loose["verdict"], raw_content=raw_content, parsed=loose, parse_error=None)
            return SelfVerificationVerdict(
                verdict="unparseable",
                raw_content=raw_content,
                parsed=None,
                parse_error=parse_error or "verdict JSON was not found",
            )
        verdict = str(parsed.get("verdict") or "").strip().lower()
        if verdict in {"correct", "incorrect"}:
            parsed["verdict"] = verdict
            return SelfVerificationVerdict(verdict=verdict, raw_content=raw_content, parsed=parsed, parse_error=None)
        return SelfVerificationVerdict(
            verdict="unparseable",
            raw_content=raw_content,
            parsed=parsed,
            parse_error="verdict must be 'correct' or 'incorrect'",
        )

    @staticmethod
    def _load_self_verification_json_verdict(text: str) -> tuple[dict[str, Any] | None, str | None]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, f"json_decode_error: {exc}"
        if not isinstance(parsed, dict):
            return None, "verdict JSON is not an object"
        return parsed, None

    @staticmethod
    def _extract_self_verification_json_verdict(text: str) -> dict[str, Any] | None:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                parsed, _end = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and str(parsed.get("verdict") or "").strip().lower() in {"correct", "incorrect"}:
                parsed["_parse_mode"] = "json_substring"
                return parsed
        return None

    @staticmethod
    def _parse_loose_self_verification_verdict(text: str) -> dict[str, str] | None:
        patterns = [
            (r"(?i)\bverdict\b\s*[:=\-]\s*[\"']?\s*(incorrect|correct)\b", "verdict_field"),
            (r"(?i)\bfinal\s+verdict\b\s*[:=\-]?\s*[\"']?\s*(incorrect|correct)\b", "final_verdict"),
            (r"(?i)\bjudg(?:e)?ment\b\s*[:=\-]\s*[\"']?\s*(incorrect|correct)\b", "judgment"),
            (r"(?i)\bcandidate\s+answer\s+(?:is|was)\s+(not\s+correct|incorrect|correct|wrong|right)\b", "candidate_statement"),
            (r"(?i)\banswer\s+(?:is|was)\s+(not\s+correct|incorrect|correct|wrong|right)\b", "answer_statement"),
        ]
        matches: list[tuple[str, str]] = []
        for pattern, mode in patterns:
            for match in re.finditer(pattern, text):
                value = match.group(1).lower()
                verdict = "incorrect" if value in {"not correct", "incorrect", "wrong"} else "correct"
                matches.append((verdict, mode))
        verdicts = {verdict for verdict, _mode in matches}
        if len(verdicts) == 1:
            verdict = matches[0][0]
            return {
                "verdict": verdict,
                "rationale": "Recovered verifier verdict from non-JSON output.",
                "_parse_mode": matches[0][1],
            }
        standalone = re.sub(r"^[`'\"\s]+|[`'\"\s.]+$", "", text).strip().lower()
        if standalone in {"correct", "incorrect"}:
            return {
                "verdict": standalone,
                "rationale": "Recovered verifier verdict from a standalone non-JSON token.",
                "_parse_mode": "standalone_token",
            }
        return None

    @staticmethod
    def _last_assistant_content(conversation: list[ConversationMessage]) -> str:
        for message in reversed(conversation):
            if message.role == MessageRole.ASSISTANT and message.content:
                return message.content
        return ""

    @staticmethod
    def _auxiliary_task_id(canonical_task_id: str | None, suffix: str) -> str:
        base = canonical_task_id or uuid4().hex
        return f"{base}_{suffix}"

    def _auxiliary_trace_ref(self, *, task_id: str, role: str) -> dict[str, Any]:
        path_parts = [self.config.name, *self.tool_path, _SELF_VERIFICATION_TRACE_DIR, f"{task_id}.json"]
        return {
            "role": role,
            "task_id": task_id,
            "trace_path": "/".join(path_parts),
        }

    @staticmethod
    def _copy_result_with_self_verification(
        result: OrchestrationResult,
        *,
        output: str,
        self_verification_metadata: dict[str, Any],
    ) -> OrchestrationResult:
        info = dict(result.info)
        metadata = dict(result.metadata)
        info["self_verification"] = self_verification_metadata
        metadata["self_verification"] = self_verification_metadata
        metadata["output"] = output
        return OrchestrationResult(
            output=output,
            reason=result.reason,
            conversation=result.conversation,
            visible_conversation=result.visible_conversation,
            num_turns=result.num_turns,
            reward=result.reward,
            done=result.done,
            info=info,
            metadata=metadata,
        )

    def _write_self_verification_metadata_to_trace(
        self,
        *,
        canonical_task_id: str | None,
        result: OrchestrationResult,
        self_verification_metadata: dict[str, Any],
    ) -> None:
        if self.task_logger is None or not canonical_task_id:
            return
        self.task_logger.update_finished_trace_metadata(
            canonical_task_id,
            metadata={
                "output": result.output,
                "self_verification": self_verification_metadata,
            },
            tool_path=[self.config.name, *self.tool_path],
            step_name="self_verification.result",
            step_message=f"final_answer_source={self_verification_metadata.get('final_answer_source')}",
            step_metadata=self_verification_metadata,
            turn_idx=result.num_turns,
        )

    async def _retry_final_answer(self, result: OrchestrationResult) -> OrchestrationResult | None:
        messages = self._model_visible_conversation(list(result.visible_conversation or result.conversation))
        for retry_idx in range(1, self.max_final_answer_attempts):
            if messages and messages[-1].role == MessageRole.ASSISTANT:
                messages.pop()
            try:
                model_response = await self.model_client.acomplete(messages)
                content = model_response.message.content or ""
            except Exception:
                logger.exception("Final-answer retry %d/%d failed", retry_idx + 1, self.max_final_answer_attempts)
                continue

            boxed = extract_boxed_content(content)
            if boxed:
                logger.info("Boxed answer found on final-answer retry %d/%d: %s", retry_idx + 1, self.max_final_answer_attempts, boxed[:100])
                return OrchestrationResult(
                    output=boxed,
                    reason=result.reason,
                    conversation=result.conversation,
                    visible_conversation=result.visible_conversation,
                    num_turns=result.num_turns,
                    reward=result.reward,
                    done=result.done,
                    info=result.info,
                    metadata=result.metadata,
                )
            logger.info("No boxed answer on final-answer retry %d/%d", retry_idx + 1, self.max_final_answer_attempts)
            messages.append(ConversationMessage.assistant(content))
        return None


def _is_empty_search(content: str) -> bool:
    try:
        data = json.loads(content) if isinstance(content, str) else content
        if not isinstance(data, dict):
            return False
        organic = data.get("organic", None)
        return isinstance(organic, list) and len(organic) == 0
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def _redact_rollback_storm_secret_params(text: str) -> str:
    text = _ROLLBACK_STORM_AUTHORIZATION_RE.sub(lambda match: f"{match.group(1)}<redacted>", text)
    text = _ROLLBACK_STORM_BEARER_RE.sub(lambda match: f"{match.group(1)}<redacted>", text)
    return _ROLLBACK_STORM_SECRET_PARAM_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", text)
