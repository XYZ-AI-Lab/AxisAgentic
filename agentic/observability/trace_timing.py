# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Reusable timing helpers for TaskLogger traces.

These utilities aggregate timing fields that the orchestrator already records per turn
(``model_client.inference`` step ``elapsed_ms`` and ``metadata.e2e_elapsed_seconds``)
and the wall-clock duration that ``TaskLogger.finish_task`` stores on the trace metadata.

They let recipes report per-task model-client latency, SGLang engine time,
and total wall-clock in a pipeline-agnostic way,
and let ``--resume`` paths recover timing from a previously persisted trace.

``TaskOrchestrator`` also exposes the same per-task totals directly on the in-memory
:class:`OrchestrationResult` - see :func:`extract_inference_timing_from_result` for the
in-memory read path that avoids a trace-file round trip in the hot loop of eval recipes.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic.observability.task_logger import TaskTrace

# Step name suffix used by ``TaskOrchestrator._log_token_usage``.
# The full ``step_name`` is prefixed with an emoji ("🧠 model_client.inference"),
# so we always match by suffix.
_MODEL_CLIENT_INFERENCE_SUFFIX = "model_client.inference"


def extract_inference_timing_from_result(info: dict[str, Any] | None) -> tuple[float | None, float | None]:
    """Return ``(sglang_e2e_elapsed_s, model_client_elapsed_s)`` from an in-memory ``result.info``.

    ``TaskOrchestrator._update_run_info`` accumulates per-turn timings into
    ``info["inference_timing"] = {"engine_e2e_elapsed_s_sum": ..., "model_client_elapsed_ms_sum": ..., "num_inference_calls": N}``
    so eval loops can read them without synchronously parsing trace JSON after every task.

    Returns ``(None, None)`` if the info block or the inference_timing section is missing.
    """
    if not info:
        return None, None
    timing = info.get("inference_timing")
    if not isinstance(timing, dict) or not timing.get("num_inference_calls"):
        return None, None
    sglang_s = timing.get("engine_e2e_elapsed_s_sum")
    client_ms = timing.get("model_client_elapsed_ms_sum")
    return (
        float(sglang_s) if isinstance(sglang_s, (int, float)) and sglang_s > 0 else None,
        float(client_ms) / 1000.0 if isinstance(client_ms, (int, float)) and client_ms > 0 else None,
    )


def extract_inference_timing(trace: TaskTrace | None) -> tuple[float | None, float | None]:
    """Return ``(sglang_e2e_elapsed_s, model_client_elapsed_s)`` summed across all inference steps in *trace*.

    ``model_client_elapsed_s`` is the controller wall-clock recorded by ``TaskOrchestrator._run_turn``
    around ``model_client.acomplete`` (queue + RPC + engine); stored as ``step.elapsed_ms``.

    ``sglang_e2e_elapsed_s`` prefers the engine-stamped ``metadata.e2e_elapsed_seconds`` that
    ``SGLangModelClient`` / ``RolloutWorkerModelClient`` thread through ``ModelResponse.raw_response``.
    This is engine generation wall-clock time
    (``e2e_elapsed_seconds = time.perf_counter() - start_time``), allowing
    consistent model-side latency comparisons across runs.
    When a step lacks engine timing, the controller wall-clock is used as a fallback —
    cross-pipeline comparisons should rely on engine-stamped values when present.

    Returns ``(None, None)`` if *trace* is ``None`` or no inference steps are recorded.
    """
    if trace is None:
        return None, None
    sglang_total = 0.0
    sglang_seen = False
    client_total_ms = 0.0
    client_seen = False
    for step in trace.steps:
        if not step.get("step_name", "").endswith(_MODEL_CLIENT_INFERENCE_SUFFIX):
            continue
        meta = step.get("metadata") or {}
        elapsed_ms = step.get("elapsed_ms")
        if isinstance(elapsed_ms, (int, float)):
            client_total_ms += float(elapsed_ms)
            client_seen = True
        e2e = meta.get("e2e_elapsed_seconds")
        if isinstance(e2e, (int, float)):
            sglang_total += float(e2e)
            sglang_seen = True
        elif isinstance(elapsed_ms, (int, float)):
            sglang_total += float(elapsed_ms) / 1000.0
            sglang_seen = True
    sglang_s = sglang_total if sglang_seen else None
    client_s = (client_total_ms / 1000.0) if client_seen else None
    return sglang_s, client_s


def recover_task_elapsed_s(trace: TaskTrace | None) -> float | None:
    """Recover per-task wall-clock from a persisted trace.

    Used by ``--resume`` paths where the original in-loop ``time.perf_counter`` measurement
    isn't replayable. ``TaskLogger.finish_task`` stores ``metadata.total_elapsed_ms`` (preferred);
    when missing we fall back to parsing the ISO-8601 ``started_at`` / ``ended_at`` timestamps.

    Returns ``None`` when the trace is missing or both sources are unreadable.
    """
    if trace is None:
        return None
    total_ms = (trace.metadata or {}).get("total_elapsed_ms")
    if isinstance(total_ms, (int, float)):
        return float(total_ms) / 1000.0
    if trace.started_at and trace.ended_at:
        try:
            start_dt = datetime.fromisoformat(trace.started_at)
            end_dt = datetime.fromisoformat(trace.ended_at)
        except ValueError:
            return None
        return (end_dt - start_dt).total_seconds()
    return None
