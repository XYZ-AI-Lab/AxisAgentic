# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Side-aware helpers for comparing arbitrary dashboard run formats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from recipe.dashboard.constants import LLM_JUDGE_CORRECT_THRESHOLD
from recipe.dashboard.extraction import _ori_per_step_tokens, _ori_tool_calls_paired
from recipe.dashboard.loading import _load_agentic_summary, _load_json_cached, _load_ori_summary, _ori_messages

if TYPE_CHECKING:
    from pathlib import Path

PipelineKind = Literal["original", "agentic"]


@dataclass(frozen=True)
class DashboardSide:
    label: str
    kind: PipelineKind
    run: Path | None
    index: dict[str, dict[int, str]]
    evals: dict[str, dict[str, Any]]
    llm_evals: dict[str, dict[str, Any]]
    exp_name: str = ""
    run_name: str = ""
    run_type: str = ""
    agentic_trace_dir: str = ""


def side_short_label(side: DashboardSide) -> str:
    return side.label.split()[0].rstrip(":")


def side_run_title(side: DashboardSide) -> str:
    return f"{side.label}: {side.exp_name or '-'} / {side.run_name or '-'}"


def load_side_summary(side: DashboardSide, path: str) -> dict[str, Any]:
    if side.kind == "original":
        summary = _load_ori_summary(path)
        return {
            "status": summary.get("status", "?"),
            "ground_truth": summary.get("ground_truth", ""),
            "answer": summary.get("final_boxed_answer", ""),
            "turns": summary.get("turns", 0),
            "input_tokens": summary.get("input_tokens", 0),
            "output_tokens": summary.get("output_tokens", 0),
            "total_tokens": summary.get("total_tokens", 0),
            "num_tool_calls": len(_ori_tool_calls_paired(_load_json_cached(path))),
        }
    return _load_agentic_summary(path)


def side_tool_calls(side: DashboardSide, data: dict[str, Any]) -> list[dict[str, Any]]:
    if side.kind == "original":
        return _ori_tool_calls_paired(data)
    return data.get("tool_calls", []) or []


def side_per_step_tokens(side: DashboardSide, data: dict[str, Any]) -> list[dict[str, int]]:
    if side.kind == "original":
        return _ori_per_step_tokens(data)
    return data.get("token_usage", {}).get("per_step", []) or []


def side_messages(side: DashboardSide, data: dict[str, Any]) -> list[dict[str, Any]]:
    if side.kind == "original":
        return _ori_messages(data)
    return data.get("conversation", []) or []


def side_answer_correct(evals: dict[str, Any] | None) -> bool | None:
    if not evals:
        return None
    if evals.get("score") is not None:
        return float(evals["score"]) > 0
    if evals.get("is_correct") is not None:
        return bool(evals["is_correct"])
    return None


def side_llm_correct(evals: dict[str, Any] | None) -> bool | None:
    if not evals or evals.get("llm_judge_score") is None:
        return None
    return float(evals["llm_judge_score"]) > LLM_JUDGE_CORRECT_THRESHOLD
