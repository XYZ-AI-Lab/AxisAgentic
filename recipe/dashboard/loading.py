# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Cached log and result loaders for the dashboard."""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import Any

import streamlit as st

_WIDESEARCH_RETRYABLE_SCORE_PREFIXES = (
    "orchestrator error:",
    "transport error:",
    "connection error:",
    "timeout error:",
    "request error:",
)


def _is_retryable_widesearch_score_sidecar(row: dict[str, Any]) -> bool:
    msg = str(row.get("msg") or "").strip().lower()
    return msg.startswith(_WIDESEARCH_RETRYABLE_SCORE_PREFIXES)


@st.cache_data(ttl=300, show_spinner=False)
def _load_json_cached(path: str) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
        return {"_load_error": f"{type(exc).__name__}: {exc}", "_path": path}


def _ori_message_history(data: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
    history = data.get("main_agent_message_history", {})
    if isinstance(history, (dict, list)):
        return history
    return {}


def _ori_messages(data: dict[str, Any]) -> list[dict[str, Any]]:
    history = _ori_message_history(data)
    if isinstance(history, list):
        return [m for m in history if isinstance(m, dict)]
    msgs = history.get("message_history", [])
    if isinstance(msgs, list):
        return [m for m in msgs if isinstance(m, dict)]
    return []


def _ori_system_prompt(data: dict[str, Any]) -> str:
    history = _ori_message_history(data)
    if isinstance(history, dict):
        return str(history.get("system_prompt", "") or "")
    return ""


def _agentic_system_prompt(run_dir: Path) -> str:
    """Load the rendered chat-template system block for an agentic run."""
    chat_template_render_path = run_dir / "chat_template_render.json"
    if chat_template_render_path.exists():
        artifact = _load_json_cached(str(chat_template_render_path))
        if "_load_error" not in artifact:
            rendered_text = artifact.get("extracted_system_block") or artifact.get("rendered_text")
            if isinstance(rendered_text, str) and rendered_text:
                return rendered_text

    chat_template_json_path = run_dir / "chat_template_system_prompt.json"
    if chat_template_json_path.exists():
        artifact = _load_json_cached(str(chat_template_json_path))
        if "_load_error" not in artifact:
            rendered_text = artifact.get("rendered_text")
            if isinstance(rendered_text, str) and rendered_text:
                return rendered_text

    path = run_dir / "system_prompt.txt"
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return ""
    return ""


def _agentic_system_prompt_source(run_dir: Path) -> str:
    source_path = run_dir / "chat_template_render.json"
    if not source_path.exists():
        source_path = run_dir / "chat_template_system_prompt.json"
    if source_path.exists():
        source = _load_json_cached(str(source_path))
        if "_load_error" not in source:
            template_path = source.get("system_prompt_render_template")
            resolved_template_path = source.get("resolved_system_prompt_render_template")
            endpoint_profile = source.get("endpoint_profile")
            status = source.get("status")
            extract_start = source.get("extract_start")
            extract_end = source.get("extract_end")
            artifact = source.get("artifact", source_path.name)
            parts = [f"`{artifact}`"]
            if status:
                parts.append(f"status: `{status}`")
            if endpoint_profile:
                parts.append(f"profile: `{endpoint_profile}`")
            if template_path:
                parts.append(f"system prompt render template: `{template_path}`")
            if resolved_template_path and resolved_template_path != template_path:
                parts.append(f"resolved: `{resolved_template_path}`")
            if extract_start and extract_end:
                parts.append(f"extract delimiters: `{extract_start}` ... `{extract_end}`")
            return "Loaded from " + " | ".join(parts)

    if (run_dir / "system_prompt.txt").exists():
        return "Loaded from `system_prompt.txt`"
    return ""


@st.cache_data(ttl=300, show_spinner=False)
def _load_ori_summary(path: str) -> dict[str, Any]:
    d = _load_json_cached(path)
    msgs = _ori_messages(d)
    turns = sum(1 for m in msgs if m.get("role") == "assistant")
    inp = out = 0
    for step in d.get("step_logs", []):
        if "Token Usage" in step.get("step_name", ""):
            m = re.search(r"Input:\s*(\d+).*Output:\s*(\d+)", step.get("message", ""))
            if m:
                inp, out = int(m.group(1)), int(m.group(2))
    return {
        "status": d.get("status", "?"),
        "ground_truth": d.get("ground_truth", ""),
        "final_boxed_answer": d.get("final_boxed_answer", ""),
        "final_judge_result": d.get("final_judge_result", ""),
        "turns": turns,
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": inp + out,
    }


# ---------------------------------------------------------------------------
# Step / timing helpers (new log format)
# ---------------------------------------------------------------------------

# Step category prefixes emitted by the orchestrator.  The step_name field
# may be prefixed with an emoji (e.g. "🧠 model_client.inference").
_STEP_CATEGORIES: dict[str, str] = {
    "model_client": "Model Client",
    "tool": "Tool Execution",
    "runtime": "Conversation Runtime",
    "orchestrator": "Orchestrator",
}


def _classify_step(step_name: str) -> str:
    """Map a step_name (possibly emoji-prefixed) to a human-readable category."""
    # Strip leading emoji / whitespace to get the bare name
    bare = re.sub(r"^[^\w]+", "", step_name)
    for prefix, label in _STEP_CATEGORIES.items():
        if bare.startswith(prefix):
            return label
    return "Other"


def _extract_step_timing(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate elapsed_ms from step metadata by category.

    Returns ``{"by_category": {cat: total_ms}, "total_ms": float, "steps": [...]}``.
    """
    by_cat: dict[str, float] = {}
    counts_by_cat: dict[str, int] = {}
    annotated: list[dict[str, Any]] = []
    for s in steps:
        cat = _classify_step(s.get("step_name", ""))
        elapsed = s.get("metadata", {}).get("elapsed_ms")
        if elapsed is not None:
            by_cat[cat] = by_cat.get(cat, 0) + elapsed
            counts_by_cat[cat] = counts_by_cat.get(cat, 0) + 1
        annotated.append(
            {
                "turn": s.get("turn_idx", 0),
                "step_name": s.get("step_name", ""),
                "category": cat,
                "message": s.get("message", ""),
                "elapsed_ms": elapsed,
                "timestamp": s.get("timestamp", ""),
            }
        )
    return {
        "by_category": by_cat,
        "counts_by_category": counts_by_cat,
        "total_ms": sum(by_cat.values()),
        "steps": annotated,
    }


@st.cache_data(ttl=300, show_spinner=False)
def _load_agentic_summary(path: str) -> dict[str, Any]:
    d = _load_json_cached(path)
    conv = d.get("conversation", [])
    # Effective turns: assistant messages minus rolled-back ones.
    # Each rollback adds a ConversationRuntime marker; the rolled-back assistant
    # message stays in full_conversation but should not count as an effective turn.
    # This aligns with the original pipeline where rolled-back messages are popped.
    total_assistant = sum(1 for m in conv if m.get("role") == "assistant")
    # Match by marker name, not content substring: compaction markers carry
    # free-form summary text that may mention "rollback". Fall back to the
    # substring heuristic only for legacy traces without a name field.
    rollbacks = sum(
        1
        for m in conv
        if m.get("role") == "ConversationRuntime"
        and (m.get("name") == "rollback" or (m.get("name") is None and "rollback" in str(m.get("content", "")).lower()))
    )
    effective_turns = total_assistant - rollbacks
    tu = d.get("token_usage", {})
    answer = ""
    for m in reversed(conv):
        if m.get("role") == "assistant" and m.get("content"):
            boxed = re.findall(r"\\boxed\{([^}]*)\}", m["content"])
            answer = boxed[-1] if boxed else m["content"][:80]
            break

    timing = _extract_step_timing(d.get("steps", []))

    return {
        "status": d.get("status", "?"),
        "answer": answer[:80],
        "turns": effective_turns,
        "input_tokens": tu.get("input_tokens", 0),
        "output_tokens": tu.get("output_tokens", 0),
        "total_tokens": tu.get("total_tokens", 0),
        "num_tool_calls": len(d.get("tool_calls", [])),
        "timing": timing,
    }


def _normalize_em_result(row: dict[str, Any], task_id: str) -> dict[str, Any]:
    normalized = dict(row)
    normalized["task_id"] = str(normalized.get("task_id", task_id))
    score = normalized.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        normalized["score"] = float(score)
        normalized.setdefault("is_correct", float(score) > 0)
        normalized.setdefault("em_result", "CORRECT" if float(score) > 0 else "INCORRECT")
    elif normalized.get("is_correct") is not None:
        normalized["score"] = 1.0 if bool(normalized["is_correct"]) else 0.0
        normalized.setdefault("em_result", "CORRECT" if normalized["score"] > 0 else "INCORRECT")
    return normalized


@st.cache_data(ttl=30, show_spinner=False)
def _load_em_results(run_dir: str) -> dict[str, dict[str, Any]]:
    run_path = Path(run_dir)
    results: dict[str, dict[str, Any]] = {}
    summary_path = run_path / "em_summary.json"
    if summary_path.exists():
        with contextlib.suppress(UnicodeDecodeError, json.JSONDecodeError, OSError):
            results["_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))

    jsonl_path = run_path / "em_results.jsonl"
    if jsonl_path.exists():
        try:
            lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            lines = []
        for line in lines:
            if not line.strip():
                continue
            with contextlib.suppress(json.JSONDecodeError):
                row = json.loads(line)
                tid = str(row.get("task_id", ""))
                if tid:
                    results[tid] = _normalize_em_result(row, tid)

    sidecar_dir = run_path / "exact_match"
    if sidecar_dir.exists():
        for path in sorted(sidecar_dir.glob("*.json")):
            with contextlib.suppress(UnicodeDecodeError, json.JSONDecodeError, OSError):
                row = json.loads(path.read_text(encoding="utf-8"))
                tid = str(row.get("task_id", path.stem))
                if tid and tid not in results:
                    results[tid] = _normalize_em_result(row, tid)
    return results


@st.cache_data(ttl=30, show_spinner=False)
def _load_llm_judge_results(run_dir: str) -> dict[str, dict[str, Any]]:
    run_path = Path(run_dir)
    results: dict[str, dict[str, Any]] = {}
    summary_path = run_path / "llm_judge_summary.json"
    if summary_path.exists():
        with contextlib.suppress(UnicodeDecodeError, json.JSONDecodeError, OSError):
            results["_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))

    jsonl_path = run_path / "llm_judge_results.jsonl"
    if jsonl_path.exists():
        try:
            lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        except (UnicodeDecodeError, OSError):
            lines = []
        for line in lines:
            if not line.strip():
                continue
            with contextlib.suppress(json.JSONDecodeError):
                row = json.loads(line)
                tid = str(row.get("task_id", ""))
                if tid:
                    results[tid] = row

    sidecar_dir = run_path / "llm_judge"
    if sidecar_dir.exists():
        for path in sorted(sidecar_dir.glob("*.json")):
            with contextlib.suppress(UnicodeDecodeError, json.JSONDecodeError, OSError):
                row = json.loads(path.read_text(encoding="utf-8"))
                tid = str(row.get("task_id", path.stem))
                if tid and tid not in results:
                    results[tid] = row
    return results


@st.cache_data(ttl=30, show_spinner=False)
def _load_widesearch_summary(run_dir: str) -> dict[str, Any]:
    """Read ``widesearch_summary.json`` written by the WideSearch evaluator."""
    path = Path(run_dir) / "widesearch_summary.json"
    if not path.exists():
        return {}
    with contextlib.suppress(UnicodeDecodeError, json.JSONDecodeError, OSError):
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@st.cache_data(ttl=30, show_spinner=False)
def _load_widesearch_per_task(run_dir: str) -> dict[str, Any]:
    """Read ``widesearch_per_task.json`` written by the WideSearch evaluator."""
    path = Path(run_dir) / "widesearch_per_task.json"
    if not path.exists():
        return {}
    with contextlib.suppress(UnicodeDecodeError, json.JSONDecodeError, OSError):
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@st.cache_data(ttl=30, show_spinner=False)
def _load_widesearch_trial_scores(run_dir: str) -> dict[str, dict[str, Any]]:
    """Read every per-trial sidecar under ``widesearch_scores/``.

    Keyed by the sidecar's ``task_id`` field (e.g. ``ws_en_001__trial-0``).
    """
    run_path = Path(run_dir)
    scores_dir = run_path / "widesearch_scores"
    results: dict[str, dict[str, Any]] = {}
    if not scores_dir.exists():
        return results
    for path in sorted(scores_dir.glob("*.json")):
        with contextlib.suppress(UnicodeDecodeError, json.JSONDecodeError, OSError):
            row = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(row, dict) or _is_retryable_widesearch_score_sidecar(row):
                continue
            tid = str(row.get("task_id", path.stem))
            if tid:
                results[tid] = row
    return results
