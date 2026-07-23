# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""WideSearch metrics tab for the benchmark dashboard."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st
import yaml

from recipe.dashboard.loading import (
    _load_widesearch_summary,
    _load_widesearch_trial_scores,
)
from recipe.web_search.agent.prompts import extract_boxed_content
from recipe.wide_search.eval.data_loader import load_gold_dataframe

if TYPE_CHECKING:
    from recipe.dashboard.sides import DashboardSide

_LEADERBOARD_LABELS: dict[str, str] = {
    "success_rate_avg@N": "Success rate avg@N",
    "success_rate_pass@N": "Success rate pass@N",
    "row_f1_avg@N": "Row F1 avg@N",
    "row_f1_max@N": "Row F1 max@N",
    "item_f1_avg@N": "Item F1 avg@N",
    "item_f1_max@N": "Item F1 max@N",
}
_AVG_LEADERBOARD_KEYS: tuple[str, ...] = ("success_rate_avg@N", "row_f1_avg@N", "item_f1_avg@N")
_MAX_LEADERBOARD_KEYS: tuple[str, ...] = ("success_rate_pass@N", "row_f1_max@N", "item_f1_max@N")

_METRIC_ORDER: tuple[str, ...] = (
    "score",
    "precision_by_row",
    "recall_by_row",
    "f1_by_row",
    "precision_by_item",
    "recall_by_item",
    "f1_by_item",
)
_MARKDOWN_BLOCK_RE = re.compile(r"```(?:markdown|md)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_PIPE_RUN_RE = re.compile(r"((?:^\s*\|.*(?:\n|$))+)", re.MULTILINE)
_TRIAL_SUFFIX_RE = re.compile(r"__trial-(?P<trial>\d+)$")
_TrialChoice = int | str


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt_score_compact(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "-"


def _instance_id_from_trial_key(key: str) -> str:
    base, sep, _ = key.partition("__trial-")
    return base if sep else key


def _load_jsonl_row(path: Path, instance_id: str) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return {}
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and str(row.get("instance_id")) == instance_id:
            return row
    return {}


def _load_evaluation_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _benchmark_paths_from_run(run_dir: Path) -> tuple[Path | None, Path | None]:
    for path in (run_dir / "run_config.effective.yaml", run_dir.parent / "run_config.effective.yaml"):
        config = _load_yaml_dict(path)
        benchmark = config.get("benchmark")
        if not isinstance(benchmark, dict):
            continue
        data_path = benchmark.get("data_path")
        gold_dir = benchmark.get("gold_dir")
        resolved_data = Path(str(data_path)).expanduser() if data_path else None
        resolved_gold = Path(str(gold_dir)).expanduser() if gold_dir else None
        return resolved_data, resolved_gold
    return None, None


def _markdown_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("|", "\\|").replace("\n", "<br>")


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    columns = [str(column) for column in df.columns]
    header = "| " + " | ".join(_markdown_cell(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(_markdown_cell(row[column]) for column in df.columns) + " |" for _, row in df.iterrows()]
    return "\n".join([header, separator, *rows])


@st.cache_data(ttl=300, show_spinner=False)
def _load_widesearch_task_reference(run_dir: str, instance_id: str) -> dict[str, str]:
    run_path = Path(run_dir)
    data_path, gold_dir = _benchmark_paths_from_run(run_path)
    if data_path is None or gold_dir is None:
        return {"question": "", "groundtruth": "", "warning": "No benchmark data_path/gold_dir found in run_config.effective.yaml."}
    if not data_path.exists() or not gold_dir.is_dir():
        return {"question": "", "groundtruth": "", "warning": f"Benchmark data/gold path is not readable: {data_path} | {gold_dir}"}

    row = _load_jsonl_row(data_path, instance_id)
    if not row:
        return {"question": "", "groundtruth": "", "warning": f"Task `{instance_id}` not found in {data_path}."}

    question = str(row.get("question") or row.get("query") or "")
    evaluation = _load_evaluation_payload(row.get("evaluation"))
    required = [str(column) for column in evaluation.get("required") or []]
    answer_df = load_gold_dataframe(gold_dir / f"{instance_id}.csv", required) if required else None
    groundtruth = _dataframe_to_markdown(answer_df) if answer_df is not None else ""
    warning = "" if groundtruth else f"Ground truth CSV is missing or incompatible for `{instance_id}`."
    return {"question": question, "groundtruth": groundtruth, "warning": warning}


def _trial_index_from_task_id(task_id: str) -> int | None:
    match = _TRIAL_SUFFIX_RE.search(task_id)
    return int(match.group("trial")) if match else None


def _trial_sort_key(task_id: str) -> tuple[int, str]:
    trial = _trial_index_from_task_id(task_id)
    return (trial if trial is not None else 10**9, task_id)


def _trial_scores_by_instance(trial_scores: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for task_id in trial_scores:
        grouped.setdefault(_instance_id_from_trial_key(task_id), []).append(task_id)
    return {instance_id: sorted(task_ids, key=_trial_sort_key) for instance_id, task_ids in sorted(grouped.items())}


def _trial_choice_from_task_id(task_id: str) -> _TrialChoice:
    trial = _trial_index_from_task_id(task_id)
    return trial if trial is not None else task_id


def _trial_choice_sort_key(choice: _TrialChoice) -> tuple[int, int, str]:
    if isinstance(choice, int):
        return (0, choice, "")
    return (1, 10**9, choice)


def _shared_task_options(
    left_grouped: dict[str, list[str]],
    right_grouped: dict[str, list[str]],
) -> list[str]:
    return sorted(set(left_grouped) | set(right_grouped))


def _shared_trial_options(
    left_grouped: dict[str, list[str]],
    right_grouped: dict[str, list[str]],
    instance_id: str,
) -> list[_TrialChoice]:
    choices = {_trial_choice_from_task_id(task_id) for grouped in (left_grouped, right_grouped) for task_id in grouped.get(instance_id, [])}
    return sorted(choices, key=_trial_choice_sort_key)


def _trial_task_id_for_choice(
    grouped: dict[str, list[str]],
    instance_id: str,
    choice: _TrialChoice,
) -> str | None:
    for task_id in grouped.get(instance_id, []):
        if _trial_choice_from_task_id(task_id) == choice:
            return task_id
    return None


def _per_task_table(per_task_payload: dict[str, Any]) -> pd.DataFrame:
    rows = per_task_payload.get("per_task")
    if not isinstance(rows, list):
        return pd.DataFrame()
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        metrics = row.get("metrics") or {}
        record: dict[str, Any] = {
            "instance_id": row.get("instance_id", ""),
            "num_trials": row.get("num_trials", 0),
        }
        for metric_key in _METRIC_ORDER:
            buckets = metrics.get(metric_key) or {}
            for bucket in ("avg_n", "max_n", "min_n"):
                record[f"{metric_key}.{bucket}"] = buckets.get(bucket)
        out.append(record)
    return pd.DataFrame(out)


def _trial_table(trial_scores: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task_id, payload in sorted(trial_scores.items()):
        rows.append(
            {
                "task_id": task_id,
                "instance_id": _instance_id_from_trial_key(task_id),
                "score": payload.get("score"),
                "f1_by_row": payload.get("f1_by_row"),
                "f1_by_item": payload.get("f1_by_item"),
                "precision_by_row": payload.get("precision_by_row"),
                "recall_by_row": payload.get("recall_by_row"),
                "precision_by_item": payload.get("precision_by_item"),
                "recall_by_item": payload.get("recall_by_item"),
                "msg": str(payload.get("msg") or "")[:120],
            }
        )
    return pd.DataFrame(rows)


def _trace_attempt_number(path: Path, trial_task_id: str) -> int | None:
    prefix = f"{trial_task_id}_attempt-"
    if not path.stem.startswith(prefix):
        return None
    suffix = path.stem[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


@st.cache_data(ttl=30, show_spinner=False)
def _load_widesearch_trial_trace(run_dir: str, trial_task_id: str) -> dict[str, Any]:
    trace_dir = Path(run_dir) / "wide-search"
    if not trace_dir.exists():
        return {}
    candidates: list[tuple[int, Path]] = []
    for path in trace_dir.glob("*.json"):
        attempt = _trace_attempt_number(path, trial_task_id)
        if attempt is not None:
            candidates.append((attempt, path))
    if not candidates:
        return {}
    _, path = max(candidates, key=lambda item: item[0])
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _stringify_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)


def _query_from_trace(trace: dict[str, Any]) -> str:
    task_input = trace.get("task_input")
    text = _stringify_content(task_input).strip()
    if text:
        return text
    conversation = trace.get("conversation")
    if not isinstance(conversation, list):
        return ""
    for message in conversation:
        if isinstance(message, dict) and message.get("role") == "user":
            text = _stringify_content(message.get("content")).strip()
            if text:
                return text
    return ""


def _last_assistant_content(trace: dict[str, Any]) -> str:
    conversation = trace.get("conversation")
    if not isinstance(conversation, list):
        return ""
    for message in reversed(conversation):
        if isinstance(message, dict) and message.get("role") == "assistant":
            text = _stringify_content(message.get("content")).strip()
            if text:
                return text
    return ""


def _looks_like_markdown_table(text: str) -> bool:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    pipe_lines = [line for line in lines if "|" in line]
    if len(pipe_lines) < 2:
        return False
    return any(re.fullmatch(r"\|?[\s:|\-]+\|?", line) for line in pipe_lines[1:])


def _markdown_table_from_response(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    boxed = extract_boxed_content(text)
    if boxed and _looks_like_markdown_table(boxed):
        return boxed.strip()

    for block in _MARKDOWN_BLOCK_RE.findall(text):
        if _looks_like_markdown_table(block):
            return block.strip()

    for block in _PIPE_RUN_RE.findall(text):
        if _looks_like_markdown_table(block):
            return block.strip()

    return ""


def _final_answer_candidates(trial_payload: dict[str, Any], trace: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    metadata = trace.get("metadata")
    if isinstance(metadata, dict):
        output = metadata.get("output")
        output_text = _stringify_content(output).strip()
        if output_text:
            candidates.append(output_text)
    assistant = _last_assistant_content(trace)
    if assistant:
        candidates.append(assistant)
    preview = trial_payload.get("raw_response_preview")
    if isinstance(preview, str) and preview.strip():
        candidates.append(preview)
    return candidates


def _final_answer_markdown(trial_payload: dict[str, Any], trace: dict[str, Any]) -> tuple[str, str]:
    raw_fallback = ""
    for candidate in _final_answer_candidates(trial_payload, trace):
        if not raw_fallback:
            raw_fallback = candidate
        table = _markdown_table_from_response(candidate)
        if table:
            return table, candidate
    return "", raw_fallback


def _render_leaderboard(label: str, summary_payload: dict[str, Any]) -> None:
    leaderboard = summary_payload.get("leaderboard") or {}
    st.markdown(f"#### {label}")
    if not leaderboard:
        st.info("No `widesearch_summary.json` found in this run.")
        return

    completed = summary_payload.get("num_complete_instances")
    incomplete = summary_payload.get("num_incomplete_instances")
    num_trials = summary_payload.get("num_trials")
    bits: list[str] = []
    if num_trials is not None:
        bits.append(f"trials={num_trials}")
    if completed is not None:
        bits.append(f"complete={completed}")
    if incomplete:
        bits.append(f"incomplete={incomplete}")
    if bits:
        st.caption(" | ".join(bits))

    cols_top = st.columns(3)
    for col, key in zip(cols_top, _AVG_LEADERBOARD_KEYS, strict=False):
        col.metric(_LEADERBOARD_LABELS[key], _fmt_pct(leaderboard.get(key)))
    cols_bot = st.columns(3)
    for col, key in zip(cols_bot, _MAX_LEADERBOARD_KEYS, strict=False):
        col.metric(_LEADERBOARD_LABELS[key], _fmt_pct(leaderboard.get(key)))


def _trial_choice_label(
    choice: _TrialChoice,
    selected_task: str,
    left_grouped: dict[str, list[str]],
    right_grouped: dict[str, list[str]],
    left_scores: dict[str, dict[str, Any]],
    right_scores: dict[str, dict[str, Any]],
) -> str:
    base = f"trial {choice}" if isinstance(choice, int) else str(choice)
    side_bits: list[str] = []
    for label, grouped, scores in (
        ("Left", left_grouped, left_scores),
        ("Right", right_grouped, right_scores),
    ):
        task_id = _trial_task_id_for_choice(grouped, selected_task, choice)
        if task_id is None:
            side_bits.append(f"{label}: -")
            continue
        payload = scores.get(task_id, {})
        side_bits.append(f"{label}: item F1 {_fmt_score_compact(payload.get('f1_by_item'))}")
    return " | ".join([base, *side_bits])


def _render_side_metrics(side: DashboardSide, summary_payload: dict[str, Any]) -> None:
    if side.run is None:
        st.info(f"{side.label}: no run selected.")
        return

    title = f"{side.label}: {side.exp_name or '-'} / {side.run_name or '-'}"
    _render_leaderboard(title, summary_payload)


def _load_side_reference(side: DashboardSide, instance_id: str) -> dict[str, str]:
    if side.run is None:
        return {"question": "", "groundtruth": "", "warning": ""}
    return _load_widesearch_task_reference(str(side.run), instance_id)


def _first_reference_value(references: list[dict[str, str]], key: str) -> str:
    for reference in references:
        value = reference.get(key, "")
        if value:
            return value
    return ""


def _render_reference(
    left: DashboardSide,
    right: DashboardSide,
    selected_task: str,
    left_trace: dict[str, Any],
    right_trace: dict[str, Any],
) -> None:
    references = [
        _load_side_reference(left, selected_task),
        _load_side_reference(right, selected_task),
    ]
    question = _first_reference_value(references, "question") or _query_from_trace(left_trace) or _query_from_trace(right_trace)
    groundtruth = _first_reference_value(references, "groundtruth")
    reference_warning = _first_reference_value(references, "warning")

    st.markdown("##### Question")
    if question:
        st.text_area(
            "Question",
            question,
            height=140,
            disabled=True,
            label_visibility="collapsed",
            key=f"widesearch-question-{left.run}-{right.run}-{selected_task}",
        )
    else:
        st.info("No question found for this task.")

    st.markdown("##### Ground Truth")
    if groundtruth:
        st.markdown(groundtruth)
    elif reference_warning:
        st.info(reference_warning)
    else:
        st.info("No ground truth found for this task.")


def _load_selected_trace(side: DashboardSide, selected_trial: str | None) -> dict[str, Any]:
    if side.run is None or selected_trial is None:
        return {}
    return _load_widesearch_trial_trace(str(side.run), selected_trial)


def _render_final_answer(
    label: str,
    side: DashboardSide,
    trial_scores: dict[str, dict[str, Any]],
    selected_trial: str | None,
    trace: dict[str, Any],
) -> None:
    st.markdown(f"##### {label} Final Answer")
    if side.run is None:
        st.info(f"{label}: no run selected.")
        return
    if selected_trial is None:
        st.info(f"{label}: no matching trial found for this task/trial selection.")
        return

    trial_payload = trial_scores.get(selected_trial, {})
    final_answer, raw_answer = _final_answer_markdown(trial_payload, trace)
    if final_answer:
        st.markdown(final_answer)
    elif raw_answer:
        st.code(raw_answer, language="markdown")
    else:
        st.info("No final answer found for this trial.")


def _render_shared_trial_viewer(
    left: DashboardSide,
    right: DashboardSide,
    left_scores: dict[str, dict[str, Any]],
    right_scores: dict[str, dict[str, Any]],
) -> None:
    left_grouped = _trial_scores_by_instance(left_scores)
    right_grouped = _trial_scores_by_instance(right_scores)
    task_options = _shared_task_options(left_grouped, right_grouped)
    if not task_options:
        st.info("No per-trial score sidecars found.")
        return

    selector_cols = st.columns(2)
    with selector_cols[0]:
        selected_task = st.selectbox(
            "Task",
            task_options,
            key=f"widesearch-task-shared-{left.run}-{right.run}",
        )

    trial_options = _shared_trial_options(left_grouped, right_grouped, selected_task)
    with selector_cols[1]:
        selected_trial_choice = st.selectbox(
            "Trial",
            trial_options,
            format_func=lambda choice: _trial_choice_label(
                choice,
                selected_task,
                left_grouped,
                right_grouped,
                left_scores,
                right_scores,
            ),
            key=f"widesearch-trial-shared-{left.run}-{right.run}-{selected_task}",
        )

    left_trial = _trial_task_id_for_choice(left_grouped, selected_task, selected_trial_choice)
    right_trial = _trial_task_id_for_choice(right_grouped, selected_task, selected_trial_choice)
    left_trace = _load_selected_trace(left, left_trial)
    right_trace = _load_selected_trace(right, right_trial)

    _render_reference(left, right, selected_task, left_trace, right_trace)
    _render_final_answer("Left", left, left_scores, left_trial, left_trace)
    _render_final_answer("Right", right, right_scores, right_trial, right_trace)


def _render_widesearch_metrics_tab(left: DashboardSide, right: DashboardSide) -> None:
    st.header("WideSearch Metrics")
    st.caption(
        "Reads `widesearch_summary.json`, `widesearch_scores/*.json`, and `wide-search/*.json` "
        "written by the WideSearch evaluator. "
        "Only meaningful for runs of run_type `wide_search`."
    )

    left_summary = _load_widesearch_summary(str(left.run)) if left.run is not None else {}
    right_summary = _load_widesearch_summary(str(right.run)) if right.run is not None else {}
    left_scores = _load_widesearch_trial_scores(str(left.run)) if left.run is not None else {}
    right_scores = _load_widesearch_trial_scores(str(right.run)) if right.run is not None else {}

    cols = st.columns(2)
    with cols[0]:
        _render_side_metrics(left, left_summary)
    with cols[1]:
        _render_side_metrics(right, right_summary)

    _render_shared_trial_viewer(left, right, left_scores, right_scores)
