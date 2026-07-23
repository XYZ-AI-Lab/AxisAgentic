# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Streamlit dashboard for comparing web-search benchmark run logs.

Usage:
    streamlit run recipe/dashboard/app.py --server.fileWatcherType none -- \
        --log-dir logs \
        --log-dir /path/to/axis-agentic/logs

If no log directory flags are provided, the root log directory defaults to
AXIS_LOG_DIR, falling back to logs. The sidebar selects root log directory,
log directory, experiments, and runs for each side.

Streamlit uses port 8501 by default; forward that port from a remote machine
to open the dashboard in a local browser.
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from recipe.dashboard.accuracy_tab import _render_accuracy_tab
from recipe.dashboard.assistant_message_tab import _render_assistant_message_tab
from recipe.dashboard.comparison_tabs import (
    _render_config_comparison_sides,
    _render_system_prompt_sides,
    _render_task_description_sides,
)
from recipe.dashboard.constants import _DEFAULT_WEB_SEARCH_LOG_DIR, _DEFAULT_WIDE_SEARCH_LOG_DIR
from recipe.dashboard.discovery import (
    _DEFAULT_AGENTIC_TRACE_DIRS,
    _all_task_ids,
    _discover_experiments,
    _discover_runs,
    _parse_cli_args,
    _scan_agentic_index,
)
from recipe.dashboard.experiments_tab import ExperimentOverviewSource, _render_experiments_tab
from recipe.dashboard.live_results import (
    _build_live_agentic_eval_results,
    _load_eval_results,
)
from recipe.dashboard.loading import _load_em_results, _load_llm_judge_results
from recipe.dashboard.sides import DashboardSide
from recipe.dashboard.task_detail_tab import _render_task_detail_sides
from recipe.dashboard.timing_tab import _render_timing_tab
from recipe.dashboard.tool_call_tab import _render_tool_call_tab
from recipe.dashboard.trace_distribution_tab import _render_trace_distribution_tab
from recipe.dashboard.widesearch_metrics_tab import _render_widesearch_metrics_tab

if TYPE_CHECKING:
    import argparse

PipelineKind = Literal["agentic"]
RunType = Literal["web_search", "wide_search"]

_TYPE_LABELS: dict[RunType, str] = {
    "web_search": "web_search",
    "wide_search": "wide_search",
}
_TYPE_KIND: dict[RunType, PipelineKind] = {
    "web_search": "agentic",
    "wide_search": "agentic",
}
_TYPE_TRACE_DIR: dict[RunType, str] = {
    "web_search": "web-search-benchmark",
    "wide_search": "wide-search",
}
_TYPE_DEFAULT_LOG_DIR: dict[RunType, str] = {
    "web_search": _DEFAULT_WEB_SEARCH_LOG_DIR,
    "wide_search": _DEFAULT_WIDE_SEARCH_LOG_DIR,
}
_CUSTOM_ROOT_LOG_DIR_OPTION = "(custom root path)"


@dataclass(frozen=True)
class _RunSelection:
    label: str
    kind: PipelineKind
    run_type: RunType
    root_log_dir: Path
    log_dir: Path
    exp: str
    exp_root: Path
    run_name: str
    agentic_trace_dir: str = _DEFAULT_AGENTIC_TRACE_DIRS[0]

    @property
    def run(self) -> Path | None:
        if self.run_name == "(none)":
            return None
        return self.exp_root / self.run_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _axis_log_root() -> Path:
    return Path(os.environ.get("AXIS_LOG_DIR", "logs"))


def _resolved_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = _resolved_path(path)
        if resolved in seen:
            continue
        deduped.append(path)
        seen.add(resolved)
    return deduped


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        _resolved_path(path).relative_to(_resolved_path(root))
    except ValueError:
        return False
    return True


def _root_log_dirs(extra_roots: list[Path] | None = None, extra_log_dirs: list[Path] | None = None) -> list[Path]:
    roots = [*(extra_roots or []), _axis_log_root()]
    roots.extend(path.parent for path in extra_log_dirs or [])
    return _dedupe_paths(roots)


def _root_for_log_dir(log_dir: Path, roots: list[Path]) -> Path | None:
    for root in roots:
        if _path_is_under(log_dir, root):
            return root
    return None


def _candidate_log_dirs(
    extra_paths: list[Path] | None = None,
    *,
    root_log_dir: Path | None = None,
) -> list[Path]:
    root = root_log_dir or _axis_log_root()
    candidates = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name) if root.exists() else []
    seen = {_resolved_path(path) for path in candidates}
    for path in extra_paths or []:
        include_external = root_log_dir is None
        if not include_external and not _path_is_under(path, root):
            continue
        if path.exists() and path.is_dir() and _resolved_path(path) not in seen:
            candidates.append(path)
            seen.add(_resolved_path(path))
    return candidates


def _format_root_log_dir_option(path: Path | str) -> str:
    if isinstance(path, str):
        return path
    return path.as_posix()


def _format_log_dir_option(path: Path | str, *, root_log_dir: Path | None = None) -> str:
    if isinstance(path, str):
        return path
    root = root_log_dir or _axis_log_root()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _default_log_dir_for_type(run_type: RunType, fallback: Path, candidates: list[Path]) -> Path:
    if run_type == "wide_search":
        prefix = "wide_search"
    elif run_type == "web_search":
        prefix = "web_search"
    else:
        prefix = "web_search"
    for path in candidates:
        if path.name.startswith(prefix):
            return path
    type_default = Path(_TYPE_DEFAULT_LOG_DIR[run_type])
    if type_default.exists():
        return type_default
    return fallback if fallback in candidates else (candidates[0] if candidates else fallback)


def _initial_log_dir_for_type(run_type: RunType, fallback: Path, candidates: list[Path], *, prefer_type_default: bool) -> Path:
    if prefer_type_default:
        return _default_log_dir_for_type(run_type, fallback, candidates)
    return fallback


def _log_dir_state_signature(default_log_dir: Path, *, prefer_type_default: bool) -> str:
    return f"{default_log_dir.expanduser().resolve()}|prefer_type_default={prefer_type_default}"


def _root_log_dir_state_signature(root_log_dirs: list[Path], default_log_dir: Path) -> str:
    roots = ",".join(str(_resolved_path(path)) for path in root_log_dirs)
    return f"{roots}|default_log_dir={_resolved_path(default_log_dir)}"


def _set_root_log_dir_state_for_path(side_label: str, path: Path, options: list[Path | str]) -> None:
    option_key = f"{side_label.lower()}_root_log_dir_option"
    custom_key = f"{side_label.lower()}_root_log_dir_custom"
    resolved_by_path = {option.resolve(): option for option in options if isinstance(option, Path)}
    matched = resolved_by_path.get(path.resolve())
    if matched is not None:
        st.session_state[option_key] = matched
    else:
        st.session_state[option_key] = _CUSTOM_ROOT_LOG_DIR_OPTION
        st.session_state[custom_key] = str(path)


def _set_log_dir_state_for_path(side_label: str, path: Path, options: list[Path | str]) -> None:
    option_key = f"{side_label.lower()}_log_dir_option"
    path_options = [option for option in options if isinstance(option, Path)]
    resolved_by_path = {option.resolve(): option for option in path_options}
    matched = resolved_by_path.get(path.resolve()) or (path_options[0] if path_options else None)
    if matched is not None:
        st.session_state[option_key] = matched


def _select_root_log_dir(side_label: str, root_log_dirs: list[Path], default_log_dir: Path) -> Path:
    option_key = f"{side_label.lower()}_root_log_dir_option"
    custom_key = f"{side_label.lower()}_root_log_dir_custom"
    default_key = f"{side_label.lower()}_root_log_dir_default"
    roots = _dedupe_paths(root_log_dirs or [_axis_log_root()])
    options: list[Path | str] = [*roots, _CUSTOM_ROOT_LOG_DIR_OPTION]
    initial_root = _root_for_log_dir(default_log_dir, roots) or roots[0]
    default_signature = _root_log_dir_state_signature(roots, default_log_dir)

    if (
        option_key not in st.session_state
        or st.session_state.get(default_key) != default_signature
        or st.session_state.get(option_key) not in options
    ):
        _set_root_log_dir_state_for_path(side_label, initial_root, options)
        st.session_state[default_key] = default_signature

    selected = st.selectbox(
        f"{side_label} root log dir",
        options,
        key=option_key,
        format_func=_format_root_log_dir_option,
    )
    if selected == _CUSTOM_ROOT_LOG_DIR_OPTION:
        if custom_key not in st.session_state:
            st.session_state[custom_key] = str(initial_root)
        return Path(st.text_input(f"{side_label} custom root log dir", key=custom_key))
    return cast("Path", selected)


def _select_log_dir(
    side_label: str,
    run_type: RunType,
    default_log_dir: Path,
    *,
    prefer_type_default: bool,
    root_log_dir: Path | None = None,
) -> Path:
    option_key = f"{side_label.lower()}_log_dir_option"
    prev_type_key = f"{side_label.lower()}_prev_type"
    default_key = f"{side_label.lower()}_log_dir_default"
    options: list[Path | str] = _candidate_log_dirs([default_log_dir, Path(_TYPE_DEFAULT_LOG_DIR[run_type])], root_log_dir=root_log_dir)
    if not options:
        options = [default_log_dir]
    path_options = [option for option in options if isinstance(option, Path)]
    type_default_log_dir = _default_log_dir_for_type(run_type, default_log_dir, path_options)
    initial_log_dir = _initial_log_dir_for_type(run_type, default_log_dir, path_options, prefer_type_default=prefer_type_default)
    default_signature = (
        _log_dir_state_signature(default_log_dir, prefer_type_default=prefer_type_default)
        + f"|root_log_dir={_resolved_path(root_log_dir) if root_log_dir is not None else '<axis>'}"
    )

    previous_type = st.session_state.get(prev_type_key)
    if previous_type is not None and previous_type != run_type:
        next_log_dir = type_default_log_dir if prefer_type_default else initial_log_dir
        _set_log_dir_state_for_path(side_label, next_log_dir, options)
        st.session_state[prev_type_key] = run_type
        st.session_state[default_key] = default_signature
    elif option_key not in st.session_state or st.session_state.get(default_key) != default_signature:
        _set_log_dir_state_for_path(side_label, initial_log_dir, options)
        st.session_state[prev_type_key] = run_type
        st.session_state[default_key] = default_signature
    elif st.session_state.get(option_key) not in options:
        _set_log_dir_state_for_path(side_label, initial_log_dir, options)
        st.session_state[default_key] = default_signature

    selected = st.selectbox(
        f"{side_label} log dir",
        options,
        key=option_key,
        format_func=lambda option: _format_log_dir_option(option, root_log_dir=root_log_dir),
    )
    return cast("Path", selected)


def _select_run(
    side_label: str,
    *,
    root_log_dirs: list[Path],
    default_log_dir: Path,
    default_exp: str,
    default_run: str,
    default_run_type: RunType | None = None,
    prefer_type_default_log_dir: bool = True,
) -> _RunSelection:
    st.subheader(side_label)
    inferred_default_type: RunType = default_run_type or "web_search"
    type_options: list[RunType] = ["web_search", "wide_search"]
    type_key = f"{side_label.lower()}_type"
    if st.session_state.get(type_key) not in type_options:
        st.session_state[type_key] = inferred_default_type
    run_type = cast(
        "RunType",
        st.radio(
            f"{side_label} type",
            type_options,
            format_func=lambda t: _TYPE_LABELS[t],
            key=type_key,
            horizontal=True,
        ),
    )

    root_log_dir = _select_root_log_dir(side_label, root_log_dirs, default_log_dir)
    log_dir = _select_log_dir(
        side_label,
        run_type,
        default_log_dir,
        prefer_type_default=prefer_type_default_log_dir,
        root_log_dir=root_log_dir,
    )
    exps = _discover_experiments(log_dir, run_type) or ["(none)"]
    exp_key = f"{side_label.lower()}_exp"
    if st.session_state.get(exp_key) not in exps:
        st.session_state[exp_key] = default_exp if default_exp in exps else exps[0]
    exp = st.selectbox(f"{side_label} exp", exps, key=exp_key)
    exp_root = log_dir / exp if exp != "(none)" else log_dir

    runs = _discover_runs(exp_root) or ["(none)"]
    run_key = f"{side_label.lower()}_run"
    if st.session_state.get(run_key) not in runs:
        st.session_state[run_key] = default_run if default_run in runs else runs[0]
    run_name = st.selectbox(f"{side_label} run", runs, key=run_key)

    kind = _TYPE_KIND[run_type]
    agentic_trace_dir = _TYPE_TRACE_DIR[run_type]
    return _RunSelection(
        label=side_label,
        kind=kind,
        run_type=run_type,
        root_log_dir=root_log_dir,
        log_dir=log_dir,
        exp=exp,
        exp_root=exp_root,
        run_name=run_name,
        agentic_trace_dir=agentic_trace_dir,
    )


def _select_runs_from_sidebar(cli: argparse.Namespace) -> tuple[_RunSelection, _RunSelection]:
    with st.sidebar:
        st.header("Settings")
        cli_log_roots = _cli_log_roots(getattr(cli, "log_dir", None))
        argv = sys.argv[1:]
        left_log_dir_explicit = "--left-log-dir" in argv
        right_log_dir_explicit = "--right-log-dir" in argv
        left_explicit_log_dir = getattr(cli, "left_log_dir", None)
        right_explicit_log_dir = getattr(cli, "right_log_dir", None)
        explicit_side_log_dirs = [Path(path) for path in (left_explicit_log_dir, right_explicit_log_dir) if path]
        root_log_dirs = _root_log_dirs(cli_log_roots, explicit_side_log_dirs)
        default_root = root_log_dirs[0] if root_log_dirs else _axis_log_root()
        left_log_dir = Path(left_explicit_log_dir) if left_explicit_log_dir else _default_log_dir_path_for_root(default_root, "web_search")
        right_log_dir = Path(right_explicit_log_dir) if right_explicit_log_dir else _default_log_dir_path_for_root(default_root, "web_search")
        left = _select_run(
            "Left",
            root_log_dirs=root_log_dirs,
            default_log_dir=left_log_dir,
            default_exp="default",
            default_run="run_1",
            default_run_type="web_search",
            prefer_type_default_log_dir=not left_log_dir_explicit,
        )
        right = _select_run(
            "Right",
            root_log_dirs=root_log_dirs,
            default_log_dir=right_log_dir,
            default_exp="default",
            default_run="run_1",
            default_run_type="web_search",
            prefer_type_default_log_dir=not right_log_dir_explicit,
        )
    return left, right


def _cli_log_roots(raw_log_dir: Any) -> list[Path]:
    if raw_log_dir is None:
        return []
    if isinstance(raw_log_dir, (list, tuple)):
        return [Path(path) for path in raw_log_dir]
    return [Path(raw_log_dir)]


def _default_log_dir_path_for_root(root_log_dir: Path, run_type: RunType) -> Path:
    return root_log_dir / Path(_TYPE_DEFAULT_LOG_DIR[run_type]).name


def _scan_index(selection: _RunSelection) -> dict[str, dict[int, str]]:
    run = selection.run
    if not run:
        return {}
    return _scan_agentic_index(str(run), (selection.agentic_trace_dir,))


def _build_live_eval_results(
    selection: _RunSelection,
    index: dict[str, dict[int, str]],
    evals: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    run = selection.run
    if not run:
        return None
    return _build_live_agentic_eval_results(str(run), index, evals)


def main() -> None:
    st.set_page_config(page_title="Benchmark Run Comparison", layout="wide")
    st.title("Benchmark Run Comparison")

    cli = _parse_cli_args()
    left, right = _select_runs_from_sidebar(cli)

    view_options = [
        "Experiments",
        "Accuracy",
        "WideSearch Metrics",
        "Timing",
        "Trace Distributions",
        "Assistant Message",
        "Tool Call",
        "Task Detail",
        "Config",
        "System Prompt",
        "Task Description",
    ]
    view = st.radio(
        "View",
        view_options,
        horizontal=True,
    )

    if view == "Experiments":
        _render_experiments_tab(
            ExperimentOverviewSource("Left", left.log_dir, left.run_type),
            ExperimentOverviewSource("Right", right.log_dir, right.run_type),
        )
        return

    left_run = left.run
    right_run = right.run

    if view == "Trace Distributions":
        # Completed runs render from precomputed artifacts in the tab. Avoid
        # scanning raw trace directories here; running runs are scanned lazily by
        # the tab when no completed-run artifact is available.
        _render_trace_distribution_tab(
            DashboardSide(
                label="Left",
                kind=left.kind,
                run=left_run,
                index={},
                evals={},
                llm_evals={},
                exp_name=left.exp,
                run_name=left.run_name,
                run_type=left.run_type,
                agentic_trace_dir=left.agentic_trace_dir,
            ),
            DashboardSide(
                label="Right",
                kind=right.kind,
                run=right_run,
                index={},
                evals={},
                llm_evals={},
                exp_name=right.exp,
                run_name=right.run_name,
                run_type=right.run_type,
                agentic_trace_dir=right.agentic_trace_dir,
            ),
        )
        return

    if view == "Assistant Message":
        # Completed runs render from precomputed artifacts in the tab. Running
        # runs are scanned lazily by the tab when no completed-run artifact is
        # available.
        _render_assistant_message_tab(
            DashboardSide(
                label="Left",
                kind=left.kind,
                run=left_run,
                index={},
                evals={},
                llm_evals={},
                exp_name=left.exp,
                run_name=left.run_name,
                run_type=left.run_type,
                agentic_trace_dir=left.agentic_trace_dir,
            ),
            DashboardSide(
                label="Right",
                kind=right.kind,
                run=right_run,
                index={},
                evals={},
                llm_evals={},
                exp_name=right.exp,
                run_name=right.run_name,
                run_type=right.run_type,
                agentic_trace_dir=right.agentic_trace_dir,
            ),
        )
        return

    if view == "WideSearch Metrics":
        _render_widesearch_metrics_tab(
            DashboardSide(
                label="Left",
                kind=left.kind,
                run=left_run,
                index={},
                evals={},
                llm_evals={},
                exp_name=left.exp,
                run_name=left.run_name,
                run_type=left.run_type,
                agentic_trace_dir=left.agentic_trace_dir,
            ),
            DashboardSide(
                label="Right",
                kind=right.kind,
                run=right_run,
                index={},
                evals={},
                llm_evals={},
                exp_name=right.exp,
                run_name=right.run_name,
                run_type=right.run_type,
                agentic_trace_dir=right.agentic_trace_dir,
            ),
        )
        return

    left_index = _scan_index(left)
    right_index = _scan_index(right)
    left_evals = _load_em_results(str(left_run)) if left_run else {}
    right_evals = _load_em_results(str(right_run)) if right_run else {}
    left_llm_evals = _load_llm_judge_results(str(left_run)) if left_run else {}
    right_llm_evals = _load_llm_judge_results(str(right_run)) if right_run else {}

    if not left_index and not right_index:
        st.warning("No task logs found. Check the log directories.")
        return

    all_ids = _all_task_ids(left_index, right_index)

    left_side = DashboardSide(
        label="Left",
        kind=left.kind,
        run=left_run,
        index=left_index,
        evals=left_evals,
        llm_evals=left_llm_evals,
        exp_name=left.exp,
        run_name=left.run_name,
        run_type=left.run_type,
        agentic_trace_dir=left.agentic_trace_dir,
    )
    right_side = DashboardSide(
        label="Right",
        kind=right.kind,
        run=right_run,
        index=right_index,
        evals=right_evals,
        llm_evals=right_llm_evals,
        exp_name=right.exp,
        run_name=right.run_name,
        run_type=right.run_type,
        agentic_trace_dir=right.agentic_trace_dir,
    )

    if view == "Accuracy":
        _render_accuracy_tab(left_side, right_side)

    elif view == "Timing":
        left_eval_results = _load_eval_results(str(left_run)) if left_run else None
        right_eval_results = _load_eval_results(str(right_run)) if right_run else None
        left_timing_results = left_eval_results or _build_live_eval_results(left, left_index, left_evals)
        right_timing_results = right_eval_results or _build_live_eval_results(right, right_index, right_evals)
        _render_timing_tab(
            left_timing_results,
            right_timing_results,
            left_label=left_side.label,
            right_label=right_side.label,
            left_side=left_side,
            right_side=right_side,
        )

    elif view == "Tool Call":
        _render_tool_call_tab(left_side, right_side)

    elif view == "Task Detail":
        selected = st.selectbox("Select task", all_ids, key="detail_task")
        if selected:
            _render_task_detail_sides(selected, left_side, right_side)

    elif view == "Config":
        _render_config_comparison_sides(left_side, right_side)

    elif view == "System Prompt":
        _render_system_prompt_sides(left_side, right_side)

    elif view == "Task Description":
        _render_task_description_sides(left_side, right_side)


if __name__ == "__main__":
    main()
