# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Run discovery and task-index helpers for the dashboard."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import streamlit as st

from recipe.common.log_processing.trace_refs import add_trace_refs, is_attempt_budget_branch_stem


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        "--log-root",
        action="append",
        dest="log_dir",
        default=None,
        help="Root log directory containing dashboard log dirs. Can be passed multiple times.",
    )
    parser.add_argument("--left-log-dir", default=None, help="Initial concrete log dir for the left dashboard side")
    parser.add_argument("--right-log-dir", default=None, help="Initial concrete log dir for the right dashboard side")
    return parser.parse_args(sys.argv[1:])


def _discover_runs(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("run_"))


def _discover_experiments(log_dir: Path, run_type: str) -> list[str]:
    del run_type
    if not log_dir.exists():
        return []
    return sorted(p.name for p in log_dir.iterdir() if p.is_dir())


def _parse_agentic_trace_stem(stem: str) -> tuple[str, int] | None:
    match = re.match(r"(?P<base>.+)_attempt-(?P<attempt>\d+)(?:$|_)", stem)
    if not match:
        return None
    return match.group("base"), int(match.group("attempt"))


@st.cache_data(ttl=30, show_spinner=False)
def _scan_ori_index(run_dir: str) -> dict[str, dict[int, str]]:
    rd = Path(run_dir)
    index: dict[str, dict[int, str]] = {}
    for f in sorted(rd.glob("task_*.json")):
        if f.stat().st_size == 0:
            continue
        m = re.match(r"task_(?P<base>[^_]+)_attempt-(?P<attempt>\d+)", f.name)
        if not m:
            continue
        base_id = m.group("base")
        attempt = int(m.group("attempt"))
        index.setdefault(base_id, {})[attempt] = str(f)
    return index


_DEFAULT_AGENTIC_TRACE_DIRS: tuple[str, ...] = ("web-search-benchmark", "wide-search")


@st.cache_data(ttl=30, show_spinner=False)
def _scan_agentic_index(run_dir: str, trace_dir_names: tuple[str, ...] = _DEFAULT_AGENTIC_TRACE_DIRS) -> dict[str, dict[int, str]]:
    rd = Path(run_dir)
    index: dict[str, dict[int, str]] = {}
    for name in trace_dir_names:
        trace_dir = rd / name
        if not trace_dir.exists():
            continue
        add_trace_refs(index, trace_dir)
        for f in sorted(trace_dir.glob("*.json")):
            if f.name == "trace_refs.json":
                continue
            if f.stat().st_size == 0:
                continue
            if is_attempt_budget_branch_stem(f.stem):
                continue
            parsed = _parse_agentic_trace_stem(f.stem)
            if parsed is None:
                continue
            base_id, att = parsed
            index.setdefault(base_id, {})[att] = str(f)
    return index


def _latest_path(attempts: dict[int, str]) -> str:
    return attempts[max(attempts)]


def _sorted_attempts(attempts: dict[int, str]) -> list[int]:
    return sorted(attempts)


def _all_task_ids(*indexes: dict[str, dict[int, str]]) -> list[str]:
    merged: set[str] = set()
    for idx in indexes:
        merged.update(idx)
    return sorted(merged, key=_task_id_sort_key)


def _task_id_sort_key(task_id: str) -> tuple[int, int, str]:
    """Type-stable sort key for mixed numeric/non-numeric task IDs.

    BrowseComp uses digit-only IDs (``"123"``); WideSearch uses string IDs
    (``"ws_en_001__trial-0"``). The previous ``int(x) if x.isdigit() else x``
    key returned mixed ``int`` and ``str`` values, and Python 3 raises
    ``TypeError`` when comparing them. This key always returns a homogeneous
    ``tuple[int, int, str]``: numeric IDs sort first by integer value, then
    string IDs follow lexicographically.
    """
    if task_id.isdigit():
        return (0, int(task_id), task_id)
    return (1, 0, task_id)
