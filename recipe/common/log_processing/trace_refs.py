# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from pathlib import Path


def is_attempt_budget_branch_stem(stem: str) -> bool:
    return re.search(r"_budget-\d+_attempt-\d+(?:$|_)", stem) is not None


def add_trace_refs(index: dict[str, dict[int, str]], trace_dir: Path) -> None:
    refs_path = trace_dir / "trace_refs.json"
    if not refs_path.exists() or refs_path.stat().st_size == 0:
        return
    try:
        payload = json.loads(refs_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return
    refs = payload.get("refs") if isinstance(payload, dict) else None
    if not isinstance(refs, list):
        return
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        task_id = ref.get("task_id")
        trace_path_value = ref.get("trace_path")
        attempt_value = ref.get("attempt")
        if not isinstance(task_id, str) or not task_id:
            continue
        if not isinstance(trace_path_value, str) or not trace_path_value:
            continue
        try:
            attempt = int(attempt_value)
        except (TypeError, ValueError):
            continue
        trace_path = Path(trace_path_value)
        if not trace_path.is_absolute():
            trace_path = trace_dir / trace_path
        index.setdefault(task_id, {})[attempt] = str(trace_path.resolve())
