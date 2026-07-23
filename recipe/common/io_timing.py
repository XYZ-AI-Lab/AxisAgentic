# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def write_json_with_timing(path: Path, payload: dict[str, Any], *, timing_key: str | None = None) -> float:
    """Write JSON and return elapsed wall-clock seconds for serialization + file IO.

    When ``timing_key`` is provided, the output JSON also records the measured write
    duration under that key. The file is written once, then the fixed-width placeholder
    for ``timing_key`` is patched in-place so the measured duration covers the full
    JSON payload rather than a smaller pre-timing payload.
    """
    placeholder = "__AXIS_IO_TIMING_PLACEHOLDER__"
    if timing_key is not None:
        payload[timing_key] = placeholder

    serialized = json.dumps(payload, indent=2).encode("utf-8")
    placeholder_offset = serialized.find(json.dumps(placeholder).encode("utf-8"))
    start = time.perf_counter()
    path.write_bytes(serialized)
    elapsed = time.perf_counter() - start

    if timing_key is not None:
        payload[timing_key] = elapsed
        if placeholder_offset < 0:
            msg = f"timing placeholder for {timing_key!r} was not found in serialized JSON"
            raise RuntimeError(msg)
        placeholder_len = len(json.dumps(placeholder).encode("utf-8"))
        replacement = f"{elapsed:.9f}".ljust(placeholder_len).encode("utf-8")
        try:
            with path.open("r+b") as f:
                f.seek(placeholder_offset)
                f.write(replacement)
        except PermissionError:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return elapsed


def write_jsonl_with_timing(path: Path, rows: list[dict[str, object]]) -> float:
    """Write JSONL rows and return elapsed wall-clock seconds."""
    start = time.perf_counter()
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return time.perf_counter() - start


def write_text_with_timing(path: Path, content: str) -> float:
    """Write text and return elapsed wall-clock seconds."""
    start = time.perf_counter()
    path.write_text(content, encoding="utf-8")
    return time.perf_counter() - start
