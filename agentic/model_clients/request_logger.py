# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import atexit
import gzip
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(mode="json"))
        except TypeError:
            return _jsonable(value.model_dump())
    return repr(value)


class ModelRequestLogger:
    """Asynchronous rolling JSONL logger for exact model request/response payloads.

    This logger can still add client-side latency when enabled because it
    serializes full prompts and responses. Keep it default-off for
    high-concurrency rollouts and enable it only when the exact payload trail is
    needed.
    """

    def __init__(
        self,
        log_dir: str | Path,
        *,
        name: str = "inference",
        max_records_per_file: int = 1000,
        max_bytes_per_file: int = 128 * 1024 * 1024,
        include_payload: bool = True,
        include_response: bool = True,
        compress: bool = False,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.dir = self.log_dir / "model_requests" / name
        self.path = self.dir / self._part_filename(1, compress=compress)
        self.manifest_path = self.dir / "manifest.json"
        self.name = name
        self.max_records_per_file = max(1, max_records_per_file)
        self.max_bytes_per_file = max(1, max_bytes_per_file)
        self.include_payload = include_payload
        self.include_response = include_response
        self.compress = compress
        self._session_id = uuid4().hex[:8]
        self._lock = threading.Lock()
        self._counter = 0
        self._closed = False
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._part_index = 1
        self._part_record_count = 0
        self._part_byte_count = 0
        self._total_records = 0
        self._parts: list[dict[str, Any]] = []
        self.dir.mkdir(parents=True, exist_ok=True)
        self._writer = threading.Thread(target=self._drain, name=f"model-request-logger-{name}", daemon=True)
        self._writer.start()
        atexit.register(self.close)

    def next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"{self.name}-{self._session_id}-{self._counter:08d}"

    @staticmethod
    def now() -> str:
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    @staticmethod
    def perf_counter() -> float:
        return time.perf_counter()

    def log(
        self,
        *,
        request_id: str,
        started_at: str,
        elapsed_ms: float,
        request: dict[str, Any],
        response: Any | None = None,
        error: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "schema_version": 1,
            "request_id": request_id,
            "logger_name": self.name,
            "started_at": started_at,
            "ended_at": self.now(),
            "elapsed_ms": elapsed_ms,
            "status": "error" if error else "success",
            "metadata": metadata or {},
            "request": request if self.include_payload else None,
            "response": response if self.include_response else None,
            "error": error,
        }
        with self._lock:
            if self._closed:
                return
            # Keep the producer path cheap; recursive payload normalization is
            # intentionally handled by the writer thread in ``_drain``.
            self._queue.put(record)

    def close(self, *, timeout: float | None = None) -> None:
        """Drain queued records and stop the background writer."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(None)
        self._writer.join(timeout=timeout)

    @staticmethod
    def _part_filename(index: int, *, compress: bool) -> str:
        suffix = ".jsonl.gz" if compress else ".jsonl"
        return f"part-{index:06d}{suffix}"

    def _open_part(self) -> Any:
        self.path = self.dir / self._part_filename(self._part_index, compress=self.compress)
        if self.compress:
            return gzip.open(self.path, "at", encoding="utf-8")
        return self.path.open("a", encoding="utf-8")

    def _record_part(self) -> None:
        if self._part_record_count == 0 and self._part_byte_count == 0:
            return
        part = {
            "path": self.path.name,
            "records": self._part_record_count,
            "bytes": self._part_byte_count,
        }
        if self._parts and self._parts[-1]["path"] == part["path"]:
            self._parts[-1] = part
        else:
            self._parts.append(part)

    def _write_manifest(self, *, closed: bool) -> None:
        self._record_part()
        payload = {
            "schema_version": 1,
            "logger_name": self.name,
            "session_id": self._session_id,
            "closed": closed,
            "compress": self.compress,
            "max_records_per_file": self.max_records_per_file,
            "max_bytes_per_file": self.max_bytes_per_file,
            "include_payload": self.include_payload,
            "include_response": self.include_response,
            "total_records": self._total_records,
            "parts": list(self._parts),
        }
        self.manifest_path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _should_roll(self, line_bytes: int) -> bool:
        if self._part_record_count == 0:
            return False
        return self._part_record_count >= self.max_records_per_file or self._part_byte_count + line_bytes > self.max_bytes_per_file

    def _roll(self, handle: Any) -> Any:
        handle.close()
        self._record_part()
        self._write_manifest(closed=False)
        self._part_index += 1
        self._part_record_count = 0
        self._part_byte_count = 0
        return self._open_part()

    def _drain(self) -> None:
        handle = self._open_part()
        try:
            while True:
                record = self._queue.get()
                try:
                    if record is None:
                        self._write_manifest(closed=True)
                        return
                    line = json.dumps(_jsonable(record), ensure_ascii=False, sort_keys=True)
                    line_bytes = len((line + "\n").encode("utf-8"))
                    if self._should_roll(line_bytes):
                        handle = self._roll(handle)
                    handle.write(line + "\n")
                    self._part_record_count += 1
                    self._part_byte_count += line_bytes
                    self._total_records += 1
                finally:
                    self._queue.task_done()
        finally:
            handle.close()
