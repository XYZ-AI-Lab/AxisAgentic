# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import logging
import queue
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic.contracts.messages import ToolResultReason, ToolResultStatus

if TYPE_CHECKING:
    from collections.abc import Iterator

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

UTC8 = timezone(timedelta(hours=8))

# Module-level default used by TaskTrace dataclass default_factory.
# Updated by TaskLogger.__init__ so traces use the configured timezone.
_active_tz: timezone = UTC8

# Cap for tool-failure content previews rendered into log lines.
_TOOL_ERROR_PREVIEW_CHARS = 240
_RAW_SCRAPE_CACHE_SUMMARY_SCHEMA_VERSION = 1
_RAW_SCRAPE_CACHE_STATUSES = ("hit", "miss", "bypass")
_RAW_SCRAPE_CACHE_COUNTERS = {
    "hit": "raw_scrape_cache_hit_count",
    "miss": "raw_scrape_cache_miss_count",
    "bypass": "raw_scrape_cache_bypass_count",
    "put": "raw_scrape_cache_put_count",
}


def _format_tool_error_preview(content: str, *, max_chars: int = _TOOL_ERROR_PREVIEW_CHARS) -> str:
    """Render *content* as a single-line error snippet for tool-failure log lines.

    When *content* is a JSON object carrying a top-level ``error``, ``message``, or ``detail`` string, prefer that message;
    otherwise fall back to the raw content. Newlines are collapsed and the result is truncated with an ellipsis.
    """
    text = content
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            data = None
        if isinstance(data, dict):
            for key in ("error", "message", "detail"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    text = value
                    break
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text


def _now_with_tz() -> str:
    return datetime.now(tz=_active_tz).isoformat()


def _single_or_mixed(values: set[Any]) -> Any:
    if not values:
        return None
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def _add_summary_value(values: set[Any], value: Any) -> None:
    try:
        hash(value)
    except TypeError:
        values.add(repr(value))
    else:
        values.add(value)


def _raw_scrape_cache_summary(tool_calls: list[ToolTrace]) -> dict[str, Any]:
    status_counts = dict.fromkeys(_RAW_SCRAPE_CACHE_STATUSES, 0)
    max_counters = dict.fromkeys(_RAW_SCRAPE_CACHE_COUNTERS, 0)
    key_hashes: list[str] = []
    saved_jina_request_count = 0
    counter_scopes: set[Any] = set()
    providers: set[Any] = set()
    scopes: set[Any] = set()
    normalize_urls: set[Any] = set()
    normalize_policies: set[Any] = set()
    observed_tool_calls = 0

    for tool_call in tool_calls:
        metadata = tool_call.metadata if isinstance(tool_call.metadata, dict) else {}
        status = metadata.get("raw_scrape_cache_status")
        if not isinstance(status, str):
            continue
        observed_tool_calls += 1
        if status in status_counts:
            status_counts[status] += 1
        if metadata.get("raw_scrape_cache_saved_jina_request") is True:
            saved_jina_request_count += 1
        key_hash = metadata.get("raw_scrape_cache_key_hash")
        if isinstance(key_hash, str) and key_hash:
            key_hashes.append(key_hash)
        for counter_name, metadata_key in _RAW_SCRAPE_CACHE_COUNTERS.items():
            value = metadata.get(metadata_key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int | float):
                max_counters[counter_name] = max(max_counters[counter_name], int(value))
        if "raw_scrape_cache_counter_scope" in metadata:
            _add_summary_value(counter_scopes, metadata.get("raw_scrape_cache_counter_scope"))
        if "raw_scrape_cache_provider" in metadata:
            _add_summary_value(providers, metadata.get("raw_scrape_cache_provider"))
        if "raw_scrape_cache_scope" in metadata:
            _add_summary_value(scopes, metadata.get("raw_scrape_cache_scope"))
        if "raw_scrape_cache_normalize_url" in metadata:
            _add_summary_value(normalize_urls, metadata.get("raw_scrape_cache_normalize_url"))
        if "raw_scrape_cache_normalize_policy" in metadata:
            _add_summary_value(normalize_policies, metadata.get("raw_scrape_cache_normalize_policy"))

    unique_key_hashes = set(key_hashes)
    duplicate_key_hash_count = len(key_hashes) - len(unique_key_hashes)
    observed = observed_tool_calls > 0
    recognized_no_hit_count = status_counts["miss"] + status_counts["bypass"]
    no_hit_evidence = observed and recognized_no_hit_count > 0 and status_counts["hit"] == 0 and duplicate_key_hash_count == 0

    return {
        "schema_version": _RAW_SCRAPE_CACHE_SUMMARY_SCHEMA_VERSION,
        "observed": observed,
        "tool_call_count": observed_tool_calls,
        "status_counts": status_counts,
        "hit_count_max": max_counters["hit"],
        "miss_count_max": max_counters["miss"],
        "bypass_count_max": max_counters["bypass"],
        "put_count_max": max_counters["put"],
        "saved_jina_request_count": saved_jina_request_count,
        "key_hash_count": len(key_hashes),
        "unique_key_hash_count": len(unique_key_hashes),
        "duplicate_key_hash_count": duplicate_key_hash_count,
        "no_hit_evidence": no_hit_evidence,
        "counter_scope": _single_or_mixed(counter_scopes),
        "provider": _single_or_mixed(providers),
        "scope": _single_or_mixed(scopes),
        "normalize_url": _single_or_mixed(normalize_urls),
        "normalize_policy": _single_or_mixed(normalize_policies),
    }


# ANSI escape sequences for coloured console output.
# Inlined from colorama so the observability stack has no optional dependency —
# ``_AnsiStrippingFormatter`` removes these codes before writing to the file handler,
# so file logs stay plain text regardless.
_ANSI_RESET = "\x1b[0m"
_ANSI_NAME = "\x1b[1;34m"  # bright blue
_ANSI_DEFAULT_LEVEL = "\x1b[1;37m"  # bright white
_LEVEL_COLORS: dict[str, str] = {
    "ERROR": "\x1b[1;31m",  # bright red
    "WARNING": "\x1b[1;33m",  # bright yellow
    "INFO": "\x1b[1;32m",  # bright green
    "DEBUG": "\x1b[1;36m",  # bright cyan
}


class _AnsiStrippingFormatter(logging.Formatter):
    """Formatter that strips ANSI escape codes so file logs stay clean."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # noqa: N802
        dt = datetime.fromtimestamp(record.created, tz=_active_tz)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat(timespec="milliseconds")

    def format(self, record: logging.LogRecord) -> str:
        return _ANSI_ESCAPE_RE.sub("", super().format(record))


class ColoredFormatter(logging.Formatter):
    """Logging formatter that applies ANSI colors to level and logger name."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        level_color = _LEVEL_COLORS.get(record.levelname, _ANSI_DEFAULT_LEVEL)
        message = record.getMessage()
        return f"[{timestamp}][{_ANSI_NAME}{record.name}{_ANSI_RESET}][{level_color}{record.levelname}{_ANSI_RESET}] {message}"


@dataclass(slots=True)
class ToolTrace:
    tool_name: str
    call_id: str
    arguments: dict[str, Any]
    status: ToolResultStatus
    latency_ms: float
    reason: ToolResultReason | None = None
    content: str = ""
    emoji: str = "🔧"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TokenUsageRecord:
    """Cumulative and per-step token usage tracked over the lifetime of a task."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    per_step: list[dict[str, int]] = field(default_factory=list)


@dataclass(slots=True)
class TaskTrace:
    task_id: str
    task_input: Any
    status: str = "running"
    started_at: str = field(default_factory=_now_with_tz)
    ended_at: str | None = None
    log_dir: str | None = None
    tool_path: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolTrace] = field(default_factory=list)
    conversation: list[dict[str, Any]] = field(default_factory=list)
    token_usage: TokenUsageRecord = field(default_factory=TokenUsageRecord)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Cumulative wall-clock time spent inside ``TaskLogger`` methods for this task
    # (bookkeeping + incremental flushes).  ``finish_task`` splits its own body
    # so the full JSON-write cost is measured separately and logged at INFO;
    # the value persisted in ``metadata.task_logger_bookkeeping_ms`` /
    # ``metadata.task_logger_finish_ms`` at shutdown tells callers whether
    # TaskLogger IO is a meaningful slice of ``timing/rollout_seconds``.
    task_logger_ms: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskTrace:
        """Reconstruct a TaskTrace from a JSON-deserialized dictionary.

        This is the inverse of ``dataclasses.asdict(trace)``.
        """
        tool_calls = [
            ToolTrace(
                tool_name=tc["tool_name"],
                call_id=tc["call_id"],
                arguments=tc.get("arguments", {}),
                status=ToolResultStatus(tc["status"]) if isinstance(tc["status"], str) else tc["status"],
                latency_ms=tc.get("latency_ms", 0.0),
                reason=ToolResultReason(tc["reason"]) if tc.get("reason") else None,
                content=tc.get("content", ""),
                emoji=tc.get("emoji", "🔧"),
                metadata=tc.get("metadata", {}),
            )
            for tc in data.get("tool_calls", [])
        ]
        tu = data.get("token_usage", {})
        token_usage = TokenUsageRecord(
            input_tokens=tu.get("input_tokens", 0),
            output_tokens=tu.get("output_tokens", 0),
            total_tokens=tu.get("total_tokens", 0),
            per_step=tu.get("per_step", []),
        )
        return cls(
            task_id=data["task_id"],
            task_input=data.get("task_input"),
            status=data.get("status", "running"),
            started_at=data.get("started_at", ""),
            ended_at=data.get("ended_at"),
            log_dir=data.get("log_dir"),
            tool_path=data.get("tool_path", []),
            steps=data.get("steps", []),
            tool_calls=tool_calls,
            conversation=data.get("conversation", []),
            token_usage=token_usage,
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class _TraceWriteJob:
    trace: TaskTrace
    timing_metadata: dict[str, float]


class TaskLogger:
    def __init__(
        self,
        name: str = "agentic",
        *,
        log_dir: str | Path = "logs/agentic",
        persist_json: bool = True,
        background_persist_json: bool = False,
        background_queue_maxsize: int = 4096,
        timezone_offset_hours: float = 8,
        incremental_flush_steps: int = 0,
    ) -> None:
        global _active_tz  # noqa: PLW0603
        _active_tz = timezone(timedelta(hours=timezone_offset_hours))
        self._incremental_flush_steps = incremental_flush_steps
        self._step_counters: dict[str, int] = {}  # task_id → steps since last flush
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            # Colored stream handler for stdout
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(
                ColoredFormatter("%(asctime)s,%(msecs)03d", datefmt="%Y-%m-%d %H:%M:%S%z"),
            )
            self._logger.addHandler(stream_handler)
            self._logger.propagate = False
        self._logger.setLevel(logging.INFO)
        self._persist_json = persist_json
        self._background_persist_json = bool(persist_json and background_persist_json)
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # File handler for persistent plain-text log at the root log dir
        if not any(isinstance(h, logging.FileHandler) for h in self._logger.handlers):
            log_file = self._log_dir / f"{name}.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(
                _AnsiStrippingFormatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s"),
            )
            self._logger.addHandler(file_handler)

        self._live_traces: dict[str, TaskTrace] = {}
        self._trace_write_queue: queue.Queue[_TraceWriteJob | None] | None = None
        self._trace_writer_thread: threading.Thread | None = None
        self._trace_writer_closed = False
        self._trace_writer_errors: list[str] = []
        if self._background_persist_json:
            maxsize = max(1, background_queue_maxsize)
            self._trace_write_queue = queue.Queue(maxsize=maxsize)
            self._trace_writer_thread = threading.Thread(
                target=self._trace_writer_loop,
                name=f"{name}-trace-writer",
                daemon=True,
            )
            self._trace_writer_thread.start()

    @contextmanager
    def _measure(self, task_id: str) -> Iterator[None]:
        """Accumulate wall-clock time spent inside TaskLogger methods into ``TaskTrace.task_logger_ms``.

        Wraps the body of every public mutating method so the trace self-records how much
        of the controller event loop was spent on bookkeeping / file IO vs. actual rollout work.
        A missing live trace (e.g. after ``finish_task`` but before the method returns) is
        silently ignored — callers should not have to guard against trace lifecycle races.
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            trace = self._live_traces.get(task_id)
            if trace is not None:
                trace.task_logger_ms += (time.perf_counter() - t0) * 1000.0

    def start_task(
        self,
        task_id: str,
        task_input: Any,
        *,
        metadata: dict[str, Any] | None = None,
        level: int = 0,
        tool_path: list[str] | None = None,
    ) -> None:
        if task_id in self._live_traces:
            raise ValueError(f"Task '{task_id}' is already being tracked.")
        # Construct the trace FIRST so ``_measure`` has somewhere to accumulate;
        # initialisation cost itself is part of the TaskLogger time budget we want to expose.
        start_counter = time.perf_counter()
        trace = TaskTrace(
            task_id=task_id,
            task_input=task_input,
            tool_path=list(tool_path or []),
            log_dir=str(self._log_dir),
        )
        if metadata:
            trace.metadata.update(metadata)
        self._live_traces[task_id] = trace
        self.info(f"start task {task_id}", level=level)
        trace.task_logger_ms += (time.perf_counter() - start_counter) * 1000.0

    def fork_live_trace(
        self,
        source_task_id: str,
        new_task_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        level: int = 0,
    ) -> None:
        if new_task_id in self._live_traces:
            raise ValueError(f"Task '{new_task_id}' is already being tracked.")
        source = self._get_live_trace(source_task_id)
        start_counter = time.perf_counter()
        trace = copy.deepcopy(source)
        trace.task_id = new_task_id
        trace.status = "running"
        trace.ended_at = None
        trace.metadata = dict(trace.metadata)
        trace.metadata.pop("task_logger_finish_ms", None)
        trace.metadata.pop("task_logger_trace_dump_ms", None)
        trace.metadata.pop("task_logger_trace_serialize_ms", None)
        trace.metadata.pop("task_logger_trace_file_write_ms", None)
        trace.metadata.pop("task_logger_partial_unlink_ms", None)
        trace.metadata.pop("task_logger_trace_enqueue_wait_ms", None)
        trace.metadata.pop("task_logger_trace_async_pending", None)
        if metadata:
            trace.metadata.update(metadata)
        self._live_traces[new_task_id] = trace
        self.info(f"fork task {source_task_id} -> {new_task_id}", level=level)
        trace.task_logger_ms += (time.perf_counter() - start_counter) * 1000.0

    def add_step(
        self,
        task_id: str,
        turn_idx: int,
        step_name: str,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
        level: int = 0,
    ) -> None:
        with self._measure(task_id):
            trace = self._get_live_trace(task_id)
            step_entry: dict[str, Any] = {
                "turn_idx": turn_idx,
                "step_name": step_name,
                "message": message,
                "timestamp": _now_with_tz(),
            }
            # Promote elapsed_ms to a top-level field for easy access while keeping it in metadata too.
            if metadata:
                step_entry["metadata"] = metadata
                if "elapsed_ms" in metadata:
                    step_entry["elapsed_ms"] = metadata["elapsed_ms"]
            else:
                step_entry["metadata"] = {}
            trace.steps.append(step_entry)
            self.info(f"Task {task_id} | Turn {turn_idx} | {step_name}: {message}", level=level)
            self._maybe_incremental_flush(task_id)

    def append_conversation(self, task_id: str, conversation: list[dict[str, Any]]) -> None:
        with self._measure(task_id):
            trace = self._get_live_trace(task_id)
            trace.conversation.extend(conversation)

    def add_token_usage(self, task_id: str, *, input_tokens: int, output_tokens: int, total_tokens: int) -> None:
        """Accumulate token usage from a single model call into the task trace."""
        with self._measure(task_id):
            trace = self._get_live_trace(task_id)
            trace.token_usage.input_tokens += input_tokens
            trace.token_usage.output_tokens += output_tokens
            trace.token_usage.total_tokens += total_tokens
            trace.token_usage.per_step.append({"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens})

    def add_tool_trace(self, task_id: str, tool_trace: ToolTrace, *, level: int = 0) -> None:
        with self._measure(task_id):
            trace = self._get_live_trace(task_id)
            trace.tool_calls.append(tool_trace)
            log = self.error if tool_trace.status != ToolResultStatus.SUCCESS else self.info
            suffix = f" reason={tool_trace.reason}" if tool_trace.reason else ""
            if tool_trace.status != ToolResultStatus.SUCCESS and tool_trace.content:
                preview = _format_tool_error_preview(tool_trace.content)
                if preview:
                    suffix += f" error={preview}"
            log(f"{tool_trace.emoji} {tool_trace.tool_name} call_id={tool_trace.call_id} latency_ms={tool_trace.latency_ms:.2f}{suffix}", level=level)
            self._maybe_incremental_flush(task_id)

    def _finalize_trace_metadata(self, trace: TaskTrace, status: str, metadata: dict[str, Any] | None) -> None:
        trace.status = status
        trace.ended_at = _now_with_tz()
        if metadata:
            trace.metadata.update(metadata)
        trace.metadata["raw_scrape_cache_summary"] = _raw_scrape_cache_summary(trace.tool_calls)
        try:
            start_dt = datetime.fromisoformat(trace.started_at)
            end_dt = datetime.fromisoformat(trace.ended_at)
            trace.metadata["total_elapsed_ms"] = round((end_dt - start_dt).total_seconds() * 1000, 1)
        except (ValueError, TypeError):
            pass

    def _persist_trace_to_file(self, trace: TaskTrace, timing_metadata: dict[str, float]) -> dict[str, float]:
        if not self._persist_json:
            return timing_metadata
        output_dir = self._log_dir.joinpath(*trace.tool_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{trace.task_id}.json"
        placeholders = {key: f"__{key.upper()}__" for key in timing_metadata}
        trace.metadata.update(placeholders)
        serialize_start = time.perf_counter()
        payload = json.dumps(asdict(trace), indent=2, sort_keys=True).encode("utf-8")
        placeholder_offsets = {key: payload.find(json.dumps(placeholder).encode("utf-8")) for key, placeholder in placeholders.items()}
        trace_serialize_ms = (time.perf_counter() - serialize_start) * 1000.0
        file_write_start = time.perf_counter()
        output_path.write_bytes(payload)
        trace_file_write_ms = (time.perf_counter() - file_write_start) * 1000.0
        trace_partial_unlink_ms = 0.0
        partial_path = output_dir / f"{trace.task_id}.partial.json"
        if partial_path.exists():
            unlink_start = time.perf_counter()
            partial_path.unlink()
            trace_partial_unlink_ms = (time.perf_counter() - unlink_start) * 1000.0
        trace_dump_ms = (time.perf_counter() - serialize_start) * 1000.0
        finish_ms = (time.perf_counter() - serialize_start) * 1000.0
        write_timing_metadata = {
            "task_logger_finish_ms": round(finish_ms, 1),
            "task_logger_trace_dump_ms": round(trace_dump_ms, 1),
            "task_logger_trace_serialize_ms": round(trace_serialize_ms, 1),
            "task_logger_trace_file_write_ms": round(trace_file_write_ms, 1),
            "task_logger_partial_unlink_ms": round(trace_partial_unlink_ms, 1),
        }
        timing_metadata = {**timing_metadata, **write_timing_metadata}
        with output_path.open("r+b") as f:
            for key, value in timing_metadata.items():
                offset = placeholder_offsets[key]
                if offset < 0:
                    continue
                placeholder_len = len(json.dumps(placeholders[key]).encode("utf-8"))
                replacement = f"{value:.1f}".ljust(placeholder_len).encode("utf-8")
                f.seek(offset)
                f.write(replacement)
        return timing_metadata

    def _enqueue_trace_write(self, trace: TaskTrace, timing_metadata: dict[str, float], *, bookkeeping_ms: float) -> dict[str, float]:
        trace_write_queue = self._trace_write_queue
        if trace_write_queue is None or self._trace_writer_closed:
            return self._persist_trace_to_file(trace, timing_metadata)

        enqueue_start = time.perf_counter()
        while True:
            enqueue_wait_ms = (time.perf_counter() - enqueue_start) * 1000.0
            timing_metadata["task_logger_trace_enqueue_wait_ms"] = round(enqueue_wait_ms, 1)
            timing_metadata["task_logger_finish_ms"] = round(enqueue_wait_ms, 1)
            timing_metadata["task_logger_trace_async_pending"] = 1.0
            trace.metadata.update(timing_metadata)
            trace.task_logger_ms = bookkeeping_ms + timing_metadata["task_logger_finish_ms"]
            try:
                trace_write_queue.put_nowait(_TraceWriteJob(trace=trace, timing_metadata=dict(timing_metadata)))
                break
            except queue.Full:
                time.sleep(0.001)
        return timing_metadata

    def _trace_writer_loop(self) -> None:
        assert self._trace_write_queue is not None
        while True:
            job = self._trace_write_queue.get()
            try:
                if job is None:
                    return
                timing_metadata = dict(job.timing_metadata)
                timing_metadata["task_logger_trace_async_pending"] = 0.0
                self._persist_trace_to_file(job.trace, timing_metadata)
            except Exception as exc:
                self._trace_writer_errors.append(repr(exc))
            finally:
                self._trace_write_queue.task_done()

    def drain(self) -> None:
        """Wait until all background trace JSON writes have completed."""
        if self._trace_write_queue is not None:
            self._trace_write_queue.join()

    def close(self, *, timeout: float | None = None) -> None:
        """Flush pending background trace writes and stop the writer thread."""
        if self._trace_write_queue is None or self._trace_writer_thread is None or self._trace_writer_closed:
            return
        self.drain()
        self._trace_writer_closed = True
        self._trace_write_queue.put(None)
        self._trace_writer_thread.join(timeout=timeout)
        if self._trace_writer_thread.is_alive():
            self.warning("TaskLogger trace writer did not stop before close timeout")
        if self._trace_writer_errors:
            self.warning(f"TaskLogger trace writer saw {len(self._trace_writer_errors)} error(s); first={self._trace_writer_errors[0]}")

    def finish_task(self, task_id: str, *, status: str, metadata: dict[str, Any] | None = None, level: int = 0) -> dict[str, float]:
        finish_start = time.perf_counter()
        trace = self._get_live_trace(task_id)
        self._finalize_trace_metadata(trace, status, metadata)
        # Snapshot bookkeeping time BEFORE the big JSON write so the persisted file can report it accurately.
        # The write itself is the expensive part on most rollouts and is surfaced as ``task_logger_finish_ms``;
        # we compute it right before writing, stash it inline, then write.
        # If the disk call itself ran long, the number we persist slightly under-counts —
        # the post-write INFO line captures the true total so operators can reconcile if needed.
        bookkeeping_ms = trace.task_logger_ms + (time.perf_counter() - finish_start) * 1000.0
        trace.metadata["task_logger_bookkeeping_ms"] = round(bookkeeping_ms, 1)
        self.info(f"finish task {trace.task_id} status={status}", level=level)
        write_start = time.perf_counter()
        timing_metadata: dict[str, float] = {
            "task_logger_finish_ms": 0.0,
            "task_logger_trace_dump_ms": 0.0,
            "task_logger_trace_serialize_ms": 0.0,
            "task_logger_trace_file_write_ms": 0.0,
            "task_logger_partial_unlink_ms": 0.0,
            "task_logger_trace_enqueue_wait_ms": 0.0,
            "task_logger_trace_async_pending": 0.0,
        }
        timing_metadata = (
            self._enqueue_trace_write(trace, timing_metadata, bookkeeping_ms=bookkeeping_ms)
            if self._background_persist_json
            else self._persist_trace_to_file(trace, timing_metadata)
        )
        finish_ms = (time.perf_counter() - write_start) * 1000.0
        if not self._background_persist_json:
            timing_metadata["task_logger_finish_ms"] = round(finish_ms, 1)
            trace.metadata.update(timing_metadata)
            trace.task_logger_ms = bookkeeping_ms + timing_metadata["task_logger_finish_ms"]
        self.info(
            f"finish task {trace.task_id} bookkeeping_ms={bookkeeping_ms:.1f} "
            f"finish_ms={timing_metadata['task_logger_finish_ms']:.1f} total_ms={trace.task_logger_ms:.1f}",
            level=level,
        )
        self._step_counters.pop(task_id, None)
        del self._live_traces[task_id]
        return timing_metadata

    def save_config(self, config: dict[str, Any], filename: str = "run_config.json", *, level: int = 0) -> Path:
        """Persist a configuration dictionary to the log directory.

        Args:
            config: Configuration dict to save.
            filename: Target filename inside the log directory.
            level: Indentation level for the log message.

        Returns:
            Path to the written file.
        """
        output_path = self._log_dir / filename
        output_path.write_text(json.dumps(config, indent=2, sort_keys=True, default=str))
        self.info(f"saved config to {output_path}", level=level)
        return output_path

    def save_text(self, content: str, filename: str) -> Path:
        """Persist arbitrary text to the log directory.

        Args:
            content: Text content to save.
            filename: Target filename inside the log directory.

        Returns:
            Path to the written file.
        """
        output_path = self._log_dir / filename
        output_path.write_text(content)
        return output_path

    # ------------------------------------------------------------------
    # Trace loading and scanning (resume support)
    # ------------------------------------------------------------------

    def load_trace(
        self,
        task_id: str,
        *,
        tool_path: list[str] | None = None,
        allow_partial: bool = False,
    ) -> TaskTrace | None:
        """Load a persisted trace from disk.

        Args:
            task_id: The task identifier.
            tool_path: Subdirectory path segments under the log directory.
                If ``None``, looks in the log root.
            allow_partial: If ``True``,
                falls back to the ``.partial.json`` file when the final trace file is not found.

        Returns:
            The loaded :class:`TaskTrace`,
            or ``None`` if no matching file exists or the file is malformed.
        """
        output_dir = self._log_dir.joinpath(*(tool_path or []))
        final_path = output_dir / f"{task_id}.json"
        if final_path.exists():
            return self._read_trace_file(final_path)
        if allow_partial:
            partial_path = output_dir / f"{task_id}.partial.json"
            if partial_path.exists():
                return self._read_trace_file(partial_path)
        return None

    def update_finished_trace_metadata(
        self,
        task_id: str,
        *,
        metadata: dict[str, Any],
        tool_path: list[str] | None = None,
        step_name: str | None = None,
        step_message: str = "",
        step_metadata: dict[str, Any] | None = None,
        turn_idx: int = 0,
    ) -> bool:
        """Merge metadata into an already-finished JSON trace.

        This is intended for orchestration-level decisions that are made only after
        an attempt trace has been finalized, such as cross-attempt retry gating.
        """
        self.drain()
        output_dir = self._log_dir.joinpath(*(tool_path or []))
        final_path = output_dir / f"{task_id}.json"
        if not final_path.exists():
            return False
        try:
            data = json.loads(final_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        trace_metadata = data.setdefault("metadata", {})
        if not isinstance(trace_metadata, dict):
            trace_metadata = {}
            data["metadata"] = trace_metadata
        trace_metadata.update(metadata)
        if step_name:
            steps = data.setdefault("steps", [])
            if isinstance(steps, list):
                steps.append(
                    {
                        "turn_idx": turn_idx,
                        "step_name": step_name,
                        "message": step_message,
                        "timestamp": _now_with_tz(),
                        "metadata": dict(step_metadata or metadata),
                    }
                )
        try:
            final_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            return False
        return True

    def scan_completed_task_ids(
        self,
        *,
        tool_path: list[str] | None = None,
        exclude_statuses: frozenset[str] = frozenset({"running"}),
    ) -> set[str]:
        """Scan a trace directory for completed task IDs.

        Args:
            tool_path: Subdirectory path segments under the log directory.
                If ``None``, scans the log root.
            exclude_statuses: Task statuses to exclude from the result set.
                Defaults to ``{"running"}``,
                so that in-flight or crashed tasks are not considered completed.

        Returns:
            Set of ``task_id`` strings for traces whose status is not excluded.
        """
        output_dir = self._log_dir.joinpath(*(tool_path or []))
        if not output_dir.exists():
            return set()
        completed: set[str] = set()
        for path in output_dir.glob("*.json"):
            if path.name.endswith(".partial.json"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                task_id = data.get("task_id")
                status = data.get("status", "running")
                if task_id and status not in exclude_statuses:
                    completed.add(task_id)
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        return completed

    @staticmethod
    def _read_trace_file(path: Path) -> TaskTrace | None:
        """Read and deserialize a trace JSON file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return TaskTrace.from_dict(data)
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def _get_live_trace(self, task_id: str) -> TaskTrace:
        try:
            return self._live_traces[task_id]
        except KeyError as exc:
            raise KeyError(f"Task '{task_id}' is not being tracked.") from exc

    def _maybe_incremental_flush(self, task_id: str) -> None:
        """Write a partial trace JSON if enough steps have accumulated since the last flush."""
        if self._incremental_flush_steps <= 0 or not self._persist_json:
            return
        self._step_counters[task_id] = self._step_counters.get(task_id, 0) + 1
        if self._step_counters[task_id] < self._incremental_flush_steps:
            return
        self._step_counters[task_id] = 0
        trace = self._live_traces.get(task_id)
        if trace is None:
            return
        output_dir = self._log_dir.joinpath(*trace.tool_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        partial_path = output_dir / f"{trace.task_id}.partial.json"
        partial_path.write_text(json.dumps(asdict(trace), indent=2, sort_keys=True))

    def info(self, message: str, *, level: int = 0) -> None:
        self._logger.info("%s%s", "  " * level, message)

    def warning(self, message: str, *, level: int = 0) -> None:
        self._logger.warning("%s%s", "  " * level, message)

    def error(self, message: str, *, level: int = 0) -> None:
        self._logger.error("%s%s", "  " * level, message)
