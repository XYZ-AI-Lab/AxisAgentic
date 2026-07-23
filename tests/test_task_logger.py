import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agentic.contracts.messages import ToolResultReason, ToolResultStatus
from agentic.observability import TaskLogger, TaskTrace, ToolTrace


def test_task_logger_tracks_live_traces_by_task_id() -> None:
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="test-tracks", log_dir=tmp_dir)
        logger.start_task("task-123", "solve it", metadata={"source": "test"})
        logger.add_step("task-123", 0, "conversation.init", "initialized")
        logger.append_conversation("task-123", [{"role": "user", "content": "solve it"}])
        logger.append_conversation("task-123", [{"role": "assistant", "content": "ok"}])
        logger.add_tool_trace(
            "task-123",
            ToolTrace(
                tool_name="echo",
                call_id="call-1",
                arguments={"text": "done"},
                status=ToolResultStatus.SUCCESS,
                latency_ms=1.25,
            ),
        )
        logger.finish_task("task-123", status="completed", metadata={"done": True})

        trace_path = Path(tmp_dir) / "task-123.json"
        payload = json.loads(trace_path.read_text(encoding="utf-8"))

        assert payload["task_id"] == "task-123"
        assert payload["status"] == "completed"
        assert payload["metadata"]["source"] == "test"
        assert payload["metadata"]["done"] is True
        assert payload["conversation"][0]["content"] == "solve it"
        assert payload["conversation"][1]["content"] == "ok"
        assert payload["tool_calls"][0]["tool_name"] == "echo"
        # TaskLogger self-reports its own bookkeeping / JSON-write time so viz
        # can tell whether TaskLogger IO is a meaningful slice of rollout wall-clock.
        assert "task_logger_bookkeeping_ms" in payload["metadata"]
        assert "task_logger_finish_ms" in payload["metadata"]
        assert payload["metadata"]["task_logger_bookkeeping_ms"] >= 0
        assert payload["metadata"]["task_logger_finish_ms"] >= 0


def test_task_logger_rejects_updates_after_finish() -> None:
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="test-reject", log_dir=tmp_dir, persist_json=False)
        logger.start_task("task-123", "solve it")
        logger.finish_task("task-123", status="completed")

        exc_message: str | None = None
        try:
            logger.add_step("task-123", 1, "unexpected", "should fail")
        except KeyError as exc:
            exc_message = str(exc)
        assert exc_message is not None
        assert "task-123" in exc_message


def test_task_logger_writes_json_to_subdirectory_based_on_tool_path() -> None:
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="test-subdir", log_dir=tmp_dir)
        logger.start_task("nested-1", "do work", tool_path=["delegate", "search"])
        logger.finish_task("nested-1", status="completed")

        expected = Path(tmp_dir) / "delegate" / "search" / "nested-1.json"
        assert expected.exists()
        payload = json.loads(expected.read_text(encoding="utf-8"))
        assert payload["tool_path"] == ["delegate", "search"]


def test_task_logger_fork_live_trace_preserves_prefix_independently() -> None:
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="test-fork", log_dir=tmp_dir)
        logger.start_task("source", "do work", tool_path=["web-search-test"])
        logger.append_conversation("source", [{"role": "user", "content": "shared prefix"}])

        logger.fork_live_trace("source", "branch", metadata={"attempt_budget_role": "budget_final_branch"})
        logger.append_conversation("branch", [{"role": "assistant", "content": "branch only"}])
        logger.finish_task("branch", status="completed")
        logger.finish_task("source", status="completed")

        source_payload = json.loads((Path(tmp_dir) / "web-search-test" / "source.json").read_text(encoding="utf-8"))
        branch_payload = json.loads((Path(tmp_dir) / "web-search-test" / "branch.json").read_text(encoding="utf-8"))

        assert source_payload["conversation"] == [{"role": "user", "content": "shared prefix"}]
        assert branch_payload["conversation"] == [
            {"role": "user", "content": "shared prefix"},
            {"role": "assistant", "content": "branch only"},
        ]
        assert branch_payload["metadata"]["attempt_budget_role"] == "budget_final_branch"
        assert branch_payload["tool_path"] == ["web-search-test"]


def test_task_logger_stores_emoji_in_tool_trace() -> None:
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="test-emoji", log_dir=tmp_dir)
        logger.start_task("task-emoji", "test")
        logger.add_tool_trace(
            "task-emoji",
            ToolTrace(tool_name="calc", call_id="c1", arguments={}, status=ToolResultStatus.SUCCESS, latency_ms=1.0, emoji="🧮"),
        )
        logger.finish_task("task-emoji", status="completed")

        payload = json.loads((Path(tmp_dir) / "task-emoji.json").read_text(encoding="utf-8"))
        assert payload["tool_calls"][0]["emoji"] == "🧮"


def _raw_cache_metadata(
    *,
    status: str,
    key_hash: str,
    hit_count: int,
    miss_count: int,
    bypass_count: int = 0,
    put_count: int = 0,
    saved_jina_request: bool = False,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "raw_scrape_cache_status": status,
        "raw_scrape_cache_provider": "web_search",
        "raw_scrape_cache_scope": "task",
        "raw_scrape_cache_counter_scope": "tool_instance_task",
        "raw_scrape_cache_normalize_url": False,
        "raw_scrape_cache_normalize_policy": "none",
        "raw_scrape_cache_key_hash": key_hash,
        "raw_scrape_cache_hit_count": hit_count,
        "raw_scrape_cache_miss_count": miss_count,
        "raw_scrape_cache_bypass_count": bypass_count,
        "raw_scrape_cache_put_count": put_count,
    }
    if saved_jina_request:
        metadata["raw_scrape_cache_saved_jina_request"] = True
    return metadata


def _scrape_tool_trace(call_id: str, metadata: dict[str, object]) -> ToolTrace:
    return ToolTrace(
        tool_name="scrape_and_extract_info",
        call_id=call_id,
        arguments={"url": "https://example.com"},
        status=ToolResultStatus.SUCCESS,
        latency_ms=1.0,
        content="ok",
        metadata=metadata,
    )


def _finish_trace_with_tool_calls(tool_calls: list[ToolTrace]) -> dict[str, Any]:
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="test-raw-cache-summary", log_dir=tmp_dir)
        logger.start_task("raw-cache-summary", "test")
        for tool_call in tool_calls:
            logger.add_tool_trace("raw-cache-summary", tool_call)
        logger.finish_task("raw-cache-summary", status="completed")
        return json.loads((Path(tmp_dir) / "raw-cache-summary.json").read_text(encoding="utf-8"))


def test_task_logger_adds_raw_scrape_cache_summary_for_hit_and_miss() -> None:
    payload = _finish_trace_with_tool_calls(
        [
            _scrape_tool_trace(
                "miss-1",
                _raw_cache_metadata(
                    status="miss",
                    key_hash="a" * 64,
                    hit_count=0,
                    miss_count=1,
                    put_count=1,
                ),
            ),
            _scrape_tool_trace(
                "hit-1",
                _raw_cache_metadata(
                    status="hit",
                    key_hash="a" * 64,
                    hit_count=1,
                    miss_count=1,
                    put_count=1,
                    saved_jina_request=True,
                ),
            ),
        ],
    )

    summary = payload["metadata"]["raw_scrape_cache_summary"]
    assert summary == {
        "schema_version": 1,
        "observed": True,
        "tool_call_count": 2,
        "status_counts": {"hit": 1, "miss": 1, "bypass": 0},
        "hit_count_max": 1,
        "miss_count_max": 1,
        "bypass_count_max": 0,
        "put_count_max": 1,
        "saved_jina_request_count": 1,
        "key_hash_count": 2,
        "unique_key_hash_count": 1,
        "duplicate_key_hash_count": 1,
        "no_hit_evidence": False,
        "counter_scope": "tool_instance_task",
        "provider": "web_search",
        "scope": "task",
        "normalize_url": False,
        "normalize_policy": "none",
    }


def test_task_logger_raw_scrape_cache_summary_marks_no_hit_evidence() -> None:
    payload = _finish_trace_with_tool_calls(
        [
            _scrape_tool_trace(
                "miss-1",
                _raw_cache_metadata(status="miss", key_hash="a" * 64, hit_count=0, miss_count=1, put_count=1),
            ),
            _scrape_tool_trace(
                "bypass-1",
                _raw_cache_metadata(
                    status="bypass",
                    key_hash="b" * 64,
                    hit_count=0,
                    miss_count=2,
                    bypass_count=1,
                    put_count=1,
                ),
            ),
        ],
    )

    summary = payload["metadata"]["raw_scrape_cache_summary"]
    assert summary["observed"] is True
    assert summary["status_counts"] == {"hit": 0, "miss": 1, "bypass": 1}
    assert summary["key_hash_count"] == 2
    assert summary["unique_key_hash_count"] == 2
    assert summary["duplicate_key_hash_count"] == 0
    assert summary["no_hit_evidence"] is True
    assert summary["miss_count_max"] == 2
    assert summary["bypass_count_max"] == 1
    assert summary["put_count_max"] == 1


def test_task_logger_raw_scrape_cache_summary_duplicate_no_hit_is_not_no_hit_evidence() -> None:
    payload = _finish_trace_with_tool_calls(
        [
            _scrape_tool_trace(
                "miss-1",
                _raw_cache_metadata(status="miss", key_hash="a" * 64, hit_count=0, miss_count=1, put_count=1),
            ),
            _scrape_tool_trace(
                "miss-2",
                _raw_cache_metadata(status="miss", key_hash="a" * 64, hit_count=0, miss_count=2, put_count=2),
            ),
        ],
    )

    summary = payload["metadata"]["raw_scrape_cache_summary"]
    assert summary["status_counts"] == {"hit": 0, "miss": 2, "bypass": 0}
    assert summary["duplicate_key_hash_count"] == 1
    assert summary["no_hit_evidence"] is False


def test_task_logger_raw_scrape_cache_summary_handles_no_raw_metadata() -> None:
    payload = _finish_trace_with_tool_calls(
        [
            ToolTrace(
                tool_name="echo",
                call_id="echo-1",
                arguments={},
                status=ToolResultStatus.SUCCESS,
                latency_ms=1.0,
                content="ok",
                metadata={"success": True},
            ),
        ],
    )

    summary = payload["metadata"]["raw_scrape_cache_summary"]
    assert summary == {
        "schema_version": 1,
        "observed": False,
        "tool_call_count": 0,
        "status_counts": {"hit": 0, "miss": 0, "bypass": 0},
        "hit_count_max": 0,
        "miss_count_max": 0,
        "bypass_count_max": 0,
        "put_count_max": 0,
        "saved_jina_request_count": 0,
        "key_hash_count": 0,
        "unique_key_hash_count": 0,
        "duplicate_key_hash_count": 0,
        "no_hit_evidence": False,
        "counter_scope": None,
        "provider": None,
        "scope": None,
        "normalize_url": None,
        "normalize_policy": None,
    }


def test_task_logger_tool_failure_log_includes_error_preview() -> None:
    """Failing tool traces should surface their error content in the log line."""
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="test-err-preview", log_dir=tmp_dir)
        logger.start_task("fail-1", "test")
        logger.add_tool_trace(
            "fail-1",
            ToolTrace(
                tool_name="scrape_and_extract_info",
                call_id="c-json",
                arguments={"url": "https://x"},
                status=ToolResultStatus.FAILED,
                latency_ms=32.1,
                content='{"success": false, "url": "https://x", "error": "Extract Info: HTTP 404", "scrape_stats": {}}',
            ),
        )
        logger.add_tool_trace(
            "fail-1",
            ToolTrace(
                tool_name="echo",
                call_id="c-plain",
                arguments={},
                status=ToolResultStatus.REJECTED,
                latency_ms=0.0,
                reason=ToolResultReason.CONSECUTIVE_SAME_ARGS,
                content="Same arguments repeated too many times.",
            ),
        )
        logger.add_tool_trace(
            "fail-1",
            ToolTrace(
                tool_name="noop",
                call_id="c-ok",
                arguments={},
                status=ToolResultStatus.SUCCESS,
                latency_ms=1.0,
                content='{"success": true, "payload": "..."}',
            ),
        )
        logger.finish_task("fail-1", status="completed")

        log_text = (Path(tmp_dir) / "test-err-preview.log").read_text(encoding="utf-8")

        # FAILED: JSON error field gets surfaced.
        assert "error=Extract Info: HTTP 404" in log_text
        # REJECTED: reason label kept, plain-text content still attached.
        assert f"reason={ToolResultReason.CONSECUTIVE_SAME_ARGS}" in log_text
        assert "error=Same arguments repeated too many times." in log_text
        # SUCCESS: content must NOT leak into the log line.
        assert "payload" not in log_text


def test_task_logger_creates_file_handler_at_root_log_dir() -> None:
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="test-filelog", log_dir=tmp_dir)
        logger.start_task("task-fh", "test")
        logger.finish_task("task-fh", status="completed")

        log_file = Path(tmp_dir) / "test-filelog.log"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "start task task-fh" in content
        assert "finish task task-fh" in content


def test_task_logger_persists_utc_plus_8_offsets_in_json() -> None:
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="test-tz", log_dir=tmp_dir)
        logger.start_task("task-tz", "test")
        logger.add_step("task-tz", 0, "conversation.init", "initialized")
        logger.finish_task("task-tz", status="completed")

        payload = json.loads((Path(tmp_dir) / "task-tz.json").read_text(encoding="utf-8"))
        assert payload["started_at"].endswith("+08:00")
        assert payload["ended_at"].endswith("+08:00")
        assert payload["steps"][0]["timestamp"].endswith("+08:00")


# ---------------------------------------------------------------------------
# TaskTrace.from_dict() tests
# ---------------------------------------------------------------------------


def test_task_trace_from_dict_roundtrip() -> None:
    """Roundtrip: asdict → from_dict should produce an equivalent TaskTrace."""
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="test-roundtrip", log_dir=tmp_dir)
        logger.start_task("rt-1", "some task", metadata={"key": "val"})
        logger.add_step("rt-1", 0, "init", "initialized")
        logger.append_conversation("rt-1", [{"role": "user", "content": "hello"}])
        logger.add_tool_trace(
            "rt-1",
            ToolTrace(
                tool_name="search",
                call_id="c-1",
                arguments={"query": "test"},
                status=ToolResultStatus.SUCCESS,
                latency_ms=42.5,
                reason=None,
                content="found it",
            ),
        )
        logger.add_token_usage("rt-1", input_tokens=10, output_tokens=5, total_tokens=15)
        logger.finish_task("rt-1", status="completed")

        # Load the JSON that was written
        trace_path = Path(tmp_dir) / "rt-1.json"
        data = json.loads(trace_path.read_text(encoding="utf-8"))

        # Reconstruct and compare
        restored = TaskTrace.from_dict(data)
        assert restored.task_id == "rt-1"
        assert restored.status == "completed"
        assert restored.metadata["key"] == "val"
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].tool_name == "search"
        assert restored.tool_calls[0].status == ToolResultStatus.SUCCESS
        assert restored.tool_calls[0].latency_ms == 42.5
        assert restored.tool_calls[0].content == "found it"
        assert restored.token_usage.input_tokens == 10
        assert restored.token_usage.output_tokens == 5
        assert restored.token_usage.total_tokens == 15
        assert len(restored.conversation) == 1
        assert restored.conversation[0]["content"] == "hello"
        assert restored.ended_at is not None


def test_task_trace_from_dict_handles_tool_result_reason() -> None:
    """from_dict should correctly reconstruct ToolResultReason enums."""
    data = {
        "task_id": "reason-test",
        "task_input": "test",
        "status": "completed",
        "tool_calls": [
            {
                "tool_name": "calc",
                "call_id": "c-1",
                "arguments": {},
                "status": "rejected",
                "latency_ms": 1.0,
                "reason": "unknown_tool",
                "content": "",
                "emoji": "🔧",
                "metadata": {},
            },
        ],
    }
    trace = TaskTrace.from_dict(data)
    assert trace.tool_calls[0].status == ToolResultStatus.REJECTED
    assert trace.tool_calls[0].reason == ToolResultReason.UNKNOWN_TOOL


# ---------------------------------------------------------------------------
# TaskLogger.load_trace() tests
# ---------------------------------------------------------------------------


def test_load_trace_returns_finished_trace() -> None:
    """load_trace should return a TaskTrace for a completed task."""
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="test-load", log_dir=tmp_dir)
        logger.start_task("load-1", "task text", metadata={"src": "test"})
        logger.finish_task("load-1", status="assistant_final_answer")

        trace = logger.load_trace("load-1")
        assert trace is not None
        assert trace.task_id == "load-1"
        assert trace.status == "assistant_final_answer"
        assert trace.metadata["src"] == "test"
        assert trace.ended_at is not None


def test_load_trace_returns_none_for_missing() -> None:
    """load_trace should return None when no trace file exists."""
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="test-missing", log_dir=tmp_dir)
        assert logger.load_trace("nonexistent") is None


def test_load_trace_with_tool_path() -> None:
    """load_trace should search in the correct subdirectory."""
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="test-tp", log_dir=tmp_dir)
        logger.start_task("tp-1", "work", tool_path=["orch", "sub"])
        logger.finish_task("tp-1", status="done")

        # Should NOT find it without tool_path
        assert logger.load_trace("tp-1") is None
        # Should find it with the correct tool_path
        trace = logger.load_trace("tp-1", tool_path=["orch", "sub"])
        assert trace is not None
        assert trace.task_id == "tp-1"


def test_load_trace_reads_partial_when_allowed() -> None:
    """load_trace with allow_partial=True should fall back to .partial.json."""
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="test-partial", log_dir=tmp_dir, incremental_flush_steps=1)
        logger.start_task("p-1", "partial task")
        logger.add_step("p-1", 0, "init", "started")  # triggers incremental flush

        # Partial file should exist, final should not
        assert not (Path(tmp_dir) / "p-1.json").exists()
        assert (Path(tmp_dir) / "p-1.partial.json").exists()

        # Without allow_partial → None
        assert logger.load_trace("p-1") is None
        # With allow_partial → returns the partial trace
        trace = logger.load_trace("p-1", allow_partial=True)
        assert trace is not None
        assert trace.task_id == "p-1"
        assert trace.status == "running"

        # Cleanup: finish the task so logger doesn't complain
        logger.finish_task("p-1", status="done")


# ---------------------------------------------------------------------------
# TaskLogger.scan_completed_task_ids() tests
# ---------------------------------------------------------------------------


def test_scan_completed_task_ids_basic() -> None:
    """scan_completed_task_ids should return IDs of completed tasks only."""
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="test-scan", log_dir=tmp_dir)

        # Create three tasks with different statuses
        logger.start_task("done-1", "t1")
        logger.finish_task("done-1", status="assistant_final_answer")

        logger.start_task("done-2", "t2")
        logger.finish_task("done-2", status="terminated_turn_limit")

        # Write a "running" partial file manually (simulates a crash)
        running_data = {"task_id": "crashed-1", "status": "running"}
        (Path(tmp_dir) / "crashed-1.json").write_text(json.dumps(running_data))

        completed = logger.scan_completed_task_ids()
        assert completed == {"done-1", "done-2"}


def test_scan_completed_task_ids_with_tool_path() -> None:
    """scan_completed_task_ids should scan the correct subdirectory."""
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="test-scan-tp", log_dir=tmp_dir)

        logger.start_task("sub-1", "t", tool_path=["myorch"])
        logger.finish_task("sub-1", status="done")

        # Without tool_path → empty (files are in subdirectory)
        assert logger.scan_completed_task_ids() == set()
        # With correct tool_path → found
        assert logger.scan_completed_task_ids(tool_path=["myorch"]) == {"sub-1"}


def test_scan_completed_excludes_partial_files() -> None:
    """scan_completed_task_ids should not include .partial.json files."""
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="test-excl", log_dir=tmp_dir)
        # Write a partial file manually
        partial_data = {"task_id": "partial-1", "status": "assistant_final_answer", "ended_at": "2026-01-01T00:00:00"}
        (Path(tmp_dir) / "partial-1.partial.json").write_text(json.dumps(partial_data))

        assert logger.scan_completed_task_ids() == set()


def test_scan_completed_custom_exclude_statuses() -> None:
    """scan_completed_task_ids should respect custom exclude_statuses."""
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="test-custom", log_dir=tmp_dir)

        logger.start_task("ok-1", "t")
        logger.finish_task("ok-1", status="assistant_final_answer")

        logger.start_task("err-1", "t")
        logger.finish_task("err-1", status="model_error: timeout")

        # Exclude model errors
        completed = logger.scan_completed_task_ids(exclude_statuses=frozenset({"running", "model_error: timeout"}))
        assert completed == {"ok-1"}


def test_scan_completed_handles_empty_directory() -> None:
    """scan_completed_task_ids should return empty set for nonexistent directory."""
    with TemporaryDirectory() as tmp_dir:
        logger = TaskLogger(name="test-empty", log_dir=tmp_dir)
        assert logger.scan_completed_task_ids(tool_path=["no", "such", "dir"]) == set()
