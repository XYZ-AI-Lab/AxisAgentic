# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Replay a persisted attempt trace into the model-visible conversation.

The attempt JSON's ``conversation`` is an append-only event log: normal
messages plus ``ConversationRuntime`` markers (``rollback`` / ``compaction``).
Replaying the markers in order reconstructs exactly what the model saw at any
point of the rollout.

Usage:
    # Timeline: each marker event and the resulting visible-message count.
    python -m recipe.web_search.runners.replay_trace <attempt.json>

    # Reconstruct the visible conversation after replaying the first N trace messages.
    python -m recipe.web_search.runners.replay_trace <attempt.json> --at 43

    # Reconstruct and print/export the final visible conversation.
    python -m recipe.web_search.runners.replay_trace <attempt.json> --show-visible
    python -m recipe.web_search.runners.replay_trace <attempt.json> --dump visible.json

    # Verify that every recorded model request for the task can be reconstructed
    # from a trace prefix. This requires --model_request_logging; keep_tool_result
    # and tool_result_role are read from run_metadata.json unless overridden.
    python -m recipe.web_search.runners.replay_trace <attempt.json> --verify-requests <run_dir>/model_requests/inference

Exit codes for --verify-requests: 0 = all requests reconstructed, 1 = some
requests unmatched, 2 = nothing verified (no requests matched this task — an
error, not a pass).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic.contracts import ConversationMessage, MessageRole
from agentic.contracts.markers import (
    parse_compaction_marker,
    parse_discard_all_marker,
    splice_compaction,
    splice_discard_all,
)
from agentic.orchestration.task_orchestrator import (
    _conversation_dicts_to_messages,
    _materialize_visible_conversation,
    _parse_rollback_marker,
)
from recipe.web_search.agent.runtime import TOOL_RESULT_OMITTED_PLACEHOLDER

if TYPE_CHECKING:
    from collections.abc import Iterator


def _head(text: str | None, width: int = 72) -> str:
    collapsed = " ".join((text or "").split())
    return collapsed[:width] + ("…" if len(collapsed) > width else "")


def _describe_message(message: ConversationMessage) -> str:
    if message.tool_calls:
        names = ",".join(tc.function.name for tc in message.tool_calls)
        return f"[tool_calls: {names}] {_head(message.content, 40)}"
    return _head(message.content)


def _print_timeline(messages: list[ConversationMessage]) -> None:
    for index, message in enumerate(messages):
        if message.role != MessageRole.CONVERSATION_RUNTIME:
            continue
        before = len(_materialize_visible_conversation(messages[:index]))
        after = len(_materialize_visible_conversation(messages[: index + 1]))
        rollback_count, reason = _parse_rollback_marker(message)
        if rollback_count:
            print(f"  [{index:4d}] ROLLBACK   visible {before} -> {after}  (hide {rollback_count}, reason={reason})")
            continue
        compaction = parse_compaction_marker(message)
        if compaction is not None:
            print(
                f"  [{index:4d}] COMPACTION visible {before} -> {after}  "
                f"(splice [{compaction.prefix_len}, {compaction.prefix_len + compaction.compressed_count}) "
                f"-> summary {len(compaction.summary)} chars)"
            )
            continue
        discard = parse_discard_all_marker(message)
        if discard is not None:
            print(f"  [{index:4d}] DISCARD_ALL visible {before} -> {after}  (truncate to prefix {discard.prefix_len})")
            continue
        print(f"  [{index:4d}] UNKNOWN runtime marker name={message.name!r} (ignored by replay)")
    final_visible = len(_materialize_visible_conversation(messages))
    print(f"  final visible: {final_visible} messages / trace total: {len(messages)}")


def _print_visible(visible: list[ConversationMessage], *, full: bool) -> None:
    for index, message in enumerate(visible):
        if full:
            print(f"--- [{index}] {message.role.value} " + "-" * 50)
            if message.tool_calls:
                for tc in message.tool_calls:
                    print(f"[tool_call] {tc.function.name}({json.dumps(tc.function.arguments, ensure_ascii=False)})")
            if message.content:
                print(message.content)
        else:
            print(f"  [{index:3d}] {message.role.value:10s} {_describe_message(message)}")


# ---------------------------------------------------------------------------
# Request verification
# ---------------------------------------------------------------------------


def _canonical_arguments(arguments: Any) -> str:
    """Canonicalize tool-call arguments (dict, or JSON string on the wire)."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, json.JSONDecodeError):
            return arguments
    try:
        return json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(arguments)


def _signature(items: list[tuple[str, str, str]]) -> bytes:
    digest = hashlib.sha256()
    for role, content, tool_calls in items:
        digest.update(role.encode())
        digest.update(b"\x00")
        digest.update(content.encode())
        digest.update(b"\x00")
        digest.update(tool_calls.encode())
        digest.update(b"\x01")
    return digest.digest()


def _render_request_view(
    visible: list[ConversationMessage],
    *,
    keep_tool_result: int,
    tool_result_role: str,
) -> list[tuple[str, str, str]]:
    """Render a replayed visible conversation as the wire view the model saw.

    Mirrors ``WebSearchConversationRuntime._build_visible_conversation``
    (keep_tool_result truncation) and
    ``TaskOrchestrator._model_visible_conversation`` (tool_result_role=user).
    """
    tool_indices = [i for i, m in enumerate(visible) if m.role == MessageRole.TOOL]
    truncated: set[int] = set()
    if keep_tool_result == 0:
        truncated = set(tool_indices)
    elif keep_tool_result > 0 and len(tool_indices) > keep_tool_result:
        truncated = set(tool_indices[:-keep_tool_result])

    rendered: list[tuple[str, str, str]] = []
    for index, message in enumerate(visible):
        role = message.role.value
        content = TOOL_RESULT_OMITTED_PLACEHOLDER if index in truncated else (message.content or "")
        if message.role == MessageRole.TOOL and tool_result_role == "user":
            role = "user"
        tool_calls = ""
        if message.tool_calls:
            tool_calls = "|".join(f"{tc.function.name}({_canonical_arguments(tc.function.arguments)})" for tc in message.tool_calls)
        rendered.append((role, content, tool_calls))
    return rendered


def _request_signature(request_messages: list[dict[str, Any]]) -> bytes:
    items: list[tuple[str, str, str]] = []
    for message in request_messages:
        tool_calls = ""
        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls:
            parts = []
            for tc in raw_tool_calls:
                function = tc.get("function", {})
                parts.append(f"{function.get('name', '')}({_canonical_arguments(function.get('arguments'))})")
            tool_calls = "|".join(parts)
        items.append((str(message.get("role", "")), message.get("content") or "", tool_calls))
    return _signature(items)


def _first_user_content(messages: list[dict[str, Any]]) -> str | None:
    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str) and message["content"]:
            return message["content"]
    return None


def _iter_prefix_visibles(messages: list[ConversationMessage]) -> Iterator[list[ConversationMessage]]:
    """Yield the visible conversation after each trace-message prefix, incrementally.

    The yielded list must be consumed immediately (it is reused/rebound in place).
    """
    visible: list[ConversationMessage] = []
    yield visible
    for message in messages:
        if message.role == MessageRole.CONVERSATION_RUNTIME:
            rollback_count, _reason = _parse_rollback_marker(message)
            if rollback_count:
                while rollback_count > 0 and visible:
                    if visible[-1].role == MessageRole.SYSTEM:
                        break
                    visible.pop()
                    rollback_count -= 1
                yield visible
                continue
            compaction = parse_compaction_marker(message)
            if compaction is not None:
                spliced = splice_compaction(visible, compaction)
                if spliced is not None:
                    visible = spliced
                yield visible
                continue
            discard = parse_discard_all_marker(message)
            if discard is not None:
                truncated = splice_discard_all(visible, discard)
                if truncated is not None:
                    visible = truncated
            yield visible
            continue
        visible.append(message)
        yield visible


def _load_run_render_defaults(attempt_json: Path) -> tuple[int | None, str | None]:
    """Best-effort read of keep_tool_result / tool_result_role from run_metadata.json."""
    for candidate_dir in (attempt_json.parent.parent, attempt_json.parent):
        metadata_path = candidate_dir / "run_metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None, None
        cli_args = metadata.get("cli_args")
        if not isinstance(cli_args, dict):
            return None, None
        keep = cli_args.get("keep_tool_result")
        role = cli_args.get("tool_result_role")
        return (keep if isinstance(keep, int) and not isinstance(keep, bool) else None, role if isinstance(role, str) else None)
    return None, None


def _verify_requests(
    messages: list[ConversationMessage],
    requests_dir: Path,
    task_user_content: str,
    *,
    keep_tool_result: int,
    tool_result_role: str,
) -> int:
    """Check every logged model request reconstructs from some trace prefix.

    Returns a process exit code: 0 when every request for this task matches a
    prefix replay, 1 when some requests are unmatched, 2 when no request
    matched the task filter at all (nothing verified — treated as an error).
    """
    prefix_signatures = set()
    for visible in _iter_prefix_visibles(messages):
        prefix_signatures.add(_signature(_render_request_view(visible, keep_tool_result=keep_tool_result, tool_result_role=tool_result_role)))

    part_files = sorted(requests_dir.glob("*.jsonl"))
    if not part_files:
        print(f"ERROR: no *.jsonl request logs found under {requests_dir}")
        return 2

    total = matched = malformed_lines = 0
    unmatched: list[str] = []
    for part in part_files:
        with part.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue
                request_messages = record.get("request", {}).get("messages", [])
                if not request_messages or _first_user_content(request_messages) != task_user_content:
                    continue
                total += 1
                if _request_signature(request_messages) in prefix_signatures:
                    matched += 1
                else:
                    started_at = record.get("started_at")
                    unmatched.append(started_at if isinstance(started_at, str) else "<no started_at>")

    print(f"requests for this task: {total}, reconstructed from trace prefixes: {matched}")
    if malformed_lines:
        print(f"skipped {malformed_lines} malformed request-log line(s) (e.g. a partially-written tail of a live run)")
    if total == 0:
        print("ERROR: no logged request matched this task's user message — nothing was verified.")
        print("Check the requests dir, and that the attempt and the request log belong to the same run.")
        return 2
    if unmatched:
        timestamps = sorted(t for t in unmatched if t != "<no started_at>")
        span = f" (started {timestamps[0]} ~ {timestamps[-1]})" if timestamps else ""
        print(f"unmatched: {len(unmatched)}{span}")
        print("note: unmatched requests usually belong to another attempt of the same task (e.g. an in-flight retry).")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("attempt_json", type=Path, help="Path to <task>_attempt-<n>.json")
    parser.add_argument("--at", type=int, default=None, help="Replay only the first N trace messages (default: all)")
    parser.add_argument("--show-visible", action="store_true", help="Print the reconstructed visible conversation in full")
    parser.add_argument("--dump", type=Path, default=None, help="Write the reconstructed visible conversation to a JSON file")
    parser.add_argument("--verify-requests", type=Path, default=None, help="model_requests/inference dir to verify against")
    parser.add_argument(
        "--keep-tool-result",
        type=int,
        default=None,
        help="keep_tool_result used by the run (for --verify-requests; default: auto from run_metadata.json, else -1)",
    )
    parser.add_argument(
        "--tool-result-role",
        choices=["tool", "user"],
        default=None,
        help="tool_result_role used by the run (for --verify-requests; default: auto from run_metadata.json, else 'tool')",
    )
    args = parser.parse_args()

    attempt = json.loads(args.attempt_json.read_text())
    conversation = attempt["conversation"]
    messages = _conversation_dicts_to_messages(conversation, include_runtime=True)
    cut = len(messages) if args.at is None else max(0, min(args.at, len(messages)))
    visible = _materialize_visible_conversation(messages[:cut])

    print(f"trace: {args.attempt_json} (status={attempt.get('status')})")
    print(f"trace messages: {len(messages)}, replay cut: {cut}, visible: {len(visible)}\n")

    if args.dump is not None:
        args.dump.write_text(json.dumps([m.to_model_message() for m in visible], ensure_ascii=False, indent=2))
        print(f"visible conversation written to {args.dump}")

    if args.verify_requests is not None:
        task_user_content = _first_user_content(conversation)
        if task_user_content is None:
            print("ERROR: trace contains no user message; cannot match requests to this task.")
            return 2
        keep_default, role_default = _load_run_render_defaults(args.attempt_json)
        keep_tool_result = args.keep_tool_result if args.keep_tool_result is not None else (keep_default if keep_default is not None else -1)
        tool_result_role = args.tool_result_role or role_default or "tool"
        print(f"verify render: keep_tool_result={keep_tool_result}, tool_result_role={tool_result_role}")
        return _verify_requests(
            messages[:cut],
            args.verify_requests,
            task_user_content,
            keep_tool_result=keep_tool_result,
            tool_result_role=tool_result_role,
        )

    print("== runtime marker timeline ==")
    _print_timeline(messages[:cut])
    print("\n== visible conversation ==")
    _print_visible(visible, full=args.show_visible)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
