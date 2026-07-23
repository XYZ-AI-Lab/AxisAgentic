# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentic.contracts.markers import parse_compaction_marker_dict, parse_discard_all_marker_dict


@dataclass(frozen=True)
class SwiftAgentExportConfig:
    include_reasoning: bool = True
    skip_empty_reasoning: bool = True
    wrap_non_json_tool_response: bool = True
    include_visible_thought: bool = True
    apply_runtime_visibility: bool = True
    include_metadata: bool = False
    ensure_ascii: bool = False


@dataclass(frozen=True)
class SwiftAgentExportWarning:
    code: str
    message: str
    path: str


@dataclass(frozen=True)
class SwiftAgentExportResult:
    sample: dict[str, Any]
    warnings: list[SwiftAgentExportWarning] = field(default_factory=list)


def conversation_to_swift_agent_sample(
    *,
    conversation: list[dict[str, Any]],
    tools: str | list[dict[str, Any]] | None,
    sample_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    config: SwiftAgentExportConfig | None = None,
) -> SwiftAgentExportResult:
    cfg = config or SwiftAgentExportConfig()
    warnings: list[SwiftAgentExportWarning] = []
    sample: dict[str, Any] = {
        "tools": _tools_to_json_string(tools, warnings=warnings, ensure_ascii=cfg.ensure_ascii),
        "messages": [],
    }
    if sample_id is not None:
        sample["id"] = sample_id
    if cfg.include_metadata and metadata:
        sample["metadata"] = metadata

    if cfg.apply_runtime_visibility:
        visible_conversation = _apply_runtime_visibility(conversation, warnings=warnings)
    else:
        visible_conversation = list(enumerate(conversation))

    for idx, message in visible_conversation:
        role = str(message.get("role", ""))
        path = f"messages[{idx}]"
        if role in {"system", "user"}:
            sample["messages"].append({"role": role, "content": _string_content(message.get("content"))})
        elif role == "assistant":
            sample["messages"].extend(_assistant_to_swift_messages(message, path=path, config=cfg, warnings=warnings))
        elif role == "tool":
            sample["messages"].append(_tool_response_to_swift_message(message, path=path, config=cfg, warnings=warnings))
        else:
            warnings.append(SwiftAgentExportWarning("unsupported_role_skipped", f"Skipped unsupported role {role!r}.", path))

    validate_swift_agent_sample(sample)
    return SwiftAgentExportResult(sample=sample, warnings=warnings)


def validate_swift_agent_sample(sample: dict[str, Any]) -> None:
    if not isinstance(sample.get("messages"), list):
        msg = "Swift Agent sample requires a list-valued 'messages' field."
        raise TypeError(msg)
    tools = sample.get("tools")
    parsed_tools = _parse_json_string(tools)
    if not isinstance(parsed_tools, list):
        msg = "Swift Agent sample requires 'tools' to be a JSON string containing a list."
        raise TypeError(msg)
    for idx, message in enumerate(sample["messages"]):
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool_call", "tool_response", "tool"}:
            msg = f"Unsupported Swift Agent message role at messages[{idx}]: {role!r}"
            raise ValueError(msg)
        content = message.get("content")
        if not isinstance(content, str):
            msg = f"Swift Agent message content must be a string at messages[{idx}]."
            raise TypeError(msg)
        if role == "tool_call":
            parsed = _parse_json_string(content)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("name"), str) or "arguments" not in parsed:
                msg = f"tool_call content must be a JSON string with 'name' and 'arguments' at messages[{idx}]."
                raise ValueError(msg)
        if role in {"tool_response", "tool"} and _parse_json_string(content) is None:
            msg = f"{role} content must be a valid JSON string at messages[{idx}]."
            raise ValueError(msg)


def _apply_runtime_visibility(
    conversation: list[dict[str, Any]],
    *,
    warnings: list[SwiftAgentExportWarning],
) -> list[tuple[int, dict[str, Any]]]:
    visible: list[tuple[int, dict[str, Any]]] = []
    for idx, message in enumerate(conversation):
        role = str(message.get("role", ""))
        if role != "ConversationRuntime":
            visible.append((idx, message))
            continue

        path = f"messages[{idx}]"
        rollback_count, rollback_reason = _parse_rollback_marker(message)
        if rollback_count > 0:
            removed = 0
            remaining = rollback_count
            while remaining > 0 and visible:
                previous_role = str(visible[-1][1].get("role", ""))
                if previous_role == "system":
                    break
                visible.pop()
                removed += 1
                remaining -= 1
            warnings.append(
                SwiftAgentExportWarning(
                    "runtime_rollback_applied",
                    f"Applied ConversationRuntime rollback marker; removed {removed} message(s), reason={rollback_reason!r}.",
                    path,
                )
            )
            continue

        compaction = parse_compaction_marker_dict(message)
        if compaction is not None:
            boundary = compaction.prefix_len + compaction.compressed_count
            if boundary <= len(visible):
                summary_message: dict[str, Any] = {"role": "assistant", "content": compaction.summary}
                visible = [*visible[: compaction.prefix_len], (idx, summary_message), *visible[boundary:]]
                warnings.append(
                    SwiftAgentExportWarning(
                        "runtime_compaction_applied",
                        f"Applied ConversationRuntime compaction marker; replaced {compaction.compressed_count} message(s) with the summary.",
                        path,
                    )
                )
            else:
                warnings.append(
                    SwiftAgentExportWarning(
                        "runtime_compaction_skipped",
                        "Compaction marker range does not fit the visible conversation; skipped.",
                        path,
                    )
                )
            continue

        discard = parse_discard_all_marker_dict(message)
        if discard is not None:
            if discard.prefix_len <= len(visible):
                removed = len(visible) - discard.prefix_len
                visible = visible[: discard.prefix_len]
                warnings.append(
                    SwiftAgentExportWarning(
                        "runtime_discard_all_applied",
                        f"Applied ConversationRuntime discard-all marker; discarded {removed} message(s), kept prefix={discard.prefix_len}.",
                        path,
                    )
                )
            else:
                warnings.append(
                    SwiftAgentExportWarning(
                        "runtime_discard_all_skipped",
                        "Discard-all marker prefix does not fit the visible conversation; skipped.",
                        path,
                    )
                )
            continue

        warnings.append(SwiftAgentExportWarning("internal_runtime_message_skipped", "Skipped internal ConversationRuntime message.", path))
    return visible


def _parse_rollback_marker(message: dict[str, Any]) -> tuple[int, str | None]:
    if message.get("name") != "rollback":
        return 0, None
    payload = _parse_json_string(message.get("content"))
    if not isinstance(payload, dict):
        return 0, None
    rollback_count = payload.get("rollback_message_count", 0)
    reason = payload.get("reason")
    count = rollback_count if isinstance(rollback_count, int) and rollback_count > 0 else 0
    return count, str(reason) if reason is not None else None


def _assistant_to_swift_messages(
    message: dict[str, Any],
    *,
    path: str,
    config: SwiftAgentExportConfig,
    warnings: list[SwiftAgentExportWarning],
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    assistant_content = _render_assistant_content(message, config=config)
    if assistant_content:
        result.append({"role": "assistant", "content": assistant_content})

    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        warnings.append(SwiftAgentExportWarning("invalid_tool_calls_skipped", "Assistant tool_calls is not a list; skipped.", f"{path}.tool_calls"))
        return result

    for call_idx, tool_call in enumerate(tool_calls):
        call_path = f"{path}.tool_calls[{call_idx}]"
        result.append({"role": "tool_call", "content": _tool_call_to_json_string(tool_call, path=call_path, config=config)})
    return result


def _render_assistant_content(message: dict[str, Any], *, config: SwiftAgentExportConfig) -> str:
    parts: list[str] = []
    reasoning = _string_content(message.get("reasoning_content")).strip()
    if config.include_reasoning and reasoning and not (config.skip_empty_reasoning and reasoning == ""):
        parts.append(f"<think>\n{reasoning}\n</think>")
    visible_thought = _string_content(message.get("visible_thought")).strip()
    if config.include_visible_thought and visible_thought:
        parts.append(visible_thought)
    content = _string_content(message.get("content")).strip()
    if content:
        parts.append(content)
    return "\n".join(parts)


def _tool_call_to_json_string(tool_call: dict[str, Any], *, path: str, config: SwiftAgentExportConfig) -> str:
    if not isinstance(tool_call, dict):
        msg = f"Tool call must be an object at {path}."
        raise TypeError(msg)
    function = tool_call.get("function") or {}
    if not isinstance(function, dict):
        msg = f"Tool call function must be an object at {path}.function."
        raise TypeError(msg)
    name = function.get("name")
    if not isinstance(name, str) or not name:
        msg = f"Tool call function.name must be a non-empty string at {path}.function.name."
        raise ValueError(msg)
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        parsed_arguments = _parse_json_string(arguments)
        arguments = parsed_arguments if isinstance(parsed_arguments, dict) else {}
    elif arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        msg = f"Tool call arguments must decode to an object at {path}.function.arguments."
        raise TypeError(msg)
    return json.dumps({"name": name, "arguments": arguments}, ensure_ascii=config.ensure_ascii, separators=(",", ":"))


def _tool_response_to_swift_message(
    message: dict[str, Any],
    *,
    path: str,
    config: SwiftAgentExportConfig,
    warnings: list[SwiftAgentExportWarning],
) -> dict[str, str]:
    content = message.get("content")
    json_content = _json_string_content(content, ensure_ascii=config.ensure_ascii)
    if json_content is None:
        if not config.wrap_non_json_tool_response:
            msg = f"Tool response content is not valid JSON at {path}.content."
            raise ValueError(msg)
        warnings.append(SwiftAgentExportWarning("wrapped_non_json_tool_response", "Wrapped non-JSON tool response content.", f"{path}.content"))
        json_content = json.dumps({"content": _string_content(content)}, ensure_ascii=config.ensure_ascii, separators=(",", ":"))
    return {"role": "tool_response", "content": json_content}


def _tools_to_json_string(
    tools: str | list[dict[str, Any]] | None,
    *,
    warnings: list[SwiftAgentExportWarning],
    ensure_ascii: bool,
) -> str:
    if tools is None:
        return "[]"
    if isinstance(tools, str):
        parsed = _parse_json_string(tools)
        if isinstance(parsed, list):
            return json.dumps(parsed, ensure_ascii=ensure_ascii, separators=(",", ":"))
        warnings.append(SwiftAgentExportWarning("invalid_tools_replaced", "Input tools string was not a JSON list; replaced with [].", "tools"))
        return "[]"
    if isinstance(tools, list):
        return json.dumps(tools, ensure_ascii=ensure_ascii, separators=(",", ":"))
    warnings.append(SwiftAgentExportWarning("invalid_tools_replaced", "Input tools value was not a list/string; replaced with [].", "tools"))
    return "[]"


def _json_string_content(value: Any, *, ensure_ascii: bool) -> str | None:
    if isinstance(value, str):
        parsed = _parse_json_string(value)
        if parsed is None:
            return None
        return json.dumps(parsed, ensure_ascii=ensure_ascii, separators=(",", ":"))
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=ensure_ascii, separators=(",", ":"))
    return None


def _parse_json_string(value: Any) -> Any | None:
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _string_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)
