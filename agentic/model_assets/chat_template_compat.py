# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any


def _decode_json_object_string(value: Any) -> tuple[Any, bool]:
    if not isinstance(value, str) or not value:
        return value, False
    current: Any = value
    for _ in range(2):
        if not isinstance(current, str):
            break
        try:
            parsed = json.loads(current)
        except json.JSONDecodeError:
            return value, False
        if isinstance(parsed, dict):
            return parsed, True
        current = parsed
    return value, False


def normalize_openai_tool_call_arguments_for_chat_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI JSON-string tool-call arguments before direct HF/Jinja rendering.

    OpenAI-compatible assistant history stores ``function.arguments`` as a JSON
    string, while many Hugging Face chat templates expect a mapping. SGLang's
    OpenAI serving path performs this normalization before applying the chat
    template; this helper mirrors that transition for AxisAgentic code that
    renders chat templates locally.
    """
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant" or not isinstance(message.get("tool_calls"), list):
            normalized.append(message)
            continue

        changed = False
        tool_calls: list[Any] = []
        for tool_call in message["tool_calls"]:
            if not isinstance(tool_call, dict):
                tool_calls.append(tool_call)
                continue
            converted_tool_call = dict(tool_call)
            function = converted_tool_call.get("function")
            if isinstance(function, dict):
                converted_function = dict(function)
                arguments, arguments_changed = _decode_json_object_string(converted_function.get("arguments"))
                if arguments_changed:
                    converted_function["arguments"] = arguments
                    converted_tool_call["function"] = converted_function
                    changed = True
            tool_calls.append(converted_tool_call)

        normalized.append({**message, "tool_calls": tool_calls} if changed else message)
    return normalized
