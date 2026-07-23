# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


def tool_property_order(tool_definition: dict[str, Any]) -> list[str]:
    """Return the prompt-visible argument order for an OpenAI-style tool definition."""
    function = tool_definition.get("function")
    if not isinstance(function, dict):
        return []
    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        return []
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return []
    return list(properties.keys())


def validate_rendered_tool_argument_order(rendered_prompt: str, tools: list[dict[str, Any]] | None) -> str | None:
    """Return an error if rendered tool arguments no longer follow schema order.

    Python dict insertion order is the source of truth for parameter schemas in this
    repo. Chat-template renderers should preserve that order in the system prompt;
    this guard catches template/filter changes that sort or otherwise reorder JSON
    object keys before the prompt reaches the model.
    """
    if not rendered_prompt or not tools:
        return None

    for idx, tool in enumerate(tools):
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue

        expected = tool_property_order(tool)
        if len(expected) < 2:
            continue

        section_start = rendered_prompt.find(name)
        if section_start < 0:
            continue
        section_end = len(rendered_prompt)
        for next_tool in tools[idx + 1 :]:
            next_function = next_tool.get("function")
            if not isinstance(next_function, dict):
                continue
            next_name = next_function.get("name")
            if not isinstance(next_name, str) or not next_name:
                continue
            next_pos = rendered_prompt.find(next_name, section_start + len(name))
            if next_pos >= 0:
                section_end = next_pos
                break

        section = rendered_prompt[section_start:section_end]
        positions: list[int] = []
        missing: list[str] = []
        for argument_name in expected:
            pos = section.find(f'"{argument_name}"')
            if pos < 0:
                pos = section.find(f"'{argument_name}'")
            if pos < 0:
                missing.append(argument_name)
            else:
                positions.append(pos)

        if missing:
            continue
        if positions != sorted(positions):
            actual = [name for _, name in sorted(zip(positions, expected, strict=False))]
            return f"rendered tool argument order changed for {name!r}: expected {expected}, rendered {actual}"

    return None
