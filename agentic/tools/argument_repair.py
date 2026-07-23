# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from agentic.contracts import ToolRequest
    from agentic.tools.base import Tool


class ToolArgumentRepairHook(Protocol):
    """Callable hook for recipe- or tool-specific argument repair."""

    def __call__(self, request: ToolRequest, tool: Tool, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Return repaired arguments, or ``None`` to leave them unchanged."""
        ...


@dataclass(slots=True)
class ToolArgumentRepairResult:
    arguments: dict[str, Any]
    changed: bool = False
    applied_rules: list[str] = field(default_factory=list)


class ToolArgumentRepairer:
    """Schema-driven pre-tool argument repair.

    This lives in the tool layer because the repair decision depends on the
    registered tool's schema and execution policy. Recipes can opt in at the
    ``ToolManager`` level and add exact per-tool hooks where schema aliases are
    not expressive enough.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        repair_strict_tools: bool = False,
        apply_schema_aliases: bool = True,
        apply_schema_defaults: bool = True,
        coerce_scalar_types: bool = True,
        normalize_key_names: bool = True,
        fuzzy_match_single_missing_property: bool = True,
        fuzzy_match_threshold: float = 0.72,
        hooks: dict[str, ToolArgumentRepairHook] | None = None,
    ) -> None:
        self.enabled = enabled
        self.repair_strict_tools = repair_strict_tools
        self.apply_schema_aliases = apply_schema_aliases
        self.apply_schema_defaults = apply_schema_defaults
        self.coerce_scalar_types = coerce_scalar_types
        self.normalize_key_names = normalize_key_names
        self.fuzzy_match_single_missing_property = fuzzy_match_single_missing_property
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self._hooks: dict[str, ToolArgumentRepairHook] = dict(hooks or {})

    def register_hook(self, tool_name: str, hook: ToolArgumentRepairHook) -> None:
        self._hooks[tool_name] = hook

    def repair(self, request: ToolRequest, tool: Tool) -> ToolArgumentRepairResult:
        original = request.arguments
        if not self.enabled or not isinstance(original, dict):
            return ToolArgumentRepairResult(arguments=original)
        if tool.strict_mode and not self.repair_strict_tools:
            return ToolArgumentRepairResult(arguments=original)

        repaired = dict(original)
        applied_rules: list[str] = []

        schema = tool.parameters if isinstance(tool.parameters, dict) else {}
        properties = schema.get("properties") if schema.get("type") == "object" else None
        if isinstance(properties, dict):
            if self.normalize_key_names:
                normalized = self._normalize_argument_keys(repaired)
                if normalized != repaired:
                    repaired = normalized
                    applied_rules.append("normalize_key_names")

            if self.apply_schema_aliases and self._apply_property_aliases(repaired, properties):
                applied_rules.append("schema_aliases")

            if self.fuzzy_match_single_missing_property and self._apply_single_fuzzy_property_match(repaired, properties):
                applied_rules.append("fuzzy_property_match")

            if self.apply_schema_defaults and self._apply_schema_defaults(repaired, properties):
                applied_rules.append("schema_defaults")

            if self.coerce_scalar_types and self._coerce_schema_values(repaired, properties):
                applied_rules.append("scalar_type_coercion")

        hook = self._hooks.get(tool.name)
        if hook is not None:
            hooked = hook(request, tool, dict(repaired))
            if isinstance(hooked, dict) and hooked != repaired:
                repaired = hooked
                applied_rules.append("custom_hook")

        return ToolArgumentRepairResult(arguments=repaired, changed=repaired != original, applied_rules=applied_rules)

    @staticmethod
    def _normalize_argument_keys(arguments: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in arguments.items():
            normalized_key = key.strip().strip("\"'")
            normalized.setdefault(normalized_key, value)
        return normalized

    @staticmethod
    def _apply_property_aliases(arguments: dict[str, Any], properties: dict[str, Any]) -> bool:
        changed = False
        alias_to_property: dict[str, str] = {}
        for property_name, prop_schema in properties.items():
            if not isinstance(prop_schema, dict):
                continue
            aliases = prop_schema.get("x-agentic-aliases", ())
            if isinstance(aliases, str):
                aliases = (aliases,)
            for alias in aliases:
                if isinstance(alias, str) and alias:
                    alias_to_property.setdefault(alias, property_name)

        for alias, property_name in alias_to_property.items():
            if property_name not in arguments and alias in arguments:
                arguments[property_name] = arguments.pop(alias)
                changed = True
        return changed

    def _apply_single_fuzzy_property_match(self, arguments: dict[str, Any], properties: dict[str, Any]) -> bool:
        remaining_keys = [key for key in arguments if key not in properties]
        missing_properties = [key for key in properties if key not in arguments]
        if len(remaining_keys) != 1 or len(missing_properties) != 1:
            return False
        unknown_key = remaining_keys[0]
        missing_key = missing_properties[0]
        similarity = SequenceMatcher(None, unknown_key, missing_key).ratio()
        if similarity < self.fuzzy_match_threshold:
            return False
        arguments[missing_key] = arguments.pop(unknown_key)
        return True

    @staticmethod
    def _apply_schema_defaults(arguments: dict[str, Any], properties: dict[str, Any]) -> bool:
        changed = False
        for key, prop_schema in properties.items():
            if key not in arguments and isinstance(prop_schema, dict) and "default" in prop_schema:
                default = prop_schema["default"]
                if not ToolArgumentRepairer._value_matches_schema_type(default, prop_schema):
                    continue
                arguments[key] = default
                changed = True
        return changed

    @classmethod
    def _coerce_schema_values(cls, arguments: dict[str, Any], properties: dict[str, Any]) -> bool:
        changed = False
        for key, prop_schema in properties.items():
            if key not in arguments or not isinstance(prop_schema, dict):
                continue
            coerced = cls._coerce_schema_value(arguments[key], prop_schema)
            if coerced != arguments[key]:
                arguments[key] = coerced
                changed = True
        return changed

    @staticmethod
    def _coerce_schema_value(value: Any, prop_schema: dict[str, Any]) -> Any:
        expected_type = prop_schema.get("type")
        if isinstance(expected_type, list):
            expected_types = set(expected_type)
        else:
            expected_types = {expected_type}

        if value is None and "null" in expected_types:
            return value
        if "string" in expected_types and not isinstance(value, str):
            if isinstance(value, bool):
                return "true" if value else "false"
            if isinstance(value, (int, float)):
                return str(value)
        if "integer" in expected_types and isinstance(value, str):
            stripped = value.strip()
            if stripped.lstrip("-").isdigit():
                return int(stripped)
        if "number" in expected_types and isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value
        if "boolean" in expected_types and isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes"}:
                return True
            if lowered in {"false", "0", "no"}:
                return False
        return value

    @staticmethod
    def _value_matches_schema_type(value: Any, prop_schema: dict[str, Any]) -> bool:
        expected_type = prop_schema.get("type")
        if expected_type is None:
            return True
        if isinstance(expected_type, list):
            expected_types = set(expected_type)
        else:
            expected_types = {expected_type}
        if value is None:
            return "null" in expected_types
        return (
            ("string" in expected_types and isinstance(value, str))
            or ("integer" in expected_types and isinstance(value, int) and not isinstance(value, bool))
            or ("number" in expected_types and isinstance(value, (int, float)) and not isinstance(value, bool))
            or ("boolean" in expected_types and isinstance(value, bool))
            or ("object" in expected_types and isinstance(value, dict))
            or ("array" in expected_types and isinstance(value, list))
        )
