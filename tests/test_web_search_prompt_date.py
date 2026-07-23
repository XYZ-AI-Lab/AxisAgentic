"""Tests for system-prompt date resolution (auto-detect today vs. pinned)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from recipe.web_search.agent.prompts import _resolve_prompt_date, generate_system_prompt

_TODAY = datetime.now(tz=timezone(timedelta(hours=8))).strftime("%Y-%m-%d")


@pytest.mark.parametrize("value", [None, "today", "TODAY", " Today ", "auto", "now"])
def test_auto_sentinels_resolve_to_today(value: str | None) -> None:
    assert _resolve_prompt_date(value) == _TODAY


@pytest.mark.parametrize("value", ["2026-06-15", "2024-01-01"])
def test_pinned_date_used_verbatim(value: str) -> None:
    assert _resolve_prompt_date(value) == value


_PROFILES = ["default", "deepsearchqa", "livebrowsecomp", "livebrowsecomp_notools"]


@pytest.mark.parametrize("profile", _PROFILES)
@pytest.mark.parametrize("sentinel", [None, "today", "auto", "now"])
def test_sentinel_renders_current_date_in_prompt(profile: str, sentinel: str | None) -> None:
    prompt = generate_system_prompt(sentinel, prompt_profile=profile)
    assert _TODAY in prompt
    assert "{date}" not in prompt


@pytest.mark.parametrize("profile", _PROFILES)
def test_pinned_date_renders_in_prompt(profile: str) -> None:
    prompt = generate_system_prompt("2026-06-15", prompt_profile=profile)
    assert "2026-06-15" in prompt
    assert _TODAY not in prompt or _TODAY == "2026-06-15"


def test_default_prompt_omits_code_exec_rules_by_default() -> None:
    prompt = generate_system_prompt("2026-06-15")
    assert "Use python_exec for Python source code" not in prompt
    assert "Use shell_exec only for shell-native work" not in prompt


def test_default_prompt_includes_code_exec_rules_when_enabled() -> None:
    prompt = generate_system_prompt("2026-06-15", code_exec_enabled=True)
    assert "Use python_exec for Python source code" in prompt
    assert "Use shell_exec only for shell-native work" in prompt
    assert "packages installed with shell_exec can be imported from python_exec" in prompt
