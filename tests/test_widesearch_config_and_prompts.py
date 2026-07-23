from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from recipe.wide_search.agent.prompts import (
    OFFICIAL_SYSTEM_PROMPT_EN,
    OFFICIAL_SYSTEM_PROMPT_ZH,
    widesearch_summary_prompt,
    widesearch_system_prompt,
)
from recipe.wide_search.config import (
    WideSearchEvalConfig,
    dump_widesearch_eval_config,
    load_widesearch_eval_config,
)

CONFIG_DIR = Path("recipe/wide_search/configs")


def test_load_default_yaml() -> None:
    cfg = load_widesearch_eval_config(CONFIG_DIR / "default.yaml")
    assert cfg.benchmark.name == "widesearch"
    assert cfg.benchmark.num_trials == 4
    assert cfg.agent_prompt.profile == "project"
    assert cfg.eval.extractor == "boxed_first"
    assert cfg.eval.prompts == "official"
    assert cfg.eval.judge_model is None
    assert cfg.eval.judge_max_concurrent == 8


def test_default_yaml_official_preset_overrides_round_trip() -> None:
    """The two fields tagged OFFICIAL PRESET in default.yaml flip cleanly."""
    cfg = load_widesearch_eval_config(CONFIG_DIR / "default.yaml")
    cfg.agent_prompt.profile = "official"
    cfg.eval.extractor = "official_only"
    rendered = dump_widesearch_eval_config(cfg)
    assert "profile: official" in rendered
    assert "extractor: official_only" in rendered


def test_default_config_round_trips_through_yaml() -> None:
    cfg = WideSearchEvalConfig()
    rendered = dump_widesearch_eval_config(cfg)
    assert "benchmark" in rendered
    assert "eval" in rendered


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown_field"):
        WideSearchEvalConfig.model_validate({"benchmark": {"unknown_field": 1}}, strict=True)


def test_project_system_prompt_mentions_boxed_table() -> None:
    prompt = widesearch_system_prompt(profile="project", date="2026-06-10")
    assert "\\boxed" in prompt
    assert "Markdown" in prompt
    assert "2026-06-10" in prompt


def test_official_system_prompt_matches_upstream() -> None:
    en = widesearch_system_prompt(profile="official", language="en")
    zh = widesearch_system_prompt(profile="official", language="zh")
    assert en == OFFICIAL_SYSTEM_PROMPT_EN
    assert zh == OFFICIAL_SYSTEM_PROMPT_ZH


def test_summary_prompt_project_includes_task_and_box() -> None:
    summary = widesearch_summary_prompt("Find top 5 universities", profile="project")
    assert "Find top 5 universities" in summary
    assert "\\boxed" in summary


def test_summary_prompt_official_no_box_directive() -> None:
    summary = widesearch_summary_prompt("any task", profile="official")
    assert "\\boxed" not in summary


def test_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match="profile"):
        widesearch_system_prompt(profile="bogus")
    with pytest.raises(ValueError, match="profile"):
        widesearch_summary_prompt("x", profile="bogus")
