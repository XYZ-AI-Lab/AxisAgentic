# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from recipe.web_search.config import (
    AgentConfig,
    ModelRuntimeConfig,
    RunConfig,
    StrictConfigModel,
    ToolsConfig,
)


class WideSearchBenchmarkConfig(StrictConfigModel):
    """Where the WideSearch dataset lives and how to schedule trials."""

    name: Literal["widesearch"] = "widesearch"
    data_path: str = "axis_data://widesearch/widesearch.jsonl"
    gold_dir: str = "axis_data://widesearch/widesearch_gold"
    num_trials: int = Field(default=4, ge=1)
    max_tasks: int | None = None
    instance_ids: list[str] | None = None
    shuffle_tasks: bool = False
    shuffle_seed: int | None = None
    max_concurrent: int = Field(default=16, ge=1)


class WideSearchEvalSettings(StrictConfigModel):
    """Judge LLM + prompt-profile knobs for the deterministic-plus-LLM scorer."""

    judge_model: str | None = None
    judge_base_url: str | None = None
    judge_api_key_env: str = "JUDGE_API_KEY"
    judge_max_tokens: int = Field(default=10240, ge=1)
    judge_temperature: float = 0.0
    request_timeout: float = 600.0
    max_retries: int = Field(default=1, ge=0)
    retryable_status_codes: list[int] = Field(default_factory=list)
    judge_max_concurrent: int = Field(default=8, ge=1)
    prompts: Literal["project", "official"] = "official"
    extractor: Literal["boxed_first", "official_only"] = "boxed_first"


class WideSearchAgentPromptConfig(StrictConfigModel):
    """Agent-side prompt selection (separate from judge prompts)."""

    profile: Literal["project", "official"] = "project"
    language: Literal["en", "zh"] = "en"
    date: str | None = None


class WideSearchEvalConfig(StrictConfigModel):
    schema_version: int = Field(default=1, ge=1)
    run: RunConfig = Field(default_factory=RunConfig)
    model: ModelRuntimeConfig = Field(default_factory=ModelRuntimeConfig)
    benchmark: WideSearchBenchmarkConfig = Field(default_factory=WideSearchBenchmarkConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    agent_prompt: WideSearchAgentPromptConfig = Field(default_factory=WideSearchAgentPromptConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    eval: WideSearchEvalSettings = Field(default_factory=WideSearchEvalSettings)


def load_widesearch_eval_config(path: str | Path) -> WideSearchEvalConfig:
    config_path = Path(path)
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return WideSearchEvalConfig.model_validate(raw, strict=True)


def dump_widesearch_eval_config(config: WideSearchEvalConfig) -> str:
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, indent=2)
