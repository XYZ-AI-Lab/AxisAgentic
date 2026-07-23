from pathlib import Path

import pytest

from agentic.config import OrchestrationConfig, RunConfig, load_run_config
from recipe.web_search.config import load_web_search_eval_config
from recipe.web_search.runners.run_eval_config import (
    _resolve_config,
    _resolve_env_file_path,
    _resolve_path,
    _runner_args,
    _write_run_config_files,
)


def test_load_run_config_template() -> None:
    config_path = Path(__file__).resolve().parents[1] / "agentic/config/runtime_template.yaml"
    config = load_run_config(config_path)

    assert isinstance(config, RunConfig)
    assert isinstance(config.orchestration, OrchestrationConfig)
    assert config.orchestration.name == "default-orchestration"
    assert config.orchestration.conversation.max_turns is None
    assert config.model_client.context_window == 128000
    assert config.reward.reward_per_success_tool_call == 1.0
    assert config.logger.name == "agentic"
    assert len(config.tools) == 1
    assert config.tools[0].name == "calculator"


def test_web_search_default_env_file_resolves_from_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "recipe/web_search/configs/default.yaml"
    config = load_web_search_eval_config(config_path)

    assert config.run.env_file == ".envs/.env"
    assert _resolve_env_file_path(config_path, config.run.env_file) == repo_root / ".envs/.env"


def test_web_search_default_config_uses_grouped_context_and_cheap_estimator() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_web_search_eval_config(repo_root / "recipe/web_search/configs/default.yaml")

    assert config.model.context.estimator == "cheap"
    assert config.model.context.limit_detection == "provider_error"
    assert config.model.context.tokenizer_path is None
    assert config.agent.retry.generation_limit_recovery.non_final_attempt == "retry"
    assert config.agent.retry.generation_limit_recovery.final_attempt == "rollback"


def test_web_search_default_config_is_single_template() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_web_search_eval_config(repo_root / "recipe/web_search/configs/default.yaml")

    assert config.benchmark.name == "browsecomp"
    assert config.benchmark.data_path == "axis_data://browsecomp/standardized_data.jsonl"
    assert config.run.output_dir == "axis_log://web_search_infer/browsecomp_default"
    assert config.run.env_file == ".envs/.env"
    assert config.model.openai_base_url is None
    assert config.judge.judge_max_tokens == 1024


def test_web_search_self_verification_config_wires_runner_args(tmp_path: Path) -> None:
    config_path = tmp_path / "self_verification.yaml"
    config_path.write_text(
        """
schema_version: 1
model:
  openai_model: test-model
  openai_base_url: http://model.example/v1
agent:
  self_verification:
    enabled: true
    max_reanswer_attempts: 2
    verification_max_turns: 17
    verdict_resample_max_attempts: 4
""",
        encoding="utf-8",
    )
    config = load_web_search_eval_config(config_path)
    args = _runner_args(config, tmp_path / "run_1")

    assert "--self_verification_enabled" in args
    assert args[args.index("--self_verification_max_reanswer_attempts") + 1] == "2"
    assert args[args.index("--self_verification_max_turns") + 1] == "17"
    assert args[args.index("--self_verification_verdict_resample_max_attempts") + 1] == "4"


def test_web_search_summary_llm_feature_toggles_wire_runner_args(tmp_path: Path) -> None:
    config_path = tmp_path / "summary_toggles.yaml"
    config_path.write_text(
        """
schema_version: 1
model:
  openai_model: test-model
  openai_base_url: http://model.example/v1
tools:
  summary_llm:
    global_anchor_enabled: false
    chunk_envelope_mode: soft
    csv_layer_b_enabled: false
""",
        encoding="utf-8",
    )
    args = _runner_args(load_web_search_eval_config(config_path), tmp_path / "run_1")

    assert args[args.index("--summary_llm_global_anchor_enabled") + 1] == "false"
    assert args[args.index("--summary_llm_chunk_envelope_mode") + 1] == "soft"
    assert args[args.index("--summary_llm_csv_layer_b_enabled") + 1] == "false"


def test_web_search_path_schemes(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("AXIS_DATA_DIR", "/axis/data")
    monkeypatch.setenv("AXIS_LOG_DIR", "/axis/logs")
    monkeypatch.setenv("AXIS_MODEL_DIR", "/axis/models")

    assert _resolve_path("repo://recipe/web_search", label="x") == repo_root / "recipe/web_search"
    assert _resolve_path("axis_data://datasets/file.jsonl", label="x") == Path("/axis/data/datasets/file.jsonl")
    assert _resolve_path("axis_log://runs", label="x") == Path("/axis/logs/runs")
    assert _resolve_path("axis_model://Qwen", label="x") == Path("/axis/models/Qwen")


def test_web_search_effective_config_dump_is_secret_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "recipe/web_search/configs/default.yaml"
    config = load_web_search_eval_config(config_path)
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://provider.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai")
    monkeypatch.setenv("SERPER_API_KEY", "secret-serper")
    monkeypatch.setenv("AXIS_DATA_DIR", "/axis/data")
    monkeypatch.setenv("AXIS_LOG_DIR", "/axis/logs")

    _write_run_config_files(config_path, _resolve_config(config), tmp_path)

    effective = (tmp_path / "run_config.effective.yaml").read_text(encoding="utf-8")
    assert "secret-openai" not in effective
    assert "secret-serper" not in effective
