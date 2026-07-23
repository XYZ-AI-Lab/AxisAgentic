from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agentic.contracts import ConversationMessage, MessageRole
from recipe.web_search.agent.prompts import FORMAT_ERROR_MESSAGE
from recipe.wide_search.config import WideSearchEvalConfig
from recipe.wide_search.runners.evaluate_widesearch import (
    ORCHESTRATOR_NAME,
    _config_to_namespace,
    _filter_scored_trials_to_completed_traces,
    _filter_scored_trials_to_schedule,
    _load_scored_trials,
    _select_response_for_scoring,
    _write_per_task_and_summary,
    trial_task_id,
)
from recipe.wide_search.runners.run_eval_config import (
    PROJECT_ROOT,
    _resolve_config,
    _resolve_env_file_path,
    _resolve_path,
)


def _assistant(content: str) -> ConversationMessage:
    return ConversationMessage(role=MessageRole.ASSISTANT, content=content)


def _user(content: str) -> ConversationMessage:
    return ConversationMessage(role=MessageRole.USER, content=content)


def _result(*, output: str, conversation: list[ConversationMessage]) -> Any:
    """Synthesize the minimum OrchestrationResult shape ``_select_response_for_scoring`` reads."""
    res = MagicMock()
    res.output = output
    res.visible_conversation = conversation
    return res


_TABLE_BOXED = "\\boxed{|name|score|\n|---|---|\n|alpha|1|\n|beta|2|}"
_TABLE_MARKDOWN_BLOCK = "```markdown\n|name|score|\n|---|---|\n|alpha|1|\n|beta|2|\n```"


def test_orchestrator_name_is_wide_search() -> None:
    assert ORCHESTRATOR_NAME == "wide-search"


def test_trial_task_id_round_trips() -> None:
    tid = trial_task_id("ws_en_001", 3)
    assert tid == "ws_en_001__trial-3"
    base, suffix = tid.rsplit("__trial-", maxsplit=1)
    assert base == "ws_en_001"
    assert int(suffix) == 3


def test_resume_score_sidecars_seed_widesearch_summary(tmp_path: Path) -> None:
    scores_dir = tmp_path / "widesearch_scores"
    scores_dir.mkdir()
    sidecars = {
        "ws_en_001__trial-0": {
            "instance_id": "ws_en_001",
            "trial_index": 0,
            "score": 0.0,
            "precision_by_row": 0.2,
            "recall_by_row": 0.2,
            "f1_by_row": 0.2,
            "precision_by_item": 0.8,
            "recall_by_item": 0.8,
            "f1_by_item": 0.8,
        },
        "ws_en_001__trial-1": {
            "instance_id": "ws_en_001",
            "trial_index": 1,
            "score": 1.0,
            "precision_by_row": 1.0,
            "recall_by_row": 1.0,
            "f1_by_row": 1.0,
            "precision_by_item": 1.0,
            "recall_by_item": 1.0,
            "f1_by_item": 1.0,
        },
        "ws_en_002__trial-0": {
            "instance_id": "ws_en_002",
            "trial_index": 0,
            "score": 1.0,
            "precision_by_row": 0.5,
            "recall_by_row": 0.5,
            "f1_by_row": 0.5,
            "precision_by_item": 0.5,
            "recall_by_item": 0.5,
            "f1_by_item": 0.5,
        },
    }
    for trial_id, payload in sidecars.items():
        (scores_dir / f"{trial_id}.json").write_text(json.dumps(payload), encoding="utf-8")

    scored = _load_scored_trials(scores_dir)
    summary = _write_per_task_and_summary(tmp_path, list(scored.values()), expected_trials=2, total_instances=3)

    assert sorted(scored) == sorted(sidecars)
    assert summary["total_instances"] == 3
    assert summary["num_complete_instances"] == 1
    assert summary["num_incomplete_instances"] == 1
    assert summary["leaderboard"]["success_rate_avg@N"] == pytest.approx(0.5)
    assert summary["leaderboard"]["item_f1_max@N"] == pytest.approx(1.0)


def test_resume_score_sidecars_ignore_retryable_orchestrator_errors(tmp_path: Path) -> None:
    scores_dir = tmp_path / "widesearch_scores"
    scores_dir.mkdir()
    (scores_dir / "ws_en_001__trial-0.json").write_text(
        json.dumps(
            {
                "instance_id": "ws_en_001",
                "trial_index": 0,
                "score": 0.0,
                "msg": "orchestrator error: APIConnectionError('wrong endpoint')",
            }
        ),
        encoding="utf-8",
    )
    (scores_dir / "ws_en_001__trial-1.json").write_text(
        json.dumps(
            {
                "instance_id": "ws_en_001",
                "trial_index": 1,
                "score": 0.0,
                "msg": "response_df is None",
            }
        ),
        encoding="utf-8",
    )

    scored = _load_scored_trials(scores_dir, quarantine_ignored=True)

    assert sorted(scored) == ["ws_en_001__trial-1"]
    assert not (scores_dir / "ws_en_001__trial-0.json").exists()
    assert (scores_dir / "_ignored" / "retryable" / "ws_en_001__trial-0.json").exists()


def test_resume_score_sidecars_filter_to_current_schedule(tmp_path: Path) -> None:
    scores_dir = tmp_path / "widesearch_scores"
    scores_dir.mkdir()
    for trial_id in ("ws_en_001__trial-0", "ws_en_001__trial-2", "ws_en_999__trial-0"):
        instance_id, trial_suffix = trial_id.rsplit("__trial-", maxsplit=1)
        (scores_dir / f"{trial_id}.json").write_text(
            json.dumps({"instance_id": instance_id, "trial_index": int(trial_suffix), "score": 1.0}),
            encoding="utf-8",
        )

    scored = _load_scored_trials(scores_dir)
    filtered = _filter_scored_trials_to_schedule(scored, {"ws_en_001__trial-0", "ws_en_001__trial-1"})

    assert sorted(filtered) == ["ws_en_001__trial-0"]


def test_resume_score_sidecars_require_completed_traces(tmp_path: Path) -> None:
    scores_dir = tmp_path / "widesearch_scores"
    scores_dir.mkdir()
    for trial_id in ("ws_en_001__trial-0", "ws_en_001__trial-1"):
        instance_id, trial_suffix = trial_id.rsplit("__trial-", maxsplit=1)
        (scores_dir / f"{trial_id}.json").write_text(
            json.dumps({"instance_id": instance_id, "trial_index": int(trial_suffix), "score": 1.0}),
            encoding="utf-8",
        )

    scored = _load_scored_trials(scores_dir)
    filtered = _filter_scored_trials_to_completed_traces(scored, {"ws_en_001__trial-1"})

    assert sorted(filtered) == ["ws_en_001__trial-1"]


def test_config_to_namespace_carries_key_fields() -> None:
    cfg = WideSearchEvalConfig()
    cfg.model.openai_model = "gpt-x"
    cfg.model.openai_base_url = "http://example.com"
    cfg.tools.serper_base_url = "http://serper"
    cfg.tools.jina_base_url = "http://jina"

    ns = _config_to_namespace(cfg)

    assert ns.model == "gpt-x"
    assert ns.base_url == "http://example.com"
    assert ns.serper_base_url == "http://serper"
    assert ns.jina_base_url == "http://jina"
    assert ns.prompt_profile == "default"
    assert ns.parallel_tool_calls == cfg.model.parallel_tool_calls
    assert ns.max_context_length == cfg.model.context.max_context_length


def test_resolve_path_repo_scheme() -> None:
    resolved = _resolve_path("repo://recipe/wide_search/configs/default.yaml", label="x")
    assert resolved == PROJECT_ROOT / "recipe/wide_search/configs/default.yaml"


def test_resolve_path_axis_log_scheme(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AXIS_LOG_DIR", str(tmp_path))
    resolved = _resolve_path("axis_log://wide_search_infer/run1", label="run.output_dir")
    assert resolved == tmp_path / "wide_search_infer/run1"


def test_resolve_path_axis_log_missing_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AXIS_LOG_DIR", raising=False)
    with pytest.raises(ValueError, match="AXIS_LOG_DIR"):
        _resolve_path("axis_log://foo", label="run.output_dir")


def test_resolve_path_absolute_passthrough(tmp_path: Path) -> None:
    abs_path = tmp_path / "x"
    resolved = _resolve_path(str(abs_path), label="x")
    assert resolved == abs_path


def test_resolve_path_relative_anchored_to_repo() -> None:
    resolved = _resolve_path("recipe/wide_search/configs/default.yaml", label="x")
    assert resolved == PROJECT_ROOT / "recipe/wide_search/configs/default.yaml"


def test_resolve_env_file_path_default_repo_envs() -> None:
    config_path = PROJECT_ROOT / "recipe/wide_search/configs/default.yaml"
    resolved = _resolve_env_file_path(config_path, ".envs/.env")
    assert resolved == PROJECT_ROOT / ".envs/.env"


def test_resolve_env_file_path_none() -> None:
    config_path = PROJECT_ROOT / "recipe/wide_search/configs/default.yaml"
    assert _resolve_env_file_path(config_path, None) is None


def test_resolve_config_pulls_required_model_fields_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "stub-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://stub.example.com")
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.delenv("JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    monkeypatch.setenv("AXIS_LOG_DIR", str(tmp_path))

    cfg = WideSearchEvalConfig()
    cfg.model.api_key_env = "AGENT_API_KEY"
    cfg.run.output_dir = "axis_log://w/run"
    cfg.benchmark.data_path = str(tmp_path / "data.jsonl")
    cfg.benchmark.gold_dir = str(tmp_path / "gold")

    resolved = _resolve_config(cfg, max_tasks_override=7)

    assert resolved.model.openai_model == "stub-model"
    assert resolved.model.openai_base_url == "http://stub.example.com"
    assert resolved.eval.judge_model == "stub-model"
    assert resolved.eval.judge_base_url == "http://stub.example.com"
    assert resolved.eval.judge_api_key_env == "AGENT_API_KEY"
    assert resolved.benchmark.max_tasks == 7
    assert Path(resolved.run.output_dir) == tmp_path / "w/run"


def test_resolve_config_prefers_judge_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "stub-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://stub.example.com")
    monkeypatch.setenv("JUDGE_MODEL", "judge-stub")
    monkeypatch.setenv("JUDGE_BASE_URL", "http://judge.example.com")
    monkeypatch.setenv("JUDGE_API_KEY", "judge-key")
    monkeypatch.setenv("AXIS_LOG_DIR", str(tmp_path))

    cfg = WideSearchEvalConfig()
    cfg.model.api_key_env = "AGENT_API_KEY"
    cfg.run.output_dir = "axis_log://w/run"
    cfg.benchmark.data_path = str(tmp_path / "data.jsonl")
    cfg.benchmark.gold_dir = str(tmp_path / "gold")

    resolved = _resolve_config(cfg, max_tasks_override=None)

    assert resolved.eval.judge_model == "judge-stub"
    assert resolved.eval.judge_base_url == "http://judge.example.com"
    assert resolved.eval.judge_api_key_env == "JUDGE_API_KEY"


def test_resolve_config_missing_required_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    cfg = WideSearchEvalConfig()
    with pytest.raises(ValueError, match="openai_model"):
        _resolve_config(cfg, max_tasks_override=None)


def test_select_response_uses_latest_assistant_when_it_parses() -> None:
    """Normal completion: latest assistant message has the table → use it."""
    result = _result(
        output="|name|score|\n|---|---|\n|alpha|1|",
        conversation=[_user("q"), _assistant(_TABLE_BOXED)],
    )
    text, df = _select_response_for_scoring(result, mode="boxed_first")
    assert text == _TABLE_BOXED
    assert df is not None
    assert not df.empty


def test_select_response_falls_back_to_output_when_latest_fails() -> None:
    """fallback_output_used path: latest assistant is the failed forced-final,
    while result.output holds an earlier intermediate boxed answer.
    """
    earlier_table = "|name|score|\n|---|---|\n|alpha|1|"
    failed_forced_final = "I could not produce the table."
    result = _result(
        output=earlier_table,
        conversation=[_user("q"), _assistant(failed_forced_final)],
    )
    text, df = _select_response_for_scoring(result, mode="boxed_first")
    assert text == earlier_table
    assert df is not None
    assert not df.empty


def test_select_response_returns_latest_when_neither_parses() -> None:
    """Total failure: keep latest assistant text for sidecar diagnostics."""
    result = _result(
        output=FORMAT_ERROR_MESSAGE,
        conversation=[_user("q"), _assistant("nonsense")],
    )
    text, df = _select_response_for_scoring(result, mode="boxed_first")
    assert text == "nonsense"
    assert df is None


def test_select_response_skips_format_error_message_when_falling_back() -> None:
    """If result.output is the FORMAT_ERROR sentinel, don't try to parse it."""
    result = _result(
        output=FORMAT_ERROR_MESSAGE,
        conversation=[_user("q"), _assistant("also nonsense")],
    )
    text, df = _select_response_for_scoring(result, mode="official_only")
    assert text == "also nonsense"
    assert df is None


def test_config_to_namespace_carries_context_estimator_fields() -> None:
    """Estimator + tokenizer + render-template fields must reach the namespace
    so _build_context_token_estimator (in the chat_template branch) can read
    them when limit_detection=='estimated'.
    """
    cfg = WideSearchEvalConfig()
    cfg.model.context.estimator = "cheap"
    cfg.model.context.tokenizer_path = "/path/to/tokenizer"
    cfg.run.system_prompt_render_template = "auto"

    ns = _config_to_namespace(cfg)

    assert ns.context_estimator == "cheap"
    assert ns.context_tokenizer_path == "/path/to/tokenizer"
    assert ns.system_prompt_render_template == "auto"
    assert ns.context_limit_detection == cfg.model.context.limit_detection
