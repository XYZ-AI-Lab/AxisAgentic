# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shlex
import sys
from pathlib import Path

from recipe.wide_search.config import (
    WideSearchEvalConfig,
    dump_widesearch_eval_config,
    load_widesearch_eval_config,
)
from recipe.wide_search.runners.evaluate_widesearch import run_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_PATH_SCHEME_ENV = {
    "axis_data": "AXIS_DATA_DIR",
    "axis_log": "AXIS_LOG_DIR",
    "axis_model": "AXIS_MODEL_DIR",
}


def _load_env_file(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    print(f"Loading env from: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        try:
            parsed = shlex.split(value, comments=False, posix=True)
            os.environ[key] = parsed[0] if parsed else ""
        except ValueError:
            os.environ[key] = value.strip().strip('"').strip("'")


def _resolve_env_file_path(config_path: Path, env_file: str | None) -> Path | None:
    if not env_file:
        return None
    path = Path(env_file).expanduser()
    if path.is_absolute():
        return path
    repo_candidate = PROJECT_ROOT / path
    if repo_candidate.exists() or path.as_posix() == ".envs/.env":
        return repo_candidate
    config_candidate = config_path.expanduser().resolve().parent / path
    return config_candidate if config_candidate.exists() else repo_candidate


def _env_path_root(env_name: str, label: str) -> Path:
    value = os.environ.get(env_name)
    if not value:
        msg = f"{label} uses {env_name}, but {env_name} is not set."
        raise ValueError(msg)
    return Path(value).expanduser()


def _resolve_path(value: str | Path, *, label: str) -> Path:
    raw = str(value)
    for scheme, env_name in _PATH_SCHEME_ENV.items():
        prefix = f"{scheme}://"
        if raw.startswith(prefix):
            return _env_path_root(env_name, label) / raw[len(prefix) :].lstrip("/")
    if raw.startswith("repo://"):
        return PROJECT_ROOT / raw[len("repo://") :].lstrip("/")
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _provider_value(value: str | None, env_names: tuple[str, ...], label: str) -> str:
    resolved = value or next((os.environ[name] for name in env_names if os.environ.get(name)), None)
    if not resolved:
        msg = f"{label} is required: set it in YAML or provider env {', '.join(env_names)}"
        raise ValueError(msg)
    return resolved


def _resolve_config(config: WideSearchEvalConfig, *, max_tasks_override: int | None) -> WideSearchEvalConfig:
    resolved = config.model_copy(deep=True)
    resolved.model.openai_model = _provider_value(resolved.model.openai_model, ("OPENAI_MODEL",), "model.openai_model")
    resolved.model.openai_base_url = _provider_value(resolved.model.openai_base_url, ("OPENAI_BASE_URL",), "model.openai_base_url")
    resolved.eval.judge_model = resolved.eval.judge_model or os.environ.get("JUDGE_MODEL") or resolved.model.openai_model
    resolved.eval.judge_base_url = resolved.eval.judge_base_url or os.environ.get("JUDGE_BASE_URL") or resolved.model.openai_base_url
    if resolved.eval.judge_api_key_env == "JUDGE_API_KEY" and "JUDGE_API_KEY" not in os.environ:
        resolved.eval.judge_api_key_env = resolved.model.api_key_env
    resolved.benchmark.data_path = str(_resolve_path(resolved.benchmark.data_path, label="benchmark.data_path"))
    resolved.benchmark.gold_dir = str(_resolve_path(resolved.benchmark.gold_dir, label="benchmark.gold_dir"))
    resolved.run.output_dir = str(_resolve_path(resolved.run.output_dir, label="run.output_dir"))
    resolved.tools.serper_base_url = resolved.tools.serper_base_url or os.environ.get("SERPER_BASE_URL")
    resolved.tools.jina_base_url = resolved.tools.jina_base_url or os.environ.get("JINA_BASE_URL")
    resolved.tools.summary_llm.base_url = resolved.tools.summary_llm.base_url or os.environ.get("SUMMARY_LLM_BASE_URL")
    resolved.tools.summary_llm.model_name = resolved.tools.summary_llm.model_name or os.environ.get("SUMMARY_LLM_MODEL_NAME")
    if max_tasks_override is not None:
        resolved.benchmark.max_tasks = max_tasks_override
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run wide-search benchmark evaluation from YAML")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None, help="Override benchmark.max_tasks")
    parser.add_argument("--num-trials", type=int, default=None, help="Override benchmark.num_trials")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_widesearch_eval_config(config_path)
    if args.resume:
        config.run.resume = True
    if args.num_trials is not None:
        config.benchmark.num_trials = args.num_trials

    env_file = _resolve_env_file_path(config_path, config.run.env_file)
    _load_env_file(env_file)

    resolved = _resolve_config(config, max_tasks_override=args.max_tasks)

    log_level = getattr(logging, resolved.run.log_level)
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    print("=== Wide Search YAML Evaluation ===")
    print(f"Config:         {config_path}")
    print(f"Model:          {resolved.model.openai_model}")
    print(f"Base URL:       {resolved.model.openai_base_url}")
    print(f"Judge model:    {resolved.eval.judge_model}")
    print(f"Data path:      {resolved.benchmark.data_path}")
    print(f"Gold dir:       {resolved.benchmark.gold_dir}")
    print(f"Output dir:     {resolved.run.output_dir}")
    print(f"Num trials:     {resolved.benchmark.num_trials}")
    print(f"Max concurrent: {resolved.benchmark.max_concurrent}")
    print(f"Agent prompt:   {resolved.agent_prompt.profile}")
    print(f"Extractor:      {resolved.eval.extractor}")
    if args.dry_run:
        print(dump_widesearch_eval_config(resolved))
        return

    output_dir = Path(resolved.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.input.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (output_dir / "run_config.effective.yaml").write_text(dump_widesearch_eval_config(resolved), encoding="utf-8")

    num_runs = max(1, resolved.run.num_runs)
    for idx in range(1, num_runs + 1):
        run_dir = output_dir / f"run_{idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_config.input.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        (run_dir / "run_config.effective.yaml").write_text(dump_widesearch_eval_config(resolved), encoding="utf-8")
        run_config = resolved.model_copy(deep=True)
        run_config.run.output_dir = str(run_dir)
        print(f"--- Starting run {idx}/{num_runs} (output: {run_dir}) ---")
        summary = asyncio.run(run_evaluation(run_config))
        print(f"--- Final widesearch summary (run {idx}/{num_runs}) ---")
        for key, value in summary.get("leaderboard", {}).items():
            print(f"  {key}: {value:.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
