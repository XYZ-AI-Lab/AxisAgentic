# Web-search recipe

This recipe runs a long-horizon native tool-calling agent against an OpenAI-compatible model endpoint. It supports BrowseComp, BrowseComp-ZH, DeepSearchQA, GAIA, Humanity's Last Exam, and LiveBrowseComp, with web search, page scraping, optional LLM extraction, optional code execution, evaluation, retry/recovery, resume, replay, and state-faithful SFT export.

## Install

The base package is enough for search and scrape. Add `sandbox` for E2B code execution and `inference` for local dataset/model integrations:

```bash
python -m pip install -e '.[dev,sandbox,inference]'
```

## Configuration

Start from [`configs/default.yaml`](configs/default.yaml) or copy one of the benchmark configurations in the repository-level [`configs/`](../../configs/) directory. The strict YAML schema covers:

- model endpoint, sampling, transport, reasoning fields, and context budget;
- benchmark data, task selection, concurrency, and repeated runs;
- agent turn limits, retry/recovery, rollback, context management, and self-verification;
- search, scrape, summary-LLM, cache, and sandbox tools;
- exact-match, LLM-judge, and DeepSearchQA F1 evaluation.

Provider credentials are loaded from the file selected by `run.env_file`, which defaults to `.envs/.env`. Storage and config paths support `repo://`, `axis_data://`, `axis_model://`, and `axis_log://`; see [Configuration](../../docs/configuration.md).

## Run

```bash
python -m recipe.web_search.runners.run_eval_config \
  --config recipe/web_search/configs/default.yaml
```

Use `--dry-run` first to print the resolved config. `--resume` reuses completed tasks; finalized run directories are protected unless `--force-resume-finalized-run` is passed deliberately. Set `benchmark.max_tasks` and `benchmark.max_concurrent` in YAML for a small validation run.

The repository-level entry configs are:

- [`configs/browsecomp.yaml`](../../configs/browsecomp.yaml)
- [`configs/browsecompzh.yaml`](../../configs/browsecompzh.yaml)
- [`configs/deepsearchqa.yaml`](../../configs/deepsearchqa.yaml)
- [`configs/gaia.yaml`](../../configs/gaia.yaml)
- [`configs/hle.yaml`](../../configs/hle.yaml)
- [`configs/livebrowsecomp.yaml`](../../configs/livebrowsecomp.yaml)

## Outputs

Each run directory contains input/effective configurations, task traces, benchmark rows, evaluation sidecars, aggregate results, and compact dashboard artifacts. Model request payloads are written only when `run.model_request_logging` is enabled.

## Post-run tools

Existing result rows can be judged without rerunning inference:

```bash
python -m recipe.web_search.runners.judge_existing --help
```

Export completed traces through the config runner or directly with:

```bash
python -m recipe.web_search.runners.export_sft --help
```

Additional maintenance entry points include:

```bash
python -m recipe.web_search.runners.replay_trace --help
python -m recipe.web_search.runners.aggregate_runs --help
```

Use the [dashboard](../dashboard/README.md) to compare live or completed outputs.

## Tests

```bash
python -m pytest -q \
  tests/test_config_loader.py \
  tests/test_context_compaction.py \
  tests/test_discard_all.py \
  tests/test_web_search_prompt_date.py \
  tests/test_web_search_retry_policy.py \
  tests/test_web_search_tools.py
```
