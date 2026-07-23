# WideSearch recipe

This recipe combines the native web-search agent with WideSearch's tabular answer evaluation. It schedules multiple trials per instance and reports success rate, row-level F1, and item-level F1 aggregates.

## Install

```bash
python -m pip install -e '.[dev,wide_search]'
```

Add the `sandbox` extra if code execution is enabled.

## Configuration

[`configs/default.yaml`](configs/default.yaml) contains the complete strict schema. Important sections are:

- `model`, `agent`, and `tools`: shared web-search runtime settings;
- `benchmark`: dataset, gold directory, concurrency, task selection, and number of trials;
- `agent_prompt`: agent prompt profile and language;
- `eval`: judge endpoint, prompt profile, extractor, and concurrency.

Provider credentials default to `.envs/.env`. Dataset, gold, and output paths support the same portable schemes as web search; see [Configuration](../../docs/configuration.md).

## Run

```bash
python -m recipe.wide_search.runners.run_eval_config \
  --config recipe/wide_search/configs/default.yaml
```

Use `--dry-run` first. Useful overrides include `--max-tasks`, `--num-trials`, and `--resume`. The repository-level reference config is [`configs/widesearch.yaml`](../../configs/widesearch.yaml).

## Output layout

```text
<output_dir>/
  run_config.input.yaml
  run_config.effective.yaml
  wide-search/
  widesearch_scores/
  widesearch_per_task.json
  widesearch_summary.json
  eval_results.json
  dashboard_summary.json
  dashboard_events.jsonl
  trace_distributions.json
  assistant_message_stats.json
```

Trial score sidecars and summaries are updated incrementally, allowing partial results to be inspected while a run is active. With `--resume`, completed instance/trial pairs are reused.

## Evaluation

The evaluator aligns response columns with gold columns, applies configured normalization, joins rows on unique columns, scores cells, and aggregates precision, recall, and F1 at row and item levels. Columns configured for LLM judging are evaluated through an OpenAI-compatible judge endpoint.

Existing outputs can be evaluated, rejudged, or exported directly:

```bash
python -m recipe.wide_search.runners.evaluate_widesearch --help
python -m recipe.wide_search.runners.rejudge_widesearch --help
python -m recipe.wide_search.runners.export_sft --help
```

Use the [dashboard](../dashboard/README.md) for live and completed metric comparisons.

## Tests

```bash
python -m pytest -q \
  tests/test_widesearch_config_and_prompts.py \
  tests/test_widesearch_evaluation.py \
  tests/test_widesearch_orchestrator.py \
  tests/test_widesearch_runner.py \
  tests/test_widesearch_runtime.py
```
