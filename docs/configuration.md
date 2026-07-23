<p align="right">
  <a href="../README.md">Home</a> ·
  <strong>English</strong> ·
  <a href="configuration.zh-CN.md">简体中文</a>
</p>

# Configuration

AxisAgentic separates machine-specific provider secrets and storage roots from versioned YAML experiment settings.

## Configuration layers

1. The active shell provides storage roots and may provide credentials.
2. `run.env_file` loads missing provider variables; it defaults to `.envs/.env`.
3. YAML selects models, datasets, policies, tools, evaluation, and output paths.
4. CLI flags override the small set of values exposed by a recipe runner, such as resume or task limits.

For provider fields that can be specified in YAML, a non-empty YAML value wins over its environment fallback. The env-file loader never replaces a variable already present in the process environment.

Recipe configuration is validated with strict Pydantic models: unknown keys fail early instead of being silently ignored. Each run writes both the input and resolved configurations to its output directory.

## Provider environment

Start from [`.env.example`](../.env.example). Common variables are:

| Variable | Purpose |
| --- | --- |
| `OPENAI_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY` | Primary OpenAI-compatible model |
| `SERPER_API_KEY`, `SERPER_BASE_URL` | Web search |
| `JINA_API_KEY`, `JINA_BASE_URL` | Page scraping |
| `JUDGE_MODEL`, `JUDGE_BASE_URL`, `JUDGE_API_KEY` | Optional evaluator model; falls back to the primary model endpoint where supported |
| `SUMMARY_LLM_MODEL_NAME`, `SUMMARY_LLM_BASE_URL`, `SUMMARY_LLM_API_KEY` | Optional scrape extraction/summarization model |
| `COMPRESSION_LLM_MODEL_NAME`, `COMPRESSION_LLM_BASE_URL`, `COMPRESSION_LLM_API_KEY` | Optional context-compression model |
| `E2B_API_KEY` | Optional code sandbox |

API-key variable names used by a run are selected in YAML, for example `model.api_key_env`, `judge.api_key_env`, or `eval.judge_api_key_env`.

## Portable paths

Recipe runners expand the following schemes:

| Scheme | Root |
| --- | --- |
| `repo://path` | repository root |
| `axis_data://path` | `AXIS_DATA_DIR` |
| `axis_model://path` | `AXIS_MODEL_DIR` |
| `axis_log://path` | `AXIS_LOG_DIR` |

Absolute paths are preserved. Plain relative paths are resolved from the repository root. A scheme fails with a clear error when its required environment variable is not set.

Example:

```yaml
run:
  output_dir: axis_log://web_search_infer/my_experiment
  env_file: .envs/.env

benchmark:
  name: browsecomp
  data_path: axis_data://benchmarks/browsecomp/data.jsonl
  max_tasks: 20
  max_concurrent: 4
```

## Recipe schemas

The full, commented schemas are the best starting point:

- [Web search](../recipe/web_search/configs/default.yaml)
- [WideSearch](../recipe/wide_search/configs/default.yaml)

Repository-level files in [`configs/`](../configs/) capture benchmark-specific example settings for BrowseComp, BrowseComp-ZH, DeepSearchQA, GAIA, Humanity's Last Exam, LiveBrowseComp, and WideSearch. Adapt their model names, endpoints, and portable dataset locations to your environment.

The major web-search sections are:

| Section | Controls |
| --- | --- |
| `run` | output directory, number of runs, resume policy, env file, request logging |
| `model` | endpoint, sampling, reasoning fields, transport retries, context window |
| `benchmark` | dataset, task selection, shuffle, concurrency |
| `agent` | turns, retry attempts, context strategy, self-verification, tool budgets |
| `tools` | search, scrape, extraction, cache, and optional code execution |
| `judge` | exact/LLM evaluation and DeepSearchQA verification |

WideSearch shares the runtime sections and adds `agent_prompt` plus an `eval` section for tabular matching and judge-assisted cells.

## Context and trace policies

The runtime retains an append-only event history while deriving the exact message list visible to each model call. Relevant controls include:

- `model.context`: context size, safety margin, token estimator, and limit detection;
- `agent.context_compression`: summarize older history while retaining recent interactions;
- `agent.discard_all`: reopen the task from a clean visible context after a threshold;
- rollback and generation-limit recovery settings under `agent.retry`.

`context_compression` and `discard_all` are mutually exclusive and validation fails if both are enabled. Context events are recorded in traces so SFT export can reconstruct the actual visible state for every assistant turn.

## Secrets and request logs

`.env`, `.envs/`, logs, outputs, and local data/model directories are excluded by [`.gitignore`](../.gitignore). Model and judge payload logging is disabled by default because payloads can be large and may contain sensitive task content. Enable it only when the output location and retention policy are appropriate.
