# Recipes

Recipes assemble the reusable `agentic` runtime into benchmark-ready applications. Each recipe owns its configuration model, prompts, runners, evaluators, output artifacts, and detailed usage notes.

## Available recipes

- [Web search](web_search/README.md): long-horizon deep-search agent for BrowseComp, BrowseComp-ZH, DeepSearchQA, GAIA, Humanity's Last Exam, and LiveBrowseComp.
- [WideSearch](wide_search/README.md): repeated-trial broad information-seeking with tabular matching and row/item F1 metrics.
- [Dashboard](dashboard/README.md): Streamlit comparison and trace-inspection interface for both recipe families.
- [`common/`](common/README.md): shared artifact, retry, evaluation-summary, and log-processing helpers.

## Typical workflow

1. Install the extras required by the selected recipe.
2. Copy its default YAML and set provider, dataset, output, concurrency, and task-limit values.
3. Run the config entry point with `--dry-run` to inspect the resolved config.
4. Launch the run, then resume, rejudge, aggregate, or export traces with that recipe's tools.
5. Inspect completed or live artifacts in the dashboard.

Environment setup and path resolution are documented in [Getting started](../docs/getting-started.md) and [Configuration](../docs/configuration.md).
