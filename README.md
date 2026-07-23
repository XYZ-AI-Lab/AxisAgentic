<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/axisagentic-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/axisagentic-logo-light.svg">
    <img alt="AxisAgentic - Agent Runtime and Evaluation Framework" src="docs/assets/axisagentic-logo-light.svg" width="680">
  </picture>

  <p>
    <strong>English</strong> ·
    <a href="README.zh-CN.md">简体中文</a>
  </p>

  <p>
    <a href="https://github.com/XYZ-AI-Lab/AxisAgentic/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/XYZ-AI-Lab/AxisAgentic/actions/workflows/ci.yml/badge.svg"></a>
    <img alt="Python 3.12+" src="https://img.shields.io/badge/Python-3.12%2B-blue.svg">
    <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
    <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
    <a href="CONTRIBUTING.md"><img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
  </p>

  <p>
    <a href="https://xyz-lab.ai">XYZ AI Lab</a> ·
    <a href="https://xyz-lab.ai/blogs/ai4ai-at-scale/">Technical Report</a>
  </p>
</div>

AxisAgentic is an extensible runtime and evaluation framework for building long-horizon AI agents with OpenAI-compatible endpoints and pluggable local model clients. It provides state-faithful multi-turn execution, tool orchestration, context management, structured traces, benchmark evaluation, and training-data export while keeping the reusable runtime separate from task-specific recipes, prompts, tools, datasets, and evaluators.

The current release demonstrates the framework through the search-focused XYZ-Aquila recipes. The core runtime is not search-specific: its extension points are designed for additional domain-specific, general-purpose, and coding-agent recipes. Model weights are not bundled in this repository.

## Core capabilities

- State-faithful conversation execution over append-only traces, with exact reconstruction of the context visible at every model turn.
- Pluggable model clients, tools, orchestrators, datasets, evaluators, rewards, and recipe-level control policies.
- Long-horizon context budgets, compaction, rollback, retry/recovery, self-verification, and tool limits.
- Structured task traces, timing and token metrics, incremental evaluation artifacts, and a comparison dashboard.
- Replayable SFT export and rollout interfaces for connecting agent execution to learning systems.
- Strict, reproducible YAML configuration with portable path schemes for data, models, and logs.

## Flagship reference: XYZ-Aquila

XYZ-Aquila is the first flagship system built with AxisAgentic. It demonstrates the framework in long-horizon search workloads using stable search, scrape, and optional code-execution semantics together with context management, recovery, verification, evaluation, and state-faithful SFT export.

The [Aquila technical report](https://xyz-lab.ai/blogs/ai4ai-at-scale/) frames agent improvement as bounded exploration under a human-defined optimization contract. Humans specify the target capability, a private development benchmark, allowed interventions, resource and risk bounds, and an acceptance policy. AI agents can then investigate changes across data, learning, runtime, context, tools, and infrastructure, while an isolated evaluator gates acceptance without exposing hidden labels.

### Results

The report evaluates two systems: XYZ-Aquila-mini, built on Qwen3.6-35B-A3B, and XYZ-Aquila-pro, built on Qwen3.5-397B-A17B. Aquila-mini leads every reported column in the report's sub-40B open-weight table, while Aquila-pro does the same in the sub-400B table. The report also records a 97.1 score for Aquila-mini on GAIA.

![XYZ-Aquila benchmark results across six agentic search benchmarks](docs/assets/aquila-benchmark-results.svg)

| Model | BrowseComp | BrowseComp-ZH | DeepSearchQA F1 | LiveBrowseComp | HLE | WideSearch Item F1 Max@4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XYZ-Aquila-mini | 78.8 | 82.9 | 89.5 | 48.7 | 51.1 | 80.8 |
| XYZ-Aquila-pro | 84.8 | 85.1 | 92.5 | 53.7 | 53.3 | 81.2 |

Some comparison values come from public reports that used different harnesses, tools, judges, and evaluation dates. These numbers should be read as benchmark-level comparisons, not as a single controlled universal ranking. See [Evaluation and reproducibility](docs/evaluation.md) for details.

## Get started

AxisAgentic requires Python 3.12 or newer and an OpenAI-compatible model endpoint.

```bash
git clone https://github.com/XYZ-AI-Lab/AxisAgentic.git
cd AxisAgentic
python3.12 -m venv .venv
source .venv/bin/activate
./setup_env.sh
source .envs/axis_agentic_env.sh
cp .env.example .envs/.env
```

To try the current web-search reference recipe, copy its configuration, set the dataset and provider values, and validate the resolved run before launching it:

```bash
cp recipe/web_search/configs/default.yaml my-search-run.yaml
python -m recipe.web_search.runners.run_eval_config \
  --config my-search-run.yaml \
  --dry-run
```

## Documentation

- [Documentation index](docs/README.md)
- [Getting started](docs/getting-started.md)
- [Configuration](docs/configuration.md)
- [Architecture](docs/architecture.md)
- [Project structure](docs/project-structure.md)
- [Evaluation and reproducibility](docs/evaluation.md)
- [Recipes](recipe/README.md)

## Contributing

Contributions are welcome. Please read the [contributing guide](CONTRIBUTING.md)
for development setup and the checks your change must pass, and note our
[Code of Conduct](CODE_OF_CONDUCT.md). To report a security vulnerability, see
our [security policy](SECURITY.md).

## License

Unless otherwise noted, AxisAgentic is licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for third-party attribution and licensing notes.

## Citation

If you use this codebase, please cite the software:

```bibtex
@software{wang2026axisagentic,
  author       = {Wang, Jinyu and Zhang, Yifei and {{XYZ Agentic Team}}},
  title        = {AxisAgentic: An Extensible Runtime and Evaluation Framework for Long-Horizon Agents},
  organization = {XYZ AI Lab},
  year         = {2026},
  url          = {https://github.com/XYZ-AI-Lab/AxisAgentic}
}
```
