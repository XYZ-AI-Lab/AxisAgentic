<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/axisagentic-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/axisagentic-logo-light.svg">
    <img alt="AxisAgentic - Runtime and Trajectory Collection Framework" src="docs/assets/axisagentic-logo-light.svg" width="680">
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

AxisAgentic is an extensible execution and trajectory-collection framework for long-horizon AI agents built with OpenAI-compatible endpoints and pluggable local model clients. It provides state-faithful multi-turn execution, tool orchestration, context management, structured traces, benchmark evaluation, and training-data export: the same trace supports recovery, replay, and analysis during inference and can be converted into filtered, replayable SFT data from the exact runtime-visible state.

The repository currently provides Web Search and WideSearch recipes as reference implementations for search workloads. XYZ-Aquila is a flagship system built with AxisAgentic, and the same extension points support additional domain-specific, general-purpose, and coding-agent recipes. Model weights are not bundled in this repository.

## ✨ Core capabilities

- State-faithful conversation execution over append-only traces, with reconstruction of the context visible at each runtime stage.
- Pluggable model clients, tools, orchestrators, datasets, evaluators, rewards, and recipe-level control policies.
- Long-horizon context budgets, compaction, rollback, retry/recovery, self-verification, and tool limits.
- Structured task traces, timing and token metrics, incremental evaluation artifacts, and provenance for data selection.
- Runtime-visibility-aware SFT export and rollout interfaces for connecting agent execution to external training systems.
- Strict, reproducible YAML configuration with portable path schemes for data, models, and logs.

## 🔁 From execution to learning

AxisAgentic links inference and trajectory collection through one semantic path:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/axisagentic-execution-learning-loop-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/axisagentic-execution-learning-loop-light.svg">
  <img alt="AxisAgentic execution-to-learning loop: runtime traces feed replay, evaluation, trajectory collection, state-faithful SFT export, and external training" src="docs/assets/axisagentic-execution-learning-loop-light.svg">
</picture>

Runtime markers record visibility changes such as rollback, context compaction, and discard-all. Replaying those markers reconstructs what the model actually saw at a given stage; the same semantics drive trace inspection and training-data export, avoiding hidden history or rolled-back actions in supervised examples.

The built-in recipe exporters produce Swift Agent and related training formats while preserving source-trace, task-status, and optional metadata. Final correctness filtering, loss masks, and the training framework remain with the external training pipeline; AxisAgentic provides the trace, replay, and export boundary that keeps inference and training interaction semantics aligned.

## 🦅 Flagship reference: XYZ-Aquila

XYZ-Aquila is a flagship search system built with AxisAgentic. It combines search, scrape, context management, recovery, evaluation, and state-faithful SFT data export to demonstrate how a complete agent system and training pipeline can reuse AxisAgentic's execution and trajectory-collection capabilities. The same interfaces can be applied to other model clients, tools, and task domains.

### 📊 Results

The [Aquila technical report](https://xyz-lab.ai/blogs/ai4ai-at-scale/) reports XYZ-Aquila-mini and XYZ-Aquila-pro across seven agentic benchmarks. The figure below reproduces the reported comparisons for six of them.

![XYZ-Aquila benchmark results across six agentic search benchmarks](docs/assets/aquila-benchmark-results.svg)

*Metrics shown in the figure: BrowseComp — LLM-judge accuracy; BrowseComp-ZH — LLM-judge accuracy; DeepSearchQA — macro F1; LiveBrowseComp — LLM-judge accuracy; Humanity's Last Exam — LLM-judge accuracy; WideSearch — item-level F1 Max@4. See [Evaluation and reproducibility](docs/evaluation.md) for details.*

Some comparison values come from public reports that used different harnesses, tools, judges, and evaluation dates. These numbers should be read as benchmark-level comparisons, not as a single controlled universal ranking.

## 🚀 Get started

AxisAgentic requires Python 3.12 or newer and an OpenAI-compatible model endpoint. For installation, provider variables, recipe configuration, dry runs, replay, and SFT export, see [Getting started](docs/getting-started.md) and [Configuration](docs/configuration.md).

```bash
git clone https://github.com/XYZ-AI-Lab/AxisAgentic.git
cd AxisAgentic
python3.12 -m venv .venv
source .venv/bin/activate
./setup_env.sh
source .envs/axis_agentic_env.sh
cp .env.example .envs/.env
```

After setting the provider and dataset values, validate the Web Search reference recipe without starting a run:

```bash
cp recipe/web_search/configs/default.yaml my-search-run.yaml
python -m recipe.web_search.runners.run_eval_config \
  --config my-search-run.yaml \
  --dry-run
```

## 📚 Documentation

- [Documentation index](docs/README.md)
- [Getting started](docs/getting-started.md)
- [Configuration](docs/configuration.md)
- [Architecture](docs/architecture.md)
- [Project structure](docs/project-structure.md)
- [Evaluation and reproducibility](docs/evaluation.md)
- [Recipes](recipe/README.md)

## 🤝 Contributing

Contributions are welcome. Please read the [contributing guide](CONTRIBUTING.md)
for development setup and the checks your change must pass, and note our
[Code of Conduct](CODE_OF_CONDUCT.md). To report a security vulnerability, see
our [security policy](SECURITY.md).

## 📜 License

Unless otherwise noted, AxisAgentic is licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for third-party attribution and licensing notes.

## 📝 Citation

If you use this codebase, please cite the software:

```bibtex
@software{wang2026axisagentic,
  author       = {Wang, Jinyu and Zhang, Yifei and {{XYZ Agentic Team}}},
  title        = {AxisAgentic: An Extensible Execution and Trajectory-Collection Framework for Long-Horizon Agents},
  organization = {XYZ AI Lab},
  year         = {2026},
  url          = {https://github.com/XYZ-AI-Lab/AxisAgentic}
}
```
