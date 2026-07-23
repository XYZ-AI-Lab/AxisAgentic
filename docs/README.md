<p align="right">
  <a href="../README.md">Home</a> ·
  <strong>English</strong> ·
  <a href="README.zh-CN.md">简体中文</a>
</p>

# AxisAgentic documentation

This directory contains the operational and design documentation for AxisAgentic. Recipe-specific commands and output formats live with the recipes under [`recipe/`](../recipe/README.md).

## Guides

- [Getting started](getting-started.md): requirements, installation, environment setup, and a first dry run.
- [Configuration](configuration.md): YAML models, environment variables, path schemes, and runtime policies.
- [Architecture](architecture.md): component boundaries, request flow, trace semantics, and extension points.
- [Project structure](project-structure.md): repository layout and module responsibilities.
- [Evaluation and reproducibility](evaluation.md): supported benchmarks, run artifacts, dashboard workflow, and comparison caveats.

## Recipe documentation

- [Recipe index](../recipe/README.md)
- [Web-search recipe](../recipe/web_search/README.md)
- [WideSearch recipe](../recipe/wide_search/README.md)
- [Dashboard](../recipe/dashboard/README.md)

The public benchmark configurations are in [`configs/`](../configs/). Complete recipe schemas are in [`recipe/web_search/configs/default.yaml`](../recipe/web_search/configs/default.yaml) and [`recipe/wide_search/configs/default.yaml`](../recipe/wide_search/configs/default.yaml).
