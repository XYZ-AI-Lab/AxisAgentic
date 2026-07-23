# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-23

### Added

- Initial public release of AxisAgentic, an extensible runtime and evaluation
  framework for long-horizon AI agents.
- State-faithful multi-turn conversation runtime over append-only traces.
- Pluggable model clients, tools, orchestrators, datasets, evaluators, rewards,
  and recipe-level control policies.
- Long-horizon context budgeting, compaction, rollback, retry/recovery,
  self-verification, and tool limits.
- Structured task traces with timing and token metrics, incremental evaluation
  artifacts, and a comparison dashboard.
- Replayable SFT export and rollout interfaces.
- Reference `web_search` and `wide_search` recipes with benchmark evaluation
  runners.
- Bilingual (English and Simplified Chinese) documentation covering
  architecture, configuration, evaluation, getting started, and project
  structure.

[Unreleased]: https://github.com/XYZ-AI-Lab/AxisAgentic/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/XYZ-AI-Lab/AxisAgentic/releases/tag/v0.1.0
