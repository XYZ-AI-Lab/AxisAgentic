<p align="right">
  <a href="../README.md">Home</a> ·
  <strong>English</strong> ·
  <a href="project-structure.zh-CN.md">简体中文</a>
</p>

# Project structure

```text
AxisAgentic/
├── agentic/                    reusable runtime library
│   ├── config/                 generic runtime config models and YAML I/O
│   ├── contracts/              messages, tool calls, results, and markers
│   ├── conversations/          conversation state and context budgets
│   ├── datasets/               dataset interfaces
│   ├── evaluation/             verifier/evaluator interfaces
│   ├── model_assets/           endpoint compatibility registry and manifest
│   ├── model_clients/          model transports, retries, and request logging
│   ├── observability/          task traces, timing, and token/tool metrics
│   ├── orchestration/          model/tool execution loop
│   ├── rewards/                reward interfaces
│   ├── rl/                     rollout facade and client boundary
│   ├── sft_export/             state-faithful training-data conversion
│   └── tools/                  tool base classes and search/sandbox tools
├── configs/                    benchmark-specific public run configs
├── docs/                       project documentation and visual assets
├── recipe/
│   ├── common/                 shared artifacts and log post-processing
│   ├── dashboard/              Streamlit run-comparison dashboard
│   ├── web_search/             deep-search agent, runners, and evaluators
│   └── wide_search/            WideSearch agent and tabular evaluator
├── tests/                      focused runtime and retained-recipe tests
├── .env.example               provider configuration template
├── setup_env.sh               local environment/bootstrap script
├── setup.py                   package metadata and dependency extras
└── pyproject.toml             build, linting, and type-checker configuration
```

## `agentic/`

This is the reusable library. It owns runtime contracts and behavior, but not benchmark-specific prompts or filesystem layouts. `model_assets/` intentionally contains only the compatibility registry/manifest needed at runtime; model chat templates, tokenizers, weights, and serving patches are not bundled.

## `recipe/`

Recipes are executable compositions of the core library. They own their typed YAML schema, prompts, task loader, runner, verifier, post-processing, and documentation. Shared output conventions and dashboard artifact builders live in `recipe/common/`.

See the [recipe index](../recipe/README.md) for entry points.

## `configs/`

These files record benchmark setups for the retained recipes. Paths use the portable `axis_data://` and `axis_log://` schemes, while model names and runtime settings remain reference values. Users should copy a config and adapt it to their environment rather than editing the reference file in place.

## `tests/`

The test suite covers the core state machine, context handling, tool execution, tracing, resume behavior, model client, RL/SFT boundaries, and the two public search recipes. It intentionally excludes internal platform, hosting, deployment, and research-only components.
