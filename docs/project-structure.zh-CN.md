<p align="right">
  <a href="../README.zh-CN.md">首页</a> ·
  <a href="project-structure.md">English</a> ·
  <strong>简体中文</strong>
</p>

# 项目结构

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

这是可复用的核心库。它负责运行时契约与行为，但不包含基准特定的提示词或文件系统布局。`model_assets/` 有意只包含运行时所需的兼容性注册表和 manifest；仓库不包含模型 chat template、tokenizer、权重或模型服务补丁。

## `recipe/`

Recipe 是核心库的可执行组合。它们负责自己的类型化 YAML schema、提示词、任务加载器、runner、验证器、后处理和文档。共享输出约定和仪表盘产物构建器位于 `recipe/common/`。

入口请参阅 [recipe 索引](../recipe/README.md)。

## `configs/`

这些文件记录当前保留 recipe 的基准设置。路径采用可移植的 `axis_data://` 和 `axis_log://` 方案，而模型名称和运行时设置保留为参考值。用户应复制配置并根据自己的环境进行调整，而不是直接修改参考文件。

## `tests/`

测试套件覆盖核心状态机、上下文处理、工具执行、轨迹记录、resume 行为、模型客户端、RL/SFT 边界和两个公开搜索 recipe。它有意不包含内部平台、托管、部署和仅用于研究的组件。
