<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/axisagentic-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/axisagentic-logo-light.svg">
    <img alt="AxisAgentic - 运行时与轨迹采集框架" src="docs/assets/axisagentic-logo-light.svg" width="680">
  </picture>

  <p>
    <a href="README.md">English</a> ·
    <strong>简体中文</strong>
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
    <a href="https://xyz-lab.ai/blogs/ai4ai-at-scale/">技术报告</a>
  </p>
</div>

AxisAgentic 是一个面向长时程 AI 智能体的可扩展执行与轨迹采集框架，用于通过兼容 OpenAI 的端点和可插拔本地模型客户端运行 Agent。它提供状态忠实的多轮执行、工具编排、上下文管理、结构化轨迹、基准评估和训练数据导出：同一条轨迹既支持 inference 阶段的恢复、回放与分析，也支持将真实可见状态转换为可筛选、可重放的 SFT 数据。

仓库当前以 Web Search 和 WideSearch recipe 提供搜索任务参考实现；XYZ-Aquila 是一个基于 AxisAgentic 构建的旗舰系统，而 AxisAgentic 的扩展点同样适用于更多特定领域、通用型和编程智能体 recipe。本仓库不包含模型权重。

## ✨ 核心能力

- 基于只追加轨迹实现状态忠实的对话执行，可重建模型各阶段实际可见的上下文。
- 可插拔的模型客户端、工具、编排器、数据集、评估器、奖励函数和 recipe 级控制策略。
- 面向长时程任务的上下文预算、压缩、回滚、重试与恢复、自我验证和工具限制。
- 结构化任务轨迹、耗时与 token 指标、增量评估产物和对比仪表盘，为数据筛选保留 provenance。
- 基于运行时可见性重放的 SFT 导出，以及用于连接外部训练系统的 rollout 接口。
- 严格、可复现的 YAML 配置，并为数据、模型和日志提供可移植路径方案。

## 🔁 从执行到学习的数据闭环

AxisAgentic 将运行时和轨迹采集放在同一条语义链路中：

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/axisagentic-execution-learning-loop-zh-CN-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/axisagentic-execution-learning-loop-zh-CN-light.svg">
  <img alt="AxisAgentic 从执行到学习的数据闭环：运行轨迹支持回放、评估、轨迹采集、状态忠实的 SFT 导出和外部训练" src="docs/assets/axisagentic-execution-learning-loop-zh-CN-light.svg">
</picture>

运行时 marker 记录 rollback、context compaction 和 discard-all 等可见性变化。重放这些 marker 可以重建模型在具体阶段实际看到的上下文；同一语义也被用于轨迹检查和训练数据导出，避免把不可见历史或已回滚的无效动作直接写入监督样本。

内置 recipe exporter 负责生成 Swift Agent 等训练格式，并保留 source trace、任务状态和可选元数据。训练样本的最终正确性筛选、loss mask 和训练框架由外部流程决定；AxisAgentic 提供的是保持 inference 与 training 交互语义一致的 trace、replay 和导出边界。

## 🦅 旗舰参考系统：XYZ-Aquila

XYZ-Aquila 是一个基于 AxisAgentic 构建的旗舰搜索系统。它将搜索、抓取、上下文管理、恢复、评估和状态忠实的 SFT 数据导出组合在一起，展示了一个完整的智能体系统和训练流程如何复用 AxisAgentic 的执行与轨迹采集能力。同样的接口也可以用于其他模型客户端、工具和任务领域。

### 📊 评测结果

[Aquila 技术报告](https://xyz-lab.ai/blogs/ai4ai-at-scale/)给出了 XYZ-Aquila-mini 和 XYZ-Aquila-pro 在七项智能体基准上的评测结果。下图复现了其中六项基准的对比。

![XYZ-Aquila 在六项智能体搜索基准上的评测结果](docs/assets/aquila-benchmark-results.svg)

*图中指标：BrowseComp — LLM judge accuracy；BrowseComp-ZH — LLM judge accuracy；DeepSearchQA — macro F1；LiveBrowseComp — LLM judge accuracy；Humanity's Last Exam — LLM judge accuracy；WideSearch — item-level F1 Max@4。详情请参阅[评估与可复现性](docs/evaluation.zh-CN.md)。*

部分对比结果来自公开报告，所使用的评测框架、工具、评判模型和评测日期可能不同。因此，这些数字适合用于基准层面的对照，不应被理解为来自单一受控实验的通用排名。

## 🚀 快速开始

AxisAgentic 需要 Python 3.12 或更高版本，以及一个兼容 OpenAI 的模型端点。安装、服务变量、recipe 配置、dry run、回放和 SFT 导出的完整说明，请参阅[快速开始](docs/getting-started.zh-CN.md)与[配置](docs/configuration.zh-CN.md)。

```bash
git clone https://github.com/XYZ-AI-Lab/AxisAgentic.git
cd AxisAgentic
python3.12 -m venv .venv
source .venv/bin/activate
./setup_env.sh
source .envs/axis_agentic_env.sh
cp .env.example .envs/.env
```

配置好模型服务和数据集后，可以在不启动任务的情况下校验 Web Search 参考 recipe：

```bash
cp recipe/web_search/configs/default.yaml my-search-run.yaml
python -m recipe.web_search.runners.run_eval_config \
  --config my-search-run.yaml \
  --dry-run
```

## 📚 文档

- [文档索引](docs/README.zh-CN.md)
- [快速开始](docs/getting-started.zh-CN.md)
- [配置](docs/configuration.zh-CN.md)
- [架构](docs/architecture.zh-CN.md)
- [项目结构](docs/project-structure.zh-CN.md)
- [评估与可复现性](docs/evaluation.zh-CN.md)
- [Recipes](recipe/README.md)

## 🤝 参与贡献

欢迎贡献。开发环境搭建及提交需通过的检查项，请阅读[贡献指南](CONTRIBUTING.md)，并遵守我们的[行为准则](CODE_OF_CONDUCT.md)。如需上报安全漏洞，请参阅[安全策略](SECURITY.md)。

## 📜 许可证

除非另有说明，AxisAgentic 采用 [Apache License 2.0](LICENSE) 许可。第三方归属和许可说明请参阅 [NOTICE](NOTICE)。

## 📝 引用

如果你使用了本代码库，请引用以下软件条目：

```bibtex
@software{wang2026axisagentic,
  author       = {Wang, Jinyu and Zhang, Yifei and {{XYZ Agentic Team}}},
  title        = {AxisAgentic: An Extensible Execution and Trajectory-Collection Framework for Long-Horizon Agents},
  organization = {XYZ AI Lab},
  year         = {2026},
  url          = {https://github.com/XYZ-AI-Lab/AxisAgentic}
}
```
