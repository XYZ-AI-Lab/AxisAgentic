<p align="right">
  <a href="../README.zh-CN.md">首页</a> ·
  <a href="README.md">English</a> ·
  <strong>简体中文</strong>
</p>

# AxisAgentic 文档

本目录包含 AxisAgentic 的操作与设计文档。Recipe 特定的命令和输出格式由 [`recipe/`](../recipe/README.md) 下的相应文档说明。

## 指南

- [快速开始](getting-started.zh-CN.md)：运行要求、安装、环境配置和首次 dry run。
- [配置](configuration.zh-CN.md)：YAML 模型、环境变量、路径方案和运行时策略。
- [架构](architecture.zh-CN.md)：组件边界、请求流程、轨迹语义和扩展点。
- [项目结构](project-structure.zh-CN.md)：仓库布局和模块职责。
- [评估与可复现性](evaluation.zh-CN.md)：支持的基准、运行产物、仪表盘工作流和对比注意事项。

## Recipe 文档

- [Recipe 索引](../recipe/README.md)
- [Web Search recipe](../recipe/web_search/README.md)
- [WideSearch recipe](../recipe/wide_search/README.md)
- [仪表盘](../recipe/dashboard/README.md)

公开的基准配置位于 [`configs/`](../configs/)。完整的 recipe schema 位于 [`recipe/web_search/configs/default.yaml`](../recipe/web_search/configs/default.yaml) 和 [`recipe/wide_search/configs/default.yaml`](../recipe/wide_search/configs/default.yaml)。
