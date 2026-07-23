<p align="right">
  <a href="../README.zh-CN.md">首页</a> ·
  <a href="evaluation.md">English</a> ·
  <strong>简体中文</strong>
</p>

# 评估与可复现性

AxisAgentic 会记录最终生效的运行配置和任务级证据，以支持检查、恢复、重新评判和对比智能体实验。当前公开评估 recipe 主要面向搜索智能体基准。

## 支持的基准 recipe

| 基准 | 入口配置 | 主要评估方式 |
| --- | --- | --- |
| BrowseComp | [`configs/browsecomp.yaml`](../configs/browsecomp.yaml) | 精确匹配/LLM 验证 |
| BrowseComp-ZH | [`configs/browsecompzh.yaml`](../configs/browsecompzh.yaml) | 精确匹配/LLM 验证 |
| DeepSearchQA | [`configs/deepsearchqa.yaml`](../configs/deepsearchqa.yaml) | LLM 验证和宏 F1 评估 |
| GAIA | [`configs/gaia.yaml`](../configs/gaia.yaml) | 基准答案验证 |
| Humanity's Last Exam | [`configs/hle.yaml`](../configs/hle.yaml) | 精确匹配/LLM 验证 |
| LiveBrowseComp | [`configs/livebrowsecomp.yaml`](../configs/livebrowsecomp.yaml) | 多次运行评判与聚合 |
| WideSearch | [`configs/widesearch.yaml`](../configs/widesearch.yaml) | 行级和 item 级 precision/recall/F1 |

[Web Search recipe](../recipe/web_search/README.md) 运行前六类基准。[WideSearch](../recipe/wide_search/README.md) 使用独立的表格答案和裁判模型流程。

## 可复现的运行记录

在完成环境变量与路径解析后，recipe 会写入输入配置和最终生效配置。根据 recipe 的不同，一次运行还可能包含：

- 只追加任务轨迹和每次尝试的元数据；
- 基准输入、预测结果和评估 sidecar；
- token、耗时、工具调用和 assistant 消息摘要；
- 增量与最终聚合指标；
- 供仪表盘使用的紧凑产物。

模型请求和裁判请求 payload 为可选日志。标准轨迹检查不依赖这些内容，而且它们可能包含敏感或体积很大的数据。

使用 `--resume` 可在运行中断后复用已经完成的任务。Web Search runner 默认保护已完成的输出目录；只有在有意重写时才应使用 `--force-resume-finalized-run`。

## 仪表盘

安装 dashboard extra，并针对一个或多个日志根目录启动 Streamlit：

```bash
python -m pip install -e '.[dashboard]'
streamlit run recipe/dashboard/app.py --server.fileWatcherType none -- \
  --log-dir "${AXIS_LOG_DIR}"
```

仪表盘可对比实验、准确率、WideSearch 指标、耗时、轨迹分布、assistant 消息、工具调用、任务详情、最终生效配置和提示词。详情请参阅[仪表盘 README](../recipe/dashboard/README.md)。

## 如何理解报告中的基准图表

顶层[中文 README](../README.zh-CN.md#结果)复现了 XYZ-Aquila 技术报告中的六项对比。报告在公开的智能体搜索基准上评估 XYZ-Aquila-mini 和 XYZ-Aquila-pro，同时不将外部基准用于常规优化决策。

部分基线来自不同的公开报告，其评测框架、Web 访问方式、工具、裁判模型和评估日期可能存在差异。因此，该图表支持基准层面的对比，而不是在完全受控条件下得出的通用排名。

还应注意报告中的以下局限：

- 当前研究尚未对每项干预提供完整的因果分解；
- 即使评估器相互隔离，仍可能出现自适应过拟合；
- 实时 Web 基准会随时间变化；
- 端到端计算量、成本和延迟尚未采用统一口径报告；
- 经过筛选的、以答案为条件的 RL 方案尚未训练至最终验收阶段；
- 当前系统的实证验证主要集中在 Deep Search。

完整协议和分析请阅读 [AI4AI at Scale: A Full-Pipeline System for Enhancing LLM Agentic Capabilities](https://xyz-lab.ai/blogs/ai4ai-at-scale/)。
