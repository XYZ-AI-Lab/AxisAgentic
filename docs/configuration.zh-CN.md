<p align="right">
  <a href="../README.zh-CN.md">首页</a> ·
  <a href="configuration.md">English</a> ·
  <strong>简体中文</strong>
</p>

# 配置

AxisAgentic 将机器特定的服务凭据和存储根目录与受版本控制的 YAML 实验设置分离。

## 配置层级

1. 当前 shell 提供存储根目录，也可以提供凭据。
2. `run.env_file` 加载尚未设置的服务变量，默认值为 `.envs/.env`。
3. YAML 选择模型、数据集、策略、工具、评估方式和输出路径。
4. CLI 参数覆盖 recipe runner 暴露的少量配置，例如 resume 或任务数量限制。

对于可在 YAML 中指定的服务字段，非空 YAML 值优先于环境变量回退值。env 文件加载器不会覆盖进程环境中已经存在的变量。

Recipe 配置由严格的 Pydantic 模型验证：未知字段会立即导致失败，而不会被静默忽略。每次运行都会将输入配置和解析后的最终配置写入输出目录。

## 服务环境变量

从 [`.env.example`](../.env.example) 开始配置。常用变量包括：

| 变量 | 用途 |
| --- | --- |
| `OPENAI_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY` | 主要的 OpenAI 兼容模型 |
| `SERPER_API_KEY`, `SERPER_BASE_URL` | Web 搜索 |
| `JINA_API_KEY`, `JINA_BASE_URL` | 页面抓取 |
| `JUDGE_MODEL`, `JUDGE_BASE_URL`, `JUDGE_API_KEY` | 可选的评估模型；在支持的场景中回退到主模型端点 |
| `SUMMARY_LLM_MODEL_NAME`, `SUMMARY_LLM_BASE_URL`, `SUMMARY_LLM_API_KEY` | 可选的抓取内容提取与摘要模型 |
| `COMPRESSION_LLM_MODEL_NAME`, `COMPRESSION_LLM_BASE_URL`, `COMPRESSION_LLM_API_KEY` | 可选的上下文压缩模型 |
| `E2B_API_KEY` | 可选的代码沙箱 |

运行所使用的 API Key 环境变量名由 YAML 选择，例如 `model.api_key_env`、`judge.api_key_env` 或 `eval.judge_api_key_env`。

## 可移植路径

Recipe runner 会展开以下路径方案：

| Scheme | 根目录 |
| --- | --- |
| `repo://path` | 仓库根目录 |
| `axis_data://path` | `AXIS_DATA_DIR` |
| `axis_model://path` | `AXIS_MODEL_DIR` |
| `axis_log://path` | `AXIS_LOG_DIR` |

绝对路径保持不变。普通相对路径从仓库根目录解析。如果路径方案所需的环境变量未设置，程序会给出清晰错误并终止。

示例：

```yaml
run:
  output_dir: axis_log://web_search_infer/my_experiment
  env_file: .envs/.env

benchmark:
  name: browsecomp
  data_path: axis_data://benchmarks/browsecomp/data.jsonl
  max_tasks: 20
  max_concurrent: 4
```

## Recipe schema

以下带完整注释的 schema 是最佳起点：

- [Web Search](../recipe/web_search/configs/default.yaml)
- [WideSearch](../recipe/wide_search/configs/default.yaml)

仓库级 [`configs/`](../configs/) 文件记录 BrowseComp、BrowseComp-ZH、DeepSearchQA、GAIA、Humanity's Last Exam、LiveBrowseComp 和 WideSearch 的基准特定示例设置。请根据自己的环境调整其中的模型名称、端点和可移植数据集位置。

Web Search 的主要配置段包括：

| 配置段 | 控制内容 |
| --- | --- |
| `run` | 输出目录、运行次数、resume 策略、env 文件、请求日志 |
| `model` | 模型端点、采样、推理字段、传输重试、上下文窗口 |
| `benchmark` | 数据集、任务选择、shuffle、并发数 |
| `agent` | 轮次、重试次数、上下文策略、自我验证、工具预算 |
| `tools` | 搜索、抓取、内容提取、缓存和可选代码执行 |
| `judge` | 精确匹配/LLM 评估和 DeepSearchQA 验证 |

WideSearch 复用运行时配置段，并增加用于表格匹配和裁判模型辅助单元格评估的 `agent_prompt` 与 `eval` 配置段。

## 上下文与轨迹策略

运行时保留只追加的完整事件历史，同时派生每次模型调用实际可见的精确消息列表。相关配置包括：

- `model.context`：上下文大小、安全余量、token 估算器和限制检测；
- `agent.context_compression`：总结较早的历史，同时保留最近交互；
- `agent.discard_all`：达到阈值后，以干净的可见上下文重新开始任务；
- `agent.retry` 下的回滚和生成限制恢复设置。

`context_compression` 与 `discard_all` 互斥；同时启用时配置验证会失败。上下文事件会被记录到轨迹中，使 SFT 导出能够重建每个 assistant 轮次实际可见的状态。

## 凭据与请求日志

`.env`、`.envs/`、日志、输出以及本地数据和模型目录均由 [`.gitignore`](../.gitignore) 排除。模型和裁判请求 payload 日志默认关闭，因为 payload 可能很大，也可能包含敏感任务内容。只有在输出位置和保留策略均合适时才应启用。
