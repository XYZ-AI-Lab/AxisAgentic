<p align="right">
  <a href="../README.zh-CN.md">首页</a> ·
  <a href="getting-started.md">English</a> ·
  <strong>简体中文</strong>
</p>

# 快速开始

## 运行要求

- Python 3.12 或更高版本
- 兼容 OpenAI chat completions 的模型端点
- Recipe 所启用的搜索、抓取、裁判模型或沙箱服务凭据

使用远程模型端点时不需要 GPU 库。只有在需要本地模型或数据集集成时才安装 `inference` extra。

## 安装

克隆仓库，创建 Python 3.12 虚拟环境，然后运行独立的初始化脚本：

```bash
git clone https://github.com/XYZ-AI-Lab/AxisAgentic.git
cd AxisAgentic
python3.12 -m venv .venv
source .venv/bin/activate
./setup_env.sh
source .envs/axis_agentic_env.sh
```

`setup_env.sh` 会检查当前 Python 版本，以 editable 模式安装仓库，创建本地数据、模型和日志目录，并写入 `.envs/axis_agentic_env.sh`。它不会修改 shell 启动文件。如果系统中的 Python 3.12 使用其他可执行文件名，请在调用脚本时设置 `PYTHON_BIN`。

该脚本会导出以下路径根目录：

| 变量 | 默认值 |
| --- | --- |
| `AXIS_DATA_DIR` | `<repo>/data` |
| `AXIS_MODEL_DIR` | `<repo>/models` |
| `AXIS_LOG_DIR` | `<repo>/logs` |

如果存储位于其他位置，可在本次初始化时覆盖相应根目录：

```bash
AXIS_DATA_DIR=/data/axis \
AXIS_MODEL_DIR=/models \
AXIS_LOG_DIR=/logs/axis \
./setup_env.sh
```

## 可选依赖

`AXIS_INSTALL_EXTRAS` 是一个以逗号分隔的列表，默认值为 `dev`。

| Extra | 用途 |
| --- | --- |
| `dev` | pytest、Ruff 和 pre-commit |
| `dashboard` | Streamlit 基准评估仪表盘 |
| `inference` | datasets、Transformers、Hugging Face Hub 和进度工具 |
| `sandbox` | 基于 E2B 的 Python 与 shell 执行 |
| `wide_search` | WideSearch 数据解析和表格评估 |

要创建可运行当前两个 recipe 和仪表盘的开发环境：

```bash
AXIS_INSTALL_EXTRAS=dev,dashboard,wide_search,sandbox ./setup_env.sh
source .envs/axis_agentic_env.sh
```

也可以直接使用 pip 安装：

```bash
python -m pip install -e '.[dev,dashboard,wide_search,sandbox]'
```

进程内 SGLang 客户端还需要兼容 CUDA 的 PyTorch、SGLang 和 NVML 环境。请根据模型服务主机单独安装合适的版本；可移植的 `inference` extra 有意不强制安装这些依赖。

## 配置服务提供方

创建本地服务配置文件，并只填写当前运行所使用的服务：

```bash
cp .env.example .envs/.env
```

至少需要设置 `OPENAI_MODEL`、`OPENAI_BASE_URL` 和 `OPENAI_API_KEY`。Web Search 运行通常还需要搜索和网页抓取服务的凭据。只有启用对应功能时，才需要裁判模型、摘要 LLM、压缩 LLM 和沙箱凭据。

`.envs/` 目录已被 Git 忽略。变量优先级和所有支持的路径方案请参阅[配置](configuration.zh-CN.md)。

## 验证 recipe

请复制默认 recipe 配置，不要直接修改原文件：

```bash
cp recipe/web_search/configs/default.yaml my-search-run.yaml
```

至少需要更新：

- `benchmark.data_path`
- `run.output_dir`
- YAML 或 `.envs/.env` 中的服务提供方配置
- 与模型端点能力相匹配的并发数和任务限制

在不启动任务的情况下解析并打印最终生效的配置：

```bash
python -m recipe.web_search.runners.run_eval_config \
  --config my-search-run.yaml \
  --dry-run
```

要执行小规模端到端检查，请将 `benchmark.max_tasks` 设置为较小值并降低 `benchmark.max_concurrent`，然后移除 `--dry-run`。

## 检查和导出轨迹

任务完成后，将 `<attempt.json>` 指向运行目录 `web-search-benchmark/` 下的 `<task>_attempt-<n>.json`，即可重建某个阶段的模型可见上下文：

```bash
python -m recipe.web_search.runners.replay_trace \
  <attempt.json> --show-visible
```

也可以将完成的 Web Search 轨迹导出为 Swift Agent SFT 数据。导出器会按运行时 marker 重放可见性，并可将 source trace、任务状态等信息写入样本元数据：

```bash
python -m recipe.web_search.runners.export_sft \
  --run-dir <run-dir> --include-metadata
```

WideSearch 的导出入口和按分数筛选选项请参阅 [WideSearch recipe](../recipe/wide_search/README.md)。

## 开发检查

```bash
python -m pytest -q
python -m ruff check agentic recipe tests
python -m ruff format --check agentic recipe tests
```

Recipe 特定命令请参阅 [recipe 索引](../recipe/README.md)。
