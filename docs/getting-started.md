<p align="right">
  <a href="../README.md">Home</a> ·
  <strong>English</strong> ·
  <a href="getting-started.zh-CN.md">简体中文</a>
</p>

# Getting started

## Requirements

- Python 3.12 or newer
- An OpenAI-compatible chat-completions endpoint
- Credentials for the search, scrape, judge, or sandbox providers enabled by your recipe

GPU libraries are not required for a remote model endpoint. Install the `inference` extra only when you need local model or dataset integrations.

## Install

Clone the repository, create a Python 3.12 virtual environment, and run the standalone setup script:

```bash
git clone https://github.com/XYZ-AI-Lab/AxisAgentic.git
cd AxisAgentic
python3.12 -m venv .venv
source .venv/bin/activate
./setup_env.sh
source .envs/axis_agentic_env.sh
```

`setup_env.sh` verifies the active Python version, installs the repository in editable mode, creates local data/model/log directories, and writes `.envs/axis_agentic_env.sh`. It does not modify shell startup files. If Python 3.12 has a different executable name on your system, set `PYTHON_BIN` when invoking the script.

The script exports these path roots:

| Variable | Default |
| --- | --- |
| `AXIS_DATA_DIR` | `<repo>/data` |
| `AXIS_MODEL_DIR` | `<repo>/models` |
| `AXIS_LOG_DIR` | `<repo>/logs` |

Override a root for the current setup invocation when storage lives elsewhere:

```bash
AXIS_DATA_DIR=/data/axis \
AXIS_MODEL_DIR=/models \
AXIS_LOG_DIR=/logs/axis \
./setup_env.sh
```

## Optional dependencies

`AXIS_INSTALL_EXTRAS` is a comma-separated list. Its default is `dev`.

| Extra | Use |
| --- | --- |
| `dev` | pytest, Ruff, and pre-commit |
| `dashboard` | Streamlit benchmark dashboard |
| `inference` | datasets, Transformers, Hugging Face Hub, and progress utilities |
| `sandbox` | E2B-backed Python and shell execution |
| `wide_search` | WideSearch data parsing and tabular evaluation |

For a development environment that can run both retained recipes and the dashboard:

```bash
AXIS_INSTALL_EXTRAS=dev,dashboard,wide_search,sandbox ./setup_env.sh
source .envs/axis_agentic_env.sh
```

You can also install directly with pip:

```bash
python -m pip install -e '.[dev,dashboard,wide_search,sandbox]'
```

The in-process SGLang client additionally requires a CUDA-compatible PyTorch, SGLang, and NVML environment. Install versions appropriate for the serving host separately; they are intentionally not forced by the portable `inference` extra.

## Configure providers

Create a local provider file and fill only the services used by your run:

```bash
cp .env.example .envs/.env
```

At minimum, set `OPENAI_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`. Web-search runs normally also require search and scrape credentials. Judge, summary-LLM, compression-LLM, and sandbox credentials are optional unless the corresponding feature is enabled.

The `.envs/` directory is ignored by Git. See [Configuration](configuration.md) for variable precedence and all supported path schemes.

## Validate a recipe

Copy a default recipe config rather than editing it in place:

```bash
cp recipe/web_search/configs/default.yaml my-search-run.yaml
```

Update at least:

- `benchmark.data_path`
- `run.output_dir`
- provider values in YAML or `.envs/.env`
- concurrency and task limits appropriate for your endpoint

Resolve and print the effective configuration without starting work:

```bash
python -m recipe.web_search.runners.run_eval_config \
  --config my-search-run.yaml \
  --dry-run
```

For a small end-to-end check, set `benchmark.max_tasks` to a low value and reduce `benchmark.max_concurrent`, then remove `--dry-run`.

## Development checks

```bash
python -m pytest -q
python -m ruff check agentic recipe tests
python -m ruff format --check agentic recipe tests
```

Recipe-specific commands are documented in the [recipe index](../recipe/README.md).
