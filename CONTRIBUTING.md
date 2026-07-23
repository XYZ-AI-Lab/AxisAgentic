# Contributing to AxisAgentic

Thanks for your interest in contributing. This document explains how to set up a
development environment, the checks your changes must pass, and how to propose
changes.

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- Report bugs and request features through [GitHub Issues](https://github.com/XYZ-AI-Lab/AxisAgentic/issues).
- Improve documentation (including the Simplified Chinese translations under `docs/`).
- Submit bug fixes and new features via pull requests.
- Add or extend recipes, tools, evaluators, and model clients through the
  framework's extension points.

If you plan a large or architectural change, please open an issue to discuss the
approach before writing code.

## Development setup

AxisAgentic requires **Python 3.12 or newer**.

```bash
git clone https://github.com/XYZ-AI-Lab/AxisAgentic.git
cd AxisAgentic
python3.12 -m venv .venv
source .venv/bin/activate
./setup_env.sh
source .envs/axis_agentic_env.sh
```

`setup_env.sh` installs the package in editable mode with the `dev` extra and
creates local `data/`, `models/`, and `logs/` directories. To install additional
optional dependency groups, set `AXIS_INSTALL_EXTRAS` before running it, for
example `AXIS_INSTALL_EXTRAS="dev,dashboard,inference" ./setup_env.sh`.

Install the git hooks once so lint and formatting run automatically on commit:

```bash
pre-commit install
```

## Checks your change must pass

CI runs the same checks on every pull request. Run them locally before pushing.

### Lint and formatting

```bash
pre-commit run --all-files
```

This runs `ruff` (lint), `ruff-format` (formatting), `shellcheck`, and a
large-file guard. Configuration lives in `.pre-commit-config.yaml` and
`pyproject.toml`.

### Tests

```bash
python -m pytest -q
```

Add tests for any bug fix or new feature. Tests live under `tests/` and use
`pytest`.

## Pull request guidelines

- Branch from `main` and keep each pull request focused on a single change.
- Ensure `pre-commit run --all-files` and `python -m pytest -q` both pass.
- Update relevant documentation, including the `docs/*.zh-CN.md` translations
  when you change their English counterparts.
- Write a clear description of what changed and why. Link any related issue.
- New source files should carry the standard header used across the codebase:

  ```python
  # Copyright 2026 XYZ AI Lab and contributors.
  # SPDX-License-Identifier: Apache-2.0
  ```

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE).
