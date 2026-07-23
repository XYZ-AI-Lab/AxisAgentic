# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import yaml

from agentic.config.models import RunConfig


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    with config_path.open() as file_obj:
        raw_config = yaml.safe_load(file_obj) or {}
    return RunConfig.model_validate(raw_config)


def dump_run_config(config: RunConfig, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file_obj:
        yaml.safe_dump(config.model_dump(mode="json"), file_obj, sort_keys=False)
