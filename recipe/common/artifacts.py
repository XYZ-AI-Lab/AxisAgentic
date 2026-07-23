# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Shared artifact / environment-variable helpers for agentic recipes.

All agentic recipes resolve datasets and model snapshots under two canonical environment-variable roots:

- ``AXIS_DATA_DIR`` — root for datasets and retrieval index / corpus files.
- ``AXIS_MODEL_DIR`` — root for tokenizer / encoder / policy model snapshots.

This module centralises the environment-variable lookups and the "resolve a model by name or absolute path" convention
so individual recipes stay download-free and fail fast on misconfiguration.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR_ENV = "AXIS_DATA_DIR"
MODEL_DIR_ENV = "AXIS_MODEL_DIR"


def require_env_dir(var: str) -> Path:
    """Return ``$var`` as a ``Path``, requiring the variable be set and point at an existing directory.

    Raises:
        RuntimeError: When *var* is unset / empty, or when its value is not an existing directory.
    """
    value = os.environ.get(var)
    if not value:
        msg = f"Environment variable {var} must be set; loading from local cache or remote download is disabled."
        raise RuntimeError(msg)
    path = Path(value)
    if not path.is_dir():
        msg = f"{var}={value} is not an existing directory."
        raise RuntimeError(msg)
    return path


def data_dir() -> Path:
    """Return the validated ``$AXIS_DATA_DIR`` root."""
    return require_env_dir(DATA_DIR_ENV)


def model_dir() -> Path:
    """Return the validated ``$AXIS_MODEL_DIR`` root."""
    return require_env_dir(MODEL_DIR_ENV)


def resolve_model_path(model_name_or_path: str) -> Path:
    """Resolve *model_name_or_path* to a local ``Path``.

    If *model_name_or_path* is an existing directory, it is returned unchanged.
    Otherwise it is joined onto ``$AXIS_MODEL_DIR`` (which must be set).
    The returned path is **not** guaranteed to exist; use :func:`ensure_model_artifact`
    when you need that guarantee.
    """
    candidate = Path(model_name_or_path)
    if candidate.is_dir():
        return candidate
    return model_dir() / model_name_or_path


def ensure_model_artifact(model_name_or_path: str) -> Path:
    """Resolve *model_name_or_path* and require the target directory exists.

    Raises:
        RuntimeError: When ``$AXIS_MODEL_DIR`` is unset (and *model_name_or_path* is not an absolute dir).
        FileNotFoundError: When the resolved directory does not exist.
    """
    target = resolve_model_path(model_name_or_path)
    if not target.is_dir():
        msg = f"Model directory not found at {target}; expected under ${MODEL_DIR_ENV}."
        raise FileNotFoundError(msg)
    logger.info("Using model at %s.", target)
    return target
