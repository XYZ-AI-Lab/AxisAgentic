# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Cached live eval-result reconstruction wrappers for dashboard use."""

from __future__ import annotations

from typing import Any

import streamlit as st

from recipe.common.log_processing import live_eval_results as _common_live_results

__all__ = ["_build_live_agentic_eval_results", "_build_live_ori_eval_results", "_load_eval_results"]

_load_eval_results = _common_live_results._load_eval_results


@st.cache_data(ttl=30, show_spinner=False)
def _build_live_ori_eval_results(run_dir: str, ori_index: dict[str, dict[int, str]], ori_evals: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    return _common_live_results._build_live_ori_eval_results(run_dir, ori_index, ori_evals)


@st.cache_data(ttl=30, show_spinner=False)
def _build_live_agentic_eval_results(
    run_dir: str,
    agn_index: dict[str, dict[int, str]],
    agn_evals: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    return _common_live_results._build_live_agentic_eval_results(run_dir, agn_index, agn_evals)
