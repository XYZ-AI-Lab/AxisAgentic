# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""Tool-call tab renderer for the benchmark dashboard."""

from __future__ import annotations

import streamlit as st

from recipe.dashboard.sides import DashboardSide, side_run_title
from recipe.dashboard.timing_charts import _render_tool_usage_distribution


def _render_tool_call_tab(left: DashboardSide, right: DashboardSide) -> None:
    st.header("Tool Call")
    header_cols = st.columns(2)
    header_cols[0].markdown(f"#### {side_run_title(left)}")
    header_cols[1].markdown(f"#### {side_run_title(right)}")
    _render_tool_usage_distribution([left, right])
