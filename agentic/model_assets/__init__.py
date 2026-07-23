# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from agentic.model_assets.chat_template_compat import normalize_openai_tool_call_arguments_for_chat_template
from agentic.model_assets.registry import (
    AutoFitModel,
    ChatTemplateRenderConfig,
    get_chat_template_render_config,
    infer_endpoint_profile,
    iter_auto_fit_models,
    match_auto_fit_model,
    resolve_asset_uri,
)

__all__ = [
    "AutoFitModel",
    "ChatTemplateRenderConfig",
    "get_chat_template_render_config",
    "infer_endpoint_profile",
    "iter_auto_fit_models",
    "match_auto_fit_model",
    "normalize_openai_tool_call_arguments_for_chat_template",
    "resolve_asset_uri",
]
