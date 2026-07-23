# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ASSET_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ChatTemplateRenderConfig:
    endpoint_profile: str
    asset_uri: str
    extract_start: str | None
    extract_end: str | None


@dataclass(frozen=True)
class AutoFitModel:
    model_id: str
    endpoint_profiles: tuple[str, ...]
    chat_template_asset: str | None
    render_extract_start: str | None
    render_extract_end: str | None
    parser_asset: str | None
    helper_assets: tuple[str, ...]
    exact: tuple[str, ...]
    regex: tuple[str, ...]
    thinking_handling: str
    status: str


# The open-source package does not bundle model-specific templates. This map
# remains as an extension point for downstream packages that choose to ship
# their own assets.
_CHAT_TEMPLATE_RENDER_PROFILES: dict[str, ChatTemplateRenderConfig] = {}

_MANIFEST_CACHE: dict | None = None


def resolve_asset_uri(uri: str) -> Path:
    prefix = "asset://"
    if not uri.startswith(prefix):
        msg = f"Unsupported asset URI: {uri}"
        raise ValueError(msg)
    relative = uri[len(prefix) :].lstrip("/")
    path = (ASSET_ROOT / relative).resolve()
    root = ASSET_ROOT.resolve()
    if root not in path.parents and path != root:
        msg = f"Asset URI escapes model asset root: {uri}"
        raise ValueError(msg)
    return path


def _load_manifest() -> dict:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        _MANIFEST_CACHE = yaml.safe_load((ASSET_ROOT / "manifest.yaml").read_text(encoding="utf-8")) or {}
    return _MANIFEST_CACHE


def iter_auto_fit_models() -> tuple[AutoFitModel, ...]:
    models = _load_manifest().get("models", [])
    records: list[AutoFitModel] = []
    for item in models:
        match = item.get("match", {}) or {}
        records.append(
            AutoFitModel(
                model_id=str(item["model_id"]),
                endpoint_profiles=tuple(item.get("endpoint_profiles", ()) or ()),
                chat_template_asset=item.get("chat_template_asset"),
                render_extract_start=item.get("render_extract_start"),
                render_extract_end=item.get("render_extract_end"),
                parser_asset=item.get("parser_asset"),
                helper_assets=tuple(item.get("helper_assets", ()) or ()),
                exact=tuple(match.get("exact", ()) or ()),
                regex=tuple(match.get("regex", ()) or ()),
                thinking_handling=str(item.get("thinking_handling", "")),
                status=str(item.get("status", "")),
            )
        )
    return tuple(records)


def _select_endpoint_profile(profiles: tuple[str, ...], base_url: str | None) -> str | None:
    del base_url
    return profiles[0] if profiles else None


def _select_render_delimiter(auto_fit_value: str | None, fallback_value: str | None) -> str | None:
    return auto_fit_value if auto_fit_value is not None else fallback_value


def match_auto_fit_model(model: str | None, base_url: str | None = None) -> AutoFitModel | None:
    del base_url
    if not model:
        return None
    model_text = model.lower()
    for record in iter_auto_fit_models():
        if any(model_text == candidate.lower() for candidate in record.exact):
            return record
    for record in iter_auto_fit_models():
        if any(re.search(pattern, model, flags=re.IGNORECASE) for pattern in record.regex):
            return record
    return None


def infer_endpoint_profile(model: str | None, base_url: str | None = None) -> str | None:
    auto_fit = match_auto_fit_model(model, base_url)
    if auto_fit is not None:
        profile = _select_endpoint_profile(auto_fit.endpoint_profiles, base_url)
        if profile is not None:
            return profile
    text = " ".join(part for part in (model, base_url) if part).lower()
    if not text:
        return None
    if "deepseek" in text:
        return "deepseek-v4-pro"
    if "kimi" in text or "moonshot" in text:
        return "kimi-k2.6"
    if "glm" in text or "z.ai" in text or "zai" in text:
        return "glm-5.1"
    if "qwen" in text:
        return "qwen3-thinking"
    if "minimax" in text:
        return "minimax"
    if "nemotron" in text:
        return "nemotron"
    if "xlam" in text:
        return "xlam"
    return None


def get_chat_template_render_config(
    *,
    endpoint_profile: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> ChatTemplateRenderConfig | None:
    profile = endpoint_profile or infer_endpoint_profile(model, base_url)
    if profile is None:
        return None
    auto_fit = match_auto_fit_model(model, base_url)
    if auto_fit is not None and auto_fit.chat_template_asset:
        selected_profile = endpoint_profile or _select_endpoint_profile(auto_fit.endpoint_profiles, base_url)
        if selected_profile is not None and selected_profile in auto_fit.endpoint_profiles:
            fallback = _CHAT_TEMPLATE_RENDER_PROFILES.get(selected_profile)
            fallback_start = fallback.extract_start if fallback else None
            fallback_end = fallback.extract_end if fallback else None
            return ChatTemplateRenderConfig(
                endpoint_profile=selected_profile,
                asset_uri=auto_fit.chat_template_asset,
                extract_start=_select_render_delimiter(auto_fit.render_extract_start, fallback_start),
                extract_end=_select_render_delimiter(auto_fit.render_extract_end, fallback_end),
            )
    return _CHAT_TEMPLATE_RENDER_PROFILES.get(profile)
