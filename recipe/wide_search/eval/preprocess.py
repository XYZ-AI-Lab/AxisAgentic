# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import dateparser

if TYPE_CHECKING:
    from collections.abc import Callable

preprocess_function_registry: dict[str, Callable[[object], str]] = {}


def register_preprocess_function(func: Callable[[object], str]) -> Callable[[object], str]:
    preprocess_function_registry[func.__name__] = func
    return func


def norm_column(col: str) -> str:
    return col.strip().lower().replace(" ", "")


@register_preprocess_function
def extract_number(content: object) -> str:
    text = str(content).replace(",", "")
    numbers = re.findall(r"[-+]?\d*\.\d+%?|[-+]?\d+\.?\d*%?", text)
    if not numbers:
        return "NULL"
    return numbers[0]


@register_preprocess_function
def norm_str(content: object) -> str:
    return str(content).lower().strip().replace(" ", "").replace("*", "")


@register_preprocess_function
def norm_date(content: object) -> str:
    text = str(content)
    parsed = dateparser.parse(text, settings={"PREFER_DAY_OF_MONTH": "first"})
    if parsed is None:
        return text
    return parsed.strftime("%Y-%m-%d")


def preprocess_call(content: object, preprocess_func_name: str) -> str:
    if preprocess_func_name not in preprocess_function_registry:
        msg = f"preprocess_func_name {preprocess_func_name!r} not in registry"
        raise KeyError(msg)
    return preprocess_function_registry[preprocess_func_name](content)
