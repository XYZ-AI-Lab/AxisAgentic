# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections.abc import Callable
from urllib.parse import urlparse

import dateparser

MetricFn = Callable[..., tuple[float, str]]

metric_function_registry: dict[str, MetricFn] = {}


def register_metric_function(func: MetricFn) -> MetricFn:
    metric_function_registry[func.__name__] = func
    return func


_URL_PATTERN = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")


@register_metric_function
def exact_match(response: str, target: str) -> tuple[float, str]:
    if response.lower() == target.lower():
        return 1.0, f"exact match, response: {response}, target: {target}"
    return 0.0, f"exact not match, response: {response}, target: {target}"


@register_metric_function
def url_match(response: str, target: str) -> tuple[float, str]:
    response_urls = [urlparse(url).netloc for url in _URL_PATTERN.findall(response)]
    target_urls = [urlparse(url).netloc for url in _URL_PATTERN.findall(target)]
    if set(response_urls) == set(target_urls):
        return 1.0, f"url match, response: {response}, target: {target}"
    return 0.0, f"url not match, response: {response}, target: {target}"


@register_metric_function
def in_match(response: str, target: str) -> tuple[float, str]:
    if response in target:
        return 1.0, f"response in target, response: {response}, target: {target}"
    return 0.0, f"response not in target, response: {response}, target: {target}"


def _coerce_number(value: str) -> float | None:
    if "%" in value:
        try:
            return float(value.replace("%", "")) / 100.0
        except (ValueError, TypeError):
            return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


@register_metric_function
def number_near(response: str, target: str, criterion: float) -> tuple[float, str]:
    response_num = _coerce_number(response)
    target_num = _coerce_number(target)
    if response_num is None or target_num is None:
        if response_num is None and target_num is None and response == target:
            return 1.0, f"number equal, response: {response}, target: {target}"
        return 0.0, f"number not convertable, response: {response_num}, target: {target_num}"
    if abs(response_num - target_num) <= abs(target_num) * float(criterion):
        return (
            1.0,
            f"number near in range {float(criterion) * 100}%, response: {response_num}, target: {target_num}",
        )
    return 0.0, f"number not near, response: {response_num}, target: {target_num}"


@register_metric_function
def date_near(response: str, target: str) -> tuple[float, str]:
    try:
        response_date = dateparser.parse(response, settings={"PREFER_DAY_OF_MONTH": "first"})
    except Exception:
        response_date = None
    try:
        target_date = dateparser.parse(target, settings={"PREFER_DAY_OF_MONTH": "first"})
    except Exception:
        target_date = None
    if response_date is None or target_date is None:
        if response_date is None and target_date is None:
            return 1.0, f"date near, response: {response}, target: {target}"
        return 0.0, f"date not convertable, response: {response}, target: {target}"
    if abs((response_date - target_date).days) <= 31:
        return 1.0, f"date near, response: {response_date}, target: {target_date}"
    return 0.0, f"date not near, response: {response_date}, target: {target_date}"


def metric_call(
    response: str,
    target: str,
    criterion: object,
    metric_func_name: str,
) -> tuple[float, str]:
    if metric_func_name not in metric_function_registry:
        msg = f"metric_func_name {metric_func_name!r} not in registry"
        raise KeyError(msg)
    metric_func = metric_function_registry[metric_func_name]
    if metric_func_name == "number_near":
        return metric_func(response, target, criterion)
    return metric_func(response, target)
