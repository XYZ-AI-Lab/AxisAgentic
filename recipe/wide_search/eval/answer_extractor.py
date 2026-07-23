# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from io import StringIO
from typing import Literal

import pandas as pd

from recipe.web_search.agent.prompts import extract_boxed_content

_MARKDOWN_BLOCK_RE = re.compile(r"```markdown(.*?)```", re.DOTALL)
_PIPE_RUN_RE = re.compile(r"((?:\|.*\n?)+)")
_PIPE_OR_DASH = set("|- :")


def _normalise_table_block(block: str) -> str:
    block = block.strip()
    if not block:
        return ""
    lines = [line.strip() for line in block.split("\n")]
    if not lines:
        return ""
    lines[0] = lines[0].replace(" ", "").lower()
    new_lines = []
    for line in lines:
        if "|" not in line or set(line).issubset(_PIPE_OR_DASH):
            continue
        new_lines.append("|".join(part.strip() for part in line.split("|")))
    return "\n".join(new_lines)


def _table_block_to_df(block: str) -> pd.DataFrame | None:
    normalised = _normalise_table_block(block)
    if not normalised:
        return None
    try:
        df = pd.read_csv(StringIO(normalised), sep="|")
    except Exception:
        return None
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def _extract_pipe_region(text: str) -> str | None:
    pipe_positions = [m.start() for m in re.finditer(r"\|", text)]
    if len(pipe_positions) < 4:
        return None
    first_pipe = pipe_positions[0]
    last_pipe = pipe_positions[-1]
    start = text.rfind("\n", 0, first_pipe)
    start = 0 if start == -1 else start
    end = text.find("\n", last_pipe)
    end = len(text) if end == -1 else end
    candidate = text[start:end]
    matches = _PIPE_RUN_RE.findall(candidate)
    if not matches:
        return None
    return matches[0]


ExtractorMode = Literal["boxed_first", "official_only"]


def has_fenced_markdown_table(text: str) -> bool:
    """Return whether ``text`` contains a Markdown-fenced table.

    The fenced block must parse to a non-empty DataFrame.

    Stricter than :func:`extract_dataframe`: the runtime uses this to detect a
    completed final answer per-message, and the pipe-region heuristic would
    otherwise mistake a "draft so far" table for a finished one.
    """
    if not text:
        return False
    for block in _MARKDOWN_BLOCK_RE.findall(text):
        df = _table_block_to_df(block)
        if df is not None and not df.empty:
            return True
    return False


def extract_dataframe(response: str, *, mode: ExtractorMode = "boxed_first") -> pd.DataFrame | None:
    r"""Extract the prediction DataFrame from a response string.

    boxed_first: first try ``\\boxed{...}`` payload, then official ```markdown``` block,
                 then pipe-region heuristic.
    official_only: matches the official ``WideSearchResponse.extract_dataframe`` order
                   (markdown block → pipe region).
    """
    if not response:
        return None

    if mode == "boxed_first":
        boxed = extract_boxed_content(response)
        if boxed:
            df = _table_block_to_df(boxed)
            if df is not None and not df.empty:
                return df

    markdown_blocks = _MARKDOWN_BLOCK_RE.findall(response)
    if markdown_blocks:
        df = _table_block_to_df(markdown_blocks[0])
        if df is not None:
            return df

    pipe_region = _extract_pipe_region(response)
    if pipe_region is not None:
        df = _table_block_to_df(pipe_region)
        if df is not None:
            return df

    return None
