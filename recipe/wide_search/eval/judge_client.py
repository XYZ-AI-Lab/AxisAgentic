# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from recipe.web_search.eval.llm_verifier import normalize_openai_base_url

if TYPE_CHECKING:
    import asyncio


@dataclass
class WideSearchJudgeConfig:
    judge_model: str
    judge_base_url: str | None = None
    judge_api_key_env: str = "JUDGE_API_KEY"
    judge_max_tokens: int = 10240
    judge_temperature: float = 0.0
    request_timeout: float = 600.0
    max_retries: int = 1
    retryable_status_codes: list[int] = field(default_factory=list)


_MARKDOWN_JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def parse_markdown_json(completion: str | None) -> dict[str, Any] | None:
    if completion is None:
        return None
    matches = _MARKDOWN_JSON_RE.findall(completion)
    if not matches:
        return None
    try:
        parsed = json.loads(matches[-1])
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


class WideSearchJudgeClient:
    def __init__(self, config: WideSearchJudgeConfig) -> None:
        self.config = config
        self._client: Any | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._calls: int = 0

    @property
    def total_calls(self) -> int:
        return self._calls

    def attach_semaphore(self, semaphore: asyncio.Semaphore) -> None:
        self._semaphore = semaphore

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            finally:
                self._client = None

    def _build_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=os.environ.get(self.config.judge_api_key_env) or os.environ.get("OPENAI_API_KEY"),
                base_url=normalize_openai_base_url(self.config.judge_base_url),
            )
        return self._client

    async def chat_completion(self, prompt: str) -> str | None:
        client = self._build_client()
        sem = self._semaphore

        async def _call() -> str | None:
            self._calls += 1
            try:
                response = await client.chat.completions.create(
                    model=self.config.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.judge_max_tokens,
                    temperature=self.config.judge_temperature,
                    timeout=self.config.request_timeout,
                )
            except Exception:
                return None
            try:
                return response.choices[0].message.content
            except (AttributeError, IndexError, TypeError):
                return None

        if sem is None:
            return await _call()
        async with sem:
            return await _call()
