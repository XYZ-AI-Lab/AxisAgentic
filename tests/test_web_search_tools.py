from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import httpx

from agentic.contracts.messages import ToolResultStatus
from agentic.model_clients.request_logger import ModelRequestLogger
from agentic.tools import ToolManager
from agentic.tools.schema_order import validate_rendered_tool_argument_order
from agentic.tools.web_search import _scrape_utils as scrape_utils
from agentic.tools.web_search import scrape, search
from agentic.tools.web_search._scrape_utils import LLMExtractionCache
from recipe.web_search.runners import evaluate_benchmark as web_search_eval

if TYPE_CHECKING:
    import pytest


def test_jina_scrape_renders_content_and_moves_stats_to_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_scrape_with_jina(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "success": True,
            "content": "page markdown",
            "total_chars": 13,
            "total_lines": 1,
            "truncated": False,
        }

    monkeypatch.setattr(scrape, "scrape_with_jina", fake_scrape_with_jina)

    tool = scrape.create_jina_scrape_tool()
    result = asyncio.run(tool._fn(url="https://example.com"))

    assert result.content == "page markdown"
    assert result.status == ToolResultStatus.SUCCESS
    assert result.metadata["success"] is True
    assert result.metadata["url"] == "https://example.com"
    assert result.metadata["total_chars"] == 13
    assert result.metadata["total_lines"] == 1
    assert result.metadata["truncated"] is False


def test_llm_extract_renders_extracted_info_and_moves_operational_fields_to_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_extract_with_llm(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "success": True,
            "extracted_info": "focused answer",
            "error": "",
            "tokens_used": 42,
        }

    monkeypatch.setattr(scrape, "extract_with_llm", fake_extract_with_llm)

    tool = scrape.create_llm_extract_tool(llm_base_url="http://llm.example/v1/chat/completions")
    result = asyncio.run(tool._fn(content="raw page", info_to_extract="answer"))

    assert result.content == "focused answer"
    assert result.status == ToolResultStatus.SUCCESS
    assert result.metadata["success"] is True
    assert result.metadata["error"] == ""
    assert result.metadata["tokens_used"] == 42


def test_llm_extract_rejects_success_with_none_content(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_extract_with_llm(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "success": True,
            "extracted_info": None,
            "error": "",
            "tokens_used": 42,
        }

    monkeypatch.setattr(scrape, "extract_with_llm", fake_extract_with_llm)

    tool = scrape.create_llm_extract_tool(llm_base_url="http://llm.example/v1/chat/completions")
    result = asyncio.run(tool._fn(content="raw page", info_to_extract="answer"))

    assert result.content == ""
    assert result.status == ToolResultStatus.FAILED
    assert result.metadata["success"] is False
    assert result.metadata["error"] == "LLM extraction returned non-text content: NoneType"
    assert result.metadata["tokens_used"] == 42


def test_web_search_summary_llm_logs_payloads(tmp_path: Any) -> None:
    class FakeClient:
        async def post(self, url: str, **_kwargs: Any) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "focused answer"}}], "usage": {"total_tokens": 5}},
                request=httpx.Request("POST", url),
            )

    request_logger = ModelRequestLogger(tmp_path, name="summary_llm")
    result = asyncio.run(
        scrape.extract_with_llm(
            content="unique raw page for logging",
            info_to_extract="answer",
            llm_base_url="http://llm.example/v1",
            llm_api_key=None,
            client=FakeClient(),  # type: ignore[arg-type]
            model="summary-model",
            max_tokens=32,
            cache=LLMExtractionCache(tmp_path / "llm-cache.json"),
            request_logger=request_logger,
        )
    )
    request_logger.close()

    assert result["success"] is True
    records = [
        json.loads(line) for line in (tmp_path / "model_requests" / "summary_llm" / "part-000001.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["metadata"]["client"] == "summary_llm"
    assert records[0]["metadata"]["recipe"] == "web_search"
    assert records[0]["request"]["model"] == "summary-model"
    assert records[0]["response"]["body"]["choices"][0]["message"]["content"] == "focused answer"


def test_web_search_summary_llm_retries_context_length_wording(tmp_path: Any) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        async def post(self, url: str, **kwargs: Any) -> httpx.Response:
            self.requests.append(json.loads(json.dumps(kwargs["json"])))
            request = httpx.Request("POST", url)
            if len(self.requests) == 1:
                return httpx.Response(
                    400,
                    json={"error": {"message": "Input length (156232) exceeds model's maximum context length (131072)."}},
                    request=request,
                )
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "focused answer"}}], "usage": {"total_tokens": 5}},
                request=request,
            )

    client = FakeClient()
    result = asyncio.run(
        scrape.extract_with_llm(
            content="x" * 50000,
            info_to_extract="answer",
            llm_base_url="http://llm.example/v1",
            llm_api_key=None,
            client=client,  # type: ignore[arg-type]
            model="summary-model",
            max_tokens=32,
            cache=LLMExtractionCache(tmp_path / "llm-cache.json"),
        )
    )

    assert result["success"] is True
    assert len(client.requests) == 2
    assert len(client.requests[1]["messages"][0]["content"]) < len(client.requests[0]["messages"][0]["content"])


def test_web_search_summary_llm_chunks_long_content_and_preserves_tail() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def post(self, url: str, **kwargs: Any) -> httpx.Response:
            prompt = kwargs["json"]["messages"][0]["content"]
            self.prompts.append(prompt)
            if "extracted findings from chunks" in prompt:
                content = "Final answer: the tail-only value is TAIL_SECRET=present."
            elif "TAIL_SECRET=present" in prompt:
                content = "TAIL_SECRET=present"
            else:
                content = "NO_RELEVANT_INFORMATION"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 5}},
                request=httpx.Request("POST", url),
            )

    client = FakeClient()
    long_content = ("irrelevant head\n" * 300) + "\nTAIL_SECRET=present\n"
    result = asyncio.run(
        scrape.extract_with_llm(
            content=long_content,
            info_to_extract="Find the tail-only value.",
            llm_base_url="http://llm.example/v1",
            llm_api_key=None,
            client=client,  # type: ignore[arg-type]
            model="summary-model",
            max_tokens=32,
            cache=None,
            max_input_chars=900,
            chunk_overlap_chars=20,
        )
    )

    assert result["success"] is True
    assert "TAIL_SECRET=present" in result["extracted_info"]
    assert result["strategy"] in {"chunked", "chunked_map_reduce"}
    assert result["chunk_count"] > 1
    assert len(client.prompts) > 1


def test_web_search_summary_llm_chunk_split_preserves_input_budget() -> None:
    budget = 1_200
    chunks = scrape_utils._split_content_for_chunked_extraction(
        "a" * 20_000,
        info_to_extract="find value",
        max_input_chars=budget,
        overlap_chars=50,
        max_chunks=2,
    )

    assert len(chunks) > 2
    assert all(scrape_utils._prompt_char_len("find value", chunk) <= budget for chunk in chunks)


def test_web_search_summary_llm_chunk_map_respects_concurrency() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        async def post(self, url: str, **kwargs: Any) -> httpx.Response:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            prompt = kwargs["json"]["messages"][0]["content"]
            content = "merged answer" if "extracted findings from chunks" in prompt else "chunk answer"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 3}},
                request=httpx.Request("POST", url),
            )

    client = FakeClient()
    result = asyncio.run(
        scrape.extract_with_llm(
            content="chunk text\n" * 1_000,
            info_to_extract="find value",
            llm_base_url="http://llm.example/v1",
            llm_api_key=None,
            client=client,  # type: ignore[arg-type]
            model="summary-model",
            max_tokens=32,
            cache=None,
            max_input_chars=2_000,
            chunk_overlap_chars=20,
            chunk_max_concurrent=2,
        )
    )

    assert result["success"] is True
    assert result["extracted_info"] == "merged answer"
    assert result["strategy"] == "chunked_map_reduce"
    assert client.max_active == 2


def test_web_search_summary_llm_switch_off_disables_chunking() -> None:
    """With chunked_extraction=False, long content takes the single-shot path."""

    class FakeClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def post(self, url: str, **kwargs: Any) -> httpx.Response:
            self.prompts.append(kwargs["json"]["messages"][0]["content"])
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "single shot answer"}}], "usage": {"total_tokens": 7}},
                request=httpx.Request("POST", url),
            )

    client = FakeClient()
    result = asyncio.run(
        scrape.extract_with_llm(
            content="chunk text\n" * 1_000,
            info_to_extract="find value",
            llm_base_url="http://llm.example/v1",
            llm_api_key=None,
            client=client,  # type: ignore[arg-type]
            model="summary-model",
            max_tokens=32,
            cache=None,
            max_input_chars=900,
            chunk_overlap_chars=20,
            chunked_extraction=False,
        )
    )

    assert result["success"] is True
    assert result["extracted_info"] == "single shot answer"
    assert "strategy" not in result  # single-shot path, not chunked
    assert len(client.prompts) == 1
    assert all("reading chunk" not in p for p in client.prompts)


def test_web_search_summary_llm_single_strategy_uses_one_reduce_and_original_prompt() -> None:
    """chunk_strategy='single' maps every chunk then does ONE reduce call (original prompt, depth 0)."""

    class FakeClient:
        def __init__(self) -> None:
            self.reduce_prompts: list[str] = []

        async def post(self, url: str, **kwargs: Any) -> httpx.Response:
            prompt = kwargs["json"]["messages"][0]["content"]
            if "extracted findings from chunks" in prompt:
                self.reduce_prompts.append(prompt)
                content = "merged single answer"
            else:
                content = "chunk finding"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 3}},
                request=httpx.Request("POST", url),
            )

    client = FakeClient()
    result = asyncio.run(
        scrape.extract_with_llm(
            content="chunk text\n" * 1_000,
            info_to_extract="find value",
            llm_base_url="http://llm.example/v1",
            llm_api_key=None,
            client=client,  # type: ignore[arg-type]
            model="summary-model",
            max_tokens=32,
            cache=None,
            max_input_chars=2_000,
            chunk_overlap_chars=20,
            chunk_strategy="single",
        )
    )

    assert result["success"] is True
    assert result["extracted_info"] == "merged single answer"
    assert result["strategy"] == "chunked_map_reduce"
    assert result["recursion_depth"] == 0
    assert result["chunk_count"] > 1
    # Exactly one reduce call, using the original prompt (not the tightened one).
    assert len(client.reduce_prompts) == 1
    assert "Synthesize a single precise answer" in client.reduce_prompts[0]
    assert "closest-matching item" not in client.reduce_prompts[0]


def test_web_search_summary_llm_recursive_strategy_recurses_on_large_findings() -> None:
    """chunk_strategy='recursive' recursively reduces when concatenated findings overflow the budget."""
    long_finding = "FINDING " * 80  # ~640 chars, so a handful overflow the budget

    class FakeClient:
        def __init__(self) -> None:
            self.reduce_prompts: list[str] = []

        async def post(self, url: str, **kwargs: Any) -> httpx.Response:
            prompt = kwargs["json"]["messages"][0]["content"]
            if "extracted findings from chunks" in prompt:
                self.reduce_prompts.append(prompt)
                content = "merged recursive answer"
            else:
                content = long_finding
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 3}},
                request=httpx.Request("POST", url),
            )

    client = FakeClient()
    result = asyncio.run(
        scrape.extract_with_llm(
            content="chunk text\n" * 1_000,
            info_to_extract="find value",
            llm_base_url="http://llm.example/v1",
            llm_api_key=None,
            client=client,  # type: ignore[arg-type]
            model="summary-model",
            max_tokens=32,
            cache=None,
            max_input_chars=2_000,
            chunk_overlap_chars=20,
            chunk_strategy="recursive",
            max_recursion_depth=3,
        )
    )

    assert result["success"] is True
    assert result["strategy"] == "chunked_map_reduce"
    assert result["recursion_depth"] >= 1  # recursion engaged
    # Recursive reduce uses the tightened, precision-first prompt.
    assert any("closest-matching item" in p for p in client.reduce_prompts)


def test_web_search_summary_llm_global_anchor_injects_into_map_and_reduce_prompts() -> None:
    """An enabled global-anchor path runs its pre-pass.

    Verifies the resulting DOCUMENT_GLOBAL_ANCHOR block is prepended to every
    map AND reduce prompt.
    """

    class FakeClient:
        def __init__(self) -> None:
            self.anchor_prompts: list[str] = []
            self.map_prompts: list[str] = []
            self.reduce_prompts: list[str] = []

        async def post(self, url: str, **kwargs: Any) -> httpx.Response:
            prompt = kwargs["json"]["messages"][0]["content"]
            if "DOCUMENT_SUMMARY:" in prompt:
                self.anchor_prompts.append(prompt)
                content = (
                    "TITLE: example doc\n"
                    "URL: unknown\n"
                    "DOC_TYPE: tabular_csv\n"
                    "PRIMARY_SCHEMA: col_a, col_b, col_c\n"
                    "UNITS_HINTS: none\n"
                    "TOC: none\n"
                    "QUESTION_KEYWORDS: tail, secret\n"
                    "GLOBAL_SCOPE_HINTS: none\n"
                )
            elif "extracted findings from chunks" in prompt:
                self.reduce_prompts.append(prompt)
                content = "Final answer: TAIL_SECRET=present."
            else:
                # Treat every chunk as relevant so the reduce step is actually
                # invoked (with only one relevant chunk, _extract_with_llm_chunked
                # short-circuits and skips reduce).
                self.map_prompts.append(prompt)
                content = "Partial finding from this chunk."
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 5}},
                request=httpx.Request("POST", url),
            )

    client = FakeClient()
    long_content = ("payload line\n" * 300) + "\nTAIL_SECRET=present\n"
    result = asyncio.run(
        scrape.extract_with_llm(
            content=long_content,
            info_to_extract="Find the tail-only value.",
            llm_base_url="http://llm.example/v1",
            llm_api_key=None,
            client=client,  # type: ignore[arg-type]
            model="summary-model",
            max_tokens=32,
            cache=None,
            max_input_chars=900,
            chunk_overlap_chars=20,
            enable_global_anchor=True,
        )
    )

    assert result["success"] is True
    assert result["anchor_used"] is True
    assert result["anchor_tokens_used"] > 0
    assert len(client.anchor_prompts) == 1
    # Anchor block appears in every map prompt and the reduce prompt.
    assert client.map_prompts, "expected at least one map call"
    for prompt in client.map_prompts:
        assert "<DOCUMENT_GLOBAL_ANCHOR>" in prompt
        assert "PRIMARY_SCHEMA: col_a, col_b, col_c" in prompt
    assert client.reduce_prompts, "expected a reduce call"
    for prompt in client.reduce_prompts:
        assert "<DOCUMENT_GLOBAL_ANCHOR>" in prompt
        assert "TITLE: example doc" in prompt


def test_web_search_summary_llm_global_anchor_disabled_keeps_legacy_prompts() -> None:
    """``enable_global_anchor=False`` skips the anchor pass.

    Per-chunk prompts are bytewise the legacy shape (no DOCUMENT_GLOBAL_ANCHOR
    block).
    """

    class FakeClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def post(self, url: str, **kwargs: Any) -> httpx.Response:
            prompt = kwargs["json"]["messages"][0]["content"]
            self.prompts.append(prompt)
            if "extracted findings from chunks" in prompt:
                content = "Final answer: TAIL_SECRET=present."
            elif "TAIL_SECRET=present" in prompt:
                content = "TAIL_SECRET=present"
            else:
                content = "NO_RELEVANT_INFORMATION"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 5}},
                request=httpx.Request("POST", url),
            )

    client = FakeClient()
    long_content = ("irrelevant head\n" * 300) + "\nTAIL_SECRET=present\n"
    result = asyncio.run(
        scrape.extract_with_llm(
            content=long_content,
            info_to_extract="Find the tail-only value.",
            llm_base_url="http://llm.example/v1",
            llm_api_key=None,
            client=client,  # type: ignore[arg-type]
            model="summary-model",
            max_tokens=32,
            cache=None,
            max_input_chars=900,
            chunk_overlap_chars=20,
            enable_global_anchor=False,
        )
    )

    assert result["success"] is True
    assert result["anchor_used"] is False
    assert result["anchor_tokens_used"] == 0
    # No prompt should be an anchor pass; no chunk prompt should carry the anchor block.
    for prompt in client.prompts:
        assert "DOCUMENT_SUMMARY:" not in prompt
        assert "<DOCUMENT_GLOBAL_ANCHOR>" not in prompt


def test_web_search_summary_llm_global_anchor_falls_back_on_unstructured_response() -> None:
    """Soft-degrade path when the anchor LLM returns garbage.

    When the response has no field markers, the map-reduce silently falls
    back to legacy no-anchor prompts.
    """

    class FakeClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def post(self, url: str, **kwargs: Any) -> httpx.Response:
            prompt = kwargs["json"]["messages"][0]["content"]
            self.prompts.append(prompt)
            if "DOCUMENT_SUMMARY:" in prompt:
                # Anchor LLM returns free prose with no field markers -> reject.
                content = "sorry I cannot help with that"
            elif "extracted findings from chunks" in prompt:
                content = "Final answer: TAIL_SECRET=present."
            elif "TAIL_SECRET=present" in prompt:
                content = "TAIL_SECRET=present"
            else:
                content = "NO_RELEVANT_INFORMATION"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 5}},
                request=httpx.Request("POST", url),
            )

    client = FakeClient()
    long_content = ("irrelevant head\n" * 300) + "\nTAIL_SECRET=present\n"
    result = asyncio.run(
        scrape.extract_with_llm(
            content=long_content,
            info_to_extract="Find the tail-only value.",
            llm_base_url="http://llm.example/v1",
            llm_api_key=None,
            client=client,  # type: ignore[arg-type]
            model="summary-model",
            max_tokens=32,
            cache=None,
            max_input_chars=900,
            chunk_overlap_chars=20,
        )
    )

    assert result["success"] is True
    assert result["anchor_used"] is False  # rejected by sanity gate
    # Map prompts must NOT have the anchor block.
    map_prompts = [p for p in client.prompts if "DOCUMENT_SUMMARY:" not in p]
    assert map_prompts, "expected map/reduce prompts"
    for prompt in map_prompts:
        assert "<DOCUMENT_GLOBAL_ANCHOR>" not in prompt


def test_web_search_summary_llm_global_anchor_prompt_builders_are_pure() -> None:
    """Direct unit-check of the prompt-shape contracts the chunked extractor relies on.

    Empty ``global_anchor`` MUST yield the pre-anchor prompt bytewise so callers
    that monkey-patch with the legacy text still match. Non-empty anchor MUST
    wrap content in the marker block.
    """
    legacy_map = scrape_utils._chunk_info_request("find X", chunk_index=2, chunk_count=5)
    assert "<DOCUMENT_GLOBAL_ANCHOR>" not in legacy_map
    assert legacy_map.startswith("find X")
    assert "chunk 2 of 5" in legacy_map

    anchored_map = scrape_utils._chunk_info_request("find X", chunk_index=2, chunk_count=5, global_anchor="TITLE: t\nDOC_TYPE: tabular_csv")
    assert anchored_map.startswith("<DOCUMENT_GLOBAL_ANCHOR>")
    assert "TITLE: t" in anchored_map
    assert "<TASK>\nfind X\n</TASK>" in anchored_map

    legacy_reduce = scrape_utils._reduce_chunk_info_request("find X", tighten=False)
    assert "<DOCUMENT_GLOBAL_ANCHOR>" not in legacy_reduce
    assert legacy_reduce.startswith("find X")

    anchored_reduce = scrape_utils._reduce_chunk_info_request("find X", tighten=True, global_anchor="TITLE: t")
    assert anchored_reduce.startswith("<DOCUMENT_GLOBAL_ANCHOR>")
    assert "find X" in anchored_reduce


def test_chunk_info_request_envelope_modes() -> None:
    """strict, soft, and strict_caveat envelopes are distinguishable.

    Each mode must yield a different prompt and the strict_caveat mode must
    keep rules 1-4 while replacing the hard-abstain rule 5 with caveat
    guidance so partial-match data is not dropped. The default is
    "strict_caveat".
    """
    anchor = "TITLE: t\nDOC_TYPE: mixed"

    strict_p = scrape_utils._chunk_info_request("find X", chunk_index=1, chunk_count=2, global_anchor=anchor, envelope_mode="strict")
    soft_p = scrape_utils._chunk_info_request("find X", chunk_index=1, chunk_count=2, global_anchor=anchor, envelope_mode="soft")
    caveat_p = scrape_utils._chunk_info_request("find X", chunk_index=1, chunk_count=2, global_anchor=anchor, envelope_mode="strict_caveat")
    default_p = scrape_utils._chunk_info_request("find X", chunk_index=1, chunk_count=2, global_anchor=anchor)

    # All three include the anchor block.
    for p in (strict_p, soft_p, caveat_p):
        assert "<DOCUMENT_GLOBAL_ANCHOR>" in p

    # Strict and strict_caveat share rules 1-4 wording (no-invent kept).
    for marker in (
        "never invent values not present in this chunk",
        "include a short locator hint",
        "<INSTRUCTIONS>",
    ):
        assert marker in strict_p
        assert marker in caveat_p

    # Soft has none of the strict scaffolding.
    assert "<INSTRUCTIONS>" not in soft_p
    assert "never invent" not in soft_p

    # strict_caveat replaces rule 5 with caveat guidance.
    assert "caveat" in caveat_p.lower()
    assert "ONLY when NO part of this chunk relates" in caveat_p
    # The pure-strict envelope (explicit mode) has no caveat language.
    assert "caveat" not in strict_p.lower()

    # The default is strict_caveat.
    assert scrape_utils.DEFAULT_CHUNK_ENVELOPE_MODE == "strict_caveat"
    assert default_p == caveat_p


def test_global_anchor_prompt_teaches_multi_schema_self_judge() -> None:
    """Anchor prompt must teach the LLM to self-declare PRIMARY_SCHEMA: NONE on multi-table docs.

    The anchor prompt explicitly instructs the LLM to output a self-declared
    NONE sentinel when the document head shows multiple distinct tables/schemas.
    """
    prompt = scrape_utils._GLOBAL_ANCHOR_PROMPT
    # Self-judge clause: must mention multi-table detection AND a NONE/multiple-tables sentinel.
    assert "MULTIPLE DISTINCT tables" in prompt
    # Exact sentinel the LLM is told to emit.
    assert "NONE - document contains multiple distinct tables" in prompt
    assert "every table is a valid data source" in prompt
    # Single-table path is still documented (degrade direction matters).
    assert "single tabular" in prompt


def test_anchor_self_declared_none_flows_into_chunk_envelope() -> None:
    """A NONE-sentinel PRIMARY_SCHEMA from the anchor LLM reaches every chunk verbatim.

    No rewriting needed: the LLM-emitted sentinel is human-readable so the
    downstream chunk LLM reading the anchor naturally degrades the
    'use PRIMARY_SCHEMA' rule when it sees 'NONE - ...'.
    """
    anchor_with_none = (
        "TITLE: EIA Multi-Table Report\n"
        "DOC_TYPE: tabular_pdf_text\n"
        "PRIMARY_SCHEMA: NONE - document contains multiple distinct tables; "
        "every table is a valid data source. Do not constrain extraction to a single schema.\n"
        "UNITS_HINTS: thousand tons\n"
        "TOC: 2.8.A, 4.7.C\n"
        "QUESTION_KEYWORDS: coal, state, capacity\n"
        "GLOBAL_SCOPE_HINTS: US states 2019-2020"
    )
    # The default envelope is strict_caveat.
    prompt = scrape_utils._chunk_info_request(
        "find coal stats",
        chunk_index=1,
        chunk_count=3,
        global_anchor=anchor_with_none,
    )
    assert "PRIMARY_SCHEMA: NONE - document contains multiple distinct tables" in prompt
    assert "every table is a valid data source" in prompt
    # No rewrite path was taken — the verbatim anchor text is present.
    assert "<multi-table:" not in prompt


# ----------------------------------------------------------------------------
# Structure-aware CSV splitter
# ----------------------------------------------------------------------------


def test_detect_csv_structure_positive_cases() -> None:
    """`_detect_csv_structure` recognizes header-row CSV and TSV layouts."""
    csv_text = (
        "year,state,production_tons\n2018,Texas,1234\n2019,Texas,2345\n2020,Texas,3194\n2018,Wyoming,9876\n2019,Wyoming,8765\n2020,Wyoming,7654\n"
    )
    detected = scrape_utils._detect_csv_structure(csv_text)
    assert detected is not None
    sep, header, body_start = detected
    assert sep == ","
    assert header == "year,state,production_tons"
    assert csv_text[body_start:].startswith("2018,Texas")

    tsv_text = "year\tstate\tval\n2020\tTX\t1\n2020\tWY\t2\n2020\tCA\t3\n"
    detected_tsv = scrape_utils._detect_csv_structure(tsv_text)
    assert detected_tsv is not None
    assert detected_tsv[0] == "\t"


def test_detect_csv_structure_rejects_prose_and_markdown() -> None:
    """Prose, markdown tables, single-line, and embedded CSV fragments are rejected."""
    assert scrape_utils._detect_csv_structure("") is None
    assert scrape_utils._detect_csv_structure("Just prose. Some text here. End.") is None
    # Markdown table — `|---|---|` separator must reject.
    md = "| col1 | col2 | col3 |\n|---|---|---|\n| 1 | 2 | 3 |\n| 4 | 5 | 6 |\n"
    assert scrape_utils._detect_csv_structure(md) is None
    # CSV fragment embedded in prose — body lines do not match header (80% gate).
    embedded = "year,state,production\nSome narrative paragraph here.\nAnother paragraph follows.\nAnd one more sentence to round things out.\n"
    assert scrape_utils._detect_csv_structure(embedded) is None
    # Too few separators in header.
    one_col = "header\nval1\nval2\nval3\n"
    assert scrape_utils._detect_csv_structure(one_col) is None


def test_split_csv_content_preserves_header_in_every_chunk() -> None:
    """Header row replicated in every chunk; rows never split mid-record."""
    rows = "\n".join(f"2020,Texas,{1000 + i}" for i in range(200))
    csv = "year,state,value\n" + rows + "\n"
    detected = scrape_utils._detect_csv_structure(csv)
    assert detected is not None
    _sep, header, body_start = detected
    chunks = scrape_utils._split_csv_content_for_chunked_extraction(
        csv,
        header=header,
        body_start=body_start,
        info_to_extract="find the production for 2020 Texas",
        max_input_chars=1_000,
        max_chunks=0,
    )
    assert len(chunks) >= 2
    for chunk in chunks:
        # Every chunk leads with the header.
        assert chunk.startswith("year,state,value\n")
        # And contains at least one data row.
        lines = chunk.splitlines()
        assert len(lines) >= 2
        # No row is split mid-record — every body line has 2 commas.
        for body_line in lines[1:]:
            if body_line.strip():
                assert body_line.count(",") == 2, f"row split mid-record: {body_line!r}"


def test_split_content_for_chunked_extraction_dispatches_csv_to_layer_b() -> None:
    """The main splitter routes CSV-like content through the header-preserving path."""
    rows = "\n".join(f"2020,Texas,{1000 + i}" for i in range(80))
    csv = "year,state,production\n" + rows + "\n"
    chunks = scrape_utils._split_content_for_chunked_extraction(
        csv,
        info_to_extract="find 2020 Texas production",
        max_input_chars=800,
        overlap_chars=50,
        max_chunks=0,
        csv_layer_b_enabled=True,
    )
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.startswith("year,state,production\n"), "Header-preserving CSV path did not preserve the header"


def test_split_content_for_chunked_extraction_can_disable_csv_layer_b() -> None:
    """When CSV structure-aware filtering is off, use char-window chunking."""
    rows = "\n".join(f"2020,Texas,{1000 + i}" for i in range(80))
    csv = "year,state,production\n" + rows + "\n"
    chunks = scrape_utils._split_content_for_chunked_extraction(
        csv,
        info_to_extract="find 2020 Texas production",
        max_input_chars=1_000,
        overlap_chars=20,
        max_chunks=0,
        csv_layer_b_enabled=False,
    )

    assert len(chunks) >= 2
    assert chunks[0].startswith("year,state,production\n")
    assert any(not chunk.startswith("year,state,production\n") for chunk in chunks[1:])


def test_split_content_for_chunked_extraction_csv_query_focuses_year_rows() -> None:
    """Query-focused CSV filtering shrinks targeted row queries before chunking."""
    rows = (
        "1,Alpha University,United States,2017,90\n"
        "2,Yale University,United States,2018,95\n"
        "3,The University of Chicago,United States,2018,94\n"
        "4,Oxford University,United Kingdom,2019,93"
    )
    csv = "Rank,Name,Country,Year,Score\n" + rows + "\n"
    chunks = scrape_utils._split_content_for_chunked_extraction(
        csv,
        info_to_extract="Find lines where Year=2018",
        max_input_chars=2_000,
        overlap_chars=50,
        max_chunks=0,
        csv_layer_b_enabled=True,
    )

    focused = "\n".join(chunks)
    assert "__row_index,Rank,Name,Country,Year,Score" in focused
    assert "Yale University" in focused
    assert "The University of Chicago" in focused
    assert "Alpha University" not in focused
    assert "Oxford University" not in focused


def test_split_content_for_chunked_extraction_csv_query_focuses_contains_term() -> None:
    rows = "1,Sierra County,California,12\n2,Orange County,California,8\n3,Sierra Vista,Arizona,3"
    csv = "id,name,state,value\n" + rows + "\n"
    chunks = scrape_utils._split_content_for_chunked_extraction(
        csv,
        info_to_extract="Find lines containing Sierra",
        max_input_chars=2_000,
        overlap_chars=50,
        max_chunks=0,
        csv_layer_b_enabled=True,
    )

    focused = "\n".join(chunks)
    assert "Sierra County" in focused
    assert "Sierra Vista" in focused
    assert "Orange County" not in focused


def test_split_content_for_chunked_extraction_csv_query_focus_falls_back_without_hits() -> None:
    rows = "\n".join(f"{year},Texas,{1000 + i}" for i, year in enumerate([2018, 2019, 2020, 2021]))
    csv = "year,state,production\n" + rows + "\n"
    chunks = scrape_utils._split_content_for_chunked_extraction(
        csv,
        info_to_extract="Find rows where year=1999",
        max_input_chars=2_000,
        overlap_chars=50,
        max_chunks=0,
    )

    focused = "\n".join(chunks)
    assert "__row_index" not in focused
    assert "2018,Texas,1000" in focused
    assert "2021,Texas,1003" in focused


def test_split_content_for_chunked_extraction_falls_back_for_prose() -> None:
    """Non-CSV content still uses the char-window splitter (no header replication)."""
    prose = ("This is a paragraph of prose content. " * 80).strip()
    chunks = scrape_utils._split_content_for_chunked_extraction(
        prose,
        info_to_extract="find the topic",
        max_input_chars=800,
        overlap_chars=50,
        max_chunks=0,
    )
    assert chunks
    # No artificial header should appear at the front of subsequent chunks.
    # (The legacy splitter just emits content slices.)
    assert chunks[0].startswith("This is a paragraph")


def test_split_content_for_chunked_extraction_falls_back_for_markdown_table() -> None:
    """Markdown tables stay on the char-window path without header replication."""
    md = "| year | state | val |\n|---|---|---|\n" + "\n".join(f"| 2020 | TX | {i} |" for i in range(50))
    chunks = scrape_utils._split_content_for_chunked_extraction(
        md,
        info_to_extract="find rows",
        max_input_chars=2_000,
        overlap_chars=50,
        max_chunks=0,
    )
    assert chunks
    # First chunk should look like the original markdown (no header-row pre-pending).
    assert chunks[0].startswith("| year")
    # No chunk except the first should restart with "| year | state | val |\n" (header
    # not replicated — structure-aware CSV splitting did not fire).
    if len(chunks) > 1:
        for chunk in chunks[1:]:
            assert not chunk.startswith("| year | state | val |\n")


def test_extract_with_llm_chunked_csv_path_replicates_header() -> None:
    """End-to-end: CSV content makes every map prompt carry the header row."""

    class FakeClient:
        def __init__(self) -> None:
            self.map_prompts: list[str] = []

        async def post(self, url: str, **kwargs: Any) -> httpx.Response:
            prompt = kwargs["json"]["messages"][0]["content"]
            if "DOCUMENT_SUMMARY:" in prompt:
                content = (
                    "TITLE: Production Stats\n"
                    "DOC_TYPE: tabular_csv\n"
                    "PRIMARY_SCHEMA: year, state, production\n"
                    "UNITS_HINTS: tons\n"
                    "TOC: none\n"
                    "QUESTION_KEYWORDS: production, 2020, Texas\n"
                    "GLOBAL_SCOPE_HINTS: US states 2018-2020\n"
                )
            elif "extracted findings from chunks" in prompt:
                content = "Final synthesized answer."
            else:
                self.map_prompts.append(prompt)
                content = "Found row: 2020,Texas,3194"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 5}},
                request=httpx.Request("POST", url),
            )

    client = FakeClient()
    rows = "\n".join(f"{2018 + (i % 3)},{['TX', 'WY', 'CA'][i % 3]},{1000 + i}" for i in range(200))
    csv_content = "year,state,production\n" + rows + "\n"
    result = asyncio.run(
        scrape.extract_with_llm(
            content=csv_content,
            info_to_extract="find production for 2020 Texas",
            llm_base_url="http://llm.example/v1",
            llm_api_key=None,
            client=client,  # type: ignore[arg-type]
            model="summary-model",
            max_tokens=32,
            cache=None,
            max_input_chars=1_200,
            chunk_overlap_chars=50,
            anchor_sample_chars=10_000,
            csv_layer_b_enabled=True,
        )
    )

    assert result["success"] is True
    assert client.map_prompts, "expected at least one CSV map call"
    for prompt in client.map_prompts:
        # Every chunk must carry the CSV header inline.
        assert "year,state,production" in prompt, "header missing from a chunk prompt"


def test_jina_scrape_structured_direct_respects_content_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_scrape_direct_typed(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        seen["max_chars"] = kwargs["max_chars"]
        return {
            "success": True,
            "is_structured": True,
            "content": "year,state,value\n2021,Texas,1",
            "content_type": "text/csv",
            "total_chars": 29,
            "total_lines": 2,
            "truncated": False,
        }

    async def fake_scrape_with_jina(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("structured direct fetch should return before Jina")

    monkeypatch.setattr(scrape, "scrape_direct_typed", fake_scrape_direct_typed)
    monkeypatch.setattr(scrape, "scrape_with_jina", fake_scrape_with_jina)

    tool = scrape.create_jina_scrape_tool(max_content_length=1_000)
    result = asyncio.run(tool._fn(url="https://example.com/data.csv"))

    assert result.status == ToolResultStatus.SUCCESS
    assert result.content.startswith("year,state,value")
    assert result.metadata["scrape_backend"] == "direct"
    assert seen["max_chars"] == 1_000


def test_scrape_and_extract_renders_extracted_info_and_moves_stats_to_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_scrape_with_jina(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "success": True,
            "content": "raw page",
            "total_chars": 8,
            "total_lines": 1,
            "truncated": False,
        }

    async def fake_extract_with_llm(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "success": True,
            "extracted_info": "extracted answer",
            "error": "",
            "tokens_used": 12,
        }

    monkeypatch.setattr(scrape, "scrape_with_jina", fake_scrape_with_jina)
    monkeypatch.setattr(scrape, "extract_with_llm", fake_extract_with_llm)

    tool = scrape.create_scrape_and_extract_tool(summary_llm_base_url="http://llm.example/v1/chat/completions")
    result = asyncio.run(tool._fn(url="https://example.com", info_to_extract="answer"))

    assert result.content == "extracted answer"
    assert result.status == ToolResultStatus.SUCCESS
    assert result.metadata["success"] is True
    assert result.metadata["url"] == "https://example.com"
    assert result.metadata["error"] == ""
    assert result.metadata["tokens_used"] == 12
    assert result.metadata["extraction_strategy"] == "direct"
    assert result.metadata["scrape_stats"] == {
        "total_chars": 8,
        "total_lines": 1,
        "truncated": False,
    }


def test_scrape_and_extract_rejects_success_with_none_extracted_info(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_scrape_with_jina(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "success": True,
            "content": "raw page",
            "total_chars": 8,
            "total_lines": 1,
            "truncated": False,
        }

    async def fake_extract_with_llm(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "success": True,
            "extracted_info": None,
            "error": "",
            "tokens_used": 12,
        }

    monkeypatch.setattr(scrape, "scrape_with_jina", fake_scrape_with_jina)
    monkeypatch.setattr(scrape, "extract_with_llm", fake_extract_with_llm)

    tool = scrape.create_scrape_and_extract_tool(summary_llm_base_url="http://llm.example/v1/chat/completions")
    result = asyncio.run(tool._fn(url="https://example.com", info_to_extract="answer"))

    assert result.content == ""
    assert result.status == ToolResultStatus.FAILED
    assert result.metadata["success"] is False
    assert result.metadata["error"] == "LLM extraction returned non-text content: NoneType"
    assert result.metadata["tokens_used"] == 12


def test_web_search_renders_results_and_moves_search_parameters_to_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "organic": [
                    {
                        "title": "Example",
                        "link": "https://example.com",
                        "snippet": "Example snippet",
                    }
                ],
                "searchParameters": {
                    "q": "example",
                    "gl": "us",
                    "hl": "en",
                    "num": 10,
                },
            }

    class FakeClient:
        async def post(self, *_args: Any, **_kwargs: Any) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(search, "create_async_client", lambda *_args, **_kwargs: FakeClient())

    tool = search.create_web_search_tool(serper_api_key="key")
    result = asyncio.run(tool._fn(query="example"))
    content = json.loads(result.content)

    assert result.status == ToolResultStatus.SUCCESS
    assert content == {
        "organic": [
            {
                "title": "Example",
                "link": "https://example.com",
                "snippet": "Example snippet",
            }
        ]
    }
    assert result.metadata["success"] is True
    assert result.metadata["search_parameters"] == {
        "q": "example",
        "gl": "us",
        "hl": "en",
        "num": 10,
    }


def test_web_search_filters_huggingface_posts(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "organic": [
                    {
                        "title": "Leaked answer",
                        "link": "https://huggingface.co/posts/m-ric/141258948203422",
                        "snippet": "direct answer",
                    },
                    {
                        "title": "Allowed",
                        "link": "https://example.com/source",
                        "snippet": "source snippet",
                    },
                ],
                "searchParameters": {"q": "example"},
            }

    class FakeClient:
        async def post(self, *_args: Any, **_kwargs: Any) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(search, "create_async_client", lambda *_args, **_kwargs: FakeClient())

    tool = search.create_web_search_tool(serper_api_key="key")
    result = asyncio.run(tool._fn(query="example"))
    content = json.loads(result.content)

    assert result.status == ToolResultStatus.SUCCESS
    assert content["organic"] == [
        {
            "title": "Allowed",
            "link": "https://example.com/source",
            "snippet": "source snippet",
        }
    ]
    assert result.metadata["attempts"][0]["filtered_count"] == 1


def test_scrape_and_extract_blocks_huggingface_posts() -> None:
    tool = scrape.create_scrape_and_extract_tool(summary_llm_base_url="http://llm.example/v1/chat/completions")
    result = asyncio.run(
        tool._fn(
            url="https://huggingface.co/posts/m-ric/141258948203422",
            info_to_extract="answer",
        )
    )

    assert result.status == ToolResultStatus.FAILED
    assert "blocked" in result.content


def test_web_search_text_output_does_not_render_search_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "organic": [
                    {
                        "title": "Example",
                        "link": "https://example.com",
                        "snippet": "Example snippet",
                    }
                ],
                "searchParameters": {"q": "example"},
            }

    class FakeClient:
        async def post(self, *_args: Any, **_kwargs: Any) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(search, "create_async_client", lambda *_args, **_kwargs: FakeClient())

    tool = search.create_web_search_tool(serper_api_key="key", output_format="text")
    result = asyncio.run(tool._fn(query="example"))

    assert result.status == ToolResultStatus.SUCCESS
    assert "Organic Results:" in result.content
    assert "searchParameters" not in result.content
    assert result.metadata["search_parameters"] == {"q": "example"}


def test_web_search_uses_num_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"organic": [], "searchParameters": {"q": "example", "num": 3}}

    captured_payloads: list[dict[str, Any]] = []

    class FakeClient:
        async def post(self, *_args: Any, **kwargs: Any) -> FakeResponse:
            captured_payloads.append(kwargs["json"])
            return FakeResponse()

    monkeypatch.setattr(search, "create_async_client", lambda *_args, **_kwargs: FakeClient())

    tool = search.create_web_search_tool(serper_api_key="key")
    result = asyncio.run(tool._fn(query="example", num=3))

    assert result.status == ToolResultStatus.SUCCESS
    assert captured_payloads[0]["num"] == 3


def test_recipe_runners_use_recipe_specific_web_tools() -> None:
    assert web_search_eval.create_web_search_tool is search.create_web_search_tool
    assert web_search_eval.create_scrape_and_extract_tool is scrape.create_scrape_and_extract_tool


def test_web_search_runner_uses_current_tool_names() -> None:
    args = type(
        "Args",
        (),
        {
            "serper_base_url": "http://serper.example",
            "jina_base_url": "http://jina.example",
            "max_content_length": 1024,
            "summary_llm_base_url": "",
            "summary_llm_model_name": None,
            "summary_llm_api_key_env": "SUMMARY_LLM_API_KEY",
            "max_turns": 3,
            "max_context_length": 4096,
            "context_safety_margin": 256,
            "keep_tool_result": -1,
            "tool_result_role": "tool",
            "max_task_retries": 0,
            "include_failure_summary_in_retry": False,
            "max_final_answer_attempts": 1,
        },
    )()

    orchestrator = web_search_eval.build_orchestrator(model_client=None, task_logger=None, args=args)

    assert orchestrator.tool_manager.list_tool_names() == ["web_search", "scrape_and_extract_info"]


def test_web_search_resume_flags_are_available() -> None:
    assert (
        web_search_eval.parse_args(
            [
                "--base_url",
                "http://model.example/v1",
                "--data_path",
                "data.jsonl",
                "--resume",
            ]
        ).resume
        is True
    )


def test_rendered_tool_argument_order_guard_matches_schema_property_order() -> None:
    tool = search.create_web_search_tool(parameters=search.SIMPLE_WEB_SEARCH_PARAMETERS, serper_api_key="key")
    tools = ToolManager(tools=[tool]).list_tool_definitions()

    rendered = '"web_search": {"query": {"type": "string"}, "num": {"type": "integer"}, "gl": {}, "hl": {}}'
    assert validate_rendered_tool_argument_order(rendered, tools) is None

    rendered_sorted = '"web_search": {"gl": {}, "hl": {}, "num": {"type": "integer"}, "query": {"type": "string"}}'
    error = validate_rendered_tool_argument_order(rendered_sorted, tools)
    assert error is not None
    assert "web_search" in error
