from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from recipe.wide_search.eval.data_loader import WideSearchQuery
from recipe.wide_search.eval.evaluation import aevaluate_single_query
from recipe.wide_search.eval.judge_client import WideSearchJudgeClient, WideSearchJudgeConfig


def _make_client(responses: list[str | None]) -> WideSearchJudgeClient:
    config = WideSearchJudgeConfig(judge_model="stub")
    client = WideSearchJudgeClient(config)
    queue = list(responses)

    async def _stub(_prompt: str) -> str | None:
        client._calls += 1
        return queue.pop(0) if queue else None

    client.chat_completion = _stub  # type: ignore[assignment]
    return client


def _query_two_col(
    answer: pd.DataFrame,
    *,
    eval_pipeline: dict,
    required: list[str] | None = None,
    unique_columns: list[str] | None = None,
) -> WideSearchQuery:
    required = required if required is not None else ["name", "city"]
    unique_columns = unique_columns if unique_columns is not None else ["name"]
    return WideSearchQuery(
        instance_id="ws_test_001",
        query="test query",
        evaluation={
            "required": required,
            "unique_columns": unique_columns,
            "eval_pipeline": eval_pipeline,
        },
        answer=answer,
        language="en",
    )


def test_response_none_returns_zero_metrics() -> None:
    answer = pd.DataFrame({"name": ["Alice"], "city": ["NYC"]})
    query = _query_two_col(answer, eval_pipeline={"city": {"metric": ["exact_match"]}})
    client = _make_client([])
    result = asyncio.run(aevaluate_single_query(query, None, client=client))
    assert result.msg == "response_df is None"
    assert result.score == 0.0
    assert result.f1_by_row == 0.0
    assert result.f1_by_item == 0.0


def test_perfect_match_yields_strict_score_one() -> None:
    answer = pd.DataFrame({"name": ["Alice", "Bob"], "city": ["NYC", "LA"]})
    response = pd.DataFrame({"name": ["Alice", "Bob"], "city": ["NYC", "LA"]})
    query = _query_two_col(answer, eval_pipeline={"city": {"metric": ["exact_match"]}})
    client = _make_client([])
    result = asyncio.run(aevaluate_single_query(query, response, client=client))
    assert result.score == 1.0
    assert result.f1_by_row == 1.0
    assert result.f1_by_item == 1.0
    assert result.precision_by_row == 1.0
    assert result.recall_by_row == 1.0


def test_partial_match_drops_strict_score() -> None:
    answer = pd.DataFrame({"name": ["Alice", "Bob"], "city": ["NYC", "LA"]})
    response = pd.DataFrame({"name": ["Alice", "Bob"], "city": ["NYC", "SF"]})
    query = _query_two_col(answer, eval_pipeline={"city": {"metric": ["exact_match"]}})
    client = _make_client([])
    result = asyncio.run(aevaluate_single_query(query, response, client=client))
    assert result.score == 0.0
    # Both rows align on "name" key, but only Alice's city matches
    # row TP = 1, num rows = 2 → P=R=0.5, F1=0.5
    assert result.precision_by_row == pytest.approx(0.5)
    assert result.recall_by_row == pytest.approx(0.5)
    assert result.f1_by_row == pytest.approx(0.5)


def test_missing_row_in_response_lowers_recall() -> None:
    answer = pd.DataFrame({"name": ["Alice", "Bob"], "city": ["NYC", "LA"]})
    response = pd.DataFrame({"name": ["Alice"], "city": ["NYC"]})
    query = _query_two_col(answer, eval_pipeline={"city": {"metric": ["exact_match"]}})
    client = _make_client([])
    result = asyncio.run(aevaluate_single_query(query, response, client=client))
    assert result.score == 0.0
    assert result.precision_by_row == pytest.approx(1.0)
    assert result.recall_by_row == pytest.approx(0.5)
    assert result.f1_by_row == pytest.approx(2 / 3)


def test_extra_row_in_response_lowers_precision() -> None:
    answer = pd.DataFrame({"name": ["Alice"], "city": ["NYC"]})
    response = pd.DataFrame({"name": ["Alice", "Charlie"], "city": ["NYC", "LA"]})
    query = _query_two_col(answer, eval_pipeline={"city": {"metric": ["exact_match"]}})
    client = _make_client([])
    result = asyncio.run(aevaluate_single_query(query, response, client=client))
    assert result.score == 0.0
    assert result.precision_by_row == pytest.approx(0.5)
    assert result.recall_by_row == pytest.approx(1.0)


def test_column_mismatch_triggers_primary_key_preprocess() -> None:
    answer = pd.DataFrame({"name": ["Alice"], "city": ["NYC"]})
    response = pd.DataFrame({"Name": ["Alice"], "City": ["NYC"]})
    # After norm_column both sides become {"name", "city"} so set match → no LLM call
    query = _query_two_col(answer, eval_pipeline={"city": {"metric": ["exact_match"]}})
    client = _make_client([])
    result = asyncio.run(aevaluate_single_query(query, response, client=client))
    assert client.total_calls == 0
    assert result.score == 1.0


def test_column_set_mismatch_invokes_judge_to_remap() -> None:
    answer = pd.DataFrame({"name": ["Alice"], "city": ["NYC"]})
    response = pd.DataFrame({"full_name": ["Alice"], "city": ["NYC"]})
    payload = """```json\n{"full_name": "name"}\n```"""
    client = _make_client([payload])
    query = _query_two_col(answer, eval_pipeline={"city": {"metric": ["exact_match"]}})
    result = asyncio.run(aevaluate_single_query(query, response, client=client))
    assert client.total_calls == 1
    assert result.score == 1.0


def test_column_set_mismatch_with_failed_judge_returns_msg() -> None:
    answer = pd.DataFrame({"name": ["Alice"], "city": ["NYC"]})
    response = pd.DataFrame({"foobar": ["Alice"], "city": ["NYC"]})
    client = _make_client(["nonsense"])
    query = _query_two_col(answer, eval_pipeline={"city": {"metric": ["exact_match"]}})
    result = asyncio.run(aevaluate_single_query(query, response, client=client))
    assert "required_columns" in result.msg
    assert result.score == 0.0
    assert result.f1_by_row == 0.0


def test_llm_judge_column_invoked_for_fuzzy_field() -> None:
    answer = pd.DataFrame({"name": ["Alice", "Bob"], "bio": ["AI researcher", "ML engineer"]})
    response = pd.DataFrame({"name": ["Alice", "Bob"], "bio": ["studies AI", "builds ML systems"]})
    payload = """```json\n{"idx_0": 1, "idx_1": 1}\n```"""
    client = _make_client([payload])
    query = _query_two_col(
        answer,
        eval_pipeline={"bio": {"metric": ["llm_judge"], "criterion": "semantic match"}},
        required=["name", "bio"],
        unique_columns=["name"],
    )
    result = asyncio.run(aevaluate_single_query(query, response, client=client))
    assert client.total_calls == 1
    # When LLM judges all rows perfect, P/R/F1 all = 1 → score gets bumped to 1
    assert result.score == 1.0
    assert result.f1_by_row == pytest.approx(1.0)
    assert result.f1_by_item == pytest.approx(1.0)


def test_preprocess_normalises_before_metric() -> None:
    answer = pd.DataFrame({"name": ["Alice"], "city": ["New York"]})
    response = pd.DataFrame({"name": ["Alice"], "city": ["new york"]})
    query = _query_two_col(
        answer,
        eval_pipeline={"city": {"metric": ["exact_match"], "preprocess": ["norm_str"]}},
    )
    client = _make_client([])
    result = asyncio.run(aevaluate_single_query(query, response, client=client))
    assert result.score == 1.0


def test_duplicate_rows_dropped_before_scoring() -> None:
    answer = pd.DataFrame({"name": ["Alice"], "city": ["NYC"]})
    response = pd.DataFrame({"name": ["Alice", "Alice"], "city": ["NYC", "NYC"]})
    query = _query_two_col(answer, eval_pipeline={"city": {"metric": ["exact_match"]}})
    client = _make_client([])
    result = asyncio.run(aevaluate_single_query(query, response, client=client))
    assert result.score == 1.0
