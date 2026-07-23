# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from recipe.wide_search.eval.llm_judge_column import allm_judge_column
from recipe.wide_search.eval.metric_utils import metric_call
from recipe.wide_search.eval.preprocess import norm_column, preprocess_call
from recipe.wide_search.eval.primary_key_preprocess import aprimary_key_preprocess

if TYPE_CHECKING:
    from recipe.wide_search.eval.data_loader import WideSearchQuery
    from recipe.wide_search.eval.judge_client import WideSearchJudgeClient

_METRIC_FIELDS: tuple[str, ...] = (
    "score",
    "precision_by_row",
    "recall_by_row",
    "f1_by_row",
    "precision_by_item",
    "recall_by_item",
    "f1_by_item",
)


@dataclass
class WideSearchEvalResult:
    instance_id: str
    score: float = 0.0
    precision_by_row: float = 0.0
    recall_by_row: float = 0.0
    f1_by_row: float = 0.0
    precision_by_item: float = 0.0
    recall_by_item: float = 0.0
    f1_by_item: float = 0.0
    msg: str = ""
    judge_calls: int = 0
    judge_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "score": self.score,
            "precision_by_row": self.precision_by_row,
            "recall_by_row": self.recall_by_row,
            "f1_by_row": self.f1_by_row,
            "precision_by_item": self.precision_by_item,
            "recall_by_item": self.recall_by_item,
            "f1_by_item": self.f1_by_item,
            "msg": self.msg,
            "judge_calls": self.judge_calls,
            "judge_errors": list(self.judge_errors),
        }


def _calc_f1(precision: float, recall: float) -> float:
    epsilon = 1e-9
    if precision + recall <= epsilon:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _maybe_align_int_float(answer_df: pd.DataFrame, response_df: pd.DataFrame, col: str) -> None:
    try:
        answer_type = answer_df[col].dtype
        response_type = response_df[col].dtype
    except Exception:
        return
    response_is_float = pd.api.types.is_float_dtype(response_type)
    response_is_int = pd.api.types.is_integer_dtype(response_type)
    answer_is_float = pd.api.types.is_float_dtype(answer_type)
    answer_is_int = pd.api.types.is_integer_dtype(answer_type)
    if (response_is_float and answer_is_int) or (response_is_int and answer_is_float):
        if response_is_int:
            response_df[col] = response_df[col].astype(float)
        elif answer_is_int:
            answer_df[col] = answer_df[col].astype(float)


async def aevaluate_single_query(
    query: WideSearchQuery,
    response_df: pd.DataFrame | None,
    *,
    client: WideSearchJudgeClient,
    prompt_profile: str = "official",
) -> WideSearchEvalResult:
    """Async port of the official ``evaluate_single_query``.

    ``response_df`` is the prediction (already extracted by the caller via
    ``answer_extractor.extract_dataframe``). When ``None``, returns the
    ``response_df is None`` sentinel that mirrors official semantics.
    """
    result = WideSearchEvalResult(instance_id=query.instance_id)
    judge_errors: list[str] = []

    if response_df is None:
        result.msg = "response_df is None"
        return result

    try:
        required_columns: list[str] = list(query.evaluation["required"])
        unique_columns: list[str] = list(query.evaluation["unique_columns"])
        eval_pipeline: dict[str, Any] = dict(query.evaluation.get("eval_pipeline") or {})

        answer_df = query.answer.copy()
        answer_df.columns = [norm_column(str(col)) for col in answer_df.columns]

        response_df = response_df.copy()
        response_df.columns = [norm_column(str(col)) for col in response_df.columns]

        # Column-name alignment via judge LLM
        if set(required_columns) != set(response_df.columns):
            column_map, err = await aprimary_key_preprocess(
                response_df.columns.tolist(),
                required_columns,
                client=client,
                prompt_profile=prompt_profile,
            )
            if err is not None:
                judge_errors.append(err)
            response_df = response_df.rename(columns=column_map)

        if set(required_columns) != set(response_df.columns):
            result.msg = f"required_columns {required_columns} != response_df {response_df.columns.tolist()}"
            result.judge_errors = judge_errors
            result.judge_calls = client.total_calls
            return result

        # int/float alignment + cast to string
        for col in required_columns:
            _maybe_align_int_float(answer_df, response_df, col)
            answer_df[col] = answer_df[col].astype(str)
            response_df[col] = response_df[col].astype(str)

        response_df = response_df.drop_duplicates(subset=unique_columns)
        answer_df = answer_df.drop_duplicates(subset=unique_columns)

        # Value-level alignment for unique columns whose pipeline uses llm_judge or exact_match
        for col in unique_columns:
            item = eval_pipeline.get(col)
            if item is None:
                continue
            metric_func_name_list = item.get("metric", [])
            if "llm_judge" in metric_func_name_list or "exact_match" in metric_func_name_list:
                primary_key_map, err = await aprimary_key_preprocess(
                    response_df[col].tolist(),
                    answer_df[col].tolist(),
                    client=client,
                    prompt_profile=prompt_profile,
                )
                if err is not None:
                    judge_errors.append(f"{col}: {err}")
                response_df[col + "_before_map"] = response_df[col]
                response_df[col] = response_df[col].apply(lambda x, mapping=primary_key_map: mapping.get(x, x))

        # Per-column preprocess
        for col, item in eval_pipeline.items():
            preprocess_func_name_list = item.get("preprocess", [])
            for preprocess_func_name in preprocess_func_name_list:
                if col in response_df.columns:
                    response_df[col] = response_df[col].apply(lambda x, fn=preprocess_func_name: preprocess_call(x, fn))
                if col in answer_df.columns:
                    answer_df[col] = answer_df[col].apply(lambda x, fn=preprocess_func_name: preprocess_call(x, fn))

        # Strict score: shapes match AND sorted DataFrames identical
        temp_score = 0.0
        if answer_df.shape == response_df.shape:
            gt_sorted = answer_df.sort_values(by=required_columns).reset_index(drop=True)
            pred_sorted = response_df.sort_values(by=required_columns).reset_index(drop=True)
            if gt_sorted.equals(pred_sorted):
                temp_score = 1.0
        score = temp_score

        df_inner = answer_df.merge(
            response_df,
            on=unique_columns,
            how="inner",
            suffixes=("_query", "_response"),
        )

        df_inner_score = pd.DataFrame(index=df_inner.index)
        df_inner_msg = pd.DataFrame(index=df_inner.index)

        for col in required_columns:
            if col in unique_columns:
                df_inner_score[f"{col}_exact_match"] = 1.0
                df_inner_msg[f"{col}_exact_match_eval_msg"] = "key_match"
                continue

            item = eval_pipeline.get(col, {})
            metric_func_name_list = item.get("metric", [])
            criterion = item.get("criterion")
            for metric_func_name in metric_func_name_list:
                if metric_func_name == "llm_judge":
                    score_list, msg_list = await allm_judge_column(
                        df_inner[col + "_response"].tolist(),
                        df_inner[col + "_query"].tolist(),
                        criterion,
                        client=client,
                        prompt_profile=prompt_profile,
                    )
                    df_inner_score[f"{col}_{metric_func_name}"] = pd.Series(score_list, index=df_inner.index)
                    df_inner_msg[f"{col}_{metric_func_name}_eval_msg"] = pd.Series(msg_list, index=df_inner.index)
                else:
                    metric_info_series = df_inner.apply(
                        lambda x, fn=metric_func_name, crit=criterion, c=col: metric_call(x[c + "_response"], x[c + "_query"], crit, fn),
                        axis=1,
                    )
                    if len(metric_info_series) == 0:
                        df_inner_score[f"{col}_{metric_func_name}"] = pd.Series(dtype=float)
                        df_inner_msg[f"{col}_{metric_func_name}_eval_msg"] = pd.Series(dtype=object)
                    else:
                        df_inner_score[f"{col}_{metric_func_name}"] = metric_info_series.apply(lambda x: x[0])
                        df_inner_msg[f"{col}_{metric_func_name}_eval_msg"] = metric_info_series.apply(lambda x: x[1])

        if df_inner_score.shape[1] == 0:
            row_scores = pd.Series(dtype=float)
        else:
            row_scores = df_inner_score.min(axis=1)
        tp_by_row = float(row_scores.sum()) if len(row_scores) else 0.0
        tp_by_item = float(df_inner_score.sum().sum()) if df_inner_score.size else 0.0

        num_pred_rows = len(response_df)
        num_gt_rows = len(answer_df)
        num_pred_items = num_pred_rows * len(required_columns)
        num_gt_items = num_gt_rows * len(required_columns)

        precision_by_row = tp_by_row / num_pred_rows if num_pred_rows > 0 else 0.0
        recall_by_row = tp_by_row / num_gt_rows if num_gt_rows > 0 else 0.0
        precision_by_item = tp_by_item / num_pred_items if num_pred_items > 0 else 0.0
        recall_by_item = tp_by_item / num_gt_items if num_gt_items > 0 else 0.0
        f1_by_row = _calc_f1(precision_by_row, recall_by_row)
        f1_by_item = _calc_f1(precision_by_item, recall_by_item)

        msg = df_inner_score.to_string() if df_inner_score.size else ""
        if precision_by_item == recall_by_item == f1_by_item == 1.0 and precision_by_row == recall_by_row == f1_by_row == 1.0:
            msg += "\nAll items match perfectly."
            score = 1.0

        result.score = float(score)
        result.precision_by_row = float(precision_by_row)
        result.recall_by_row = float(recall_by_row)
        result.f1_by_row = float(f1_by_row)
        result.precision_by_item = float(precision_by_item)
        result.recall_by_item = float(recall_by_item)
        result.f1_by_item = float(f1_by_item)
        result.msg = msg
        result.judge_errors = judge_errors
        result.judge_calls = client.total_calls
        return result

    except Exception:
        result.msg = f"evaluator error: \n{traceback.format_exc()}"
        result.judge_errors = judge_errors
        result.judge_calls = client.total_calls
        return result
