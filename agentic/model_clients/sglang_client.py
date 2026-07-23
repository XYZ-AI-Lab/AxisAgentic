# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""SGLang-based model client for local inference.

Loads a model from a local checkpoint (downloaded from HuggingFace Hub) and runs
inference in-process via the SGLang Engine — no API server required.

The client exposes the standard agentic ``ModelClient`` interface.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from agentic.config.models import ModelClientConfig
from agentic.contracts import ConversationMessage, ModelResponse, TokenUsage
from agentic.model_clients.base import ModelClient
from agentic.model_clients.gpu_utils import select_least_utilized_gpus
from agentic.tools.schema_order import validate_rendered_tool_argument_order

logger = logging.getLogger(__name__)


def _coerce_tool_arguments(raw: Any) -> dict[str, Any]:
    r"""Return a dict for ``ToolCallSpec.arguments`` no matter how the parser hands them back.

    sglang's function-call parser returns ``call.parameters`` as a string for the Qwen grammar,
    but when the model emits a double-encoded shape (``"arguments": "{\"query\": ...}"``)
    — which happens when the HF chat template renders a pre-serialized JSON-string via its ``tojson`` filter,
    teaching the model that ``arguments`` is a string-typed field —
    one ``json.loads`` unwraps to a ``str`` rather than a ``dict``.
    We retry the decode once and collapse any remaining non-dict value to ``{}``
    so the rollout continues through the normal ``PARAMETER_ERROR`` tool-result path
    instead of terminating with ``model_error``.
    """
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(args, str):
            args = json.loads(args)
    except (json.JSONDecodeError, TypeError):
        return {}
    return args if isinstance(args, dict) else {}


def _raise_rendered_tool_order_error(message: str) -> None:
    raise RuntimeError(message)


def _chat_template_messages(messages: list[ConversationMessage]) -> list[dict[str, Any]]:
    r"""Render messages for an HF chat template, with tool-call arguments as dicts.

    ``ConversationMessage.to_model_message`` emits ``tool_calls[*].function.arguments`` as a JSON string so OpenAI-compatible APIs accept it,
    but HuggingFace chat templates (e.g. Qwen2.5) apply their own ``tojson`` filter on that value —
    a pre-serialized string gets re-escaped into ``"{\"query\": ...}"`` in the rendered prompt.
    The model then mimics that broken shape on later turns and emits ``arguments`` as a JSON string,
    which sglang's parser returns as a double-encoded ``str`` that fails ``ToolCallSpec`` validation and terminates the rollout with ``model_error``.
    Unwrapping the JSON-string back into a dict here lets the chat template emit a proper ``<args-json-object>``,
    matching the contract documented in Qwen's own system prompt.
    """
    rendered: list[dict[str, Any]] = []
    for msg in messages:
        d = msg.to_model_message()
        tool_calls = d.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function")
                if not isinstance(fn, dict):
                    continue
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        parsed = {}
                    fn["arguments"] = parsed if isinstance(parsed, dict) else {}
        rendered.append(d)
    return rendered


class SGLangModelClientConfig(ModelClientConfig):
    """Configuration for SGLangModelClient (local in-process inference)."""

    model_name_or_path: str
    context_window: int = Field(default=8192, gt=0)
    max_output_tokens: int = Field(default=4096, gt=0)
    temperature: float = 0.6
    top_p: float = 1.0
    tp_size: int = Field(default=1, ge=1)
    dp_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = 0.7
    max_running_requests: int = 128
    enable_tool_calls: bool = False
    download_dir: str | None = None
    trust_remote_code: bool = True
    log_level: str = "warning"
    decode_log_interval: int = 40
    cuda_visible_devices: str | None = None
    """Comma-separated GPU IDs (e.g. ``"4,5"``). ``None`` = auto-select least-utilized GPUs."""


class SGLangModelClient(ModelClient):
    """ModelClient backed by a local SGLang Engine.

    The model is loaded from a local directory (which can be auto-downloaded from
    HuggingFace Hub). Inference runs in the current process on available GPUs.
    """

    def __init__(self, config: SGLangModelClientConfig) -> None:
        super().__init__(
            context_window=config.context_window,
            max_output_tokens=config.max_output_tokens,
            enable_tool_calls=config.enable_tool_calls,
        )
        self._config = config

        self._engine: Any = None
        self._processor: Any = None

    @property
    def model_dir(self) -> Path:
        """Resolved local model directory.

        Resolution order:
        1. If model_name_or_path is an existing directory, use it directly.
        2. If AXIS_MODEL_DIR is set, check AXIS_MODEL_DIR / model_name.
        3. Fall back to download_dir / model_name.
        """
        path = Path(self._config.model_name_or_path)
        if path.is_dir():
            return path
        axis_model_dir = os.environ.get("AXIS_MODEL_DIR")
        if axis_model_dir:
            axis_path = Path(axis_model_dir) / self._config.model_name_or_path
            if axis_path.is_dir():
                return axis_path
        download_dir = Path(self._config.download_dir) if self._config.download_dir else Path.home() / ".cache" / "agentic" / "models"
        return download_dir / self._config.model_name_or_path

    def ensure_model(self) -> Path:
        """Download the model from HuggingFace Hub if not already present. Returns the local path."""
        model_dir = self.model_dir
        if model_dir.is_dir():
            logger.info("Model found at %s.", model_dir)
            return model_dir
        # Download to AXIS_MODEL_DIR if set, otherwise to download_dir.
        axis_model_dir = os.environ.get("AXIS_MODEL_DIR")
        download_dir = Path(self._config.download_dir) if self._config.download_dir else Path.home() / ".cache" / "agentic" / "models"
        target_dir = Path(axis_model_dir) / self._config.model_name_or_path if axis_model_dir else download_dir / self._config.model_name_or_path
        from huggingface_hub import snapshot_download

        logger.info("Downloading model %s to %s ...", self._config.model_name_or_path, target_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=self._config.model_name_or_path, local_dir=str(target_dir))
        logger.info("Model downloaded to %s.", target_dir)
        return target_dir

    def initialize(self) -> None:
        """Download model if needed and start the SGLang engine."""
        if self._engine is not None:
            return

        # Suppress verbose sglang INFO logs (e.g. server_args).
        logging.getLogger("sglang").setLevel(logging.WARNING)

        model_dir = self.ensure_model()

        import torch
        from sglang.srt.entrypoints.engine import Engine
        from sglang.srt.server_args import ServerArgs
        from transformers import AutoProcessor

        # --- GPU selection ---
        num_gpus_needed = self._config.tp_size * self._config.dp_size
        if self._config.cuda_visible_devices is not None:
            selected_devices = self._config.cuda_visible_devices
        else:
            selected_ids = select_least_utilized_gpus(num_gpus_needed)
            selected_devices = ",".join(str(g) for g in selected_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_devices
        logger.info("CUDA_VISIBLE_DEVICES set to %s (need %d GPUs).", selected_devices, num_gpus_needed)

        attention_backend = None
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:
            attention_backend = "triton"

        engine_args = ServerArgs(
            model_path=str(model_dir),
            mem_fraction_static=self._config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            tp_size=self._config.tp_size,
            dp_size=self._config.dp_size,
            enable_memory_saver=True,
            max_running_requests=self._config.max_running_requests,
            attention_backend=attention_backend,
            log_level=self._config.log_level,
            decode_log_interval=self._config.decode_log_interval,
        )
        self._engine = Engine(**dataclasses.asdict(engine_args))
        self._processor = AutoProcessor.from_pretrained(
            str(model_dir),
            use_fast=True,
            trust_remote_code=self._config.trust_remote_code,
        )
        logger.info(
            "SGLang engine initialized for %s (tp=%d, dp=%d).",
            self._config.model_name_or_path,
            self._config.tp_size,
            self._config.dp_size,
        )

    async def acomplete(
        self,
        messages: list[ConversationMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> ModelResponse:
        input_ids = self.messages_to_input_ids(messages, tools=tools)
        return await self.acomplete_input_ids(input_ids=input_ids, tools=tools, tool_choice=tool_choice)

    def messages_to_input_ids(self, messages: list[ConversationMessage], *, tools: list[dict[str, Any]] | None = None) -> list[int]:
        """Render chat messages and return token ids without issuing a generation request.

        This is intentionally public so recipes can pre-tokenize immutable seed prompts
        before their rollout/eval hot path while still using the exact same chat template
        and tokenizer as :meth:`acomplete`.
        """
        self.initialize()
        chat_messages = _chat_template_messages(messages)
        prompt_text = self._processor.apply_chat_template(
            chat_messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
        )
        order_error = validate_rendered_tool_argument_order(prompt_text, tools)
        if order_error is not None:
            raise RuntimeError(order_error)
        return self._processor(prompt_text, return_tensors="np", add_special_tokens=False)["input_ids"][0].tolist()

    async def acomplete_input_ids(
        self,
        input_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> ModelResponse:
        """Generate from pre-tokenized input ids.

        ``tools`` and ``tool_choice`` are still accepted so future tool-aware callers can
        pre-render/tokenize the prompt but keep the same constrained-decoding and parser
        behaviour as :meth:`acomplete`.
        """
        self.initialize()

        from sglang.srt.managers.io_struct import GenerateReqInput

        # Cap max_new_tokens by the remaining context budget.
        max_new_tokens = min(self.max_output_tokens, self._config.context_window - len(input_ids))
        if max_new_tokens <= 0:
            logger.info(
                "Input length (%d) exhausts context window (%d), returning empty response.",
                len(input_ids),
                self._config.context_window,
            )
            empty_message = ConversationMessage.assistant("", tool_calls=None)
            return ModelResponse(
                message=empty_message,
                finish_reason="length",
                usage=TokenUsage(input_tokens=len(input_ids), output_tokens=0, total_tokens=len(input_ids)),
                raw_response={"e2e_elapsed_seconds": 0.0, "skipped_generation": True},
            )

        sampling_params: dict[str, Any] = {
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "max_new_tokens": max_new_tokens,
            "skip_special_tokens": True,
        }

        # Apply tool-call constraints for constrained decoding.
        tool_call_parser: str | None = None
        if tools and tool_choice != "none":
            tool_call_parser = self._apply_tool_call_constraint(tools, tool_choice, sampling_params)

        import uuid

        rid = uuid.uuid4().hex
        req = GenerateReqInput(
            rid=rid,
            input_ids=input_ids,
            return_logprob=False,
            sampling_params=sampling_params,
        )

        # Surface in-process generation wall-clock time in ``raw_response`` so
        # the orchestrator can record a consistent model-side latency metric.
        generate_start = time.perf_counter()
        results = [result async for result in self._engine.tokenizer_manager.generate_request(req, None)]
        e2e_elapsed_seconds = time.perf_counter() - generate_start
        if not results:
            msg = "SGLang engine returned no results."
            raise RuntimeError(msg)

        last_result = results[-1]
        output_text = last_result["text"]
        finish_reason = last_result.get("meta_info", {}).get("finish_reason", {}).get("type", "stop")

        # Parse tool calls from the generated output text.
        parsed_tool_calls = None
        if tools and tool_call_parser:
            output_text, parsed_tool_calls, finish_reason = self._parse_tool_calls(
                output_text,
                tools=tools,
                tool_choice=tool_choice,
                tool_call_parser=tool_call_parser,
                finish_reason=finish_reason,
            )

        usage_info = last_result.get("meta_info", {})
        usage = TokenUsage(
            input_tokens=usage_info.get("prompt_tokens", len(input_ids)),
            output_tokens=usage_info.get("completion_tokens", 0),
            total_tokens=usage_info.get("prompt_tokens", len(input_ids)) + usage_info.get("completion_tokens", 0),
        )

        assistant_message = ConversationMessage.assistant(output_text or None, tool_calls=parsed_tool_calls)
        return ModelResponse(
            message=assistant_message,
            finish_reason=finish_reason,
            usage=usage,
            raw_response={"e2e_elapsed_seconds": e2e_elapsed_seconds},
        )

    def _apply_tool_call_constraint(
        self,
        tools: list[dict[str, Any]],
        tool_choice: str | None,
        sampling_params: dict[str, Any],
    ) -> str | None:
        """Apply a structural-tag constraint and return the parser name."""
        from sglang.srt.entrypoints.openai.protocol import Tool as SGLangTool
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        from sglang.utils import convert_json_schema_to_str

        resolved = tool_choice or "auto"
        if resolved == "none":
            return None

        tool_models = [SGLangTool.model_validate(t) for t in tools]

        if resolved == "auto":
            # Use "qwen" parser for Qwen models (default); detect from model name
            parser_name = "qwen"
            parser = FunctionCallParser(tool_models, parser_name)
            constraint = parser.get_structure_constraint(resolved)
            if constraint is not None:
                constraint_type, constraint_value = constraint
                if constraint_type == "structural_tag":
                    sampling_params[constraint_type] = convert_json_schema_to_str(constraint_value.model_dump(by_alias=True))
                elif constraint_type == "json_schema":
                    sampling_params[constraint_type] = convert_json_schema_to_str(dict(constraint_value))
                else:
                    sampling_params[constraint_type] = constraint_value
            return parser_name

        return None

    @staticmethod
    def _parse_tool_calls(
        text: str,
        *,
        tools: list[dict[str, Any]],
        tool_choice: str | None,
        tool_call_parser: str,
        finish_reason: str,
    ) -> tuple[str, list[Any] | None, str]:
        """Parse tool calls from generated text."""
        import uuid

        from agentic.contracts.messages import ToolCall as AgenticToolCall
        from agentic.contracts.messages import ToolCallSpec

        if not tools:
            return text, None, finish_reason

        resolved = tool_choice or "auto"
        if resolved == "none":
            return text, None, finish_reason

        # "required" / named-function: json_schema constraint was applied, output is raw JSON
        if resolved == "required" or (resolved not in {"auto", "none"}):
            try:
                tool_call_data = json.loads(text)
            except json.JSONDecodeError:
                return text, None, finish_reason
            if isinstance(tool_call_data, dict):
                tool_call_data = [tool_call_data]
            agentic_calls = [
                AgenticToolCall(
                    id=f"call_{uuid.uuid4().hex}",
                    function=ToolCallSpec(name=tc["name"], arguments=_coerce_tool_arguments(tc.get("parameters", tc.get("arguments", {})))),
                )
                for tc in tool_call_data
            ]
            return "", agentic_calls, "tool_calls" if finish_reason == "stop" else finish_reason

        # "auto": model may or may not have called a tool using native format (e.g. <tool_call>)
        from sglang.srt.entrypoints.openai.protocol import Tool as SGLangTool
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        tool_models = [SGLangTool.model_validate(t) for t in tools]
        parser = FunctionCallParser(tool_models, tool_call_parser)
        if not parser.has_tool_call(text):
            return text, None, finish_reason

        try:
            normal_text, parsed_calls = parser.parse_non_stream(text)
        except Exception:
            logger.warning("Failed to parse tool calls from text: %r", text, exc_info=True)
            return text, None, finish_reason

        if not parsed_calls:
            return text, None, finish_reason

        agentic_tool_calls: list[AgenticToolCall] = []
        for call in parsed_calls:
            agentic_tool_calls.append(
                AgenticToolCall(
                    id=f"call_{uuid.uuid4().hex}",
                    function=ToolCallSpec(name=call.name or "", arguments=_coerce_tool_arguments(call.parameters)),
                )
            )
        return normal_text, agentic_tool_calls, "tool_calls" if finish_reason == "stop" else finish_reason

    def estimate_tokens(self, messages: list[ConversationMessage], tools: list[dict[str, Any]] | None = None) -> int:
        if self._processor is not None:
            chat_messages = _chat_template_messages(messages)
            try:
                text = self._processor.apply_chat_template(
                    chat_messages,
                    tools=tools or None,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                order_error = validate_rendered_tool_argument_order(text, tools)
                if order_error is not None:
                    _raise_rendered_tool_order_error(order_error)
                tokens = self._processor(text, return_tensors="np", add_special_tokens=False)["input_ids"][0]
                return len(tokens)
            except Exception as e:
                print(e)
        return super().estimate_tokens(messages)

    def shutdown(self) -> None:
        """Shutdown the engine and release resources."""
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None
            self._processor = None
            logger.info("SGLang engine shut down.")
