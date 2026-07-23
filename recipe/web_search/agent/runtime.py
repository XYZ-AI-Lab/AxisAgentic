# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import Field

from agentic.config import AssistantRollbackConfig, ConversationConfig, FormatErrorConfig
from agentic.contracts import ConversationMessage, ConversationStage, FinalizationTrigger, FormatErrorStrategy, MessageRole
from agentic.conversations.conversation_runtime import ConversationRuntime
from recipe.web_search.agent.prompts import extract_boxed_content

TOOL_RESULT_OMITTED_PLACEHOLDER = "Tool result is omitted to save tokens."


class WebSearchConversationConfig(ConversationConfig):
    keep_tool_result: int = Field(
        default=5,
        description="How many recent tool-result messages to keep untruncated. -1 keeps all, 0 replaces every tool message.",
    )
    format_error: FormatErrorConfig = Field(
        default_factory=lambda: FormatErrorConfig(strategy=FormatErrorStrategy.IGNORE),
        description="Native tool calls do not require text-format recovery by default.",
    )
    assistant_rollback: AssistantRollbackConfig = Field(
        default_factory=lambda: AssistantRollbackConfig(
            refusal_keywords=[
                "time constraint",
                "I'm sorry, but I can't",
                "I'm sorry, I cannot solve",
            ],
            max_rollbacks=4,
        ),
    )


class WebSearchConversationRuntime(ConversationRuntime):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        config = kwargs.get("config")
        self.keep_tool_result: int = getattr(config, "keep_tool_result", 5)
        self.skip_turn_limit_final_response: bool = False
        self.skip_prompted_final_response: bool = False

    def _clone_extra_state_to(self, cloned: ConversationRuntime) -> None:
        super()._clone_extra_state_to(cloned)
        cloned_any: Any = cloned
        cloned_any.keep_tool_result = self.keep_tool_result
        cloned_any.skip_turn_limit_final_response = self.skip_turn_limit_final_response
        cloned_any.skip_prompted_final_response = self.skip_prompted_final_response

    def _inject_force_finalization(
        self,
        trigger: FinalizationTrigger,
        *,
        stage_path: list[ConversationStage],
        appended_messages: list[ConversationMessage],
    ) -> dict[str, object]:
        skip_prompt = self.skip_prompted_final_response or self.skip_turn_limit_final_response
        if trigger in {FinalizationTrigger.TURN_LIMIT, FinalizationTrigger.CONTEXT_LIMIT, FinalizationTrigger.TOOLS_EXHAUSTED} and skip_prompt:
            self._force_final_trigger = trigger
            self._force_final_has_prompt = False
            self._transit_stage(ConversationStage.FORCE_FINAL_AWAIT_ASSISTANT, stage_path=stage_path)
            info: dict[str, object] = {
                "force_finalize_stage": self.stage.value,
                "finalization_trigger": trigger.value,
                "has_early_stop_prompt": False,
                "skipped_budget_limit_final_response": True,
            }
            if trigger == FinalizationTrigger.TURN_LIMIT:
                info["skipped_turn_limit_final_response"] = True
            elif trigger == FinalizationTrigger.CONTEXT_LIMIT:
                info["skipped_context_limit_final_response"] = True
            else:
                info["skipped_tools_exhausted_final_response"] = True
            return info

        return super()._inject_force_finalization(trigger, stage_path=stage_path, appended_messages=appended_messages)

    def _extract_direct_final_answer(self, message: ConversationMessage) -> str | None:
        if not message.content:
            return None
        boxed = extract_boxed_content(message.content)
        return boxed or None

    def _should_prompt_for_final_response(self, _message: ConversationMessage, _prior_stage: ConversationStage) -> bool:
        return False

    def _force_finalize_due_to_context_limit(
        self,
        *,
        stage_path: list[ConversationStage],
        appended_messages: list[ConversationMessage],
        context_info: dict[str, object],
        rollback: bool = True,
    ) -> dict[str, object]:
        if self.skip_prompted_final_response or self.skip_turn_limit_final_response:
            info = self._inject_force_finalization(FinalizationTrigger.CONTEXT_LIMIT, stage_path=stage_path, appended_messages=appended_messages)
            info["rollback_message_count"] = 0
            info["context_limit_estimate"] = context_info
            return info
        return super()._force_finalize_due_to_context_limit(
            stage_path=stage_path,
            appended_messages=appended_messages,
            context_info=context_info,
            rollback=rollback,
        )

    def _compute_rollback_message_count(self, visible_conversation: list[ConversationMessage]) -> int:
        if self.stage == ConversationStage.TOOL_MESSAGES_AWAIT_ASSISTANT:
            rollback_count = 0
            for message in reversed(visible_conversation):
                if message.role == MessageRole.TOOL:
                    rollback_count += 1
                    continue
                break
            if rollback_count > 0:
                return rollback_count
        return super()._compute_rollback_message_count(visible_conversation)

    def _build_visible_conversation(self) -> list[ConversationMessage]:
        visible = super()._build_visible_conversation()
        if self.keep_tool_result < 0:
            return visible

        tool_indices = [i for i, msg in enumerate(visible) if msg.role == MessageRole.TOOL]
        if not tool_indices:
            return visible

        if self.keep_tool_result == 0:
            to_truncate = set(tool_indices)
        elif len(tool_indices) > self.keep_tool_result:
            to_truncate = set(tool_indices[: -self.keep_tool_result])
        else:
            return visible

        result: list[ConversationMessage] = []
        for idx, msg in enumerate(visible):
            if idx in to_truncate:
                result.append(
                    ConversationMessage(
                        role=MessageRole.TOOL,
                        content=TOOL_RESULT_OMITTED_PLACEHOLDER,
                        tool_call_id=msg.tool_call_id,
                        name=msg.name,
                    )
                )
            else:
                result.append(msg)
        return result
