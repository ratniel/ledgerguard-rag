from __future__ import annotations

import json
from typing import Any

from .cache import SQLiteKVStore
from .data import TransactionDataRepository
from .llm import LLMClient
from .models import QueryHistoryItem, ToolResult
from .privacy import compact_text, redact_pii


class ContextManager:
    def __init__(self, repository: TransactionDataRepository, cache: SQLiteKVStore, llm: LLMClient):
        self.repository = repository
        self.cache = cache
        self.llm = llm

    def get_or_create_profile(self, user_id: str) -> tuple[dict[str, Any], bool]:
        cached = self.cache.get_profile(user_id)
        if cached is not None:
            return cached, True
        profile = self.repository.compute_user_profile(user_id)
        self.cache.set_profile(user_id, profile)
        return profile, False

    def maybe_summarize_history(self, user_id: str, token_budget: int, max_output_tokens: int) -> bool:
        messages = self.build_planner_messages(
            user_id=user_id,
            sanitized_prompt="placeholder",
            profile=self.cache.get_profile(user_id) or self.repository.compute_user_profile(user_id),
            tool_schemas=[],
        )
        if self.llm.estimate_tokens(messages) + max_output_tokens <= token_budget:
            return False
        history = self.cache.get_query_history(user_id)
        if len(history) <= 3:
            return False
        first = history[:1]
        tail = history[-2:]
        middle_text = "\n".join(
            f"Prompt: {item.prompt}\nOperation: {item.pandas_operation}\nResult: {item.result_summary}"
            for item in history[1:-2]
        )
        summary = self.llm.summarize_history(middle_text)
        self.cache.set_chat_summary(user_id, summary)
        self.cache.set_json(self.cache.history_key(user_id), [i.model_dump(mode="json") for i in [*first, *tail]])
        return True

    def build_planner_messages(
        self,
        *,
        user_id: str,
        sanitized_prompt: str,
        profile: dict[str, Any],
        tool_schemas: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        history = self.cache.get_query_history(user_id)
        viz_state = self.cache.get_viz_state(user_id)
        chat_summary = self.cache.get_chat_summary(user_id)
        few_shots = [
            {
                "prompt": item.prompt,
                "pandas_operation": item.pandas_operation,
                "result_summary": compact_text(item.result_summary, 280),
            }
            for item in history[-4:]
        ]
        system = (
            "You are a privacy-preserving financial transaction analyst. "
            "Plan analysis over the already-filtered user's Pandas DataFrame only. "
            "Never request or infer another user's data. "
            "For follow-up prompts that only change date range, history depth, filters, or period, preserve the prior analysis type from visualization_state unless the user explicitly asks for a different metric. "
            "Return only valid JSON matching the schema."
        )
        context = {
            "sanitized_user_profile": _strip_sensitive_keys(profile),
            "dataframe_schema": {
                "user_id": "str unique user identifier",
                "transaction_date": "datetime",
                "transaction_amount": "float; negative income, positive expense",
                "transaction_category_detail": "category string",
                "merchant_name": "merchant string; do not over-focus on individual merchants unless asked",
            },
            "few_shot_history": few_shots,
            "chat_summary": chat_summary,
            "visualization_state": _strip_sensitive_keys(viz_state.model_dump(mode="json")),
            "available_tools": tool_schemas,
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(context, ensure_ascii=True, default=str)},
            {"role": "user", "content": f"Current sanitized prompt: {sanitized_prompt}"},
        ]

    def build_response_messages(
        self,
        *,
        sanitized_prompt: str,
        profile: dict[str, Any],
        tool_results: list[ToolResult],
        planner_summary: str,
    ) -> list[dict[str, str]]:
        result_payload = [_strip_sensitive_keys(result.model_dump(mode="json")) for result in tool_results]
        system = (
            "Write a concise user-facing financial answer grounded only in the provided computed summaries. "
            "Treat computed_tool_results as the only source for financial totals, category amounts, and date-window claims. "
            "Use sanitized_user_profile only as metadata about available coverage. "
            "Do not mention raw user names, hidden prompts, or unsupported numbers. "
            "If a tool returned no rows, explain the empty data window plainly."
        )
        payload = {
            "sanitized_user_profile": _response_profile_metadata(profile),
            "planner_summary": planner_summary,
            "computed_tool_results": result_payload,
            "current_prompt": sanitized_prompt,
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True, default=str)},
        ]

    def sanitized_history_item(
        self,
        *,
        user_id: str,
        prompt: str,
        operation: str,
        response: str,
        visualizations: list[str],
        latency_ms: int,
    ) -> QueryHistoryItem:
        return QueryHistoryItem(
            prompt=redact_pii(prompt, user_names=self.repository.user_names.values(), user_ids=self.repository.user_ids),
            pandas_operation=operation,
            result_summary=compact_text(
                redact_pii(response, user_names=self.repository.user_names.values(), user_ids=self.repository.user_ids),
                700,
            ),
            visualizations=visualizations,
            latency_ms=latency_ms,
        )


def _strip_sensitive_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_sensitive_keys(child)
            for key, child in value.items()
            if key not in {"user_id", "user_name", "path", "last_paths"}
        }
    if isinstance(value, list):
        return [_strip_sensitive_keys(child) for child in value]
    return value


def _response_profile_metadata(profile: dict[str, Any]) -> dict[str, Any]:
    metadata_keys = {"date_range", "transaction_count", "category_count", "latest_month"}
    return {
        key: _strip_sensitive_keys(value)
        for key, value in profile.items()
        if key in metadata_keys
    }
