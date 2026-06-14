from __future__ import annotations

import json
import re
from typing import Any

import tiktoken
from openai import OpenAI
from pydantic import ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import Settings
from .errors import CircuitOpenError, LLMTimeoutError, LLMUnavailableError, MalformedLLMOutputError
from .models import GuardrailFlag, PlannerOutput, ToolCall, ToolName


def _is_retryable_llm_error(error: BaseException) -> bool:
    return isinstance(error, LLMUnavailableError | MalformedLLMOutputError) and not isinstance(error, CircuitOpenError)


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._failures = 0
        self._client: OpenAI | None = None
        self.planner_models = _dedupe_models(_split_models(settings.openrouter_planner_models))
        self.response_models = _dedupe_models(
            [
                *_split_models(settings.openrouter_response_models),
                settings.openrouter_primary_model,
                settings.openrouter_fallback_model,
            ]
        )
        if settings.enable_llm and settings.openrouter_api_key:
            self._client = OpenAI(
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
                timeout=settings.request_timeout_seconds,
                max_retries=0,
            )

    @property
    def available(self) -> bool:
        return self._client is not None and self._failures < self.settings.circuit_breaker_threshold

    def estimate_tokens(self, value: str | list[dict[str, Any]]) -> int:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            if isinstance(value, str):
                return max(1, len(value) // 4)
            return max(1, len(json.dumps(value, default=str)) // 4)
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=True, default=str)
        return len(enc.encode(text))

    def plan(self, messages: list[dict[str, str]]) -> tuple[PlannerOutput, dict[str, int]]:
        if not self.available:
            raise CircuitOpenError("LLM is unavailable or circuit is open")
        return self._run_planner(messages)

    def compose_response(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, int]]:
        if not self.available:
            raise CircuitOpenError("LLM is unavailable or circuit is open")
        return self._run_text_completion(messages)

    def summarize_history(self, history_text: str) -> str:
        if not self.available:
            return self._deterministic_summary(history_text)
        messages = [
            {
                "role": "system",
                "content": "Summarize this user's prior financial chat history without PII. Keep facts, date windows, tool results, and unresolved follow-ups. Use under 160 words.",
            },
            {"role": "user", "content": history_text},
        ]
        try:
            response, _ = self.compose_response(messages)
            return response.strip()
        except Exception:
            return self._deterministic_summary(history_text)

    @retry(
        retry=retry_if_exception(_is_retryable_llm_error),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        stop=stop_after_attempt(2),
        reraise=True,
    )
    def _run_planner(self, messages: list[dict[str, str]]) -> tuple[PlannerOutput, dict[str, int]]:
        assert self._client is not None
        errors: list[Exception] = []
        for model in self.planner_models:
            try:
                completion = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.settings.llm_temperature,
                    max_tokens=self.settings.llm_max_output_tokens,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "planner_output",
                            "strict": True,
                            "schema": PlannerOutput.model_json_schema(),
                        },
                    },
                    extra_headers={
                        "HTTP-Referer": "http://localhost",
                        "X-Title": "LedgerGuard RAG",
                    },
                )
                content = completion.choices[0].message.content or "{}"
                planner = PlannerOutput.model_validate_json(content)
                self._failures = 0
                return planner, self._usage_dict(completion)
            except (json.JSONDecodeError, ValidationError) as exc:
                errors.append(exc)
            except Exception as exc:
                errors.append(exc)
        self._failures += 1
        if self._failures >= self.settings.circuit_breaker_threshold:
            raise CircuitOpenError(_format_model_errors(errors))
        if _has_timeout_error(errors):
            raise LLMTimeoutError(_format_model_errors(errors))
        if errors and all(isinstance(error, (json.JSONDecodeError, ValidationError)) for error in errors):
            raise MalformedLLMOutputError(_format_model_errors(errors))
        raise LLMUnavailableError(_format_model_errors(errors))

    @retry(
        retry=retry_if_exception(_is_retryable_llm_error),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        stop=stop_after_attempt(2),
        reraise=True,
    )
    def _run_text_completion(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, int]]:
        assert self._client is not None
        errors: list[Exception] = []
        for model in self.response_models:
            try:
                completion = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.settings.llm_temperature,
                    max_tokens=self.settings.llm_max_output_tokens,
                    extra_headers={
                        "HTTP-Referer": "http://localhost",
                        "X-Title": "LedgerGuard RAG",
                    },
                )
                self._failures = 0
                return completion.choices[0].message.content or "", self._usage_dict(completion)
            except Exception as exc:
                errors.append(exc)
        self._failures += 1
        if self._failures >= self.settings.circuit_breaker_threshold:
            raise CircuitOpenError(_format_model_errors(errors))
        if _has_timeout_error(errors):
            raise LLMTimeoutError(_format_model_errors(errors))
        raise LLMUnavailableError(_format_model_errors(errors))

    @staticmethod
    def heuristic_plan(prompt: str) -> PlannerOutput:
        lowered = prompt.lower()
        tool_calls: list[ToolCall] = []
        intent = "general_financial_analysis"
        if any(term in lowered for term in ["full report", "financial report", "full financial report", "financially", "how am i doing"]):
            intent = "full_financial_report"
            tool_calls = [
                ToolCall(name=ToolName.INCOME_VS_EXPENSE, arguments={"months": 6}, rationale="Assess savings trend."),
                ToolCall(name=ToolName.CATEGORY_BREAKDOWN, arguments={"period": "last_3_months", "top_n": 7}, rationale="Show where money goes."),
                ToolCall(name=ToolName.MONTHLY_SPENDING_TREND, arguments={"months": 6}, rationale="Show spending over time."),
            ]
        elif any(term in lowered for term in ["saving", "savings", "bleeding", "income vs", "income versus"]):
            intent = "savings_assessment"
            arguments: dict[str, Any] = {"months": 6}
            if any(term in lowered for term in ["last month", "previous month", "prior month"]):
                arguments = {"months": 1, "period": "last_month"}
            elif any(term in lowered for term in ["this month", "current month"]):
                arguments = {"months": 1, "period": "current_month"}
            tool_calls = [ToolCall(name=ToolName.INCOME_VS_EXPENSE, arguments=arguments, rationale="Compare income, expenses, and net savings.")]
        elif any(term in lowered for term in ["trend", "changed over time", "over time"]):
            intent = "spending_trend"
            tool_calls = [ToolCall(name=ToolName.MONTHLY_SPENDING_TREND, arguments={"months": 6}, rationale="Show monthly spending trend.")]
        elif any(term in lowered for term in ["most", "where", "category", "breakdown", "money going", "spent the most"]):
            intent = "category_breakdown"
            period = "last_month" if "last month" in lowered else "last_3_months"
            tool_calls = [ToolCall(name=ToolName.CATEGORY_BREAKDOWN, arguments={"period": period, "top_n": 7}, rationale="Identify top spending categories.")]
        elif "food" in lowered:
            intent = "food_spending"
            tool_calls = [ToolCall(name=ToolName.MONTHLY_SPENDING_TREND, arguments={"months": 6, "category_filter": "food"}, rationale="Show food spending over time.")]
        elif any(term in lowered for term in ["spend", "expense", "expenses", "month"]):
            intent = "spending_summary"
            tool_calls = [ToolCall(name=ToolName.CATEGORY_BREAKDOWN, arguments={"period": "last_3_months", "top_n": 7}, rationale="Summarize recent spending categories.")]
        return PlannerOutput(
            user_intent=intent,
            data_focus="Filtered user DataFrame only.",
            response_plan="Use deterministic Pandas summaries and generated tool outputs. Avoid unsupported claims.",
            tool_calls=tool_calls,
            confidence=0.68 if not tool_calls else 0.86,
        )

    @staticmethod
    def _usage_dict(completion: Any) -> dict[str, int]:
        usage = getattr(completion, "usage", None)
        if usage is None:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        return {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _deterministic_summary(text: str) -> str:
        compact = " ".join(text.split())
        if len(compact) <= 700:
            return compact
        return compact[:697].rstrip() + "..."


def _split_models(raw: str) -> list[str]:
    return [model.strip() for model in re.split(r"[,\n]", raw or "") if model.strip()]


def _dedupe_models(models: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for model in models:
        if model not in seen:
            seen.add(model)
            unique.append(model)
    return unique


def _format_model_errors(errors: list[Exception]) -> str:
    if not errors:
        return "No OpenRouter models were configured."
    compact_errors = [f"{type(error).__name__}: {str(error)[:220]}" for error in errors[-3:]]
    return " | ".join(compact_errors)


def _has_timeout_error(errors: list[Exception]) -> bool:
    return any(_is_timeout_error(error) for error in errors)


def _is_timeout_error(error: Exception) -> bool:
    name = type(error).__name__.lower()
    message = str(error).lower()
    return isinstance(error, TimeoutError) or "timeout" in name or "timed out" in message
