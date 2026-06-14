from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolName(str, Enum):
    MONTHLY_SPENDING_TREND = "plot_monthly_spending_trend"
    CATEGORY_BREAKDOWN = "plot_category_breakdown"
    INCOME_VS_EXPENSE = "plot_income_vs_expense"


class GuardrailFlag(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    OFF_TOPIC = "off_topic"
    INPUT_LENGTH_EXCEEDED = "input_length_exceeded"
    CROSS_USER_LEAKAGE = "cross_user_leakage"
    OUTPUT_UNGROUNDED = "output_ungrounded"
    TOXIC_OUTPUT = "toxic_output"
    LOW_CONFIDENCE = "low_confidence"
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"
    TIMEOUT = "timeout"
    CIRCUIT_OPEN = "circuit_open"
    MALFORMED_LLM_OUTPUT = "malformed_llm_output"
    LLM_UNAVAILABLE = "llm_unavailable"


class ErrorCode(str, Enum):
    INVALID_USER = "invalid_user"
    GUARDRAIL_BLOCKED = "guardrail_blocked"
    INTERNAL_ERROR = "internal_error"


class ErrorDetails(BaseModel):
    code: ErrorCode
    message: str
    retryable: bool = False


class ToolCall(BaseModel):
    name: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class PlannerOutput(BaseModel):
    guardrail_triggered: bool = False
    guardrail_flags: list[GuardrailFlag] = Field(default_factory=list)
    user_intent: str = ""
    data_focus: str = ""
    response_plan: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)


class ToolResult(BaseModel):
    name: ToolName
    path: str | None = None
    rows: int = 0
    parameters: dict[str, Any] = Field(default_factory=dict)
    data_summary: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


class QueryHistoryItem(BaseModel):
    prompt: str
    pandas_operation: str
    result_summary: str
    visualizations: list[str] = Field(default_factory=list)
    latency_ms: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


class VizState(BaseModel):
    last_chart_type: ToolName | None = None
    axes: dict[str, str] = Field(default_factory=dict)
    filters: dict[str, Any] = Field(default_factory=dict)
    last_parameters: dict[str, Any] = Field(default_factory=dict)
    last_paths: list[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


class PipelineResult(BaseModel):
    user_name: str | None = None
    response: str
    data_summary: dict[str, Any] = Field(default_factory=dict)
    visualizations: list[str] = Field(default_factory=list)
    cache_hit: bool = False
    latency_ms: int = 0
    guardrail_flags: list[GuardrailFlag] = Field(default_factory=list)
    error: ErrorDetails | None = None
