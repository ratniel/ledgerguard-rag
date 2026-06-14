from __future__ import annotations

import re
import time
from typing import Any

import pandas as pd
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

from .cache import AuditLogger, SQLiteKVStore
from .config import Settings
from .context import ContextManager
from .data import TransactionDataRepository
from .errors import CircuitOpenError, InvalidUserError, LLMTimeoutError, LLMUnavailableError, MalformedLLMOutputError
from .guardrails import InputGuardrails, OutputGuardrails
from .llm import LLMClient
from .models import (
    ErrorCode,
    ErrorDetails,
    GuardrailFlag,
    PipelineResult,
    PlannerOutput,
    QueryHistoryItem,
    ToolCall,
    ToolName,
    ToolResult,
    VizState,
)
from .observability import Observability
from .responses import collect_allowed_numbers, deterministic_response
from .tools import VisualizationToolkit


class ValidatedEvent(Event):
    user_id: str
    prompt: str
    sanitized_prompt: str
    start_time: float
    flags: list[GuardrailFlag]


class ContextReadyEvent(Event):
    user_id: str
    prompt: str
    sanitized_prompt: str
    start_time: float
    flags: list[GuardrailFlag]
    profile: dict[str, Any]
    cache_hit: bool


class FollowUpResolvedEvent(Event):
    user_id: str
    prompt: str
    sanitized_prompt: str
    start_time: float
    flags: list[GuardrailFlag]
    profile: dict[str, Any]
    cache_hit: bool
    planner_override: PlannerOutput | None
    resolution: dict[str, Any]


class PlannedEvent(Event):
    user_id: str
    prompt: str
    sanitized_prompt: str
    start_time: float
    flags: list[GuardrailFlag]
    profile: dict[str, Any]
    cache_hit: bool
    planner: PlannerOutput
    usage: dict[str, int]
    resolution: dict[str, Any]


class ToolsDoneEvent(Event):
    user_id: str
    prompt: str
    sanitized_prompt: str
    start_time: float
    flags: list[GuardrailFlag]
    profile: dict[str, Any]
    cache_hit: bool
    planner: PlannerOutput
    tool_results: list[ToolResult]
    usage: dict[str, int]
    resolution: dict[str, Any]


class ResponseReadyEvent(Event):
    user_id: str
    prompt: str
    sanitized_prompt: str
    start_time: float
    flags: list[GuardrailFlag]
    profile: dict[str, Any]
    cache_hit: bool
    planner: PlannerOutput
    tool_results: list[ToolResult]
    response: str
    usage: dict[str, int]
    resolution: dict[str, Any]


class TransactionRAGWorkflow(Workflow):
    def __init__(
        self,
        *,
        settings: Settings,
        repository: TransactionDataRepository,
        cache: SQLiteKVStore,
        audit_logger: AuditLogger,
        context_manager: ContextManager,
        toolkit: VisualizationToolkit,
        input_guardrails: InputGuardrails,
        output_guardrails: OutputGuardrails,
        llm: LLMClient,
        observability: Observability,
    ):
        super().__init__(timeout=settings.request_timeout_seconds + 15, disable_validation=True)
        self.settings = settings
        self.repository = repository
        self.cache = cache
        self.audit_logger = audit_logger
        self.context_manager = context_manager
        self.toolkit = toolkit
        self.input_guardrails = input_guardrails
        self.output_guardrails = output_guardrails
        self.llm = llm
        self.observability = observability

    @step
    async def validate_input(self, ev: StartEvent) -> ValidatedEvent | StopEvent:
        start_time = time.perf_counter()
        user_id = ev.get("user_id")
        prompt = ev.get("prompt")
        span_user_id = user_id if user_id in self.repository.user_ids else ""
        with self.observability.span("validate_input", **{"user.id": span_user_id}):
            if not user_id or not prompt:
                return self._stop_error(
                    user_id=user_id,
                    response="Please provide both user_id and prompt.",
                    start_time=start_time,
                    code=ErrorCode.INTERNAL_ERROR,
                    flags=[],
                    retryable=False,
                )
            try:
                self.repository.validate_user_id(user_id)
            except InvalidUserError:
                return self._stop_error(
                    user_id=None,
                    response="I could not find transactions for that user_id.",
                    start_time=start_time,
                    code=ErrorCode.INVALID_USER,
                    flags=[],
                    retryable=False,
                )
            decision = self.input_guardrails.validate(
                user_id=user_id,
                prompt=prompt,
                valid_user_ids=self.repository.user_ids,
                user_names=self.repository.user_names,
            )
            if decision.blocked:
                if (
                    GuardrailFlag.OFF_TOPIC in decision.flags
                    and _is_contextual_financial_follow_up(
                        decision.sanitized_prompt,
                        self.cache.get_viz_state(user_id),
                    )
                ):
                    return ValidatedEvent(
                        user_id=user_id,
                        prompt=decision.prompt,
                        sanitized_prompt=decision.sanitized_prompt,
                        start_time=start_time,
                        flags=[flag for flag in decision.flags if flag != GuardrailFlag.OFF_TOPIC],
                    )
                return self._stop_error(
                    user_id=user_id,
                    response=decision.response or "That request was blocked by the safety policy.",
                    start_time=start_time,
                    code=ErrorCode.GUARDRAIL_BLOCKED,
                    flags=decision.flags,
                    retryable=False,
                )
            return ValidatedEvent(
                user_id=user_id,
                prompt=decision.prompt,
                sanitized_prompt=decision.sanitized_prompt,
                start_time=start_time,
                flags=decision.flags,
            )

    @step
    async def assemble_context(self, ev: ValidatedEvent) -> ContextReadyEvent:
        with self.observability.span("context_assembly", **{"user.id": ev.user_id}):
            profile, cache_hit = self.context_manager.get_or_create_profile(ev.user_id)
            summarized = self.context_manager.maybe_summarize_history(
                ev.user_id,
                token_budget=self.settings.token_budget,
                max_output_tokens=self.settings.llm_max_output_tokens,
            )
            flags = list(ev.flags)
            if summarized:
                flags.append(GuardrailFlag.TOKEN_BUDGET_EXCEEDED)
            return ContextReadyEvent(
                user_id=ev.user_id,
                prompt=ev.prompt,
                sanitized_prompt=ev.sanitized_prompt,
                start_time=ev.start_time,
                flags=flags,
                profile=profile,
                cache_hit=cache_hit,
            )

    @step
    async def resolve_follow_up(self, ev: ContextReadyEvent) -> FollowUpResolvedEvent:
        with self.observability.span("follow_up_resolution", **{"user.id": ev.user_id}):
            override, resolution = _resolve_follow_up_window_request(
                prompt=ev.sanitized_prompt,
                user_id=ev.user_id,
                repository=self.repository,
                viz_state=self.cache.get_viz_state(ev.user_id),
            )
            return FollowUpResolvedEvent(
                user_id=ev.user_id,
                prompt=ev.prompt,
                sanitized_prompt=ev.sanitized_prompt,
                start_time=ev.start_time,
                flags=ev.flags,
                profile=ev.profile,
                cache_hit=ev.cache_hit,
                planner_override=override,
                resolution=resolution,
            )

    @step
    async def plan_response(self, ev: FollowUpResolvedEvent) -> PlannedEvent:
        flags = list(ev.flags)
        usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        with self.observability.span("planning_llm", **{"user.id": ev.user_id}):
            if ev.planner_override is not None:
                planner = ev.planner_override
            else:
                messages = self.context_manager.build_planner_messages(
                    user_id=ev.user_id,
                    sanitized_prompt=ev.sanitized_prompt,
                    profile=ev.profile,
                    tool_schemas=self.toolkit.tool_schemas(),
                )
                if self.llm.estimate_tokens(messages) + self.settings.llm_max_output_tokens > self.settings.token_budget:
                    flags.append(GuardrailFlag.TOKEN_BUDGET_EXCEEDED)
                    self.context_manager.maybe_summarize_history(
                        ev.user_id,
                        token_budget=self.settings.token_budget,
                        max_output_tokens=self.settings.llm_max_output_tokens,
                    )
                try:
                    planner, usage = self.llm.plan(messages)
                except CircuitOpenError:
                    planner = self.llm.heuristic_plan(ev.sanitized_prompt)
                    flags.append(GuardrailFlag.CIRCUIT_OPEN)
                except LLMTimeoutError:
                    planner = self.llm.heuristic_plan(ev.sanitized_prompt)
                    flags.append(GuardrailFlag.TIMEOUT)
                except MalformedLLMOutputError:
                    planner = self.llm.heuristic_plan(ev.sanitized_prompt)
                    flags.append(GuardrailFlag.MALFORMED_LLM_OUTPUT)
                except LLMUnavailableError:
                    planner = self.llm.heuristic_plan(ev.sanitized_prompt)
                    flags.append(GuardrailFlag.LLM_UNAVAILABLE)
                except Exception:
                    planner = self.llm.heuristic_plan(ev.sanitized_prompt)
                    flags.append(GuardrailFlag.LLM_UNAVAILABLE)
                if planner.guardrail_triggered:
                    flags.extend(planner.guardrail_flags)
                heuristic_planner = self.llm.heuristic_plan(ev.sanitized_prompt)
                if not planner.tool_calls and heuristic_planner.tool_calls:
                    planner.tool_calls = heuristic_planner.tool_calls
                    if not planner.user_intent:
                        planner.user_intent = heuristic_planner.user_intent
                    if not planner.data_focus:
                        planner.data_focus = heuristic_planner.data_focus
                    planner.response_plan = (
                        planner.response_plan
                        or "Use deterministic Pandas summaries and generated tool outputs. Avoid unsupported claims."
                    )
                else:
                    _apply_heuristic_tool_constraints(planner, heuristic_planner, ev.sanitized_prompt)
            return PlannedEvent(
                user_id=ev.user_id,
                prompt=ev.prompt,
                sanitized_prompt=ev.sanitized_prompt,
                start_time=ev.start_time,
                flags=_unique_flags(flags),
                profile=ev.profile,
                cache_hit=ev.cache_hit,
                planner=planner,
                usage=usage,
                resolution=ev.resolution,
            )

    @step
    async def dispatch_tools(self, ev: PlannedEvent) -> ToolsDoneEvent:
        with self.observability.span(
            "tool_dispatch",
            **{
                "user.id": ev.user_id,
                "tool.names": [call.name.value for call in ev.planner.tool_calls],
            },
        ):
            tool_results = self.toolkit.dispatch(ev.user_id, ev.planner.tool_calls)
            return ToolsDoneEvent(
                user_id=ev.user_id,
                prompt=ev.prompt,
                sanitized_prompt=ev.sanitized_prompt,
                start_time=ev.start_time,
                flags=ev.flags,
                profile=ev.profile,
                cache_hit=ev.cache_hit,
                planner=ev.planner,
                tool_results=tool_results,
                usage=ev.usage,
                resolution=ev.resolution,
            )

    @step
    async def compose_response(self, ev: ToolsDoneEvent) -> ResponseReadyEvent:
        flags = list(ev.flags)
        with self.observability.span("final_response", **{"user.id": ev.user_id}):
            if ev.tool_results:
                response = deterministic_response(ev.prompt, ev.tool_results, ev.profile, ev.resolution)
                try:
                    messages = self.context_manager.build_response_messages(
                        sanitized_prompt=ev.sanitized_prompt,
                        profile=ev.profile,
                        tool_results=ev.tool_results,
                        planner_summary=ev.planner.response_plan,
                    )
                    llm_response, response_usage = self.llm.compose_response(messages)
                    candidate_response = llm_response.strip()
                    if _is_useful_llm_response(candidate_response):
                        response = candidate_response
                    ev.usage.update({f"response_{k}": v for k, v in response_usage.items()})
                except CircuitOpenError:
                    flags.append(GuardrailFlag.CIRCUIT_OPEN)
                except LLMTimeoutError:
                    flags.append(GuardrailFlag.TIMEOUT)
                except LLMUnavailableError:
                    flags.append(GuardrailFlag.LLM_UNAVAILABLE)
                except Exception:
                    flags.append(GuardrailFlag.LLM_UNAVAILABLE)
            else:
                response = deterministic_response(ev.prompt, ev.tool_results, ev.profile, ev.resolution)
            return ResponseReadyEvent(
                user_id=ev.user_id,
                prompt=ev.prompt,
                sanitized_prompt=ev.sanitized_prompt,
                start_time=ev.start_time,
                flags=_unique_flags(flags),
                profile=ev.profile,
                cache_hit=ev.cache_hit,
                planner=ev.planner,
                tool_results=ev.tool_results,
                response=response,
                usage=ev.usage,
                resolution=ev.resolution,
            )

    @step
    async def finalize(self, ev: ResponseReadyEvent) -> StopEvent:
        with self.observability.span("guardrail_cache_audit", **{"user.id": ev.user_id}):
            visualizations = [result.path for result in ev.tool_results if result.path]
            data_summary = _merge_data_summary(ev.tool_results, ev.profile)
            allowed_numbers = collect_allowed_numbers(ev.tool_results, ev.profile)
            response, output_flags = self.output_guardrails.validate(
                ev.response,
                data_available=any(result.rows > 0 for result in ev.tool_results) or not ev.tool_results,
                allowed_numbers=allowed_numbers,
            )
            if GuardrailFlag.OUTPUT_UNGROUNDED in output_flags and ev.tool_results:
                fallback = deterministic_response(ev.prompt, ev.tool_results, ev.profile, ev.resolution)
                fallback_response, fallback_flags = self.output_guardrails.validate(
                    fallback,
                    data_available=True,
                    allowed_numbers=allowed_numbers,
                )
                if GuardrailFlag.OUTPUT_UNGROUNDED not in fallback_flags:
                    response = fallback_response
                    output_flags = fallback_flags
            flags = _unique_flags([*ev.flags, *output_flags])
            latency_ms = int((time.perf_counter() - ev.start_time) * 1000)
            if ev.tool_results:
                last = ev.tool_results[-1]
                self.cache.set_viz_state(
                    ev.user_id,
                    VizState(
                        last_chart_type=last.name,
                        axes=_viz_axes(last.name),
                        filters=last.parameters,
                        last_parameters=last.parameters,
                        last_paths=visualizations,
                    ),
                )
            operation = _operation_summary(ev.tool_results)
            item = self.context_manager.sanitized_history_item(
                user_id=ev.user_id,
                prompt=ev.prompt,
                operation=operation,
                response=response,
                visualizations=visualizations,
                latency_ms=latency_ms,
            )
            self.cache.append_query_history(ev.user_id, item)
            audit_payload = {
                "prompt_summary": item.prompt,
                "response_summary": item.result_summary,
                "latency_ms": latency_ms,
                "guardrail_flags": [flag.value for flag in flags],
                "visualization_count": len(visualizations),
                "tool_names": [result.name.value for result in ev.tool_results],
                "usage": ev.usage,
            }
            self.audit_logger.log(user_id=ev.user_id, event="pipeline.run", payload=audit_payload)
            result = PipelineResult(
                user_name=None,
                response=response,
                data_summary=data_summary,
                visualizations=visualizations,
                cache_hit=ev.cache_hit,
                latency_ms=latency_ms,
                guardrail_flags=flags,
            )
            return StopEvent(result=result)

    def _stop_error(
        self,
        *,
        user_id: str | None,
        response: str,
        start_time: float,
        code: ErrorCode,
        flags: list[GuardrailFlag],
        retryable: bool,
        user_name: str | None = None,
    ) -> StopEvent:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        result = PipelineResult(
            user_name=user_name,
            response=response,
            cache_hit=False,
            latency_ms=latency_ms,
            guardrail_flags=_unique_flags(flags),
            error=ErrorDetails(code=code, message=response, retryable=retryable),
        )
        self.audit_logger.log(
            user_id=user_id,
            event="pipeline.error",
            payload={
                "error_code": code.value,
                "response_summary": response,
                "latency_ms": latency_ms,
                "guardrail_flags": [flag.value for flag in _unique_flags(flags)],
            },
        )
        return StopEvent(result=result)


def _unique_flags(flags: list[GuardrailFlag]) -> list[GuardrailFlag]:
    seen: set[GuardrailFlag] = set()
    unique: list[GuardrailFlag] = []
    for flag in flags:
        if flag not in seen:
            seen.add(flag)
            unique.append(flag)
    return unique


def _resolve_follow_up_window_request(
    *,
    prompt: str,
    user_id: str,
    repository: TransactionDataRepository,
    viz_state: VizState,
) -> tuple[PlannerOutput | None, dict[str, Any]]:
    last_tool = viz_state.last_chart_type
    if last_tool is None:
        return None, {}

    lowered = prompt.lower()
    if not _looks_like_window_follow_up(lowered):
        return None, {}

    explicit_tool = _explicit_tool_intent(lowered)
    if explicit_tool is not None and explicit_tool != last_tool:
        return None, {}

    available_months = _available_month_count(repository, user_id)
    previous_months = _parameter_window_months(viz_state.last_parameters, available_months)
    requested = _requested_window(lowered, previous_months=previous_months, available_months=available_months)
    if requested is None:
        return None, {}

    requested_months = max(1, min(24, requested["months"]))
    resolved_months = min(requested_months, available_months)
    period = requested.get("period")
    if period is None and resolved_months == 1 and _mentions_last_month(lowered):
        period = "last_month"
    elif period is None and resolved_months == 1 and _mentions_current_month(lowered):
        period = "current_month"

    args = _follow_up_tool_args(
        tool_name=last_tool,
        last_parameters=viz_state.last_parameters,
        months=resolved_months,
        period=period,
    )
    start_period = repository.anchor_month(user_id) - (resolved_months - 1)
    resolution = {
        "type": "follow_up_window",
        "preserved_tool": last_tool.value,
        "requested_months": requested_months,
        "resolved_months": resolved_months,
        "window_start_month": str(start_period),
        "window_end_month": str(repository.anchor_month(user_id)),
    }
    coverage_note = _coverage_note(
        requested_months=requested_months,
        resolved_months=resolved_months,
        start_period=start_period,
        reason=str(requested.get("reason", "")),
    )
    if coverage_note:
        resolution["coverage_note"] = coverage_note

    planner = PlannerOutput(
        user_intent="follow_up_window_adjustment",
        data_focus="Same authenticated user's filtered transaction DataFrame only.",
        response_plan=(
            "Preserve the prior analysis type and adjust only the requested date window. "
            "Ground the answer in computed tool results and mention any available-history limit."
        ),
        tool_calls=[
            ToolCall(
                name=last_tool,
                arguments=args,
                rationale="Follow-up window request; preserve prior metric unless the user explicitly changes it.",
            )
        ],
        confidence=0.92,
    )
    return planner, resolution


def _is_contextual_financial_follow_up(prompt: str, viz_state: VizState) -> bool:
    return viz_state.last_chart_type is not None and _looks_like_window_follow_up(prompt.lower())


def _looks_like_window_follow_up(lowered: str) -> bool:
    window_markers = [
        "month",
        "months",
        "history",
        "period",
        "window",
        "range",
        "date",
        "earlier",
        "back",
    ]
    follow_up_markers = [
        "what about",
        "how about",
        "same",
        "that",
        "this",
        "those",
        "more",
        "another",
        "additional",
        "extra",
        "further",
        "earlier",
        "back",
        "extend",
        "longer",
        "full",
        "all available",
        "entire",
        "previous",
        "prior",
        "can you",
        "could you",
    ]
    return any(marker in lowered for marker in window_markers) and any(
        marker in lowered for marker in follow_up_markers
    )


def _explicit_tool_intent(lowered: str) -> ToolName | None:
    if any(term in lowered for term in ["saving", "savings", "income vs", "income versus", "income and expense"]):
        return ToolName.INCOME_VS_EXPENSE
    if any(term in lowered for term in ["trend", "changed over time", "over time"]):
        return ToolName.MONTHLY_SPENDING_TREND
    if any(term in lowered for term in ["most", "category", "categories", "breakdown", "money going", "spent the most"]):
        return ToolName.CATEGORY_BREAKDOWN
    return None


def _requested_window(lowered: str, *, previous_months: int, available_months: int) -> dict[str, Any] | None:
    if any(phrase in lowered for phrase in ["full history", "all history", "entire history", "all available"]):
        return {"months": available_months, "reason": "full_history"}
    if _mentions_last_month(lowered):
        return {"months": 1, "period": "last_month", "reason": "last_month"}
    if _mentions_current_month(lowered):
        return {"months": 1, "period": "current_month", "reason": "current_month"}

    extension = _parse_extension_months(lowered)
    if extension is not None:
        return {"months": previous_months + extension, "reason": "extension"}

    explicit_total = _parse_total_window_months(lowered)
    if explicit_total is not None:
        return {"months": explicit_total, "reason": "explicit_window"}

    if any(phrase in lowered for phrase in ["further back", "go back", "more history", "earlier", "extend"]):
        return {"months": available_months, "reason": "extend_to_available_history"}
    return None


def _parse_extension_months(lowered: str) -> int | None:
    number = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    extension_patterns = [
        rf"(?:more|another|additional|extra)\s+{number}\s+months?",
        rf"(?:go|look|extend|back|further|earlier)[^\n.?!]*?{number}\s+months?",
        rf"{number}\s+more\s+months?",
    ]
    for pattern in extension_patterns:
        match = re.search(pattern, lowered)
        if match:
            return _number_text_to_int(match.group(1))
    return None


def _parse_total_window_months(lowered: str) -> int | None:
    number = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    total_patterns = [
        rf"(?:last|past|recent|latest)\s+{number}\s+months?",
        rf"{number}\s+months?",
    ]
    for pattern in total_patterns:
        match = re.search(pattern, lowered)
        if match:
            return _number_text_to_int(match.group(1))
    return None


def _number_text_to_int(value: str) -> int:
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
    }
    return words.get(value, int(value) if value.isdigit() else 1)


def _mentions_last_month(lowered: str) -> bool:
    return "last month" in lowered or "previous month" in lowered or "prior month" in lowered


def _mentions_current_month(lowered: str) -> bool:
    return "current month" in lowered or "this month" in lowered


def _available_month_count(repository: TransactionDataRepository, user_id: str) -> int:
    user_df = repository.get_user_df(user_id)
    start = user_df["transaction_date"].min().to_period("M")
    end = user_df["transaction_date"].max().to_period("M")
    return _months_inclusive(start, end)


def _parameter_window_months(parameters: dict[str, Any], available_months: int) -> int:
    if "months" in parameters:
        try:
            return max(1, int(parameters["months"]))
        except (TypeError, ValueError):
            return 1
    period = str(parameters.get("period") or "").lower()
    if period == "all":
        return available_months
    if period in {"last_month", "current_month"}:
        return 1
    match = re.match(r"last_(\d+)_months", period)
    if match:
        return max(1, int(match.group(1)))
    return 3


def _follow_up_tool_args(
    *,
    tool_name: ToolName,
    last_parameters: dict[str, Any],
    months: int,
    period: str | None,
) -> dict[str, Any]:
    base = {key: value for key, value in last_parameters.items() if key != "user_id" and value is not None}
    if tool_name == ToolName.CATEGORY_BREAKDOWN:
        base.pop("months", None)
        base["period"] = period or f"last_{months}_months"
        base.setdefault("top_n", 7)
    else:
        base.pop("period", None)
        base["months"] = months
        if period:
            base["period"] = period
        if tool_name == ToolName.INCOME_VS_EXPENSE:
            base.setdefault("show_net_line", True)
    return base


def _coverage_note(
    *,
    requested_months: int,
    resolved_months: int,
    start_period: pd.Period,
    reason: str,
) -> str:
    if requested_months > resolved_months:
        return f"I could only extend back to {start_period} because that is the first month in the available data."
    if reason in {"full_history", "extend_to_available_history"}:
        return f"I extended this to the full available history starting {start_period}."
    return ""


def _months_inclusive(start: pd.Period, end: pd.Period) -> int:
    return (end.year - start.year) * 12 + end.month - start.month + 1


def _merge_data_summary(tool_results: list[ToolResult], profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile": profile,
        "tools": [result.model_dump(mode="json") for result in tool_results],
    }


def _operation_summary(tool_results: list[ToolResult]) -> str:
    if not tool_results:
        return "No visualization tool selected; profile-level summary only."
    return "; ".join(
        f"{result.name.value}({', '.join(f'{k}={v}' for k, v in result.parameters.items() if k != 'user_id')})"
        for result in tool_results
    )


def _viz_axes(tool_name: Any) -> dict[str, str]:
    name = getattr(tool_name, "value", str(tool_name))
    if name == "plot_monthly_spending_trend":
        return {"x": "month", "y": "expense"}
    if name == "plot_income_vs_expense":
        return {"x": "month", "y": "income/expense/net_savings"}
    if name == "plot_category_breakdown":
        return {"label": "category", "value": "expense"}
    return {}


def _apply_heuristic_tool_constraints(planner: PlannerOutput, heuristic_planner: PlannerOutput, prompt: str) -> None:
    if not planner.tool_calls or not heuristic_planner.tool_calls:
        return
    lowered = prompt.lower()
    explicit_window = any(
        phrase in lowered
        for phrase in [
            "last month",
            "this month",
            "current month",
            "last 3 months",
            "last three months",
            "last 6 months",
            "last six months",
        ]
    )
    if not explicit_window:
        return
    heuristic_by_name = {call.name: call for call in heuristic_planner.tool_calls}
    for call in planner.tool_calls:
        heuristic_call = heuristic_by_name.get(call.name)
        if not heuristic_call:
            continue
        for key in ("period", "months", "category_filter"):
            if key in heuristic_call.arguments:
                call.arguments[key] = heuristic_call.arguments[key]


def _is_useful_llm_response(response: str) -> bool:
    lowered = response.lower().strip()
    if len(lowered) < 40:
        return False
    if lowered in {"safe", "user safety: safe", "safety: safe"}:
        return False
    financial_terms = ["$", "spend", "spent", "income", "expense", "saving", "category", "month"]
    return any(term in lowered for term in financial_terms)
