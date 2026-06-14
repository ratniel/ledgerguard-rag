from __future__ import annotations

import math
from typing import Any
import re

from .models import ToolName, ToolResult


def deterministic_response(
    prompt: str,
    tool_results: list[ToolResult],
    profile: dict[str, Any],
    resolution: dict[str, Any] | None = None,
) -> str:
    if not tool_results:
        return (
            "I can help analyze your spending, income, savings, and transaction categories. "
            f"I have data from {profile['date_range']['start']} to {profile['date_range']['end']}."
        )
    parts: list[str] = []
    for result in tool_results:
        if result.rows == 0:
            parts.append(result.message or "I could not find matching transactions for that request.")
            continue
        summary = result.data_summary
        if result.name == ToolName.CATEGORY_BREAKDOWN:
            top = summary.get("top_category") or {}
            total = summary.get("total_spend", 0)
            if top:
                category = str(top.get("category", "")).replace("_", " ").title()
                amount = float(top.get("amount", 0))
                share = float(top.get("share", 0)) * 100
                parts.append(
                    f"For {summary.get('window')}, your largest spending category was {category} at ${amount:,.0f} "
                    f"({share:.0f}% of ${float(total):,.0f} total spend)."
                )
        elif result.name == ToolName.MONTHLY_SPENDING_TREND:
            totals = summary.get("monthly_totals") or []
            if totals:
                first = totals[0]
                last = totals[-1]
                delta = float(last["amount"]) - float(first["amount"])
                direction = "up" if delta > 0 else "down" if delta < 0 else "flat"
                parts.append(
                    f"Your spending trend for {summary.get('window')} ended at ${float(last['amount']):,.0f}, "
                    f"{direction} ${abs(delta):,.0f} from the first month in the window."
                )
        elif result.name == ToolName.INCOME_VS_EXPENSE:
            income = float(summary.get("total_income", 0))
            expense = float(summary.get("total_expense", 0))
            net = float(summary.get("total_net_savings", 0))
            verdict = "saved money" if net >= 0 else "spent more than you brought in"
            parts.append(
                f"Across {summary.get('window')}, you had ${income:,.0f} of income and ${expense:,.0f} of expenses, "
                f"so you {verdict} by ${abs(net):,.0f}."
            )
    response = " ".join(parts) if parts else "I could not find enough matching transaction data to answer that."
    note = str((resolution or {}).get("coverage_note") or "").strip()
    return f"{note} {response}" if note else response


def collect_allowed_numbers(tool_results: list[ToolResult], profile: dict[str, Any]) -> set[str]:
    numbers: set[str] = set()

    def add_numeric(numeric: float) -> None:
        for value in {numeric, abs(numeric)}:
            numbers.add(str(int(value)))
            numbers.add(str(math.floor(value)))
            numbers.add(str(math.ceil(value)))
            numbers.add(str(int(round(value))))
            numbers.add(str(round(value, 2)))
            numbers.add(f"{value:.2f}")
            if 0 < value <= 1:
                numbers.add(str(int(round(value * 100))))
                numbers.add(str(round(value * 100, 1)))
                numbers.add(f"{value * 100:.2f}")

    def add_difference(value: dict[str, Any], left: str, right: str) -> None:
        try:
            difference = float(value[left]) - float(value[right])
        except (KeyError, TypeError, ValueError):
            return
        add_numeric(difference)
        add_numeric(abs(difference))

    def walk(value: Any) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, int | float):
            add_numeric(float(value))
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
            add_difference(value, "total_income", "total_spend")
            add_difference(value, "total_income", "total_expense")
            add_difference(value, "income", "expense")
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, str):
            for numeric_text in re.findall(r"\b\d+\b", value):
                numbers.add(str(int(numeric_text)))
                numbers.add(numeric_text)

    walk(profile)
    for result in tool_results:
        walk(result.parameters)
        walk(result.data_summary)
    return numbers
