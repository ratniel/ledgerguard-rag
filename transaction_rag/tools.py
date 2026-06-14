from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic import BaseModel, Field

from .data import TransactionDataRepository
from .models import ToolCall, ToolName, ToolResult


class MonthlySpendingArgs(BaseModel):
    user_id: str
    months: int = Field(default=1, ge=1, le=24)
    period: str | None = None
    category_filter: str | None = None


class CategoryBreakdownArgs(BaseModel):
    user_id: str
    period: str = "last_3_months"
    top_n: int = Field(default=7, ge=1, le=12)


class IncomeExpenseArgs(BaseModel):
    user_id: str
    months: int = Field(default=6, ge=1, le=24)
    period: str | None = None
    show_net_line: bool = True


class VisualizationToolkit:
    def __init__(self, repository: TransactionDataRepository, output_dir: Path):
        self.repository = repository
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def dispatch(self, user_id: str, tool_calls: list[ToolCall]) -> list[ToolResult]:
        results: list[ToolResult] = []
        for call in tool_calls:
            args = dict(call.arguments)
            args["user_id"] = user_id
            if call.name == ToolName.MONTHLY_SPENDING_TREND:
                validated = MonthlySpendingArgs.model_validate(args)
                results.append(self.plot_monthly_spending_trend(**validated.model_dump()))
            elif call.name == ToolName.CATEGORY_BREAKDOWN:
                validated = CategoryBreakdownArgs.model_validate(args)
                results.append(self.plot_category_breakdown(**validated.model_dump()))
            elif call.name == ToolName.INCOME_VS_EXPENSE:
                validated = IncomeExpenseArgs.model_validate(args)
                results.append(self.plot_income_vs_expense(**validated.model_dump()))
        return results

    def tool_schemas(self, *, include_user_id: bool = False) -> list[dict[str, Any]]:
        return [
            self._tool_schema(
                ToolName.MONTHLY_SPENDING_TREND,
                "Line chart of monthly expense totals with a rolling average overlay.",
                MonthlySpendingArgs,
                include_user_id=include_user_id,
            ),
            self._tool_schema(
                ToolName.CATEGORY_BREAKDOWN,
                "Donut chart showing top spending categories for a selected period.",
                CategoryBreakdownArgs,
                include_user_id=include_user_id,
            ),
            self._tool_schema(
                ToolName.INCOME_VS_EXPENSE,
                "Grouped bars comparing income and expenses with optional net savings line.",
                IncomeExpenseArgs,
                include_user_id=include_user_id,
            ),
        ]

    @staticmethod
    def _tool_schema(name: ToolName, description: str, model: type[BaseModel], *, include_user_id: bool) -> dict[str, Any]:
        schema = model.model_json_schema()
        if not include_user_id:
            schema.get("properties", {}).pop("user_id", None)
            required = schema.get("required")
            if isinstance(required, list):
                schema["required"] = [item for item in required if item != "user_id"]
        return {
            "type": "function",
            "function": {
                "name": name.value,
                "description": description,
                "parameters": schema,
            },
        }

    def plot_monthly_spending_trend(
        self,
        user_id: str,
        months: int = 1,
        period: str | None = None,
        category_filter: str | None = None,
    ) -> ToolResult:
        df, window = self.repository.filter_user_transactions(
            user_id, period=period, months=months, category_filter=category_filter
        )
        expenses = df.loc[df["transaction_amount"] > 0].copy()
        params = {"user_id": user_id, "months": months, "category_filter": category_filter}
        if period:
            params["period"] = period
        if expenses.empty:
            return ToolResult(
                name=ToolName.MONTHLY_SPENDING_TREND,
                rows=0,
                parameters=params,
                data_summary={**window, "monthly_totals": []},
                message="No spending transactions found for the requested window.",
            )
        monthly = (
            expenses.assign(month=expenses["transaction_date"].dt.to_period("M").dt.to_timestamp())
            .groupby("month")["transaction_amount"]
            .sum()
            .sort_index()
        )
        rolling = monthly.rolling(window=min(3, len(monthly)), min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        ax.plot(monthly.index, monthly.values, marker="o", linewidth=2.5, label="Monthly spend", color="#2f80a3")
        ax.plot(rolling.index, rolling.values, linestyle="--", linewidth=2, label="Rolling average", color="#d99021")
        ax.set_title(f"Monthly Spending Trend\n{window['window']}", fontsize=15, pad=12)
        ax.set_xlabel("Month")
        ax.set_ylabel("Spend ($)")
        ax.margins(x=0.08, y=0.12)
        ax.legend(loc="best", framealpha=0.95)
        ax.yaxis.set_major_formatter(lambda x, _: f"${x:,.0f}")
        ax.annotate(
            f"${monthly.iloc[-1]:,.0f}",
            xy=(monthly.index[-1], monthly.iloc[-1]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            color="#1f5f78",
        )
        fig.autofmt_xdate()
        path = self._save_fig(fig, ToolName.MONTHLY_SPENDING_TREND, params)
        monthly_totals = [
            {"month": idx.strftime("%Y-%m"), "amount": round(float(amount), 2)}
            for idx, amount in monthly.items()
        ]
        spend_delta = round(float(monthly.iloc[-1] - monthly.iloc[0]), 2)
        return ToolResult(
            name=ToolName.MONTHLY_SPENDING_TREND,
            path=str(path),
            rows=int(len(expenses)),
            parameters=params,
            data_summary={
                **window,
                "monthly_totals": monthly_totals,
                "latest_month_spend": monthly_totals[-1]["amount"],
                "average_monthly_spend": round(float(monthly.mean()), 2),
                "spend_delta_from_first_month": spend_delta,
            },
            message="Monthly spending trend generated.",
        )

    def plot_category_breakdown(
        self,
        user_id: str,
        period: str = "last_3_months",
        top_n: int = 7,
    ) -> ToolResult:
        df, window = self.repository.filter_user_transactions(user_id, period=period)
        expenses = df.loc[df["transaction_amount"] > 0].copy()
        params = {"user_id": user_id, "period": period, "top_n": top_n}
        if expenses.empty:
            return ToolResult(
                name=ToolName.CATEGORY_BREAKDOWN,
                rows=0,
                parameters=params,
                data_summary={**window, "categories": [], "total_spend": 0.0},
                message="No spending transactions found for the requested period.",
            )
        categories = (
            expenses.groupby("transaction_category_detail")["transaction_amount"]
            .sum()
            .sort_values(ascending=False)
        )
        top = categories.head(top_n)
        other = categories.iloc[top_n:].sum()
        if other > 0:
            top = pd.concat([top, pd.Series({"OTHER": other})])
        total = float(categories.sum())
        fig, ax = plt.subplots(figsize=(9.2, 5.4))
        colors = sns.color_palette("Set2", n_colors=len(top))
        wedges, _ = ax.pie(
            top.values,
            startangle=90,
            colors=colors,
            wedgeprops={"width": 0.42, "edgecolor": "white"},
        )
        ax.text(0, 0, f"${total:,.0f}\nspent", ha="center", va="center", fontsize=13, weight="bold")
        ax.set_title(f"Category Breakdown\n{window['window']}", fontsize=15, pad=12)
        legend_labels = [
            f"{str(category).replace('_', ' ').title()} - ${float(amount):,.0f} ({float(amount / total) * 100:.0f}%)"
            for category, amount in top.items()
        ]
        ax.legend(
            wedges,
            legend_labels,
            title="Categories",
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            fontsize=9,
            title_fontsize=10,
        )
        path = self._save_fig(fig, ToolName.CATEGORY_BREAKDOWN, params)
        category_rows = [
            {
                "category": str(category),
                "amount": round(float(amount), 2),
                "share": round(float(amount / total), 4) if total else 0.0,
            }
            for category, amount in categories.items()
        ]
        return ToolResult(
            name=ToolName.CATEGORY_BREAKDOWN,
            path=str(path),
            rows=int(len(expenses)),
            parameters=params,
            data_summary={
                **window,
                "total_spend": round(total, 2),
                "top_category": category_rows[0] if category_rows else None,
                "categories": category_rows,
            },
            message="Category breakdown generated.",
        )

    def plot_income_vs_expense(
        self,
        user_id: str,
        months: int = 6,
        period: str | None = None,
        show_net_line: bool = True,
    ) -> ToolResult:
        df, window = self.repository.filter_user_transactions(user_id, period=period, months=months)
        params = {"user_id": user_id, "months": months, "show_net_line": show_net_line}
        if period:
            params["period"] = period
        if df.empty:
            return ToolResult(
                name=ToolName.INCOME_VS_EXPENSE,
                rows=0,
                parameters=params,
                data_summary={**window, "monthly": []},
                message="No transactions found for the requested window.",
            )
        working = df.assign(
            month=df["transaction_date"].dt.to_period("M").dt.to_timestamp(),
            expense=df["transaction_amount"].clip(lower=0),
            income=-df["transaction_amount"].clip(upper=0),
        )
        monthly = (
            working.groupby("month")
            .agg(expense=("expense", "sum"), income=("income", "sum"))
            .sort_index()
        )
        monthly["net_savings"] = monthly["income"] - monthly["expense"]
        total_net_savings = round(float(monthly["net_savings"].sum()), 2)
        x = range(len(monthly))
        fig, ax = plt.subplots(figsize=(9.2, 5.0))
        width = 0.36
        ax.bar([i - width / 2 for i in x], monthly["income"], width=width, label="Income", color="#24855b")
        ax.bar([i + width / 2 for i in x], monthly["expense"], width=width, label="Expense", color="#c84c4c")
        if show_net_line:
            ax2 = ax.twinx()
            ax2.plot(list(x), monthly["net_savings"], marker="o", linewidth=2, color="#2f80a3", label="Net savings")
            ax2.axhline(0, color="#555555", linewidth=0.8, alpha=0.5)
            ax2.set_ylabel("Net savings")
            ax2.yaxis.set_major_formatter(lambda y, _: f"${y:,.0f}")
        ax.set_xticks(list(x))
        ax.set_xticklabels([idx.strftime("%Y-%m") for idx in monthly.index], rotation=30, ha="right")
        ax.set_title(f"Income vs Expense\n{window['window']}", fontsize=15, pad=12)
        ax.set_ylabel("Amount ($)")
        ax.yaxis.set_major_formatter(lambda y, _: f"${y:,.0f}")
        handles, labels = ax.get_legend_handles_labels()
        if show_net_line:
            line_handles, line_labels = ax2.get_legend_handles_labels()
            handles += line_handles
            labels += line_labels
        ax.margins(x=0.05, y=0.12)
        ax.legend(handles, labels, loc="upper left", framealpha=0.95)
        path = self._save_fig(fig, ToolName.INCOME_VS_EXPENSE, params)
        monthly_rows = [
            {
                "month": idx.strftime("%Y-%m"),
                "income": round(float(row["income"]), 2),
                "expense": round(float(row["expense"]), 2),
                "net_savings": round(float(row["net_savings"]), 2),
            }
            for idx, row in monthly.iterrows()
        ]
        return ToolResult(
            name=ToolName.INCOME_VS_EXPENSE,
            path=str(path),
            rows=int(len(df)),
            parameters=params,
            data_summary={
                **window,
                "monthly": monthly_rows,
                "total_income": round(float(monthly["income"].sum()), 2),
                "total_expense": round(float(monthly["expense"].sum()), 2),
                "total_net_savings": total_net_savings,
                "absolute_total_net_savings": abs(total_net_savings),
            },
            message="Income vs expense chart generated.",
        )

    def _save_fig(self, fig: plt.Figure, tool_name: ToolName, params: dict[str, Any]) -> Path:
        safe_params = {k: v for k, v in params.items() if k != "user_id"}
        digest = hashlib.md5(json.dumps(safe_params, sort_keys=True, default=str).encode()).hexdigest()[:8]
        user_id = str(params.get("user_id", "user")).replace("/", "_")
        path = self.output_dir / f"{user_id}_{tool_name.value}_{digest}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return path
