from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .errors import InvalidUserError


REQUIRED_COLUMNS = {
    "user_id",
    "user_name",
    "transaction_date",
    "transaction_amount",
    "transaction_category_detail",
    "merchant_name",
}


class TransactionDataRepository:
    def __init__(self, df: pd.DataFrame):
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")
        self.df = df.copy()
        self.df["transaction_date"] = pd.to_datetime(self.df["transaction_date"], errors="coerce")
        if self.df["transaction_date"].isna().any():
            raise ValueError("transaction_date contains unparseable values")
        self.df["transaction_amount"] = self.df["transaction_amount"].astype(float)
        self.df = self.df.sort_values(["user_id", "transaction_date", "merchant_name"]).reset_index(drop=True)
        self._user_names = (
            self.df.groupby("user_id")["user_name"].first().astype(str).to_dict()
        )

    @classmethod
    def from_csv(cls, path: str | Path) -> "TransactionDataRepository":
        return cls(pd.read_csv(path))

    @property
    def user_ids(self) -> list[str]:
        return sorted(self._user_names)

    @property
    def user_names(self) -> dict[str, str]:
        return dict(self._user_names)

    def validate_user_id(self, user_id: str) -> None:
        if user_id not in self._user_names:
            raise InvalidUserError(f"Unknown user_id: {user_id}")

    def get_user_name(self, user_id: str) -> str:
        self.validate_user_id(user_id)
        return self._user_names[user_id]

    def get_user_df(self, user_id: str) -> pd.DataFrame:
        self.validate_user_id(user_id)
        return self.df.loc[self.df["user_id"] == user_id].copy()

    def anchor_month(self, user_id: str) -> pd.Period:
        user_df = self.get_user_df(user_id)
        return user_df["transaction_date"].max().to_period("M")

    def compute_user_profile(self, user_id: str) -> dict[str, Any]:
        user_df = self.get_user_df(user_id)
        expenses = user_df.loc[user_df["transaction_amount"] > 0].copy()
        income = user_df.loc[user_df["transaction_amount"] < 0].copy()
        monthly_expense = (
            expenses.assign(month=expenses["transaction_date"].dt.to_period("M").astype(str))
            .groupby("month")["transaction_amount"]
            .sum()
            .sort_index()
        )
        top_categories = (
            expenses.groupby("transaction_category_detail")["transaction_amount"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        return {
            "user_id": user_id,
            "date_range": {
                "start": user_df["transaction_date"].min().date().isoformat(),
                "end": user_df["transaction_date"].max().date().isoformat(),
            },
            "transaction_count": int(len(user_df)),
            "category_count": int(user_df["transaction_category_detail"].nunique()),
            "avg_monthly_spend": round(float(monthly_expense.mean() if not monthly_expense.empty else 0.0), 2),
            "total_spend": round(float(expenses["transaction_amount"].sum()), 2),
            "total_income": round(float(-income["transaction_amount"].sum()), 2),
            "top_categories": [
                {"category": str(category), "amount": round(float(amount), 2)}
                for category, amount in top_categories.items()
            ],
            "latest_month": str(self.anchor_month(user_id)),
        }

    def period_window(self, user_id: str, *, period: str | None = None, months: int | None = None) -> tuple[pd.Timestamp, pd.Timestamp, str]:
        anchor = self.anchor_month(user_id)
        normalized = (period or "").lower().strip()
        if normalized == "last_month":
            start_period = anchor - 1
            end_period = anchor - 1
        elif normalized == "current_month":
            start_period = anchor
            end_period = anchor
        elif normalized.startswith("last_") and normalized.endswith("_months"):
            try:
                count = int(normalized.removeprefix("last_").removesuffix("_months"))
            except ValueError:
                count = 3
            count = max(1, count)
            end_period = anchor
            start_period = anchor - (count - 1)
        elif normalized == "all":
            user_df = self.get_user_df(user_id)
            return (
                user_df["transaction_date"].min().normalize(),
                user_df["transaction_date"].max().normalize(),
                "all available transactions",
            )
        else:
            count = max(1, int(months or 1))
            end_period = anchor
            start_period = anchor - (count - 1)
        start = start_period.to_timestamp(how="start")
        end = end_period.to_timestamp(how="end").normalize()
        return start, end, f"{start_period} to {end_period}"

    def filter_user_transactions(
        self,
        user_id: str,
        *,
        period: str | None = None,
        months: int | None = None,
        category_filter: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        user_df = self.get_user_df(user_id)
        start, end, label = self.period_window(user_id, period=period, months=months)
        mask = (user_df["transaction_date"] >= start) & (user_df["transaction_date"] <= end)
        filtered = user_df.loc[mask].copy()
        if category_filter:
            needle = category_filter.lower().replace(" ", "_")
            cat = filtered["transaction_category_detail"].str.lower()
            filtered = filtered.loc[cat.str.contains(needle, regex=False) | cat.str.endswith(f"_{needle}")]
        summary = {
            "window": label,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "category_filter": category_filter,
            "rows": int(len(filtered)),
        }
        return filtered, summary

    @staticmethod
    def summarize_frame(df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            return {"rows": 0, "expense": 0.0, "income": 0.0, "net_savings": 0.0}
        expense = float(df.loc[df["transaction_amount"] > 0, "transaction_amount"].sum())
        income = float(-df.loc[df["transaction_amount"] < 0, "transaction_amount"].sum())
        return {
            "rows": int(len(df)),
            "expense": round(expense, 2),
            "income": round(income, 2),
            "net_savings": round(income - expense, 2),
        }
