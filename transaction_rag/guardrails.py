from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Settings
from .models import GuardrailFlag
from .privacy import compact_text, redact_pii


FINANCE_KEYWORDS = {
    "spend",
    "spending",
    "spent",
    "expense",
    "expenses",
    "income",
    "salary",
    "saving",
    "savings",
    "budget",
    "transaction",
    "transactions",
    "merchant",
    "category",
    "categories",
    "money",
    "financial",
    "finance",
    "rent",
    "food",
    "travel",
    "cashback",
    "refund",
    "trend",
    "month",
    "months",
    "last month",
    "report",
}

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior) instructions",
    r"reveal (the )?(system|developer) prompt",
    r"show (the )?(system|developer) prompt",
    r"act as (a )?system",
    r"you are now",
    r"override (your|the) instructions",
    r"jailbreak",
    r"developer mode",
]

TOXIC_TERMS = {"idiot", "stupid", "hate speech"}


@dataclass
class InputGuardrailDecision:
    prompt: str
    sanitized_prompt: str
    flags: list[GuardrailFlag] = field(default_factory=list)
    blocked: bool = False
    response: str | None = None


class InputGuardrails:
    def __init__(self, settings: Settings):
        self.settings = settings

    def validate(
        self,
        *,
        user_id: str,
        prompt: str,
        valid_user_ids: list[str],
        user_names: dict[str, str],
    ) -> InputGuardrailDecision:
        flags: list[GuardrailFlag] = []
        original_prompt = prompt.strip()
        working_prompt = original_prompt
        if len(working_prompt) > self.settings.max_prompt_chars:
            working_prompt = working_prompt[: self.settings.max_prompt_chars].rstrip()
            flags.append(GuardrailFlag.INPUT_LENGTH_EXCEEDED)

        security_lowered = original_prompt.lower()
        lowered = working_prompt.lower()
        if any(re.search(pattern, security_lowered) for pattern in INJECTION_PATTERNS):
            return InputGuardrailDecision(
                prompt=working_prompt,
                sanitized_prompt=redact_pii(working_prompt, user_names=user_names.values(), user_ids=valid_user_ids),
                flags=flags + [GuardrailFlag.PROMPT_INJECTION],
                blocked=True,
                response="I can help with financial transaction analysis, but I cannot reveal or override system instructions.",
            )

        mentioned_ids = {m.lower() for m in re.findall(r"\busr_[a-z0-9]+\b", original_prompt, flags=re.IGNORECASE)}
        generic_users = {m.lower() for m in re.findall(r"\buser[_-][a-z0-9_]+\b", original_prompt, flags=re.IGNORECASE)}
        if any(uid != user_id.lower() for uid in mentioned_ids) or generic_users:
            return InputGuardrailDecision(
                prompt=working_prompt,
                sanitized_prompt=redact_pii(working_prompt, user_names=user_names.values(), user_ids=valid_user_ids),
                flags=flags + [GuardrailFlag.CROSS_USER_LEAKAGE],
                blocked=True,
                response="I can only analyze the authenticated user's own transactions.",
            )

        current_name = user_names.get(user_id, "")
        other_names = [name for uid, name in user_names.items() if uid != user_id]
        if any(name and re.search(re.escape(name), original_prompt, re.IGNORECASE) for name in other_names):
            return InputGuardrailDecision(
                prompt=working_prompt,
                sanitized_prompt=redact_pii(working_prompt, user_names=user_names.values(), user_ids=valid_user_ids),
                flags=flags + [GuardrailFlag.CROSS_USER_LEAKAGE],
                blocked=True,
                response="I can only analyze the authenticated user's own transactions.",
            )
        other_name_parts = _unique_other_name_parts(user_id=user_id, user_names=user_names)
        if any(re.search(rf"\b{re.escape(part)}\b", original_prompt, re.IGNORECASE) for part in other_name_parts):
            return InputGuardrailDecision(
                prompt=working_prompt,
                sanitized_prompt=redact_pii(working_prompt, user_names=user_names.values(), user_ids=valid_user_ids),
                flags=flags + [GuardrailFlag.CROSS_USER_LEAKAGE],
                blocked=True,
                response="I can only analyze the authenticated user's own transactions.",
            )

        if not self._is_financial_prompt(lowered):
            return InputGuardrailDecision(
                prompt=working_prompt,
                sanitized_prompt=redact_pii(working_prompt, user_names=user_names.values(), user_ids=valid_user_ids),
                flags=flags + [GuardrailFlag.OFF_TOPIC],
                blocked=True,
                response="I can help with spending, income, savings, and transaction questions. Please ask about your financial data.",
            )

        sanitized = redact_pii(
            working_prompt,
            user_names=[current_name, *other_names],
            user_ids=valid_user_ids,
        )
        return InputGuardrailDecision(prompt=working_prompt, sanitized_prompt=sanitized, flags=flags)

    @staticmethod
    def _is_financial_prompt(lowered: str) -> bool:
        if len(lowered.split()) <= 2 and any(word in lowered for word in {"hi", "hello", "hey"}):
            return True
        return any(keyword in lowered for keyword in FINANCE_KEYWORDS)


def _unique_other_name_parts(*, user_id: str, user_names: dict[str, str]) -> set[str]:
    current_parts = _name_parts(user_names.get(user_id, ""))
    part_to_user_ids: dict[str, set[str]] = {}
    for candidate_user_id, name in user_names.items():
        for part in _name_parts(name):
            part_to_user_ids.setdefault(part, set()).add(candidate_user_id)
    return {
        part
        for part, owner_ids in part_to_user_ids.items()
        if user_id not in owner_ids and len(owner_ids) == 1 and part not in current_parts
    }


def _name_parts(name: str) -> set[str]:
    return {
        part.lower()
        for part in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", name)
    }


class OutputGuardrails:
    def validate(self, response: str, *, data_available: bool, allowed_numbers: set[str]) -> tuple[str, list[GuardrailFlag]]:
        flags: list[GuardrailFlag] = []
        lowered = response.lower()
        if any(term in lowered for term in TOXIC_TERMS):
            return (
                "I generated an inappropriate response, so I am returning a safe fallback. Please rephrase your financial question.",
                [GuardrailFlag.TOXIC_OUTPUT],
            )
        if not data_available:
            flags.append(GuardrailFlag.LOW_CONFIDENCE)
        if any(marker in lowered for marker in ["not sure", "cannot determine", "insufficient data"]):
            flags.append(GuardrailFlag.LOW_CONFIDENCE)

        unsupported = []
        amount_pattern = r"(?<![\w.])\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\d,])"
        for match in re.findall(amount_pattern, response):
            normalized = match.replace("$", "").replace(",", "")
            if normalized not in allowed_numbers and not normalized.startswith("2025"):
                unsupported.append(match)
        if unsupported and allowed_numbers:
            flags.append(GuardrailFlag.OUTPUT_UNGROUNDED)
            response = compact_text(response, 1200) + "\n\nNote: I only rely on the computed transaction summaries returned by the tools."
        return response, flags
