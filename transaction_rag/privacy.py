from __future__ import annotations

import re
from collections.abc import Iterable


USER_ID_RE = re.compile(r"\busr_[a-z0-9]+\b", re.IGNORECASE)
GENERIC_USER_RE = re.compile(r"\buser[_-][a-z0-9_]+\b", re.IGNORECASE)


def redact_pii(text: str, *, user_names: Iterable[str] = (), user_ids: Iterable[str] = ()) -> str:
    redacted = text
    redacted = USER_ID_RE.sub("[USER_ID]", redacted)
    redacted = GENERIC_USER_RE.sub("[USER_ID]", redacted)
    for name in sorted({n for n in user_names if n}, key=len, reverse=True):
        redacted = re.sub(re.escape(name), "[USER_NAME]", redacted, flags=re.IGNORECASE)
    for user_id in sorted({u for u in user_ids if u}, key=len, reverse=True):
        redacted = re.sub(re.escape(user_id), "[USER_ID]", redacted, flags=re.IGNORECASE)
    return redacted


def compact_text(text: str, limit: int = 500) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."
