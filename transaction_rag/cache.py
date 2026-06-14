from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .models import QueryHistoryItem, VizState


class Base(DeclarativeBase):
    pass


class KVCacheRow(Base):
    __tablename__ = "kv_cache"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


class AuditLogRow(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[str | None] = mapped_column(String, nullable=True)
    event: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[str] = mapped_column(Text, nullable=False)


class SQLiteKVStore:
    def __init__(self, path: Path, *, max_history_items: int = 6):
        self.path = path
        self.max_history_items = max_history_items
        self.path.parent.mkdir(parents=True, exist_ok=True) if self.path.parent != Path(".") else None
        self._lock = threading.Lock()
        self.engine = create_engine(f"sqlite:///{self.path}", future=True)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False, future=True)
        self._init_db()

    def _init_db(self) -> None:
        Base.metadata.create_all(self.engine)

    def get_json(self, key: str, default: Any = None) -> Any:
        with self._lock, self.session_factory() as session:
            row = session.get(KVCacheRow, key)
        if row is None:
            return default
        return json.loads(row.value)

    def set_json(self, key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=True, default=str)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self._lock, self.session_factory() as session:
            row = session.get(KVCacheRow, key)
            if row is None:
                session.add(KVCacheRow(key=key, value=payload, updated_at=now))
            else:
                row.value = payload
                row.updated_at = now
            session.commit()

    def profile_key(self, user_id: str) -> str:
        return f"user:{user_id}:profile"

    def history_key(self, user_id: str) -> str:
        return f"user:{user_id}:query_history"

    def viz_key(self, user_id: str) -> str:
        return f"user:{user_id}:viz_state"

    def summary_key(self, user_id: str) -> str:
        return f"user:{user_id}:chat_summary"

    def get_profile(self, user_id: str) -> dict[str, Any] | None:
        return self.get_json(self.profile_key(user_id))

    def set_profile(self, user_id: str, profile: dict[str, Any]) -> None:
        self.set_json(self.profile_key(user_id), profile)

    def get_query_history(self, user_id: str) -> list[QueryHistoryItem]:
        items = self.get_json(self.history_key(user_id), default=[]) or []
        return [QueryHistoryItem.model_validate(item) for item in items]

    def append_query_history(self, user_id: str, item: QueryHistoryItem) -> None:
        items = self.get_query_history(user_id)
        items.append(item)
        items = items[-self.max_history_items :]
        self.set_json(self.history_key(user_id), [i.model_dump(mode="json") for i in items])

    def get_viz_state(self, user_id: str) -> VizState:
        payload = self.get_json(self.viz_key(user_id), default={}) or {}
        return VizState.model_validate(payload)

    def set_viz_state(self, user_id: str, state: VizState) -> None:
        self.set_json(self.viz_key(user_id), state.model_dump(mode="json"))

    def get_chat_summary(self, user_id: str) -> str:
        return str(self.get_json(self.summary_key(user_id), default="") or "")

    def set_chat_summary(self, user_id: str, summary: str) -> None:
        self.set_json(self.summary_key(user_id), summary)

    def insert_audit(self, user_id: str | None, event: str, payload: dict[str, Any]) -> None:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self._lock, self.session_factory() as session:
            session.add(
                AuditLogRow(
                    created_at=now,
                    user_id=user_id,
                    event=event,
                    payload=json.dumps(payload, ensure_ascii=True, default=str),
                )
            )
            session.commit()

    def list_audit_events(self, user_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock, self.session_factory() as session:
            statement = select(AuditLogRow).order_by(AuditLogRow.id.desc()).limit(limit)
            if user_id is not None:
                statement = statement.where(AuditLogRow.user_id == user_id)
            rows = session.scalars(statement).all()
        return [
            {
                "id": row.id,
                "created_at": row.created_at,
                "user_id": row.user_id,
                "event": row.event,
                "payload": json.loads(row.payload),
            }
            for row in rows
        ]


class AuditLogger:
    def __init__(self, path: Path, kv_store: SQLiteKVStore):
        self.path = path
        self.kv_store = kv_store
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, *, user_id: str | None, event: str, payload: dict[str, Any]) -> None:
        record = {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "user_id": user_id,
            "event": event,
            "payload": payload,
        }
        line = json.dumps(record, ensure_ascii=True, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        self.kv_store.insert_audit(user_id, event, payload)
