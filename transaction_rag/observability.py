from __future__ import annotations

import json
import socket
from contextlib import contextmanager
from typing import Any, Iterator
from urllib.parse import urlparse

from opentelemetry import trace

from .config import Settings


class Observability:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.tracer = trace.get_tracer("transaction_rag")
        self.enabled = False
        if settings.tracing_enabled and self._collector_reachable(settings.phoenix_collector_endpoint):
            try:
                from phoenix.otel import register

                register(
                    endpoint=settings.phoenix_collector_endpoint,
                    project_name=settings.phoenix_project_name,
                    protocol="http/protobuf",
                    batch=False,
                    verbose=False,
                    auto_instrument=False,
                )
                self.tracer = trace.get_tracer("transaction_rag")
                self.enabled = True
            except Exception:
                self.enabled = False

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Any]:
        if not self.enabled:
            yield None
            return
        with self.tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                self.set_attribute(span, key, value)
            yield span

    @staticmethod
    def set_attribute(span: Any, key: str, value: Any) -> None:
        if span is None:
            return
        if isinstance(value, (str, bool, int, float)) or value is None:
            span.set_attribute(key, "" if value is None else value)
        else:
            span.set_attribute(key, json.dumps(value, ensure_ascii=True, default=str))

    @staticmethod
    def _collector_reachable(endpoint: str) -> bool:
        parsed = urlparse(endpoint)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            return False
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return True
        except OSError:
            return False


def current_trace_id() -> str | None:
    span = trace.get_current_span()
    context = span.get_span_context()
    if not context or not context.is_valid:
        return None
    return f"{context.trace_id:032x}"
