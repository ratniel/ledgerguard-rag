from __future__ import annotations


class TransactionRAGError(Exception):
    """Base exception for expected pipeline failures."""


class InvalidUserError(TransactionRAGError):
    pass


class LLMUnavailableError(TransactionRAGError):
    pass


class LLMTimeoutError(LLMUnavailableError):
    pass


class CircuitOpenError(LLMUnavailableError):
    pass


class MalformedLLMOutputError(LLMUnavailableError):
    pass
