from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd

from .cache import AuditLogger, SQLiteKVStore
from .config import Settings, get_settings
from .context import ContextManager
from .data import TransactionDataRepository
from .guardrails import InputGuardrails, OutputGuardrails
from .llm import LLMClient
from .models import PipelineResult
from .observability import Observability
from .tools import VisualizationToolkit
from .workflow import TransactionRAGWorkflow


class TransactionRAGPipeline:
    def __init__(self, df: pd.DataFrame, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.settings.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.repository = TransactionDataRepository(df)
        self.cache = SQLiteKVStore(self.settings.sqlite_path, max_history_items=self.settings.max_history_items)
        self.audit_logger = AuditLogger(self.settings.audit_log_path, self.cache)
        self.llm = LLMClient(self.settings)
        self.context_manager = ContextManager(self.repository, self.cache, self.llm)
        self.toolkit = VisualizationToolkit(self.repository, self.settings.outputs_dir)
        self.input_guardrails = InputGuardrails(self.settings)
        self.output_guardrails = OutputGuardrails()
        self.observability = Observability(self.settings)
        self.workflow = TransactionRAGWorkflow(
            settings=self.settings,
            repository=self.repository,
            cache=self.cache,
            audit_logger=self.audit_logger,
            context_manager=self.context_manager,
            toolkit=self.toolkit,
            input_guardrails=self.input_guardrails,
            output_guardrails=self.output_guardrails,
            llm=self.llm,
            observability=self.observability,
        )

    @classmethod
    def from_csv(cls, path: str | Path, settings: Settings | None = None) -> "TransactionRAGPipeline":
        return cls(pd.read_csv(path), settings=settings)

    def run(self, user_id: str, prompt: str, session_id: str | None = None) -> PipelineResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(user_id=user_id, prompt=prompt, session_id=session_id))
        raise RuntimeError("TransactionRAGPipeline.run() cannot be called inside an active event loop; use arun().")

    async def arun(self, user_id: str, prompt: str, session_id: str | None = None) -> PipelineResult:
        handler = self.workflow.run(user_id=user_id, prompt=prompt, session_id=session_id)
        result = await handler
        self.workflow.ctx = handler.ctx
        if isinstance(result, PipelineResult):
            return result
        return PipelineResult.model_validate(result)

    def generate_workflow_diagrams(self, output_dir: str | Path = "docs") -> list[Path]:
        from llama_index.utils.workflow import draw_all_possible_flows, draw_most_recent_execution

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        all_flows = target / "workflow_all_flows.html"
        recent = target / "workflow_recent_execution.html"
        draw_all_possible_flows(self.workflow, filename=str(all_flows))
        try:
            draw_most_recent_execution(self.workflow, filename=str(recent))
        except Exception:
            return [all_flows]
        return [all_flows, recent]
