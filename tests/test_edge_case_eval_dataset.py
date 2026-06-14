from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from transaction_rag.config import Settings
from transaction_rag.pipeline import TransactionRAGPipeline


DATA_PATH = Path("data/assessment_transaction_data.xlsx - Transactions.csv")
EVAL_PATHS = [
    Path("evals/edge_cases.jsonl"),
    Path("evals/custom_edge_cases.jsonl"),
]


def load_eval_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in EVAL_PATHS:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            case["dataset_path"] = str(path)
            cases.append(case)
    return cases


def make_pipeline(tmp_path: Path, case: dict[str, Any]) -> TransactionRAGPipeline:
    settings = Settings(
        enable_llm=False,
        tracing_enabled=False,
        max_prompt_chars=int(case.get("max_prompt_chars", 1200)),
        token_budget=int(case.get("token_budget", 8000)),
        llm_max_output_tokens=int(case.get("llm_max_output_tokens", 900)),
        max_history_items=int(case.get("max_history_items", 6)),
        sqlite_path=tmp_path / f"{case['id']}.sqlite3",
        outputs_dir=tmp_path / "outputs",
        audit_log_path=tmp_path / "audit" / f"{case['id']}.jsonl",
    )
    return TransactionRAGPipeline(pd.read_csv(DATA_PATH), settings=settings)


@pytest.mark.parametrize("case", load_eval_cases(), ids=lambda case: case["id"])
def test_edge_case_eval_dataset(case: dict[str, Any], tmp_path: Path):
    pipeline = make_pipeline(tmp_path, case)
    for setup_prompt in case.get("setup_prompts", []):
        pipeline.run(case["user_id"], setup_prompt)

    result = pipeline.run(case["user_id"], case["prompt"])

    expected_error_code = case.get("expected_error_code")
    if expected_error_code is None:
        assert result.error is None
    else:
        assert result.error is not None
        assert result.error.code.value == expected_error_code

    assert result.user_name == case.get("expected_user_name")

    flags = {flag.value for flag in result.guardrail_flags}
    for expected_flag in case.get("expected_flags", []):
        assert expected_flag in flags
    for forbidden_flag in case.get("forbidden_flags", []):
        assert forbidden_flag not in flags

    expected_tool = case.get("expected_tool")
    tools = result.data_summary.get("tools", [])
    if expected_tool:
        assert tools
        assert tools[0]["name"] == expected_tool
    for key, value in case.get("expected_tool_arguments", {}).items():
        assert tools
        assert tools[0]["parameters"][key] == value
    for key, value in case.get("expected_tool_data_summary", {}).items():
        assert tools
        assert tools[0]["data_summary"][key] == value

    if "expect_visualization" in case:
        assert bool(result.visualizations) is bool(case["expect_visualization"])

    for expected_text in case.get("response_contains", []):
        assert expected_text in result.response
    for forbidden_text in case.get("response_not_contains", []):
        assert forbidden_text not in result.response

    history = pipeline.cache.get_query_history(case["user_id"])
    if any(key.startswith("history_prompt_") for key in case):
        assert history
        prompt = history[-1].prompt
        for expected_text in case.get("history_prompt_contains", []):
            assert expected_text in prompt
        for forbidden_text in case.get("history_prompt_not_contains", []):
            assert forbidden_text not in prompt
        if "history_prompt_max_chars" in case:
            assert len(prompt) <= int(case["history_prompt_max_chars"])
    if "expected_history_max_items" in case:
        assert len(history) <= int(case["expected_history_max_items"])
    if case.get("expect_chat_summary"):
        assert pipeline.cache.get_chat_summary(case["user_id"])
