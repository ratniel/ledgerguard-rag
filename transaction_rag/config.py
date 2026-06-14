from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_OPENROUTER_PLANNER_MODELS = ",".join(
    [
        "nex-agi/nex-n2-pro:free",
        "google/gemma-4-26b-a4b-it:free",
        "google/gemma-4-31b-it:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "nvidia/nemotron-nano-9b-v2:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    ]
)

DEFAULT_OPENROUTER_RESPONSE_MODELS = ",".join(
    [
        "nex-agi/nex-n2-pro:free",
        "google/gemma-4-26b-a4b-it:free",
        "google/gemma-4-31b-it:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "openai/gpt-oss-120b:free",
        "openai/gpt-oss-20b:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "nousresearch/hermes-3-llama-3.1-405b:free",
    ]
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_primary_model: str = "qwen/qwen3-next-80b-a3b-instruct:free"
    openrouter_fallback_model: str = "openrouter/free"
    openrouter_planner_models: str = Field(
        default=DEFAULT_OPENROUTER_PLANNER_MODELS,
        validation_alias="OPENROUTER_PLANNER_MODELS",
    )
    openrouter_response_models: str = Field(
        default=DEFAULT_OPENROUTER_RESPONSE_MODELS,
        validation_alias="OPENROUTER_RESPONSE_MODELS",
    )
    enable_llm: bool = True
    prefer_heuristic_planner: bool = True
    enable_response_llm: bool = False
    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 900
    request_timeout_seconds: float = 6.0
    llm_model_attempt_limit: int = 2
    circuit_breaker_threshold: int = 3

    max_prompt_chars: int = 1200
    token_budget: int = 8000
    max_history_items: int = 6

    sqlite_path: Path = Path("cache.sqlite3")
    outputs_dir: Path = Path("outputs")
    audit_log_path: Path = Path("audit/audit.jsonl")

    tracing_enabled: bool = True
    phoenix_project_name: str = "ledgerguard-rag"
    phoenix_collector_endpoint: str = "http://localhost:16006/v1/traces"

    gradio_host: str = "127.0.0.1"
    gradio_port: int = 7860


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
