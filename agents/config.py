"""Central runtime configuration, loaded from environment variables / .env."""

from __future__ import annotations

import json
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    llm_model: str = Field(default="llama3.1:8b", alias="LLM_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")

    # Embeddings
    embeddings_provider: str = Field(default="fake", alias="EMBEDDINGS_PROVIDER")
    embeddings_model: str = Field(default="nomic-embed-text", alias="EMBEDDINGS_MODEL")
    embeddings_dim: int = Field(default=384, alias="EMBEDDINGS_DIM")

    # Postgres
    database_url: str = Field(
        default="postgresql+asyncpg://agent_runtime:agent_runtime@localhost:5432/agent_runtime",
        alias="DATABASE_URL",
    )
    checkpoint_database_url: str = Field(
        default="postgresql://agent_runtime:agent_runtime@localhost:5432/agent_runtime",
        alias="CHECKPOINT_DATABASE_URL",
    )

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="legal_docs", alias="QDRANT_COLLECTION")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Search
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")

    # A2A
    a2a_registry_raw: str = Field(
        default='{"researcher": "http://localhost:8000", '
        '"coder": "http://localhost:8000", "reviewer": "http://localhost:8000"}',
        alias="A2A_REGISTRY",
    )

    # Observability
    otel_exporter: str = Field(default="console", alias="OTEL_EXPORTER")
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str | None = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="ai-agent-runtime", alias="LANGCHAIN_PROJECT")

    @property
    def a2a_registry(self) -> dict[str, str]:
        try:
            return json.loads(self.a2a_registry_raw)
        except json.JSONDecodeError:
            return {}


@lru_cache
def get_settings() -> Settings:
    return Settings()
