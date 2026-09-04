"""
Application settings using Pydantic settings for environment-based configuration.

All settings are loaded from environment variables with sensible defaults.
"""
import secrets
from typing import List
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    ENVIRONMENT: str = Field(default="development", description="Environment name (development/production)")
    DEBUG: bool = Field(default=True, description="Debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # API
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8000, description="API port")
    API_PREFIX: str = Field(default="/api/v1", description="API route prefix")
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="CORS allowed origins"
    )

    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/finance_agent",
        description="Database connection URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=20, description="Database connection pool size")
    DATABASE_MAX_OVERFLOW: int = Field(default=10, description="Database connection pool max overflow")

    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    REDIS_CACHE_TTL: int = Field(default=3600, description="Redis cache TTL in seconds")

    # Vector DB Service (external)
    VECTOR_DB_URL: str = Field(default="http://localhost:6333", description="Vector database service URL")
    VECTOR_COLLECTION_NAME: str = Field(default="documents", description="Vector collection name")
    VECTOR_API_KEY: str | None = Field(default=None, description="Vector database API key")

    # LLM Providers
    OPENAI_API_KEY: str | None = Field(default=None, description="OpenAI API key")
    OPENAI_MODEL: str = Field(default="gpt-4-turbo-preview", description="OpenAI model name")
    OPENAI_EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )

    ANTHROPIC_API_KEY: str | None = Field(default=None, description="Anthropic API key")
    ANTHROPIC_MODEL: str = Field(default="claude-3-opus-20240229", description="Anthropic model name")

    DEEPSEEK_API_KEY: str | None = Field(default=None, description="DeepSeek API key")
    DEEPSEEK_MODEL: str = Field(default="deepseek-chat", description="DeepSeek model name")
    DEEPSEEK_BASE_URL: str = Field(
        default="https://api.deepseek.com",
        description="DeepSeek OpenAI-compatible API base URL",
    )

    # Embedding Models
    EMBEDDING_PROVIDER: str = Field(default="local", description="Embedding provider (local, openai)")
    EMBEDDING_MODEL: str = Field(default="bge-m3-v2-zh", description="Embedding model name")
    EMBEDDING_CACHE_TTL: int = Field(default=604800, description="Embedding cache TTL in seconds (7 days)")
    EMBEDDING_DEVICE: str = Field(default="cpu", description="Device for local embeddings (cpu, cuda)")

    # Security
    SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secret key for JWT signing"
    )
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="JWT access token expiration")
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="JWT refresh token expiration")

    # Memory Management
    MEMORY_TYPE: str = Field(default="optimized", description="Memory strategy (sliding_window, summarization, hybrid, optimized)")
    MEMORY_MAX_RECENT: int = Field(default=3, description="Max recent messages to always include")
    MEMORY_MAX_RELEVANT: int = Field(default=5, description="Max relevant historical messages")
    MEMORY_RELEVANCE_THRESHOLD: float = Field(default=0.5, description="Minimum similarity for relevance filtering")
    MEMORY_TOKEN_BUDGET: int = Field(default=4096, description="Max tokens for chat history")

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, description="Rate limit per minute")
    RATE_LIMIT_PER_HOUR: int = Field(default=1000, description="Rate limit per hour")

    # Neo4j Graph Database
    NEO4J_URI: str = Field(default="bolt://localhost:7687", description="Neo4j bolt URI")
    NEO4J_USER: str = Field(default="neo4j", description="Neo4j username")
    NEO4J_PASSWORD: str = Field(default="password", description="Neo4j password")
    NEO4J_DATABASE: str = Field(default="neo4j", description="Neo4j database name")
    NEO4J_MAX_CONNECTION_POOL_SIZE: int = Field(
        default=50, description="Neo4j connection pool size"
    )
    NEO4J_CONNECTION_TIMEOUT: float = Field(
        default=30.0, description="Neo4j connection timeout seconds"
    )

    # GraphRAG Feature Flags
    GRAPH_RAG_ENABLED: bool = Field(
        default=False, description="Enable GraphRAG features"
    )
    GRAPH_RAG_EXTRACTION_ENABLED: bool = Field(
        default=True, description="Enable entity extraction during ingestion"
    )
    GRAPH_RAG_COMMUNITY_ENABLED: bool = Field(
        default=False, description="Enable community detection and summarization"
    )
    GRAPH_RAG_COMMUNITY_MIN_SIZE: int = Field(
        default=3, description="Minimum community size for Leiden"
    )
    GRAPH_RAG_COMMUNITY_MAX_LEVELS: int = Field(
        default=5, description="Max hierarchy levels for community detection"
    )
    GRAPH_RAG_FUSION_WEIGHT: float = Field(
        default=0.5, description="Weight for graph results in fusion (0.0-1.0)"
    )
    GRAPH_RAG_TEXT_TO_CYPHER_ENABLED: bool = Field(
        default=True, description="Enable Text-to-Cypher query mode"
    )
    GRAPH_RAG_MAX_HOPS: int = Field(
        default=3, description="Maximum traversal depth for graph queries"
    )

    # Slot Filling
    SLOT_FILLING_ENABLED: bool = Field(
        default=True, description="Enable slot filling after intent detection"
    )
    SLOT_FILLING_TYPE: str = Field(
        default="hybrid", description="Slot filler type: rule_based, hybrid"
    )

    # Guardrails
    GUARDRAILS_ENABLED: bool = Field(
        default=True, description="Enable input/output guardrails"
    )
    GUARDRAILS_INPUT_ENABLED: bool = Field(
        default=True, description="Enable input guardrail (injection detection, PII redaction)"
    )
    GUARDRAILS_OUTPUT_ENABLED: bool = Field(
        default=True, description="Enable output guardrail (PII redaction in responses)"
    )
    GUARDRAILS_PII_REDACTION_ENABLED: bool = Field(
        default=True, description="Enable PII redaction in input and output"
    )
    PII_MASK_STYLE: str = Field(
        default="full", description="PII mask style: full ([REDACTED]) or partial (keep last 4 digits)"
    )
    COMPLIANCE_CHECK_ENABLED: bool = Field(
        default=True, description="Enable financial compliance check on output (return promises)"
    )
    RISK_DISCLAIMER_ENABLED: bool = Field(
        default=True, description="Append risk disclaimer to investment-related responses"
    )

    # Audit Logging
    AUDIT_LOG_ENABLED: bool = Field(
        default=True, description="Enable structured audit logging of tool executions and guardrail events"
    )

    # Reranker
    RERANKER_ENABLED: bool = Field(
        default=True, description="Enable reranking in retrieval pipeline"
    )
    RERANKER_TYPE: str = Field(
        default="cross_encoder",
        description="Reranker type: cross_encoder, llm, chained, noop",
    )
    RERANKER_MODEL: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking",
    )
    RERANKER_DEVICE: str = Field(
        default="cpu", description="Device for cross-encoder model (cpu, cuda)"
    )
    RERANKER_TOP_N: int = Field(
        default=5, description="Number of results after first-stage reranking"
    )
    RERANKER_LLM_SECOND_STAGE: bool = Field(
        default=False,
        description="Enable LLM as second-stage reranker after CrossEncoder",
    )
    RERANKER_LLM_TOP_N: int = Field(
        default=3, description="Number of results after LLM second-stage reranking"
    )

    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, description="Enable Prometheus metrics")
    METRICS_PORT: int = Field(default=9090, description="Metrics port")
    ENABLE_TRACING: bool = Field(default=False, description="Enable OpenTelemetry tracing")
    OTEL_ENDPOINT: str = Field(default="http://jaeger:4317", description="OpenTelemetry endpoint")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | List[str]) -> List[str]:
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL format"""
        if not v.startswith(("postgresql+asyncpg://", "postgresql://")):
            raise ValueError("Database URL must use postgresql+asyncpg:// scheme")
        return v

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate secret key is not default in production"""
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters")
        return v


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    This function caches the settings to avoid reloading from environment variables
    on every access. The cache is created once per process.

    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Global settings instance
settings = get_settings()
