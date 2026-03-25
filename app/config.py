"""
Configuration for toolproxy.
"""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Upstream LLM (any OpenAI-compatible endpoint)
    upstream_url: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL of the upstream OpenAI-compatible API",
    )
    upstream_model: str = Field(
        default="your-model-name",
        description="Model name to forward to upstream",
    )
    upstream_api_key: str = Field(
        default="dummy-key",
        description="API key for upstream (LiteLLM master key or dummy)",
    )

    # HTTP
    request_timeout: int = Field(default=180, description="Upstream request timeout in seconds")
    max_retries: int = Field(default=2, description="Max retries on upstream 5xx errors")
    retry_on_timeout: bool = Field(
        default=False,
        description=(
            "Retry upstream request on timeout. Default False — OCI timeouts are usually "
            "caused by the model being stuck, not a transient network issue. Retrying just "
            "multiplies the wait time without benefit. Set True only if your upstream is "
            "known to have intermittent timeouts."
        ),
    )

    # Logging
    log_level: str = Field(default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR")
    audit_log_path: str = Field(
        default="/logs/toolproxy-audit.jsonl",
        description="Path for the anonymized audit log (JSON Lines). Empty string disables it.",
    )


settings = Settings()
