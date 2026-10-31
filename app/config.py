from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_provider: Literal["poolside", "groq", "ollama"] = "poolside"
    poolside_api_key: SecretStr = SecretStr("")
    poolside_base_url: str = "https://inference.poolside.ai/v1"
    poolside_model: str = "poolside/laguna-s-2.1"
    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = "openai/gpt-oss-20b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    llm_temperature: float = Field(default=0.2, ge=0, le=1)
    llm_timeout_seconds: float = Field(default=90, ge=1, le=300)
    llm_max_retries: int = Field(default=3, ge=0, le=6)
    llm_min_interval_seconds: float = Field(default=3.0, ge=0, le=30)
    llm_max_tokens: int = Field(default=4096, ge=256, le=32000)
    groq_reasoning_effort: Literal["low", "medium", "high", ""] = "low"
    max_revisions: int = Field(default=1, ge=0, le=3)
    agent_timeout_seconds: float = Field(default=600, ge=10, le=1800)
    output_dir: Path = BASE_DIR / "outputs"

    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr = SecretStr("")
    langsmith_project: str = "autonomous-docx-agent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    @field_validator("output_dir")
    @classmethod
    def resolve_output_dir(cls, value: Path) -> Path:
        return (value if value.is_absolute() else BASE_DIR / value).resolve()

    @property
    def model_name(self) -> str:
        if self.llm_provider == "poolside":
            return self.poolside_model
        return self.groq_model if self.llm_provider == "groq" else self.ollama_model
