import os
from pathlib import Path
from typing import Literal
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM configuration
    llm_provider: Literal["gemini", "openai", "mock"] = "openai"
    llm_model: str = "gpt-4o-mini"
    gemini_api_key: str | None = None
    google_api_key: str | None = None
    openai_api_key: str | None = None

    # Embedding configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    surveys_file: Path = data_dir / "surveys.json"
    faq_file: Path = data_dir / "faq.txt"
    faiss_index_dir: Path = data_dir / "faiss_index"

    @model_validator(mode="after")
    def resolve_provider_and_keys(self) -> "Settings":
        """Auto-resolve LLM provider and API keys from environment if not explicit."""
        openai_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        gemini_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if openai_key and not self.openai_api_key:
            self.openai_api_key = openai_key
        if gemini_key and not self.gemini_api_key:
            self.gemini_api_key = gemini_key

        # If provider wasn't explicitly overridden to mock, pick based on available key
        if self.llm_provider == "openai" and not self.openai_api_key:
            if self.gemini_api_key:
                self.llm_provider = "gemini"
                if self.llm_model == "gpt-4o-mini":
                    self.llm_model = "gemini-1.5-flash"
            else:
                self.llm_provider = "mock"
        elif self.llm_provider == "gemini" and not self.gemini_api_key:
            if self.openai_api_key:
                self.llm_provider = "openai"
                if self.llm_model == "gemini-1.5-flash":
                    self.llm_model = "gpt-4o-mini"
            else:
                self.llm_provider = "mock"

        return self

    @property
    def has_llm_credentials(self) -> bool:
        """Check whether valid LLM credentials are ready for invocation."""
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        if self.llm_provider == "gemini":
            return bool(self.gemini_api_key)
        return False


settings = Settings()
