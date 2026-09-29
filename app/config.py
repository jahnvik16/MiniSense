from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM configuration
    llm_provider: Literal["gemini", "openai", "mock"] = "gemini"
    llm_model: str = "gemini-1.5-flash"
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


settings = Settings()
