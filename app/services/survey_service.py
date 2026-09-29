"""Survey data access service."""

import json
from pathlib import Path
from typing import Any
from app.config import settings
from app.models.schemas import SurveyRecord


class SurveyService:
    """Service for loading and querying survey datasets."""

    def __init__(self, data_path: Path | None = None) -> None:
        self.data_path = data_path or settings.surveys_file
        self._cache: list[dict[str, Any]] | None = None

    def get_all_surveys(self) -> list[dict[str, Any]]:
        """Return raw survey dictionary records."""
        if self._cache is None:
            if not self.data_path.exists():
                return []
            with open(self.data_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        return self._cache

    def get_response_count(self) -> int:
        """Return total count of loaded survey records."""
        return len(self.get_all_surveys())

    def get_survey_models(self) -> list[SurveyRecord]:
        """Return validated SurveyRecord Pydantic models."""
        raw = self.get_all_surveys()
        return [SurveyRecord(**item) for item in raw]
