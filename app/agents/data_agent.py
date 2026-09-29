"""DataAgent executing deterministic survey analytics tools."""

from typing import Any
from app.models.schemas import DataAgentInput, DataAgentResult, TaskSpec
from app.services.survey_service import SurveyService
from app.tools.data_tools import (
    compute_average_rating,
    compute_csat,
    count_responses,
    filter_by_date,
    filter_surveys,
    get_top_themes,
)


class DataAgent:
    """Sub-agent responsible for exact, deterministic survey calculations.

    Explicitly delegates all arithmetic to standalone data tools to avoid
    LLM mathematical hallucinations.
    """

    def __init__(self, service: SurveyService | None = None) -> None:
        self.service = service or SurveyService()
        # Explicit tool registry exposed by the agent
        self.tools = {
            "compute_csat": compute_csat,
            "compute_average_rating": compute_average_rating,
            "count_responses": count_responses,
            "get_top_themes": get_top_themes,
            "filter_by_date": filter_by_date,
            "filter_surveys": filter_surveys,
        }

    def run(self, task_or_input: TaskSpec | DataAgentInput) -> DataAgentResult:
        """Execute deterministic analytical tools on survey records."""
        # Normalize input to TaskSpec parameters
        if isinstance(task_or_input, TaskSpec):
            task_id = task_or_input.task_id
            question = task_or_input.question
            params = task_or_input.parameters
            start_date = task_or_input.start_date or params.get("start_date")
            end_date = task_or_input.end_date or params.get("end_date")
            cohort = params.get("cohort")
            category = params.get("category") or params.get("theme")
            top_n = int(params.get("top_n", 5))
        else:
            task_id = "legacy_data_task"
            question = f"Compute {task_or_input.metric_name} metrics"
            cohort = task_or_input.cohort
            category = task_or_input.category
            start_date = None
            end_date = None
            top_n = 5
            params = {}

        tools_invoked: list[str] = []
        warnings: list[str] = []

        # 1. Fetch raw survey records
        records = self.service.get_all_surveys()

        # 2. Apply cohort and category filtering deterministically
        if cohort or category:
            records = self.tools["filter_surveys"](records, cohort=cohort, category=category)
            tools_invoked.append("filter_surveys")

        # 3. Apply date filtering deterministically with graceful error handling
        if start_date or end_date:
            try:
                date_filtered = self.tools["filter_by_date"](records, start_date=start_date, end_date=end_date)
                records = date_filtered
                tools_invoked.append("filter_by_date")
            except Exception as e:
                warnings.append(f"Date filtering encountered an issue ({e}); proceeded with unfiltered dates")

        # 4. Determine theme sorting criterion from question semantics
        param_metric = params.get("metric") or params.get("theme_metric")
        if param_metric:
            theme_metric = param_metric
        else:
            question_lower = question.lower()
            if any(term in question_lower for term in ("worst", "lowest csat", "worst performing", "lowest rating", "poor")):
                theme_metric = "worst_csat"
            elif any(term in question_lower for term in ("complaint", "complaints", "negative", "issues", "dissatisfied", "unhappy")):
                theme_metric = "negative_volume"
            elif any(term in question_lower for term in ("best", "highest", "top satisfaction")):
                theme_metric = "best_csat"
            else:
                theme_metric = "volume"

        # 5. Call explicit deterministic analytics tools
        resp_count = self.tools["count_responses"](records)
        tools_invoked.append("count_responses")

        csat_score = self.tools["compute_csat"](records)
        tools_invoked.append("compute_csat")

        avg_rating = self.tools["compute_average_rating"](records)
        tools_invoked.append("compute_average_rating")

        top_themes = self.tools["get_top_themes"](records, top_n=top_n, metric=theme_metric)
        tools_invoked.append("get_top_themes")

        # 6. Assemble supporting metadata
        filters_applied: dict[str, Any] = {}
        if cohort:
            filters_applied["cohort"] = cohort
        if category:
            filters_applied["category"] = category
        if start_date:
            filters_applied["start_date"] = str(start_date)
        if end_date:
            filters_applied["end_date"] = str(end_date)

        metadata: dict[str, Any] = {
            "task_id": task_id,
            "filters_applied": filters_applied,
            "tools_called": tools_invoked,
            "theme_ranking_strategy": theme_metric,
        }
        if warnings:
            metadata["warnings"] = warnings

        return DataAgentResult(
            response_count=resp_count,
            average_rating=avg_rating,
            csat=csat_score,
            top_themes=top_themes,
            supporting_metadata=metadata,
        )
