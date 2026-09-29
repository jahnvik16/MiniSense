"""ComparisonAgent executing deterministic period and cohort metric delta analysis."""

from typing import Any
from app.models.schemas import (
    ComparisonAgentInput,
    ComparisonAgentResult,
    PeriodMetrics,
    TaskSpec,
    ThemeMetric,
)
from app.services.survey_service import SurveyService
from app.tools.data_tools import (
    compare_period_metrics,
    compute_average_rating,
    compute_csat,
    compute_sentiment_breakdown,
    count_responses,
    filter_by_date,
    filter_surveys,
    get_top_themes,
)


class ComparisonAgent:
    """Sub-agent responsible for deterministic period and cohort delta comparisons.

    Computes absolute and relative shifts for CSAT, average ratings, response counts,
    and top themes using pure Python math with safe zero-denominator handling.
    """

    def __init__(
        self,
        service: SurveyService | None = None,
        data_agent: Any | None = None,
    ) -> None:
        if service is not None:
            self.service = service
        elif data_agent is not None and hasattr(data_agent, "service"):
            self.service = data_agent.service
        else:
            self.service = SurveyService()
        self.data_agent = data_agent

    def run(self, task_or_input: TaskSpec | ComparisonAgentInput) -> ComparisonAgentResult:
        """Execute period or cohort comparison deterministically."""
        records = self.service.get_all_surveys()

        if isinstance(task_or_input, TaskSpec):
            params = task_or_input.parameters
            theme = params.get("theme") or params.get("category")
            cohort = params.get("cohort")
            top_n = int(params.get("top_n", 5))

            # Current period bounds
            current_start = task_or_input.start_date or params.get("current_start_date") or params.get("start_date")
            current_end = task_or_input.end_date or params.get("current_end_date") or params.get("end_date")
            current_label = params.get("current_label", "Current Period")

            # Previous period bounds
            previous_start = (
                task_or_input.comparison_start_date
                or params.get("previous_start_date")
                or params.get("comparison_start_date")
            )
            previous_end = (
                task_or_input.comparison_end_date
                or params.get("previous_end_date")
                or params.get("comparison_end_date")
            )
            previous_label = params.get("previous_label", "Previous Period")

            # Cohort comparison fallback if comparison periods are not specified
            cohort_a = params.get("cohort_a")
            cohort_b = params.get("cohort_b")

            if cohort_a and cohort_b and not (previous_start and previous_end):
                return self._compare_cohorts(
                    records=records,
                    cohort_a=cohort_a,
                    cohort_b=cohort_b,
                    metric_name=params.get("metric_name", "csat"),
                    top_n=top_n,
                )

            return self._compare_periods(
                records=records,
                current_start=current_start,
                current_end=current_end,
                previous_start=previous_start,
                previous_end=previous_end,
                theme=theme,
                cohort=cohort,
                current_label=current_label,
                previous_label=previous_label,
                top_n=top_n,
            )

        # Legacy ComparisonAgentInput payload
        return self._compare_cohorts(
            records=records,
            cohort_a=task_or_input.cohort_a,
            cohort_b=task_or_input.cohort_b,
            metric_name=task_or_input.metric_name,
            top_n=5,
        )

    def _compare_periods(
        self,
        records: list[dict[str, Any]],
        current_start: str | None,
        current_end: str | None,
        previous_start: str | None,
        previous_end: str | None,
        theme: str | None = None,
        cohort: str | None = None,
        current_label: str = "Current Period",
        previous_label: str = "Previous Period",
        top_n: int = 5,
    ) -> ComparisonAgentResult:
        """Compare performance across two distinct time windows."""
        pool = records
        if theme:
            pool = [
                r for r in pool
                if (r.get("theme", "").lower() == theme.lower() or r.get("category", "").lower() == theme.lower())
            ]
        if cohort:
            pool = [r for r in pool if r.get("cohort", "").lower() == cohort.lower()]

        # Filter records for both periods
        current_records = filter_by_date(pool, current_start, current_end)
        previous_records = filter_by_date(pool, previous_start, previous_end)

        # Current period metrics
        curr_count = count_responses(current_records)
        curr_avg = compute_average_rating(current_records)
        curr_csat = compute_csat(current_records)
        curr_sentiments = compute_sentiment_breakdown(current_records)
        curr_themes = get_top_themes(current_records, top_n=top_n)

        current_period_metrics = PeriodMetrics(
            period_label=current_label,
            start_date=str(current_start) if current_start else None,
            end_date=str(current_end) if current_end else None,
            response_count=curr_count,
            average_rating=curr_avg,
            csat=curr_csat,
            sentiment_breakdown=curr_sentiments,
            top_themes=curr_themes,
        )

        # Previous period metrics
        prev_count = count_responses(previous_records)
        prev_avg = compute_average_rating(previous_records)
        prev_csat = compute_csat(previous_records)
        prev_sentiments = compute_sentiment_breakdown(previous_records)
        prev_themes = get_top_themes(previous_records, top_n=top_n)

        previous_period_metrics = PeriodMetrics(
            period_label=previous_label,
            start_date=str(previous_start) if previous_start else None,
            end_date=str(previous_end) if previous_end else None,
            response_count=prev_count,
            average_rating=prev_avg,
            csat=prev_csat,
            sentiment_breakdown=prev_sentiments,
            top_themes=prev_themes,
        )

        # Deterministic delta calculations with zero denominator safety
        csat_diff = round(curr_csat - prev_csat, 2)
        csat_pct_change = round(((curr_csat - prev_csat) / prev_csat) * 100, 2) if prev_csat > 0 else 0.0

        avg_diff = round(curr_avg - prev_avg, 2)
        avg_pct_change = round(((curr_avg - prev_avg) / prev_avg) * 100, 2) if prev_avg > 0 else 0.0

        count_diff = curr_count - prev_count
        count_pct_change = round(((curr_count - prev_count) / prev_count) * 100, 2) if prev_count > 0 else 0.0

        metric_changes: dict[str, float] = {
            "csat_delta": csat_diff,
            "csat_pct_change": csat_pct_change,
            "average_rating_delta": avg_diff,
            "average_rating_pct_change": avg_pct_change,
            "count_delta": float(count_diff),
            "count_pct_change": count_pct_change,
        }

        # Executive summary construction
        csat_status = "improved" if csat_diff > 0 else ("declined" if csat_diff < 0 else "remained flat")
        scope_str = f" for '{theme}'" if theme else ""
        cohort_str = f" in {cohort}" if cohort else ""
        summary = (
            f"Comparing {current_label} vs {previous_label}{scope_str}{cohort_str}: "
            f"CSAT {csat_status} by {abs(csat_diff):.1f}% ({prev_csat}% -> {curr_csat}%), "
            f"Average Rating moved {avg_diff:+.2f} ({prev_avg:.2f} -> {curr_avg:.2f}), "
            f"and Volume changed by {count_diff:+d} responses ({count_pct_change:+.1f}%)."
        )

        return ComparisonAgentResult(
            current_period=current_period_metrics,
            previous_period=previous_period_metrics,
            metric_changes=metric_changes,
            summary=summary,
        )

    def _compare_cohorts(
        self,
        records: list[dict[str, Any]],
        cohort_a: str,
        cohort_b: str,
        metric_name: str = "csat",
        top_n: int = 5,
    ) -> ComparisonAgentResult:
        """Compare performance across two distinct customer cohorts."""
        records_a = filter_surveys(records, cohort=cohort_a)
        records_b = filter_surveys(records, cohort=cohort_b)

        count_a = count_responses(records_a)
        avg_a = compute_average_rating(records_a)
        csat_a = compute_csat(records_a)
        sentiments_a = compute_sentiment_breakdown(records_a)
        themes_a = get_top_themes(records_a, top_n=top_n)

        metrics_a = PeriodMetrics(
            period_label=cohort_a,
            response_count=count_a,
            average_rating=avg_a,
            csat=csat_a,
            sentiment_breakdown=sentiments_a,
            top_themes=themes_a,
        )

        count_b = count_responses(records_b)
        avg_b = compute_average_rating(records_b)
        csat_b = compute_csat(records_b)
        sentiments_b = compute_sentiment_breakdown(records_b)
        themes_b = get_top_themes(records_b, top_n=top_n)

        metrics_b = PeriodMetrics(
            period_label=cohort_b,
            response_count=count_b,
            average_rating=avg_b,
            csat=csat_b,
            sentiment_breakdown=sentiments_b,
            top_themes=themes_b,
        )

        csat_diff = round(csat_a - csat_b, 2)
        avg_diff = round(avg_a - avg_b, 2)
        count_diff = count_a - count_b

        metric_changes: dict[str, float] = {
            "csat_delta": csat_diff,
            "average_rating_delta": avg_diff,
            "count_delta": float(count_diff),
        }

        higher = cohort_a if csat_a > csat_b else (cohort_b if csat_b > csat_a else "equal")
        summary = (
            f"Cohort comparison {cohort_a.capitalize()} vs {cohort_b.capitalize()}: "
            f"CSAT delta is {csat_diff:+.1f}% ({csat_a}% vs {csat_b}%), "
            f"Average Rating delta is {avg_diff:+.2f} ({avg_a:.2f} vs {avg_b:.2f}). Higher: {higher}."
        )

        return ComparisonAgentResult(
            current_period=metrics_a,
            previous_period=metrics_b,
            metric_changes=metric_changes,
            summary=summary,
        )
