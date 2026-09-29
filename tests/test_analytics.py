"""Comprehensive unit tests for the deterministic survey analytics layer."""

from datetime import datetime, timezone
import pytest
from app.tools.data_tools import (
    compute_average_rating,
    compute_csat,
    compute_nps,
    compute_sentiment_breakdown,
    count_responses,
    filter_by_date,
    filter_surveys,
    get_top_themes,
    compare_period_metrics,
)
from app.models.schemas import PeriodComparisonResult, PeriodMetrics, ThemeMetric


@pytest.fixture
def sample_records() -> list[dict]:
    """Controlled fixture containing 8 multi-theme, multi-period survey records."""
    return [
        {
            "id": "s1",
            "rating": 5,
            "csat_score": 5,
            "theme": "Food Quality",
            "category": "Food Quality",
            "cohort": "enterprise",
            "sentiment": "positive",
            "timestamp": "2026-04-05T10:00:00Z",
        },
        {
            "id": "s2",
            "rating": 4,
            "csat_score": 4,
            "theme": "Food Quality",
            "category": "Food Quality",
            "cohort": "enterprise",
            "sentiment": "positive",
            "timestamp": "2026-04-10T11:00:00Z",
        },
        {
            "id": "s3",
            "rating": 2,
            "csat_score": 2,
            "theme": "Wait Time",
            "category": "Wait Time",
            "cohort": "self_serve",
            "sentiment": "negative",
            "timestamp": "2026-04-15T12:00:00Z",
        },
        {
            "id": "s4",
            "rating": 1,
            "csat_score": 1,
            "theme": "Wait Time",
            "category": "Wait Time",
            "cohort": "self_serve",
            "sentiment": "negative",
            "timestamp": "2026-04-20T13:00:00Z",
        },
        {
            "id": "s5",
            "rating": 3,
            "csat_score": 3,
            "theme": "Pricing",
            "category": "Pricing",
            "cohort": "self_serve",
            "sentiment": "neutral",
            "timestamp": "2026-04-25T14:00:00Z",
        },
        {
            "id": "s6",
            "rating": 5,
            "csat_score": 5,
            "theme": "Wait Time",
            "category": "Wait Time",
            "cohort": "self_serve",
            "sentiment": "positive",
            "timestamp": "2026-05-05T15:00:00Z",
        },
        {
            "id": "s7",
            "rating": 4,
            "csat_score": 4,
            "theme": "Wait Time",
            "category": "Wait Time",
            "cohort": "self_serve",
            "sentiment": "positive",
            "timestamp": "2026-05-10T16:00:00Z",
        },
        {
            "id": "s8",
            "rating": 2,
            "csat_score": 2,
            "theme": "Pricing",
            "category": "Pricing",
            "cohort": "self_serve",
            "sentiment": "negative",
            "timestamp": "2026-05-15T17:00:00Z",
        },
    ]


# 1. Empty data safety tests
def test_empty_data_handling() -> None:
    assert compute_csat([]) == 0.0
    assert compute_average_rating([]) == 0.0
    assert count_responses([]) == 0
    assert compute_nps([]) == 0.0
    assert compute_sentiment_breakdown([]) == {"positive": 0, "neutral": 0, "negative": 0}
    assert filter_by_date([], "2026-04-01", "2026-04-30") == []
    assert get_top_themes([]) == []

    comp = compare_period_metrics([], "2026-04-01", "2026-04-30", "2026-05-01", "2026-05-31")
    assert isinstance(comp, PeriodComparisonResult)
    assert comp.period_a.response_count == 0
    assert comp.period_b.response_count == 0
    assert comp.csat_delta == 0.0
    assert comp.average_rating_delta == 0.0


# 2. Date filtering tests
def test_date_filtering(sample_records: list[dict]) -> None:
    # Filter April (5 records: s1-s5)
    april = filter_by_date(sample_records, start_date="2026-04-01", end_date="2026-04-30")
    assert len(april) == 5
    assert {r["id"] for r in april} == {"s1", "s2", "s3", "s4", "s5"}

    # Filter May (3 records: s6-s8)
    may = filter_by_date(sample_records, start_date="2026-05-01", end_date="2026-05-31")
    assert len(may) == 3
    assert {r["id"] for r in may} == {"s6", "s7", "s8"}

    # Filter exact single day
    day = filter_by_date(sample_records, start_date="2026-04-10", end_date="2026-04-10")
    assert len(day) == 1
    assert day[0]["id"] == "s2"

    # Out of range
    none_found = filter_by_date(sample_records, start_date="2026-06-01", end_date="2026-06-30")
    assert len(none_found) == 0


# 3. CSAT computation tests
def test_compute_csat(sample_records: list[dict]) -> None:
    # Definition: rating >= 4
    # All 8 records: ratings = [5, 4, 2, 1, 3, 5, 4, 2] -> 4 are >= 4 -> 50.0%
    assert compute_csat(sample_records) == 50.0

    # Raw numbers list
    assert compute_csat([5, 4, 4, 5]) == 100.0
    assert compute_csat([1, 2, 3]) == 0.0
    assert compute_csat([4, 3]) == 50.0

    # April only: ratings [5, 4, 2, 1, 3] -> 2 are >= 4 -> 40.0%
    april = filter_by_date(sample_records, "2026-04-01", "2026-04-30")
    assert compute_csat(april) == 40.0

    # May only: ratings [5, 4, 2] -> 2 are >= 4 -> 66.67%
    may = filter_by_date(sample_records, "2026-05-01", "2026-05-31")
    assert compute_csat(may) == 66.67


# 4. Average rating tests
def test_compute_average_rating(sample_records: list[dict]) -> None:
    # All 8 ratings: sum = 5+4+2+1+3+5+4+2 = 26 / 8 = 3.25
    assert compute_average_rating(sample_records) == 3.25

    # Raw numbers list
    assert compute_average_rating([5, 5, 5]) == 5.0
    assert compute_average_rating([1, 2]) == 1.5

    # Single value
    assert compute_average_rating([4]) == 4.0


# 5. Response count tests
def test_count_responses(sample_records: list[dict]) -> None:
    assert count_responses(sample_records) == 8
    assert count_responses([]) == 0


# 6. Top themes tests
def test_get_top_themes(sample_records: list[dict]) -> None:
    # Group breakdown in sample_records:
    # Wait Time: 4 records (s3, s4, s6, s7) -> ratings [2, 1, 5, 4] -> avg = 3.0, csat = 50.0%
    # Food Quality: 2 records (s1, s2) -> ratings [5, 4] -> avg = 4.5, csat = 100.0%
    # Pricing: 2 records (s5, s8) -> ratings [3, 2] -> avg = 2.5, csat = 0.0%
    top = get_top_themes(sample_records, top_n=3)
    assert len(top) == 3
    assert all(isinstance(t, ThemeMetric) for t in top)

    # First by volume is Wait Time (4 records)
    assert top[0].theme == "Wait Time"
    assert top[0].count == 4
    assert top[0].average_rating == 3.0
    assert top[0].csat == 50.0

    # Worst CSAT sorting
    worst = get_top_themes(sample_records, top_n=3, metric="worst_csat")
    assert worst[0].theme == "Pricing"  # CSAT 0.0%
    assert worst[0].csat == 0.0


# 7. Period comparison tests
def test_compare_period_metrics(sample_records: list[dict]) -> None:
    # Compare April vs May for Wait Time
    # April Wait Time: s3 (2), s4 (1) -> count=2, avg=1.5, csat=0.0%
    # May Wait Time: s6 (5), s7 (4) -> count=2, avg=4.5, csat=100.0%
    comp = compare_period_metrics(
        sample_records,
        period_a_start="2026-04-01",
        period_a_end="2026-04-30",
        period_b_start="2026-05-01",
        period_b_end="2026-05-31",
        theme="Wait Time",
        label_a="April 2026",
        label_b="May 2026",
    )

    assert isinstance(comp, PeriodComparisonResult)
    assert comp.period_a.response_count == 2
    assert comp.period_a.average_rating == 1.5
    assert comp.period_a.csat == 0.0

    assert comp.period_b.response_count == 2
    assert comp.period_b.average_rating == 4.5
    assert comp.period_b.csat == 100.0

    assert comp.count_delta == 0
    assert comp.average_rating_delta == 3.0
    assert comp.csat_delta == 100.0
    assert "improved by 100.0%" in comp.summary
