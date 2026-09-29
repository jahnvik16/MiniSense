"""Deterministic numerical tools for survey analytics."""

from datetime import datetime, timezone
from typing import Any, Union
from app.models.schemas import PeriodComparisonResult, PeriodMetrics, ThemeMetric


def _extract_score(item: Any) -> float | None:
    """Helper to extract a numeric rating/score from an int, float, dict, or object."""
    if isinstance(item, (int, float)):
        return float(item)
    if isinstance(item, dict):
        for key in ("rating", "csat_score", "score", "value"):
            if key in item and item[key] is not None:
                try:
                    return float(item[key])
                except (ValueError, TypeError):
                    pass
        return None
    if hasattr(item, "rating"):
        return float(item.rating)
    if hasattr(item, "csat_score"):
        return float(item.csat_score)
    return None


def _parse_timestamp(ts: Any) -> datetime | None:
    """Parse various timestamp representations into a UTC datetime."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, str):
        cleaned = ts.strip()
        if not cleaned:
            return None
        # Handle ISO with Z
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(cleaned)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            # Fallback to date-only string YYYY-MM-DD
            try:
                dt = datetime.strptime(cleaned[:10], "%Y-%m-%d")
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                return None
    return None


def filter_by_date(
    records: list[dict[str, Any]],
    start_date: str | datetime | None = None,
    end_date: str | datetime | None = None,
) -> list[dict[str, Any]]:
    """Filter survey records deterministically by an inclusive date range.

    Handles empty datasets, date-only strings ('2026-04-01'), ISO strings, and datetimes.
    """
    if not records:
        return []

    start_dt = _parse_timestamp(start_date)
    end_dt = _parse_timestamp(end_date)

    # If end_date is date-only string (length 10 e.g. '2026-04-30'), extend to end of day
    if isinstance(end_date, str) and len(end_date.strip()) == 10 and end_dt:
        end_dt = end_dt.replace(hour=23, minute=59, second=59, microsecond=999999)

    filtered: list[dict[str, Any]] = []
    for r in records:
        record_dt = _parse_timestamp(r.get("timestamp"))
        if record_dt is None:
            continue
        if start_dt and record_dt < start_dt:
            continue
        if end_dt and record_dt > end_dt:
            continue
        filtered.append(r)

    return filtered


def compute_csat(records_or_scores: list[Any], scale: int = 5) -> float:
    """Compute Customer Satisfaction (CSAT) percentage.

    Definition: percentage of responses with rating >= 4 (for 5-point scale).
    Returns 0.0 safely on empty datasets.
    """
    if not records_or_scores:
        return 0.0

    scores: list[float] = []
    for item in records_or_scores:
        score = _extract_score(item)
        if score is not None:
            scores.append(score)

    if not scores:
        return 0.0

    threshold = 4.0 if scale == 5 else (0.8 * scale)
    positive_count = sum(1 for s in scores if s >= threshold)
    return round((positive_count / len(scores)) * 100, 2)


def compute_average_rating(records_or_scores: list[Any]) -> float:
    """Compute the arithmetic mean of survey ratings (1-5).

    Returns 0.0 safely on empty datasets.
    """
    if not records_or_scores:
        return 0.0

    scores: list[float] = []
    for item in records_or_scores:
        score = _extract_score(item)
        if score is not None:
            scores.append(score)

    if not scores:
        return 0.0

    return round(sum(scores) / len(scores), 2)


def count_responses(records: list[Any]) -> int:
    """Return the total count of responses safely."""
    if not records:
        return 0
    return len(records)


def compute_sentiment_breakdown(sentiments_or_records: list[Any]) -> dict[str, int]:
    """Compute frequency counts for survey sentiments (positive, neutral, negative)."""
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    if not sentiments_or_records:
        return counts

    for item in sentiments_or_records:
        sentiment_val = ""
        if isinstance(item, str):
            sentiment_val = item
        elif isinstance(item, dict):
            sentiment_val = item.get("sentiment", "")
        elif hasattr(item, "sentiment"):
            sentiment_val = item.sentiment

        norm = sentiment_val.lower().strip()
        if norm in counts:
            counts[norm] += 1
        elif norm:
            counts[norm] = counts.get(norm, 0) + 1
    return counts


def get_top_themes(
    records: list[dict[str, Any]],
    top_n: int = 5,
    metric: str = "volume",
) -> list[ThemeMetric]:
    """Extract aggregated theme metrics ranked deterministically.

    Supported sorting metrics:
      - 'volume': highest count first (default)
      - 'csat_asc' / 'worst_csat': lowest CSAT first (complaint driver analysis)
      - 'csat_desc' / 'best_csat': highest CSAT first
      - 'negative_volume': highest negative sentiment count first
    """
    if not records:
        return []

    # Group by theme / category
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        theme = r.get("theme") or r.get("category") or "General"
        groups.setdefault(theme, []).append(r)

    metrics_list: list[ThemeMetric] = []
    for theme_name, theme_records in groups.items():
        count = len(theme_records)
        avg_rating = compute_average_rating(theme_records)
        csat = compute_csat(theme_records)
        sentiments = compute_sentiment_breakdown(theme_records)

        metrics_list.append(
            ThemeMetric(
                theme=theme_name,
                count=count,
                average_rating=avg_rating,
                csat=csat,
                sentiment_breakdown=sentiments,
            )
        )

    # Sort deterministically
    metric_lower = metric.lower()
    if metric_lower in ("csat_asc", "worst_csat", "lowest_csat"):
        metrics_list.sort(key=lambda x: (x.csat, -x.count, x.theme))
    elif metric_lower in ("csat_desc", "best_csat", "highest_csat"):
        metrics_list.sort(key=lambda x: (-x.csat, -x.count, x.theme))
    elif metric_lower in ("negative_volume", "negative"):
        metrics_list.sort(
            key=lambda x: (-x.sentiment_breakdown.get("negative", 0), x.csat, x.theme)
        )
    else:  # default 'volume'
        metrics_list.sort(key=lambda x: (-x.count, -x.csat, x.theme))

    return metrics_list[:top_n]


def compare_period_metrics(
    records: list[dict[str, Any]],
    period_a_start: str | datetime | None,
    period_a_end: str | datetime | None,
    period_b_start: str | datetime | None,
    period_b_end: str | datetime | None,
    theme: str | None = None,
    cohort: str | None = None,
    label_a: str = "Period A",
    label_b: str = "Period B",
) -> PeriodComparisonResult:
    """Compare performance metrics between two periods with delta analysis."""
    # Optional preliminary filtering by theme and/or cohort
    pool = records
    if theme:
        pool = [
            r for r in pool
            if (r.get("theme", "").lower() == theme.lower() or r.get("category", "").lower() == theme.lower())
        ]
    if cohort:
        pool = [r for r in pool if r.get("cohort", "").lower() == cohort.lower()]

    records_a = filter_by_date(pool, period_a_start, period_a_end)
    records_b = filter_by_date(pool, period_b_start, period_b_end)

    count_a = count_responses(records_a)
    count_b = count_responses(records_b)

    avg_a = compute_average_rating(records_a)
    avg_b = compute_average_rating(records_b)

    csat_a = compute_csat(records_a)
    csat_b = compute_csat(records_b)

    sentiment_a = compute_sentiment_breakdown(records_a)
    sentiment_b = compute_sentiment_breakdown(records_b)

    themes_a = get_top_themes(records_a, top_n=5)
    themes_b = get_top_themes(records_b, top_n=5)

    period_a_metrics = PeriodMetrics(
        period_label=label_a,
        start_date=str(period_a_start) if period_a_start else None,
        end_date=str(period_a_end) if period_a_end else None,
        response_count=count_a,
        average_rating=avg_a,
        csat=csat_a,
        sentiment_breakdown=sentiment_a,
        top_themes=themes_a,
    )

    period_b_metrics = PeriodMetrics(
        period_label=label_b,
        start_date=str(period_b_start) if period_b_start else None,
        end_date=str(period_b_end) if period_b_end else None,
        response_count=count_b,
        average_rating=avg_b,
        csat=csat_b,
        sentiment_breakdown=sentiment_b,
        top_themes=themes_b,
    )

    count_delta = count_b - count_a
    avg_delta = round(avg_b - avg_a, 2)
    csat_delta = round(csat_b - csat_a, 2)

    qualifier = "improved" if csat_delta > 0 else ("declined" if csat_delta < 0 else "held steady")
    scope_desc = f" for '{theme}'" if theme else ""
    summary = (
        f"{label_b} vs {label_a}{scope_desc}: CSAT {qualifier} by {abs(csat_delta):.1f}% "
        f"({csat_a}% -> {csat_b}%), Average Rating shifted {avg_delta:+.2f} "
        f"({avg_a:.2f} -> {avg_b:.2f}) across {count_b} vs {count_a} responses."
    )

    return PeriodComparisonResult(
        period_a=period_a_metrics,
        period_b=period_b_metrics,
        count_delta=count_delta,
        average_rating_delta=avg_delta,
        csat_delta=csat_delta,
        summary=summary,
    )


def compute_nps(scores_or_records: list[Any]) -> float:
    """Compute Net Promoter Score (NPS) deterministically.

    Definition: % Promoters (9-10) - % Detractors (0-6). Range: -100 to +100.
    """
    if not scores_or_records:
        return 0.0

    scores: list[float] = []
    for item in scores_or_records:
        if isinstance(item, (int, float)):
            scores.append(float(item))
        elif isinstance(item, dict) and "nps_score" in item:
            scores.append(float(item["nps_score"]))
        elif hasattr(item, "nps_score"):
            scores.append(float(item.nps_score))

    if not scores:
        return 0.0

    total = len(scores)
    promoters = sum(1 for s in scores if s >= 9)
    detractors = sum(1 for s in scores if s <= 6)
    return round(((promoters - detractors) / total) * 100, 2)


def filter_surveys(
    records: list[dict[str, Any]],
    cohort: str | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Filter survey records deterministically by cohort or category."""
    if not records:
        return []
    filtered = records
    if cohort:
        filtered = [r for r in filtered if r.get("cohort", "").lower() == cohort.lower()]
    if category:
        filtered = [
            r for r in filtered
            if (r.get("category", "").lower() == category.lower() or r.get("theme", "").lower() == category.lower())
        ]
    return filtered
