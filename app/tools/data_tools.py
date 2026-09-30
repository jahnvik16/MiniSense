"""Deterministic numerical tools for survey analytics."""

import re
from datetime import date, datetime, timezone
from typing import Any, Union
from app.models.schemas import PeriodComparisonResult, PeriodMetrics, ThemeMetric

# 8 Controlled Synthetic Themes
THEMES = [
    "Food Quality",
    "Wait Time",
    "Staff",
    "Cleanliness",
    "Pricing",
    "Membership",
    "Facilities",
    "App Experience",
]

# Controlled Vocabulary Keywords for Deterministic Theme Classification
THEME_KEYWORDS: dict[str, list[str]] = {
    "Wait Time": [
        "wait", "waited", "waiting", "wait time", "turnaround", "line", "lines",
        "queue", "delay", "delays", "delayed", "slow", "lightning fast", "minutes",
        "mins", "express pickup", "counter wait", "kitchen delay", "long wait"
    ],
    "Food Quality": [
        "food", "taste", "tasted", "burger", "salad", "espresso", "toast", "avocado",
        "grain bowl", "panini", "smoothie", "matcha", "seasoning", "stale", "undercooked",
        "overcooked", "culinary", "flavor", "flavorful", "ingredients", "meal", "coffee",
        "freshly", "bland", "lukewarm", "delicious", "salty", "portion", "recipe", "dish"
    ],
    "Staff": [
        "staff", "cashier", "server", "hospitality", "attentive", "courteous", "unhelpful",
        "dismissive", "friendly", "rushed", "team", "clerk", "worker", "welcoming",
        "smiles", "service", "customer service", "manager", "barista"
    ],
    "Cleanliness": [
        "clean", "cleanliness", "dirty", "messy", "tables", "trash", "sanitation",
        "sanitized", "sticky", "hygiene", "tidy", "spotless", "restroom", "bathrooms",
        "maintenance", "unbussed"
    ],
    "Pricing": [
        "price", "prices", "pricing", "expensive", "bill", "cost", "value for money",
        "cheap", "affordable", "hike", "fee", "fees", "rates", "charge", "charged",
        "cost-to-benefit", "overpriced"
    ],
    "Membership": [
        "member", "members", "membership", "tier", "rewards", "loyalty", "perks",
        "points", "discount code", "vouchers", "vip", "portal", "credits"
    ],
    "Facilities": [
        "facility", "facilities", "amenity", "amenities", "ac", "air conditioning",
        "seating", "wi-fi", "wifi", "parking", "restroom", "room", "lighting",
        "ventilated", "decor", "cramped", "lounge", "patio", "workspaces"
    ],
    "App Experience": [
        "app", "mobile app", "crashed", "crash", "checkout", "glitch", "glitchy",
        "digital wallet", "notifications", "order tracking", "interface", "ui",
        "login", "loading spinner", "v2.0", "v1.8", "reorder"
    ],
}


# Compiled keyword-to-theme mapping sorted by phrase length descending
KEYWORD_TO_THEME: list[tuple[str, str]] = []
for theme_name, kw_list in THEME_KEYWORDS.items():
    for kw in kw_list:
        KEYWORD_TO_THEME.append((kw.lower(), theme_name))
KEYWORD_TO_THEME.sort(key=lambda x: -len(x[0]))


def derive_sentiment(rating: int | float | None) -> str:
    """Deterministically derive sentiment polarity from rating per specification.

    1-2 -> negative
    3   -> neutral
    4-5 -> positive
    """
    if rating is None:
        return "neutral"
    try:
        r = float(rating)
    except (ValueError, TypeError):
        return "neutral"

    if r <= 2.0:
        return "negative"
    elif r == 3.0:
        return "neutral"
    else:
        return "positive"


# Backwards compatibility alias
classify_sentiment = derive_sentiment


def normalize_theme_name(theme: str | None) -> str | None:
    """Normalize input theme expression to canonical controlled taxonomy.

    Strips any trailing brackets, braces, commas, or quotes and matches against
    the 8 controlled taxonomy themes:
      - Food Quality, Wait Time, Staff, Cleanliness, Pricing, Membership, Facilities, App Experience
    """
    if not theme or not str(theme).strip():
        return None
    cleaned = re.sub(r"[^\w\s-]", "", str(theme)).strip()
    if not cleaned:
        return None
    cleaned_lower = cleaned.lower()

    for canonical in THEMES:
        if cleaned_lower == canonical.lower():
            return canonical

    if "wait" in cleaned_lower or "pickup" in cleaned_lower or "queue" in cleaned_lower or "delay" in cleaned_lower:
        return "Wait Time"
    if "food" in cleaned_lower or "meal" in cleaned_lower or "taste" in cleaned_lower or "salad" in cleaned_lower or "burger" in cleaned_lower or "flavor" in cleaned_lower:
        return "Food Quality"
    if "clean" in cleaned_lower or "dirty" in cleaned_lower or "restroom" in cleaned_lower or "hygiene" in cleaned_lower or "sanit" in cleaned_lower:
        return "Cleanliness"
    if "price" in cleaned_lower or "pricing" in cleaned_lower or "expensive" in cleaned_lower or "cost" in cleaned_lower or "bill" in cleaned_lower:
        return "Pricing"
    if "staff" in cleaned_lower or "employee" in cleaned_lower or "service" in cleaned_lower or "clerk" in cleaned_lower or "cashier" in cleaned_lower:
        return "Staff"
    if "app" in cleaned_lower or "mobile" in cleaned_lower or "crash" in cleaned_lower or "glitch" in cleaned_lower or "digital" in cleaned_lower:
        return "App Experience"
    if "member" in cleaned_lower or "loyalty" in cleaned_lower or "rewards" in cleaned_lower or "points" in cleaned_lower:
        return "Membership"
    if "facilit" in cleaned_lower or "patio" in cleaned_lower or "parking" in cleaned_lower or "seating" in cleaned_lower or "lounge" in cleaned_lower:
        return "Facilities"

    return cleaned


def classify_theme(free_text: str | None) -> str:
    """Deterministically classify customer free text comment into domain themes.

    Applies exact rule-based keyword matching over the controlled synthetic vocabulary.
    Covers the 8 core operational themes:
    - Food Quality, Wait Time, Staff, Cleanliness, Pricing, Membership, Facilities, App Experience.
    """
    if not free_text or not str(free_text).strip():
        return "General"

    text_lower = f" {str(free_text).lower()} "
    scores: dict[str, int] = {}

    for kw, theme in KEYWORD_TO_THEME:
        if kw in text_lower:
            scores[theme] = scores.get(theme, 0) + 1

    if not scores:
        return "General"

    return max(scores.items(), key=lambda x: x[1])[0]


def classify_record_theme(record: dict[str, Any] | Any) -> str:
    """Classify the theme of a raw Appendix-A survey record from free_text.

    Caches the derived theme in-memory on the record dict (_theme) to avoid redundant scans.
    Never mutates persisted data files.
    """
    if isinstance(record, dict):
        if "_theme" in record:
            return record["_theme"]
        # Primary: Appendix A free_text
        free_text = record.get("free_text")
        if free_text:
            theme = classify_theme(free_text)
            record["_theme"] = theme
            return theme
        # Secondary fallback for legacy compatibility
        explicit = record.get("theme") or record.get("category") or record.get("feedback")
        theme = classify_theme(explicit) if explicit else "General"
        record["_theme"] = theme
        return theme

    if hasattr(record, "_theme") and record._theme:
        return record._theme
    text = getattr(record, "free_text", None) or getattr(record, "feedback", None)
    theme = classify_theme(text)
    try:
        setattr(record, "_theme", theme)
    except Exception:
        pass
    return theme


def normalize_appendix_a_record(record: dict[str, Any]) -> dict[str, Any]:
    """Runtime analytical normalization of a raw Appendix-A survey record.

    Derives theme from free_text and sentiment from rating without modifying persisted data.
    """
    rating = record.get("rating", 3)
    free_text = record.get("free_text", "")
    return {
        "response_id": record.get("response_id", ""),
        "date": str(record.get("date", "")),
        "business_id": record.get("business_id", ""),
        "business_name": record.get("business_name", ""),
        "survey_id": record.get("survey_id", ""),
        "survey_name": record.get("survey_name", ""),
        "rating": rating,
        "response_channel": record.get("response_channel", ""),
        "free_text": free_text,
        "derived_theme": classify_record_theme(record),
        "derived_sentiment": derive_sentiment(rating),
    }


def extract_themes(free_text: str | None) -> list[str]:
    """Extract all relevant themes mentioned in free text comment."""
    if not free_text or not str(free_text).strip():
        return ["General"]

    text_lower = f" {str(free_text).lower()} "
    matched = []
    for theme, keywords in THEME_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(theme)

    return matched if matched else ["General"]


def _extract_score(item: Any) -> float | None:
    """Helper to extract a numeric rating from an int, float, dict, or object."""
    if isinstance(item, (int, float)):
        return float(item)
    if isinstance(item, dict):
        if "rating" in item and item["rating"] is not None:
            try:
                return float(item["rating"])
            except (ValueError, TypeError):
                pass
        for key in ("rating", "score", "value"):
            if key in item and item[key] is not None:
                try:
                    return float(item[key])
                except (ValueError, TypeError):
                    pass
        return None
    if hasattr(item, "rating"):
        try:
            return float(item.rating)
        except (ValueError, TypeError):
            pass
    return None


def _parse_date(d: Any) -> date | None:
    """Parse date-only strings (YYYY-MM-DD), ISO datetimes, or dates into a date object."""
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        cleaned = d.strip()
        if not cleaned:
            return None
        # Handle standard YYYY-MM-DD
        if len(cleaned) >= 10 and cleaned[4] == "-" and cleaned[7] == "-":
            try:
                return datetime.strptime(cleaned[:10], "%Y-%m-%d").date()
            except ValueError:
                pass
        # Handle full ISO format
        try:
            return datetime.fromisoformat(cleaned.replace("Z", "+00:00")).date()
        except ValueError:
            pass
    return None


def filter_by_date(
    records: list[dict[str, Any]],
    start_date: str | datetime | date | None = None,
    end_date: str | datetime | date | None = None,
) -> list[dict[str, Any]]:
    """Filter survey records deterministically by inclusive date range.

    Operates primarily on the Appendix A 'date' field (YYYY-MM-DD).
    Does NOT depend on a 'timestamp' field.
    """
    if not records:
        return []

    start_d = _parse_date(start_date)
    end_d = _parse_date(end_date)

    if start_d is None and end_d is None:
        return records

    filtered: list[dict[str, Any]] = []
    for r in records:
        # Primary: Appendix A 'date' field
        raw_date = r.get("date") if isinstance(r, dict) else getattr(r, "date", None)
        if not raw_date:
            # Secondary fallback for legacy payloads
            raw_date = r.get("timestamp") if isinstance(r, dict) else getattr(r, "timestamp", None)

        record_d = _parse_date(raw_date)
        if record_d is None:
            continue
        if start_d and record_d < start_d:
            continue
        if end_d and record_d > end_d:
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
    """Compute frequency counts for survey sentiments (positive, neutral, negative).

    Derives sentiment from rating deterministically on raw Appendix-A records.
    """
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    if not sentiments_or_records:
        return counts

    for item in sentiments_or_records:
        sentiment_val = ""
        if isinstance(item, str):
            sentiment_val = item
        elif isinstance(item, dict):
            # Primary: derive sentiment from rating (Appendix A)
            if "rating" in item and item["rating"] is not None:
                sentiment_val = derive_sentiment(item["rating"])
            elif "sentiment" in item:
                sentiment_val = item["sentiment"]
        elif hasattr(item, "rating"):
            sentiment_val = derive_sentiment(item.rating)
        elif hasattr(item, "sentiment"):
            sentiment_val = item.sentiment

        norm = (sentiment_val or "").lower().strip()
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

    Derives themes from free_text deterministically when analyzing raw Appendix-A records.
    Never collapses all records into 'General' merely because the old 'theme' field is absent.
    Supported sorting metrics:
      - 'volume': highest count first (default)
      - 'csat_asc' / 'worst_csat': lowest CSAT first (complaint driver analysis)
      - 'csat_desc' / 'best_csat': highest CSAT first
      - 'negative_volume': highest negative sentiment count first
    """
    if not records:
        return []

    # Group by theme (derived dynamically from free_text using in-memory cache)
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        theme = classify_record_theme(r)
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
    elif metric_lower in ("negative_volume", "negative", "complaint", "complaints"):
        metrics_list.sort(
            key=lambda x: (-x.sentiment_breakdown.get("negative", 0), x.csat, x.theme)
        )
    else:  # default 'volume'
        metrics_list.sort(key=lambda x: (-x.count, -x.csat, x.theme))

    return metrics_list[:top_n]


def compare_period_metrics(
    records: list[dict[str, Any]],
    period_a_start: str | datetime | date | None,
    period_a_end: str | datetime | date | None,
    period_b_start: str | datetime | date | None,
    period_b_end: str | datetime | date | None,
    theme: str | None = None,
    response_channel: str | None = None,
    business_id: str | None = None,
    channel: str | None = None,
    cohort: str | None = None,
    label_a: str = "Period A",
    label_b: str = "Period B",
) -> PeriodComparisonResult:
    """Compare performance metrics between two periods with delta analysis on raw Appendix-A records."""
    pool = records

    # 1. Theme filtering using runtime classifier over free_text
    norm_theme = normalize_theme_name(theme)
    if norm_theme:
        target_lower = norm_theme.lower()
        pool = [
            r for r in pool
            if classify_record_theme(r).lower() == target_lower
        ]

    # 2. Channel filtering using Appendix A response_channel
    target_channel = (response_channel or channel or cohort or "").lower().strip()
    if target_channel:
        pool = [
            r for r in pool
            if (
                str(r.get("response_channel", "")).lower() == target_channel
                or str(r.get("channel", "")).lower() == target_channel
                or str(r.get("cohort", "")).lower() == target_channel
            )
        ]

    # 3. Location filtering using Appendix A business_id
    if business_id:
        target_biz = business_id.lower().strip()
        pool = [
            r for r in pool
            if str(r.get("business_id", "")).lower() == target_biz
        ]

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
        f"({avg_a:.2f} -> {avg_b:.2f}) across {count_b:,} vs {count_a:,} responses."
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
    theme: str | None = None,
    channel: str | None = None,
    response_channel: str | None = None,
    business_id: str | None = None,
    business_name: str | None = None,
    survey_id: str | None = None,
    survey_name: str | None = None,
) -> list[dict[str, Any]]:
    """Filter survey records deterministically by Appendix A fields or runtime derived themes.

    Primary Appendix A filters:
      - response_channel / channel
      - business_id
      - business_name
      - survey_id
      - survey_name
      - theme / category (evaluated dynamically via free_text runtime derivation)
    """
    if not records:
        return []
    filtered = records

    norm_theme = normalize_theme_name(theme or category)
    if norm_theme:
        target_lower = norm_theme.lower()
        filtered = [
            r for r in filtered
            if classify_record_theme(r).lower() == target_lower
        ]

    target_chan = (response_channel or channel or cohort or "").lower().strip()
    if target_chan:
        filtered = [
            r for r in filtered
            if (
                str(r.get("response_channel", "")).lower() == target_chan
                or str(r.get("channel", "")).lower() == target_chan
                or str(r.get("cohort", "")).lower() == target_chan
            )
        ]

    if business_id:
        target_biz = business_id.lower().strip()
        filtered = [
            r for r in filtered
            if str(r.get("business_id", "")).lower() == target_biz
        ]

    if business_name:
        target_biz_name = business_name.lower().strip()
        filtered = [
            r for r in filtered
            if str(r.get("business_name", "")).lower() == target_biz_name
        ]

    if survey_id:
        target_survey_id = survey_id.lower().strip()
        filtered = [
            r for r in filtered
            if str(r.get("survey_id", "")).lower() == target_survey_id
        ]

    if survey_name:
        target_survey_name = survey_name.lower().strip()
        filtered = [
            r for r in filtered
            if str(r.get("survey_name", "")).lower() == target_survey_name
        ]

    return filtered


# Explicit Data Tools Specification Registry for Agentic Execution
DATA_AGENT_TOOLS = [
    {
        "name": "compute_csat",
        "description": "Calculates Customer Satisfaction (CSAT) percentage deterministically as the percentage of survey ratings >= 4 on a 1-5 scale.",
        "parameters": {
            "type": "object",
            "properties": {
                "records_or_scores": {"type": "array", "description": "List of survey records or integer ratings"}
            },
            "required": ["records_or_scores"]
        }
    },
    {
        "name": "compute_average_rating",
        "description": "Calculates the arithmetic mean rating (1.0 to 5.0) deterministically across survey records.",
        "parameters": {
            "type": "object",
            "properties": {
                "records_or_scores": {"type": "array", "description": "List of survey records or integer ratings"}
            },
            "required": ["records_or_scores"]
        }
    },
    {
        "name": "count_responses",
        "description": "Calculates total sample size and response count deterministically.",
        "parameters": {
            "type": "object",
            "properties": {
                "records": {"type": "array", "description": "List of survey records"}
            },
            "required": ["records"]
        }
    },
    {
        "name": "get_top_themes",
        "description": "Extracts aggregated theme metrics ranked deterministically by volume, lowest CSAT, or complaint frequency.",
        "parameters": {
            "type": "object",
            "properties": {
                "records": {"type": "array", "description": "Survey records to group and rank"},
                "top_n": {"type": "integer", "description": "Number of ranked themes to return"},
                "metric": {"type": "string", "enum": ["volume", "negative_volume", "worst_csat", "best_csat"]}
            },
            "required": ["records"]
        }
    },
    {
        "name": "filter_by_date",
        "description": "Filters survey records deterministically by an inclusive start and end date range.",
        "parameters": {
            "type": "object",
            "properties": {
                "records": {"type": "array", "description": "Raw survey records"},
                "start_date": {"type": "string", "description": "Inclusive start date (YYYY-MM-DD)"},
                "end_date": {"type": "string", "description": "Inclusive end date (YYYY-MM-DD)"}
            },
            "required": ["records"]
        }
    },
    {
        "name": "filter_surveys",
        "description": "Filters survey records deterministically by theme, channel, cohort, or location.",
        "parameters": {
            "type": "object",
            "properties": {
                "records": {"type": "array", "description": "Survey records"},
                "theme": {"type": "string", "description": "Operational theme filter"},
                "channel": {"type": "string", "description": "Response channel filter"}
            },
            "required": ["records"]
        }
    }
]
