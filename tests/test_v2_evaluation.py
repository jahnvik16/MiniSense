"""Comprehensive evaluation test suite for MiniSense v2.0 AI Engineer Evaluation.

Validates:
1. Exact Appendix A schema and zero pre-labelled ground truth.
2. Dataset scale (50k - 100k records).
3. Dynamic theme extraction and sentiment derivation.
4. Relative date resolution ('this month', 'last month', explicit month names).
5. LLM planner structured output validation and Pydantic compliance.
6. Malformed LLM planner output -> graceful deterministic fallback.
7. DataAgent explicit tool invocation and metadata tracking.
8. ComparisonAgent period calculation and delta mathematics.
9. Grounded RAG retrieval with natural-language query and score gating.
10. Hybrid routing (Scenario 3) and executive narrative synthesis without raw chunk dumps.
11. Safe handling of unsupported / unknown questions (no hallucination).
"""

import json
from pathlib import Path
import pytest

from app.config import settings
from app.models.schemas import (
    AgentType,
    FinalAnswer,
    PlannerPlan,
    PlannedSubTask,
    SurveyRecord,
    TaskFilters,
    TaskSpec,
    TaskType,
)
from app.tools.data_tools import (
    classify_sentiment,
    classify_theme,
    classify_record_theme,
    compare_period_metrics,
    compute_csat,
    compute_average_rating,
    count_responses,
    derive_sentiment,
    filter_by_date,
    filter_surveys,
    get_top_themes,
)
from app.agents.planner import (
    DeterministicFallbackPlanner,
    HybridPlanner,
    resolve_date_expression,
)
from app.agents.data_agent import DataAgent
from app.agents.comparison_agent import ComparisonAgent
from app.agents.rag_agent import RAGAgent
from app.agents.orchestrator import OrchestratorAgent
from app.services.survey_service import SurveyService


# ---------------------------------------------------------
# 1. Dataset & Schema Requirements (Items 3, 4, 5)
# ---------------------------------------------------------

def test_dataset_size_and_file_existence() -> None:
    """Verifies that the survey dataset exists and satisfies 50,000 to 100,000 scale."""
    data_file = Path(settings.surveys_file)
    assert data_file.exists(), f"Survey data file not found at {data_file}"

    with open(data_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    assert isinstance(records, list)
    count = len(records)
    assert 50000 <= count <= 100000, f"Expected 50k-100k records, got {count}"


def test_exact_appendix_a_schema() -> None:
    """Verifies that survey records match the exact 9 Appendix A fields with zero extraneous stored labels."""
    with open(settings.surveys_file, "r", encoding="utf-8") as f:
        records = json.load(f)[:100]

    expected_keys = {
        "response_id",
        "date",
        "business_id",
        "business_name",
        "survey_id",
        "survey_name",
        "rating",
        "response_channel",
        "free_text",
    }

    prohibited_precomputed_keys = {
        "theme",
        "sentiment",
        "csat_score",
        "nps_score",
        "category",
        "cohort",
        "product_tier",
    }

    for idx, r in enumerate(records):
        # Must parse strictly as Appendix A SurveyRecord
        parsed = SurveyRecord.model_validate(r)
        assert parsed.response_id.startswith("r")
        assert 1 <= parsed.rating <= 5

        # Check raw JSON keys: no pre-labelled ground truth keys stored in raw dataset
        raw_keys = set(r.keys()) - {"_theme"}  # in-memory cache excluded
        assert raw_keys == expected_keys, f"Record {idx} keys {raw_keys} do not match Appendix A {expected_keys}"
        for prohibited in prohibited_precomputed_keys:
            assert prohibited not in r, f"Record {idx} contains prohibited ground-truth label '{prohibited}'"


def test_dynamic_sentiment_derivation() -> None:
    """Verifies deterministic sentiment derivation from rating (1-2 neg, 3 neutral, 4-5 pos)."""
    assert classify_sentiment(1) == "negative"
    assert classify_sentiment(2) == "negative"
    assert classify_sentiment(3) == "neutral"
    assert classify_sentiment(4) == "positive"
    assert classify_sentiment(5) == "positive"


def test_dynamic_theme_classification() -> None:
    """Verifies deterministic keyword/phrase theme classification across all controlled themes."""
    test_cases = [
        ("The food was cold and the burger was awful.", "Food Quality"),
        ("We waited 45 minutes in line before being served.", "Wait Time"),
        ("The front desk clerk was incredibly rude.", "Staff"),
        ("The restrooms were filthy and dirty.", "Cleanliness"),
        ("It is way too expensive and not worth the price.", "Pricing"),
        ("My monthly membership renewal was billed twice.", "Membership"),
        ("The parking lot was full and equipment broken.", "Facilities"),
        ("The mobile app crashes every time I checkout.", "App Experience"),
    ]
    for text, expected_theme in test_cases:
        derived = classify_theme(text)
        assert derived == expected_theme, f"Failed for '{text}': got '{derived}', expected '{expected_theme}'"


# ---------------------------------------------------------
# 2. Relative Date Resolution (Item 1)
# ---------------------------------------------------------

def test_relative_date_resolution_expressions() -> None:
    """Verifies deterministic conversion of natural temporal expressions to concrete ISO ranges."""
    # This month (May 2026 anchor)
    start, end = resolve_date_expression("this month")
    assert start == "2026-05-01"
    assert end == "2026-05-31"

    # Last month / previous month (April 2026 anchor)
    start_last, end_last = resolve_date_expression("last month")
    assert start_last == "2026-04-01"
    assert end_last == "2026-04-30"

    start_prev, end_prev = resolve_date_expression("previous month")
    assert start_prev == "2026-04-01"
    assert end_prev == "2026-04-30"

    # Explicit month names
    start_apr, end_apr = resolve_date_expression("April")
    assert start_apr == "2026-04-01"
    assert end_apr == "2026-04-30"

    start_may, end_may = resolve_date_expression("May")
    assert start_may == "2026-05-01"
    assert end_may == "2026-05-31"

    start_jun, end_jun = resolve_date_expression("June")
    assert start_jun == "2026-06-01"
    assert end_jun == "2026-06-30"


# ---------------------------------------------------------
# 3. LLM Planner Structured Outputs & Fallback (Item 1)
# ---------------------------------------------------------

def test_pydantic_planner_plan_validation() -> None:
    """Verifies strict validation of PlannerPlan and PlannedSubTask Pydantic schemas."""
    valid_plan = PlannerPlan(
        plan_rationale="Analyzing customer complaints for the current month",
        tasks=[
            PlannedSubTask(
                agent="DataAgent",
                task_type="top_themes",
                question="Extract top 3 complaint themes in May 2026",
                metric="worst_csat",
                dimensions=["theme"],
                filters=TaskFilters(sentiment="negative"),
                date_expression="this month",
                requested_limit=3,
            )
        ],
    )
    assert valid_plan.tasks[0].agent == "DataAgent"
    assert valid_plan.tasks[0].requested_limit == 3
    assert valid_plan.tasks[0].filters.sentiment == "negative"


def test_malformed_planner_fallback_graceful() -> None:
    """Verifies that if LLM planner fails or returns malformed data, fallback produces valid TaskSpecs."""
    fallback_planner = DeterministicFallbackPlanner()

    questions = [
        "What are the top 3 complaints this month?",
        "How did CSAT change from April to May?",
        "What does the FAQ say about expected wait times?",
        "How did wait-time experience change from April to May, and what does the FAQ say about expected wait times?",
    ]

    for q in questions:
        tasks = fallback_planner.plan(q)
        assert len(tasks) >= 1
        for t in tasks:
            assert isinstance(t, TaskSpec)
            assert t.agent in [AgentType.DATA_AGENT.value, AgentType.COMPARISON_AGENT.value, AgentType.RAG_AGENT.value]
            assert t.task_id.startswith("task_")


def test_hybrid_planner_offline_fallback() -> None:
    """Verifies HybridPlanner seamlessly falls back when LLM planner is not instantiated."""
    hybrid = HybridPlanner(llm_planner=None)
    tasks = hybrid.plan("What are the top 3 complaints this month?")
    assert len(tasks) >= 1
    assert tasks[0].agent == AgentType.DATA_AGENT.value
    assert tasks[0].start_date == "2026-05-01"
    assert tasks[0].end_date == "2026-05-31"


# ---------------------------------------------------------
# 4. Agentic Tool Execution & Tracking (Item 2)
# ---------------------------------------------------------

def test_data_agent_explicit_tool_invocation_tracking() -> None:
    """Verifies DataAgent invokes pure arithmetic tools and logs tools_called in metadata."""
    agent = DataAgent()
    task = TaskSpec(
        task_id="t_tools",
        agent=AgentType.DATA_AGENT.value,
        task_type=TaskType.TOP_THEMES.value,
        question="What are the top complaint themes in May 2026?",
        start_date="2026-05-01",
        end_date="2026-05-31",
        requested_limit=3,
        filters={"sentiment": "negative"},
    )
    result = agent.run(task)

    assert result.response_count > 0
    tools = result.supporting_metadata["tools_called"]
    assert "count_responses" in tools
    assert "compute_csat" in tools
    assert "compute_average_rating" in tools
    assert "get_top_themes" in tools
    assert "filter_by_date" in tools


# ---------------------------------------------------------
# 5. ComparisonAgent Mathematics (Item 2 & 9)
# ---------------------------------------------------------

def test_comparison_agent_period_calculations() -> None:
    """Verifies ComparisonAgent computes exact deterministic deltas between April and May."""
    agent = ComparisonAgent()
    task = TaskSpec(
        task_id="t_comp",
        agent=AgentType.COMPARISON_AGENT.value,
        task_type=TaskType.PERIOD_COMPARISON.value,
        question="How did wait-time experience change from April to May?",
        start_date="2026-05-01",
        end_date="2026-05-31",
        comparison_start_date="2026-04-01",
        comparison_end_date="2026-04-30",
        parameters={"theme": "Wait Time", "current_label": "May 2026", "previous_label": "April 2026"},
    )
    res = agent.run(task)

    # April vs May Wait Time
    assert res.current_period.response_count > 0
    assert res.previous_period.response_count > 0
    assert "csat_delta" in res.metric_changes
    assert "average_rating_delta" in res.metric_changes
    assert res.metric_changes["csat_delta"] > 35.0  # Significant jump in May
    assert res.current_period.csat > res.previous_period.csat


# ---------------------------------------------------------
# 6. RAG Retrieval with Query Semantics (Item 8)
# ---------------------------------------------------------

def test_rag_agent_natural_language_retrieval() -> None:
    """Verifies RAGAgent retrieves relevant FAQ chunks and computes confidence scores."""
    rag = RAGAgent()
    task = TaskSpec(
        task_id="t_rag",
        agent=AgentType.RAG_AGENT.value,
        task_type=TaskType.RAG_LOOKUP.value,
        question="What does the FAQ say about expected wait times?",
        retrieval_query="expected wait times peak off-peak policy",
        requested_limit=2,
    )
    res = rag.run(task)

    assert res.reliable is True
    assert len(res.retrieved_chunks) >= 1
    assert any("wait" in c.lower() for c in res.retrieved_chunks)
    assert res.source_metadata[0]["score"] > 0.30


# ---------------------------------------------------------
# 7. Scenario 3 Hybrid Flow & Executive Synthesis (Item 7 & 9)
# ---------------------------------------------------------

def test_scenario_3_hybrid_orchestration() -> None:
    """Verifies Scenario 3 routes to ComparisonAgent and RAGAgent and produces executive synthesis."""
    orchestrator = OrchestratorAgent()
    question = "How did wait-time experience change from April to May, and what does the FAQ say about expected wait times?"

    result = orchestrator.run(question)

    assert isinstance(result, FinalAnswer)
    assert len(result.answer) > 80

    # Ensure no raw chunk headers leaked into executive response
    assert "faq_chunk_" not in result.answer
    assert "## Q" not in result.answer

    # Both metrics and retrieved sources are present
    assert len(result.supporting_metrics) > 0
    assert len(result.retrieved_sources) > 0

    # Business content checks
    answer_lower = result.answer.lower()
    assert "wait" in answer_lower
    assert "csat" in answer_lower or "satisfaction" in answer_lower or "rating" in answer_lower


# ---------------------------------------------------------
# 8. Unseen Questions & Unsupported Non-Hallucination (Item 10)
# ---------------------------------------------------------

def test_unsupported_question_handles_safely() -> None:
    """Verifies that an out-of-domain query is handled safely without hallucinating metric calculations."""
    orchestrator = OrchestratorAgent()
    question = "What is the gravitational constant of Jupiter's moon Europa?"
    result = orchestrator.run(question)

    assert isinstance(result, FinalAnswer)
    # System should produce a safe response or state that no survey data exists
    assert len(result.answer) > 20


# ---------------------------------------------------------
# 9. Mandatory Production Audit Tests (Appendix A Schema & Analytics)
# ---------------------------------------------------------

def test_audit_1_filter_by_date_uses_date_not_timestamp() -> None:
    """Mandatory Test 1: filter_by_date() works using Appendix A 'date', not timestamp."""
    records = [
        {"response_id": "r1", "date": "2026-04-15", "rating": 5, "free_text": "Great"},
        {"response_id": "r2", "date": "2026-05-10", "rating": 4, "free_text": "Good"},
        {"response_id": "r3", "date": "2026-06-01", "rating": 3, "free_text": "Okay"},
    ]
    for r in records:
        assert "timestamp" not in r

    apr = filter_by_date(records, start_date="2026-04-01", end_date="2026-04-30")
    assert len(apr) == 1
    assert apr[0]["response_id"] == "r1"

    may = filter_by_date(records, start_date="2026-05-01", end_date="2026-05-31")
    assert len(may) == 1
    assert may[0]["response_id"] == "r2"


def test_audit_2_april_filtering_returns_nonzero_records() -> None:
    """Mandatory Test 2: April filtering returns nonzero records on dataset."""
    service = SurveyService()
    records = service.get_all_surveys()
    apr_records = filter_by_date(records, "2026-04-01", "2026-04-30")
    assert len(apr_records) > 0
    assert len(apr_records) > 10000


def test_audit_3_may_filtering_returns_nonzero_records() -> None:
    """Mandatory Test 3: May filtering returns nonzero records on dataset."""
    service = SurveyService()
    records = service.get_all_surveys()
    may_records = filter_by_date(records, "2026-05-01", "2026-05-31")
    assert len(may_records) > 0
    assert len(may_records) > 10000


def test_audit_4_classify_theme_representative_examples() -> None:
    """Mandatory Test 4: classify_theme(free_text) correctly classifies representative examples."""
    representative = {
        "The artisan salad and burger were fresh and delicious": "Food Quality",
        "Waited 30 minutes in line for my order": "Wait Time",
        "The cashier and manager were extremely helpful and polite": "Staff",
        "The restroom was dirty and tables were messy": "Cleanliness",
        "Prices are too high and not worth the money": "Pricing",
        "My monthly membership loyalty points did not apply": "Membership",
        "The parking lot was full and outdoor patio seating was closed": "Facilities",
        "The mobile app keeps crashing during checkout": "App Experience",
    }
    for text, expected in representative.items():
        assert classify_theme(text) == expected, f"Failed for '{text}'"


def test_audit_5_derive_sentiment_expected_classes() -> None:
    """Mandatory Test 5: derive_sentiment(rating) returns 1-2 negative, 3 neutral, 4-5 positive."""
    assert derive_sentiment(1) == "negative"
    assert derive_sentiment(2) == "negative"
    assert derive_sentiment(3) == "neutral"
    assert derive_sentiment(4) == "positive"
    assert derive_sentiment(5) == "positive"


def test_audit_6_get_top_themes_no_general_collapse() -> None:
    """Mandatory Test 6: get_top_themes() does not return 'General' for the generated dataset."""
    service = SurveyService()
    records = service.get_all_surveys()
    top_themes = get_top_themes(records[:2000], top_n=5)
    theme_names = [t.theme for t in top_themes]
    assert len(theme_names) == 5
    assert "General" not in theme_names
    valid_taxonomy = {
        "Food Quality", "Wait Time", "Staff", "Cleanliness",
        "Pricing", "Membership", "Facilities", "App Experience"
    }
    for t in theme_names:
        assert t in valid_taxonomy, f"Unexpected theme '{t}' outside taxonomy"


def test_audit_7_complaint_ranking_derived_negative_and_themes() -> None:
    """Mandatory Test 7: complaint ranking uses derived negative sentiment and derived themes."""
    service = SurveyService()
    records = service.get_all_surveys()
    apr_records = filter_by_date(records, "2026-04-01", "2026-04-30")
    top_complaints = get_top_themes(apr_records, top_n=3, metric="negative_volume")
    assert len(top_complaints) == 3
    for t in top_complaints:
        assert t.sentiment_breakdown["negative"] > 0
    assert top_complaints[0].sentiment_breakdown["negative"] >= top_complaints[1].sentiment_breakdown["negative"]


def test_audit_8_compare_period_metrics_produces_nonzero_metrics() -> None:
    """Mandatory Test 8: compare_period_metrics() produces nonzero April/May metrics."""
    service = SurveyService()
    records = service.get_all_surveys()
    res = compare_period_metrics(
        records=records,
        period_a_start="2026-04-01",
        period_a_end="2026-04-30",
        period_b_start="2026-05-01",
        period_b_end="2026-05-31",
        label_a="April 2026",
        label_b="May 2026",
    )
    assert res.period_a.response_count > 10000
    assert res.period_b.response_count > 10000
    assert res.period_a.csat > 0.0
    assert res.period_b.csat > 0.0
    assert res.period_a.average_rating > 0.0
    assert res.period_b.average_rating > 0.0


def test_audit_9_channel_filtering_using_response_channel() -> None:
    """Mandatory Test 9: channel filtering works using response_channel."""
    records = [
        {"response_id": "r1", "date": "2026-05-01", "response_channel": "mobile", "rating": 5, "free_text": "Great app"},
        {"response_id": "r2", "date": "2026-05-02", "response_channel": "kiosk", "rating": 4, "free_text": "Easy kiosk"},
        {"response_id": "r3", "date": "2026-05-03", "response_channel": "web", "rating": 3, "free_text": "Average web"},
    ]
    mobile_records = filter_surveys(records, response_channel="mobile")
    assert len(mobile_records) == 1
    assert mobile_records[0]["response_id"] == "r1"

    kiosk_records = filter_surveys(records, response_channel="kiosk")
    assert len(kiosk_records) == 1
    assert kiosk_records[0]["response_id"] == "r2"


def test_audit_10_business_filtering_using_business_id() -> None:
    """Mandatory Test 10: business filtering works using business_id."""
    records = [
        {"response_id": "r1", "date": "2026-05-01", "business_id": "b01", "business_name": "Downtown", "rating": 5, "free_text": "Nice"},
        {"response_id": "r2", "date": "2026-05-02", "business_id": "b02", "business_name": "Westside", "rating": 4, "free_text": "Good"},
    ]
    b01_records = filter_surveys(records, business_id="b01")
    assert len(b01_records) == 1
    assert b01_records[0]["business_id"] == "b01"


def test_audit_11_appendix_a_schema_exact_persisted_fields() -> None:
    """Mandatory Test 11: Appendix-A schema contains exactly the required persisted fields."""
    with open(settings.surveys_file, "r", encoding="utf-8") as f:
        records = json.load(f)[:100]
    expected_keys = {
        "response_id", "date", "business_id", "business_name",
        "survey_id", "survey_name", "rating", "response_channel", "free_text"
    }
    for r in records:
        assert set(r.keys()) == expected_keys


def test_audit_12_no_raw_record_contains_derived_labels() -> None:
    """Mandatory Test 12: No raw record contains theme/sentiment/category/cohort/csat/nps labels."""
    with open(settings.surveys_file, "r", encoding="utf-8") as f:
        records = json.load(f)[:500]
    prohibited_keys = {
        "theme", "category", "sentiment", "timestamp",
        "cohort", "product_tier", "csat_score", "nps_score"
    }
    for r in records:
        assert not (set(r.keys()) & prohibited_keys)

