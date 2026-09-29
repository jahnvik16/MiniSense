"""Unit tests for Pydantic structured agent contracts."""

import pytest
from pydantic import ValidationError
from app.models.schemas import (
    TaskSpec,
    DataAgentResult,
    RAGAgentResult,
    ComparisonAgentResult,
    FinalAnswer,
    PeriodMetrics,
    ThemeMetric,
)


def test_task_spec_contract() -> None:
    spec = TaskSpec(
        task_id="task_001",
        agent="DataAgent",
        task_type="data_analysis",
        question="What was our CSAT in April 2026?",
        start_date="2026-04-01",
        end_date="2026-04-30",
        comparison_start_date="2026-05-01",
        comparison_end_date="2026-05-31",
        parameters={"cohort": "enterprise", "metric": "csat"},
    )
    assert spec.task_id == "task_001"
    assert spec.agent == "DataAgent"
    assert spec.start_date == "2026-04-01"
    assert spec.parameters["cohort"] == "enterprise"

    # Validation: empty task_id / question
    with pytest.raises(ValidationError):
        TaskSpec(task_id="", agent="DataAgent", question="valid question")

    with pytest.raises(ValidationError):
        TaskSpec(task_id="t1", agent="DataAgent", question="   ")


def test_data_agent_result_contract() -> None:
    theme = ThemeMetric(
        theme="Wait Time",
        count=150,
        average_rating=4.2,
        csat=75.5,
        sentiment_breakdown={"positive": 110, "neutral": 25, "negative": 15},
    )

    result = DataAgentResult(
        response_count=150,
        average_rating=4.204,
        csat=75.501,
        top_themes=[theme],
        supporting_metadata={"cohort": "enterprise"},
    )
    # Verification of rounding validators
    assert result.response_count == 150
    assert result.average_rating == 4.2
    assert result.csat == 75.5
    assert len(result.top_themes) == 1
    assert result.top_themes[0].theme == "Wait Time"

    # Validation: CSAT out of bounds
    with pytest.raises(ValidationError):
        DataAgentResult(response_count=10, average_rating=3.0, csat=105.0)

    # Validation: average rating out of bounds
    with pytest.raises(ValidationError):
        DataAgentResult(response_count=10, average_rating=5.5, csat=50.0)


def test_rag_agent_result_contract() -> None:
    result = RAGAgentResult(
        query="What is the refund policy for downgrades?",
        retrieved_chunks=["## Q1: Refund Policy\nAll subscriptions..."],
        scores=[0.875],
        source_metadata=[{"chunk_id": "faq_chunk_1", "source": "data/faq.txt"}],
    )
    assert result.query == "What is the refund policy for downgrades?"
    assert len(result.retrieved_chunks) == 1
    assert result.scores[0] == 0.875
    assert result.source_metadata[0]["chunk_id"] == "faq_chunk_1"

    # Validation: mismatch between scores and retrieved_chunks count
    with pytest.raises(ValidationError):
        RAGAgentResult(
            query="test",
            retrieved_chunks=["chunk 1", "chunk 2"],
            scores=[0.9],  # only 1 score for 2 chunks
        )


def test_comparison_agent_result_contract() -> None:
    period_a = PeriodMetrics(
        period_label="April 2026",
        start_date="2026-04-01",
        end_date="2026-04-30",
        response_count=36886,
        average_rating=2.11,
        csat=13.47,
    )
    period_b = PeriodMetrics(
        period_label="May 2026",
        start_date="2026-05-01",
        end_date="2026-05-31",
        response_count=38114,
        average_rating=3.81,
        csat=65.84,
    )

    comp = ComparisonAgentResult(
        current_period=period_b,
        previous_period=period_a,
        metric_changes={
            "csat_delta": 52.37,
            "average_rating_delta": 1.70,
            "count_delta": 1228,
        },
    )
    assert comp.current_period.period_label == "May 2026"
    assert comp.previous_period.period_label == "April 2026"
    assert comp.metric_changes["csat_delta"] == 52.37


def test_final_answer_contract() -> None:
    answer = FinalAnswer(
        answer="In May 2026, Wait Time CSAT improved by 52.4% following the rollout of express pickup lanes.",
        supporting_metrics={"april_csat": 13.5, "may_csat": 65.8, "csat_delta": 52.4},
        retrieved_sources=["faq_chunk_3"],
        assumptions=["Analysis assumes 1-5 star scale where ratings >= 4 define satisfied responses."],
    )
    assert "improved by 52.4%" in answer.answer
    assert answer.supporting_metrics["csat_delta"] == 52.4
    assert "faq_chunk_3" in answer.retrieved_sources
    assert len(answer.assumptions) == 1

    # Validation: empty answer rejected
    with pytest.raises(ValidationError):
        FinalAnswer(answer="")
