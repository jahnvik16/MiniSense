"""Unit tests for ComparisonAgent behavior, delta calculation, and zero denominator safety."""

import pytest
from app.agents.comparison_agent import ComparisonAgent
from app.models.schemas import ComparisonAgentResult, TaskSpec, AgentType, TaskType


@pytest.fixture
def comp_agent() -> ComparisonAgent:
    return ComparisonAgent()


def test_comparison_agent_month_over_month_wait_time(comp_agent: ComparisonAgent) -> None:
    task = TaskSpec(
        task_id="task_comp_mom",
        agent=AgentType.COMPARISON_AGENT.value,
        task_type=TaskType.PERIOD_COMPARISON.value,
        question="How did Wait Time performance compare between May 2026 and April 2026?",
        start_date="2026-05-01",
        end_date="2026-05-31",
        comparison_start_date="2026-04-01",
        comparison_end_date="2026-04-30",
        parameters={
            "theme": "Wait Time",
            "current_label": "May 2026",
            "previous_label": "April 2026",
        },
    )
    result = comp_agent.run(task)

    assert isinstance(result, ComparisonAgentResult)
    assert result.current_period.period_label == "May 2026"
    assert result.previous_period.period_label == "April 2026"

    # Both periods should have ~4,500 - 4,800 responses for Wait Time
    assert result.current_period.response_count > 4000
    assert result.previous_period.response_count > 4000

    # April Wait Time had ~13.5% CSAT; May jumped to ~65.8% CSAT
    assert result.previous_period.csat < 20.0
    assert result.current_period.csat > 60.0

    # Verify deterministic delta calculations
    changes = result.metric_changes
    assert changes["csat_delta"] > 40.0
    assert changes["average_rating_delta"] > 1.5
    assert "improved by" in result.summary


def test_comparison_agent_app_experience_shift(comp_agent: ComparisonAgent) -> None:
    task = TaskSpec(
        task_id="task_comp_app",
        agent=AgentType.COMPARISON_AGENT.value,
        task_type=TaskType.PERIOD_COMPARISON.value,
        question="Compare App Experience metrics between May and April 2026",
        start_date="2026-05-01",
        end_date="2026-05-31",
        comparison_start_date="2026-04-01",
        comparison_end_date="2026-04-30",
        parameters={
            "theme": "App Experience",
            "current_label": "May 2026",
            "previous_label": "April 2026",
        },
    )
    result = comp_agent.run(task)

    assert result.metric_changes["csat_delta"] > 40.0
    assert result.current_period.average_rating > result.previous_period.average_rating


def test_comparison_agent_zero_denominator_safety(comp_agent: ComparisonAgent) -> None:
    # Compare with a future date window with 0 records
    task = TaskSpec(
        task_id="task_comp_empty",
        agent=AgentType.COMPARISON_AGENT.value,
        task_type=TaskType.PERIOD_COMPARISON.value,
        question="Compare May 2026 with a future empty period",
        start_date="2026-05-01",
        end_date="2026-05-31",
        comparison_start_date="2026-07-01",
        comparison_end_date="2026-07-31",
        parameters={
            "current_label": "May 2026",
            "previous_label": "July 2026",
        },
    )
    # Must not raise ZeroDivisionError
    result = comp_agent.run(task)

    assert isinstance(result, ComparisonAgentResult)
    assert result.previous_period.response_count == 0
    assert result.previous_period.csat == 0.0
    assert result.metric_changes["csat_pct_change"] == 0.0
    assert result.metric_changes["count_pct_change"] == 0.0


def test_comparison_agent_cohort_comparison(comp_agent: ComparisonAgent) -> None:
    task = TaskSpec(
        task_id="task_comp_cohorts",
        agent=AgentType.COMPARISON_AGENT.value,
        task_type=TaskType.COHORT_COMPARISON.value,
        question="Compare mobile vs web channel satisfaction",
        parameters={
            "cohort_a": "mobile",
            "cohort_b": "web",
        },
    )
    result = comp_agent.run(task)

    assert isinstance(result, ComparisonAgentResult)
    assert result.current_period.period_label == "mobile"
    assert result.previous_period.period_label == "web"
    assert "csat_delta" in result.metric_changes
    assert len(result.current_period.top_themes) > 0
