"""Unit tests for DataAgent behavior and structured output validation."""

import pytest
from app.agents.data_agent import DataAgent
from app.models.schemas import DataAgentResult, TaskSpec, AgentType, TaskType


@pytest.fixture
def data_agent() -> DataAgent:
    return DataAgent()


def test_data_agent_overall_metrics(data_agent: DataAgent) -> None:
    task = TaskSpec(
        task_id="task_overall",
        agent=AgentType.DATA_AGENT.value,
        task_type=TaskType.DATA_ANALYSIS.value,
        question="What are the overall survey metrics across all customers?",
    )
    result = data_agent.run(task)

    assert isinstance(result, DataAgentResult)
    assert result.response_count == 75000
    assert 1.0 <= result.average_rating <= 5.0
    assert 0.0 <= result.csat <= 100.0
    assert len(result.top_themes) > 0

    # Verify explicit tool calling tracking
    tools_called = result.supporting_metadata["tools_called"]
    assert "compute_csat" in tools_called
    assert "compute_average_rating" in tools_called
    assert "count_responses" in tools_called
    assert "get_top_themes" in tools_called


def test_data_agent_top_complaint_themes(data_agent: DataAgent) -> None:
    task = TaskSpec(
        task_id="task_complaints",
        agent=AgentType.DATA_AGENT.value,
        task_type=TaskType.DATA_ANALYSIS.value,
        question="What are our top complaint themes and lowest CSAT areas?",
        parameters={"top_n": 3},
    )
    result = data_agent.run(task)

    assert isinstance(result, DataAgentResult)
    assert len(result.top_themes) == 3
    assert result.supporting_metadata["theme_ranking_strategy"] == "worst_csat"

    # Themes should be ordered by lowest CSAT first
    assert result.top_themes[0].csat <= result.top_themes[1].csat
    assert result.top_themes[1].csat <= result.top_themes[2].csat


def test_data_agent_date_range_filtering(data_agent: DataAgent) -> None:
    # April 2026
    task_apr = TaskSpec(
        task_id="task_apr",
        agent=AgentType.DATA_AGENT.value,
        task_type=TaskType.DATA_ANALYSIS.value,
        question="What were the metrics for April 2026?",
        start_date="2026-04-01",
        end_date="2026-04-30",
    )
    result_apr = data_agent.run(task_apr)
    assert result_apr.response_count == 36886
    assert "filter_by_date" in result_apr.supporting_metadata["tools_called"]

    # May 2026
    task_may = TaskSpec(
        task_id="task_may",
        agent=AgentType.DATA_AGENT.value,
        task_type=TaskType.DATA_ANALYSIS.value,
        question="What were the metrics for May 2026?",
        start_date="2026-05-01",
        end_date="2026-05-31",
    )
    result_may = data_agent.run(task_may)
    assert result_may.response_count == 38114
    assert (result_apr.response_count + result_may.response_count) == 75000


def test_data_agent_cohort_filtering(data_agent: DataAgent) -> None:
    task = TaskSpec(
        task_id="task_enterprise",
        agent=AgentType.DATA_AGENT.value,
        task_type=TaskType.DATA_ANALYSIS.value,
        question="What is the CSAT for enterprise customers?",
        parameters={"cohort": "enterprise"},
    )
    result = data_agent.run(task)

    assert isinstance(result, DataAgentResult)
    assert result.response_count > 10000
    assert result.supporting_metadata["filters_applied"]["cohort"] == "enterprise"
    assert "filter_surveys" in result.supporting_metadata["tools_called"]


def test_data_agent_graceful_invalid_date_handling(data_agent: DataAgent) -> None:
    # Invalid date format should not crash; falls back gracefully
    task = TaskSpec(
        task_id="task_bad_date",
        agent=AgentType.DATA_AGENT.value,
        task_type=TaskType.DATA_ANALYSIS.value,
        question="Show metrics for an invalid date window",
        start_date="invalid-date-string",
        end_date="2026-99-99",
    )
    result = data_agent.run(task)

    assert isinstance(result, DataAgentResult)
    # Should safely return data without crashing
    assert result.response_count >= 0
