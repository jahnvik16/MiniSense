"""Integration tests for the LangGraph orchestration layer.

Tests:
1. Selective Planner routing (no blind invocation of every agent):
   - "What are the top 3 complaints in May?" -> DataAgent only
   - "How did CSAT change from April to May?" -> ComparisonAgent + DataAgent
   - "Why are wait-time complaints increasing and what does the FAQ say?" -> DataAgent + RAGAgent
2. End-to-end graph execution:
   Question -> Planner -> Task Routing -> Agents -> Collect -> Synthesis -> FinalAnswer
"""

import pytest
from app.agents.orchestrator import OrchestratorAgent
from app.models.schemas import AgentType, FinalAnswer, TaskType


@pytest.fixture
def orchestrator() -> OrchestratorAgent:
    return OrchestratorAgent()


def test_planner_selective_routing_data_only(orchestrator: OrchestratorAgent) -> None:
    """Verifies that single-domain survey queries route only to DataAgent."""
    question = "What are the top 3 complaints in May?"
    tasks = orchestrator.plan(question)

    agents_assigned = {t.agent for t in tasks}
    assert AgentType.DATA_AGENT.value in agents_assigned
    assert AgentType.RAG_AGENT.value not in agents_assigned
    assert AgentType.COMPARISON_AGENT.value not in agents_assigned

    data_task = next(t for t in tasks if t.agent == AgentType.DATA_AGENT.value)
    assert data_task.task_type == TaskType.TOP_THEMES.value
    assert data_task.parameters.get("top_n") == 3
    assert data_task.parameters.get("sentiment") == "negative"
    assert data_task.start_date == "2026-05-01"
    assert data_task.end_date == "2026-05-31"


def test_planner_selective_routing_comparison_and_data(orchestrator: OrchestratorAgent) -> None:
    """Verifies that month-over-month shift queries route to ComparisonAgent + DataAgent."""
    question = "How did CSAT change from April to May?"
    tasks = orchestrator.plan(question)

    agents_assigned = {t.agent for t in tasks}
    assert AgentType.COMPARISON_AGENT.value in agents_assigned
    assert AgentType.DATA_AGENT.value in agents_assigned
    assert AgentType.RAG_AGENT.value not in agents_assigned

    comp_task = next(t for t in tasks if t.agent == AgentType.COMPARISON_AGENT.value)
    assert comp_task.task_type == TaskType.PERIOD_COMPARISON.value
    assert comp_task.start_date == "2026-05-01"
    assert comp_task.comparison_start_date == "2026-04-01"


def test_planner_selective_routing_data_and_rag(orchestrator: OrchestratorAgent) -> None:
    """Verifies that feedback + FAQ inquiries route to DataAgent + RAGAgent."""
    question = "Why are wait-time complaints increasing and what does the FAQ say?"
    tasks = orchestrator.plan(question)

    agents_assigned = {t.agent for t in tasks}
    assert AgentType.DATA_AGENT.value in agents_assigned
    assert AgentType.RAG_AGENT.value in agents_assigned
    assert AgentType.COMPARISON_AGENT.value not in agents_assigned

    rag_task = next(t for t in tasks if t.agent == AgentType.RAG_AGENT.value)
    assert rag_task.task_type == TaskType.RAG_LOOKUP.value
    assert len(rag_task.parameters.get("query", "")) > 0


def test_end_to_end_orchestration_flow(orchestrator: OrchestratorAgent) -> None:
    """Tests the full LangGraph flow:

    START -> Planner -> Task routing -> Agents -> Collect -> Synthesis -> FinalAnswer -> END
    """
    question = "What is the customer satisfaction (CSAT) for our enterprise customers, and what SLA guarantees do they receive under their support contract?"

    # Execute graph flow
    result = orchestrator.run(question)

    # 1. Type contract check
    assert isinstance(result, FinalAnswer)
    assert len(result.answer) > 50

    # 2. Supporting metrics verification (exact, not hallucinated)
    assert "survey_metrics" in result.supporting_metrics
    survey_metrics = result.supporting_metrics["survey_metrics"]
    assert survey_metrics["csat"] == 58.24  # Exact enterprise CSAT in 75k survey dataset
    assert survey_metrics["response_count"] == 11363

    # 3. Grounded FAQ sources verification
    assert len(result.retrieved_sources) > 0
    top_source = result.retrieved_sources[0]
    assert "SLA" in top_source or "Enterprise" in top_source or "hour" in top_source

    # 4. Assumptions verification
    assert any("CSAT" in a for a in result.assumptions)

    # 5. Coherent business narrative verification
    answer_text = result.answer
    assert "58.2" in answer_text or "CSAT" in answer_text
    assert "policy" in answer_text.lower() or "sla" in answer_text.lower() or "enterprise" in answer_text.lower()


def test_end_to_end_period_comparison_flow(orchestrator: OrchestratorAgent) -> None:
    """Tests end-to-end period comparison through LangGraph."""
    question = "How did CSAT change from April to May?"
    result = orchestrator.run(question)

    assert isinstance(result, FinalAnswer)
    assert "comparison" in result.supporting_metrics
    comp_metrics = result.supporting_metrics["comparison"]
    assert "metric_changes" in comp_metrics
    assert "csat_delta" in comp_metrics["metric_changes"]
    assert len(result.answer) > 30
