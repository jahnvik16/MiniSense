"""Unit and integration tests for MiniSense scaffold."""

import pytest
from app.config import settings
from app.models.schemas import (
    TaskSpec,
    TaskType,
    DataAgentInput,
    RAGAgentInput,
    ComparisonAgentInput,
)
from app.tools.data_tools import (
    compute_csat,
    compute_nps,
    compute_sentiment_breakdown,
    filter_surveys,
)
from app.services.survey_service import SurveyService
from app.rag.ingest import chunk_faq_sections, load_faq_text
from app.rag.retriever import FAQRetriever
from app.agents.orchestrator import OrchestratorAgent
from app.agents.data_agent import DataAgent
from app.agents.rag_agent import RAGAgent
from app.agents.comparison_agent import ComparisonAgent


def test_deterministic_csat_calculation() -> None:
    # 5 ratings: [5, 4, 3, 2, 1] -> 2 are >= 4 -> 40.0%
    scores = [5, 4, 3, 2, 1]
    assert compute_csat(scores) == 40.0
    assert compute_csat([]) == 0.0
    assert compute_csat([5, 5, 4]) == 100.0


def test_deterministic_nps_calculation() -> None:
    # Promoters (>=9): 2, Detractors (<=6): 2, Passives (7-8): 1 -> NPS = ((2 - 2) / 5) * 100 = 0.0
    scores = [10, 9, 8, 5, 2]
    assert compute_nps(scores) == 0.0

    # All promoters
    assert compute_nps([9, 10, 10]) == 100.0

    # All detractors
    assert compute_nps([1, 2, 3]) == -100.0


def test_sentiment_breakdown() -> None:
    sentiments = ["positive", "positive", "negative", "neutral", "positive"]
    res = compute_sentiment_breakdown(sentiments)
    assert res == {"positive": 3, "neutral": 1, "negative": 1}


def test_survey_service_loading() -> None:
    service = SurveyService()
    surveys = service.get_all_surveys()
    assert len(surveys) == 75000

    # Filter enterprise
    enterprise = filter_surveys(surveys, cohort="enterprise")
    assert len(enterprise) > 10000


def test_faq_chunking_and_retrieval() -> None:
    text = load_faq_text(settings.faq_file)
    chunks = chunk_faq_sections(text)
    assert len(chunks) >= 4

    retriever = FAQRetriever()
    results = retriever.retrieve("refund policy for cancellations", top_k=2)
    assert len(results) > 0
    assert "refund" in results[0].content.lower()


def test_agents_scaffold_execution() -> None:
    # 1. Orchestrator
    orchestrator = OrchestratorAgent()
    tasks = orchestrator.plan("Compare CSAT between enterprise and self_serve")
    assert len(tasks) > 0
    assert tasks[0].task_type == TaskType.COMPARISON

    # 2. DataAgent
    data_agent = DataAgent()
    data_out = data_agent.run(DataAgentInput(metric_name="csat", cohort="enterprise"))
    assert isinstance(data_out.value, float)
    assert 0.0 <= data_out.value <= 100.0
    assert data_out.sample_size == 11363

    # 3. RAGAgent
    rag_agent = RAGAgent()
    rag_out = rag_agent.run(RAGAgentInput(query="support SLA", top_k=1))
    assert len(rag_out.retrieved_chunks) == 1
    assert "SLA" in rag_out.grounding_context

    # 4. ComparisonAgent
    comp_agent = ComparisonAgent(data_agent=data_agent)
    comp_out = comp_agent.run(
        ComparisonAgentInput(metric_name="csat", cohort_a="enterprise", cohort_b="self_serve")
    )
    assert isinstance(comp_out.delta, float)
    assert comp_out.cohort_a == "enterprise"
    assert comp_out.cohort_b == "self_serve"
