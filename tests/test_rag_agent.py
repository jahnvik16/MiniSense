"""Unit tests for RAGAgent behavior, grounded retrieval, and confidence gating."""

import pytest
from app.agents.rag_agent import RAGAgent
from app.models.schemas import RAGAgentResult, TaskSpec, AgentType, TaskType


@pytest.fixture(scope="module")
def rag_agent() -> RAGAgent:
    agent = RAGAgent()
    agent.retriever.build_index()
    return agent


def test_rag_agent_refund_policy_retrieval(rag_agent: RAGAgent) -> None:
    task = TaskSpec(
        task_id="task_refund_query",
        agent=AgentType.RAG_AGENT.value,
        task_type=TaskType.RAG_LOOKUP.value,
        question="How do you handle customer complaints and refunds for quality issues?",
        parameters={"top_k": 2},
    )
    result = rag_agent.run(task)

    assert isinstance(result, RAGAgentResult)
    assert result.reliable is True
    assert len(result.retrieved_chunks) >= 1
    assert len(result.scores) == len(result.retrieved_chunks)
    assert len(result.source_metadata) == len(result.retrieved_chunks)

    # Top chunk should be faq_chunk_3 (complaints & refunds policy)
    top_meta = result.source_metadata[0]
    assert top_meta["chunk_id"] == "faq_chunk_3"
    assert "refund" in result.retrieved_chunks[0].lower()
    assert top_meta["score"] >= 0.35


def test_rag_agent_wait_time_policy_retrieval(rag_agent: RAGAgent) -> None:
    task = TaskSpec(
        task_id="task_wait_time_query",
        agent=AgentType.RAG_AGENT.value,
        task_type=TaskType.RAG_LOOKUP.value,
        question="What is the average wait time during off-peak and peak hours?",
        parameters={"top_k": 1},
    )
    result = rag_agent.run(task)

    assert result.reliable is True
    assert len(result.retrieved_chunks) == 1
    assert result.source_metadata[0]["chunk_id"] == "faq_chunk_2"
    assert "wait time" in result.retrieved_chunks[0].lower()


def test_rag_agent_low_confidence_handling(rag_agent: RAGAgent) -> None:
    # Query completely unrelated to product policies or FAQs
    task = TaskSpec(
        task_id="task_irrelevant",
        agent=AgentType.RAG_AGENT.value,
        task_type=TaskType.RAG_LOOKUP.value,
        question="How do black holes bend space time in quantum gravitation?",
        parameters={"min_confidence": 0.50},
    )
    result = rag_agent.run(task)

    assert isinstance(result, RAGAgentResult)
    # Must explicitly declare no reliable context found rather than hallucinating
    assert result.reliable is False
    assert len(result.retrieved_chunks) == 0
    assert len(result.scores) == 0


def test_rag_agent_strict_threshold_rejection(rag_agent: RAGAgent) -> None:
    # Even on a related topic, an impossibly high threshold (e.g. 0.99) should safely return no reliable context
    task = TaskSpec(
        task_id="task_strict",
        agent=AgentType.RAG_AGENT.value,
        task_type=TaskType.RAG_LOOKUP.value,
        question="What is the refund policy?",
        parameters={"min_confidence": 0.99},
    )
    result = rag_agent.run(task)

    assert result.reliable is False
    assert result.retrieved_chunks == []


def test_rag_agent_empty_query_handling(rag_agent: RAGAgent) -> None:
    # 1. Via TaskSpec with empty query parameter
    task = TaskSpec(
        task_id="task_empty_query",
        agent=AgentType.RAG_AGENT.value,
        task_type=TaskType.RAG_LOOKUP.value,
        question="Find relevant FAQ policies",
        parameters={"query": "   \n\t  "},
    )
    result = rag_agent.run(task)

    assert isinstance(result, RAGAgentResult)
    assert result.reliable is False
    assert len(result.retrieved_chunks) == 0

    # 2. Via direct RAGAgentInput with empty query
    from app.models.schemas import RAGAgentInput
    result_direct = rag_agent.run(RAGAgentInput(query=""))
    assert result_direct.reliable is False
    assert len(result_direct.retrieved_chunks) == 0
