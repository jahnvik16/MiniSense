"""Unit and integration tests for the FAQ RAG pipeline."""

import shutil
from pathlib import Path
import pytest

from app.config import settings
from app.rag.ingest import chunk_faq_sections, load_faq_text
from app.rag.retriever import FAQRetriever
from app.models.schemas import DocumentChunk


@pytest.fixture(scope="module")
def persistent_retriever() -> FAQRetriever:
    """Module-level retriever fixture using the project's standard FAISS index."""
    retriever = FAQRetriever()
    retriever.build_index(force=True)
    return retriever


def test_chunk_faq_sections() -> None:
    text = load_faq_text(settings.faq_file)
    assert len(text) > 0

    chunks = chunk_faq_sections(text)
    # The GreenLeaf Bistro FAQ contains 7 distinct self-contained Q&A sections
    assert len(chunks) == 7

    for c in chunks:
        assert isinstance(c, DocumentChunk)
        assert c.chunk_id.startswith("faq_chunk_")
        assert "## Q" in c.content
        assert c.source == "data/faq.txt"
        assert len(c.content.split("\n")) >= 2  # Has question header and body answer


def test_empty_text_chunking() -> None:
    assert chunk_faq_sections("") == []
    assert chunk_faq_sections("   ") == []


def test_index_creation_and_persistence(tmp_path: Path) -> None:
    """Verify that the index creates and saves files to custom directory."""
    custom_index_dir = tmp_path / "custom_faiss"
    retriever = FAQRetriever(index_dir=custom_index_dir)

    assert not (custom_index_dir / "index.faiss").exists()
    assert not (custom_index_dir / "chunks.json").exists()

    retriever.build_index(force=True)

    assert (custom_index_dir / "index.faiss").exists()
    assert (custom_index_dir / "chunks.json").exists()
    assert len(retriever._chunks) == 7

    # Reload from disk into a fresh instance
    new_retriever = FAQRetriever(index_dir=custom_index_dir)
    loaded = new_retriever.load_index()
    assert loaded is True
    assert len(new_retriever._chunks) == 7
    assert new_retriever._index is not None


def test_retrieval_relevance(persistent_retriever: FAQRetriever) -> None:
    # 1. Refund / complaint handling policy query
    results_refund = persistent_retriever.retrieve("How do you handle complaints and refunds for quality issues?", top_k=2)
    assert len(results_refund) == 2
    assert results_refund[0].chunk_id == "faq_chunk_3"
    assert "refund" in results_refund[0].content.lower()
    assert results_refund[0].score > 0.4

    # 2. Wait time & peak hours query
    results_wait = persistent_retriever.retrieve("What is your average wait time during off-peak and peak hours?", top_k=2)
    assert len(results_wait) == 2
    assert results_wait[0].chunk_id == "faq_chunk_2"
    assert "wait time" in results_wait[0].content.lower()

    # 3. Popular menu items query
    results_menu = persistent_retriever.retrieve("What are your most popular menu items?", top_k=2)
    assert len(results_menu) == 2
    assert results_menu[0].chunk_id == "faq_chunk_1"
    assert "avocado toast" in results_menu[0].content.lower()


def test_top_k_behavior(persistent_retriever: FAQRetriever) -> None:
    # top_k = 1
    res_1 = persistent_retriever.retrieve("average wait time and express pickup", top_k=1)
    assert len(res_1) == 1

    # top_k = 3
    res_3 = persistent_retriever.retrieve("average wait time and express pickup", top_k=3)
    assert len(res_3) == 3
    # Scores must be in non-ascending order
    assert res_3[0].score >= res_3[1].score >= res_3[2].score

    # top_k exceeds total chunks (total is 7)
    res_excess = persistent_retriever.retrieve("wait time", top_k=50)
    assert len(res_excess) == len(persistent_retriever._chunks)

    # Non-positive top_k
    assert persistent_retriever.retrieve("pricing tiers", top_k=0) == []
    assert persistent_retriever.retrieve("pricing tiers", top_k=-2) == []


def test_empty_and_irrelevant_query_handling(persistent_retriever: FAQRetriever) -> None:
    # Empty string
    assert persistent_retriever.retrieve("") == []
    assert persistent_retriever.retrieve("   \t  \n  ") == []

    # Out-of-domain / irrelevant query: should not crash and return gracefully
    irrelevant = persistent_retriever.retrieve("supernova astrophysics dark matter quantum gravitation", top_k=2)
    assert len(irrelevant) == 2
    for chunk in irrelevant:
        assert isinstance(chunk, DocumentChunk)
        assert isinstance(chunk.score, float)
