"""RAGAgent responsible for grounded knowledge retrieval from local FAQ store."""

from typing import Any
from app.models.schemas import RAGAgentInput, RAGAgentResult, TaskSpec
from app.rag.retriever import FAQRetriever

# Default minimum cosine similarity score to qualify as reliable context
DEFAULT_MIN_CONFIDENCE = 0.35


class RAGAgent:
    """Sub-agent executing grounded knowledge retrieval against local FAISS index.

    Strictly retrieves existing FAQ policy chunks without fabricating facts or
    generating ungrounded narrative text.
    """

    def __init__(
        self,
        retriever: FAQRetriever | None = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        self.retriever = retriever or FAQRetriever()
        self.min_confidence = min_confidence

    def run(self, task_or_input: TaskSpec | RAGAgentInput) -> RAGAgentResult:
        """Process structured retrieval task and return validated RAGAgentResult."""
        if isinstance(task_or_input, TaskSpec):
            query = task_or_input.parameters.get("query") or task_or_input.question
            top_k = int(task_or_input.parameters.get("top_k", 3))
            threshold = float(task_or_input.parameters.get("min_confidence", self.min_confidence))
        else:
            query = task_or_input.query
            top_k = int(task_or_input.top_k)
            threshold = self.min_confidence

        cleaned_query = query.strip()
        if not cleaned_query or top_k <= 0:
            return RAGAgentResult(
                query=query,
                retrieved_chunks=[],
                scores=[],
                source_metadata=[],
                reliable=False,
            )

        # Retrieve candidate chunks from local FAISS vector store
        raw_chunks = self.retriever.retrieve(cleaned_query, top_k=top_k)

        # Confidence verification: verify top result exceeds minimum similarity threshold
        if not raw_chunks or raw_chunks[0].score < threshold:
            return RAGAgentResult(
                query=query,
                retrieved_chunks=[],
                scores=[],
                source_metadata=[],
                reliable=False,
            )

        # Filter down to chunks meeting the confidence threshold
        confident_chunks = [c for c in raw_chunks if c.score >= threshold]

        chunk_texts: list[str] = [c.content for c in confident_chunks]
        scores: list[float] = [c.score for c in confident_chunks]
        metadata_list: list[dict[str, Any]] = [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "score": c.score,
                "title": c.content.split("\n")[0].replace("## ", "").strip(),
            }
            for c in confident_chunks
        ]

        return RAGAgentResult(
            query=query,
            retrieved_chunks=chunk_texts,
            scores=scores,
            source_metadata=metadata_list,
            reliable=True,
        )
