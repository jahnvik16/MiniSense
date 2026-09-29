"""Local FAISS vector retriever using sentence-transformers embeddings."""

import json
from pathlib import Path
from typing import Any
import numpy as np

from app.config import settings
from app.models.schemas import DocumentChunk
from app.rag.ingest import chunk_faq_sections, load_faq_text


class FAQRetriever:
    """Independent local vector retriever backed by FAISS and sentence-transformers."""

    def __init__(
        self,
        index_dir: Path | None = None,
        faq_file: Path | None = None,
        model_name: str | None = None,
    ) -> None:
        self.index_dir = Path(index_dir or settings.faiss_index_dir)
        self.faq_file = Path(faq_file or settings.faq_file)
        self.model_name = model_name or settings.embedding_model

        self.index_path = self.index_dir / "index.faiss"
        self.chunks_path = self.index_dir / "chunks.json"

        self._model = None
        self._index = None
        self._chunks: list[DocumentChunk] = []

    def _get_model(self):
        """Lazy load sentence-transformers embedding model on CPU."""
        if self._model is None:
            import os
            import logging
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)

            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def build_index(self, force: bool = False) -> None:
        """Ingest FAQ document, embed Q&A chunks, and persist FAISS index locally."""
        if not force and self.index_path.exists() and self.chunks_path.exists():
            self.load_index()
            return

        import faiss

        text = load_faq_text(self.faq_file)
        chunks = chunk_faq_sections(text, source_path=str(self.faq_file))
        if not chunks:
            self._chunks = []
            self._index = None
            return

        # Embed chunk contents using sentence-transformers
        model = self._get_model()
        contents = [c.content for c in chunks]
        embeddings = model.encode(contents, normalize_embeddings=True, show_progress_bar=False)
        embeddings_np = np.asarray(embeddings, dtype=np.float32)

        dimension = embeddings_np.shape[1]
        # IndexFlatIP computes exact Inner Product (cosine similarity on normalized vectors)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_np)

        # Persist FAISS index and chunk metadata locally
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))

        chunks_data = [c.model_dump() for c in chunks]
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2)

        self._index = index
        self._chunks = chunks

    def load_index(self) -> bool:
        """Load persisted FAISS index and chunk metadata from disk."""
        if not (self.index_path.exists() and self.chunks_path.exists()):
            return False

        import faiss

        self._index = faiss.read_index(str(self.index_path))
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
        self._chunks = [DocumentChunk(**c) for c in raw_chunks]
        return True

    def retrieve(self, query: str, top_k: int = 3) -> list[DocumentChunk]:
        """Retrieve top-k relevant FAQ chunks for a natural language query."""
        cleaned_query = query.strip()
        if not cleaned_query or top_k <= 0:
            return []

        # Ensure index is ready
        if self._index is None or not self._chunks:
            if not self.load_index():
                self.build_index()

        if self._index is None or not self._chunks:
            return []

        # Encode query
        model = self._get_model()
        query_vec = model.encode([cleaned_query], normalize_embeddings=True, show_progress_bar=False)
        query_vec_np = np.asarray(query_vec, dtype=np.float32)

        actual_k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(query_vec_np, actual_k)

        results: list[DocumentChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            orig = self._chunks[idx]
            results.append(
                DocumentChunk(
                    chunk_id=orig.chunk_id,
                    content=orig.content,
                    score=round(float(score), 4),
                    source=orig.source,
                )
            )

        return results
