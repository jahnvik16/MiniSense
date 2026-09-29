import os
import logging
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import settings
from app.rag.retriever import FAQRetriever


def build_and_verify_index() -> None:
    print("=" * 60)
    print("MiniSense RAG: Building Local FAISS Vector Index")
    print(f"Source FAQ Document : {settings.faq_file}")
    print(f"Target Index Dir    : {settings.faiss_index_dir}")
    print(f"Embedding Model     : {settings.embedding_model}")
    print("=" * 60)

    retriever = FAQRetriever()
    print("\nGenerating embeddings and building FAISS index...")
    retriever.build_index(force=True)

    print("\nVerifying persisted index:")
    print(f"  - FAISS index file : {retriever.index_path} (exists: {retriever.index_path.exists()})")
    print(f"  - Chunks metadata  : {retriever.chunks_path} (exists: {retriever.chunks_path.exists()})")
    print(f"  - Total Q&A chunks : {len(retriever._chunks)}")

    for c in retriever._chunks:
        title = c.content.split("\n")[0]
        print(f"    * [{c.chunk_id}] {title}")

    # Test sample retrieval
    sample_query = "What is the refund policy for downgrades?"
    print(f"\nTesting sample query: '{sample_query}'")
    results = retriever.retrieve(sample_query, top_k=2)
    for idx, r in enumerate(results, 1):
        snippet = r.content.split("\n")[1][:80]
        print(f"  Rank {idx} [{r.chunk_id}] (Score: {r.score:.4f}): {snippet}...")

    print("\nFAISS Index built and verified successfully.")


if __name__ == "__main__":
    build_and_verify_index()
