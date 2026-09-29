"""FAQ document ingestion and semantic Q&A chunking."""

from pathlib import Path
from app.config import settings
from app.models.schemas import DocumentChunk


def load_faq_text(file_path: Path | None = None) -> str:
    """Read raw text from FAQ documentation file."""
    path = file_path or settings.faq_file
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_faq_sections(text: str, source_path: str = "data/faq.txt") -> list[DocumentChunk]:
    """Chunk FAQ text into self-contained Question-Answer semantic units.

    Each markdown section starting with '## Q' is parsed as an individual
    retrieval chunk containing both the question prompt and complete answer text.
    """
    if not text.strip():
        return []

    sections = text.split("## ")
    chunks: list[DocumentChunk] = []

    for idx, sec in enumerate(sections):
        cleaned = sec.strip()
        # Skip top-level headers (# MiniSense...)
        if not cleaned or cleaned.startswith("# "):
            continue

        chunk_id = f"faq_chunk_{idx}"
        content = "## " + cleaned

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                content=content,
                score=1.0,
                source=source_path,
            )
        )

    return chunks
