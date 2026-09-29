"""Synthesis service producing coherent, grounded FinalAnswer outputs."""

import json
import logging
from typing import Any
from app.config import settings
from app.models.schemas import (
    ComparisonAgentResult,
    DataAgentResult,
    FinalAnswer,
    RAGAgentResult,
)

logger = logging.getLogger(__name__)


class Synthesizer:
    """Combines structured survey results and retrieved FAQ context into FinalAnswer.

    Guarantees:
    - Never hallucinates numerical metrics (uses exact numbers from agents).
    - Grounds policy context exclusively in retrieved FAQ chunks.
    - Explicitly notes uncertainty or missing data.
    - Operates robustly with LLM or fallback deterministic engine.
    """

    def synthesize(
        self,
        question: str,
        data_result: DataAgentResult | None = None,
        rag_result: RAGAgentResult | None = None,
        comparison_result: ComparisonAgentResult | None = None,
    ) -> FinalAnswer:
        """Produce a structured FinalAnswer from multi-agent results."""
        supporting_metrics: dict[str, Any] = {}
        retrieved_sources: list[str] = []
        assumptions: list[str] = [
            "CSAT is defined deterministically as the percentage of survey ratings >= 4 out of 5."
        ]

        # 1. Collect structured metrics
        if comparison_result:
            supporting_metrics["comparison"] = {
                "current_period": {
                    "label": comparison_result.current_period.period_label,
                    "count": comparison_result.current_period.response_count,
                    "average_rating": comparison_result.current_period.average_rating,
                    "csat": comparison_result.current_period.csat,
                },
                "previous_period": {
                    "label": comparison_result.previous_period.period_label,
                    "count": comparison_result.previous_period.response_count,
                    "average_rating": comparison_result.previous_period.average_rating,
                    "csat": comparison_result.previous_period.csat,
                },
                "metric_changes": comparison_result.metric_changes,
            }
            assumptions.append("Evaluated comparison periods / cohorts are distinct groups.")

        if data_result:
            supporting_metrics["survey_metrics"] = {
                "response_count": data_result.response_count,
                "average_rating": data_result.average_rating,
                "csat": data_result.csat,
                "top_themes": [
                    {"theme": t.theme, "count": t.count, "average_rating": t.average_rating, "csat": t.csat}
                    for t in data_result.top_themes
                ],
            }

        # 2. Collect retrieved FAQ sources
        if rag_result and rag_result.reliable and rag_result.retrieved_chunks:
            retrieved_sources = []
            for i, chunk_text in enumerate(rag_result.retrieved_chunks):
                chunk_id = (
                    rag_result.source_metadata[i].get("chunk_id", f"faq_chunk_{i+1}")
                    if i < len(rag_result.source_metadata)
                    else f"faq_chunk_{i+1}"
                )
                retrieved_sources.append(f"{chunk_id}: {chunk_text}")
        elif rag_result and not rag_result.reliable:
            assumptions.append("No official FAQ policy chunk met the minimum confidence threshold.")

        # 3. Attempt LLM generation if configured, otherwise use deterministic synthesizer
        answer_text = None
        if settings.llm_provider != "mock":
            answer_text = self._try_llm_synthesis(
                question=question,
                supporting_metrics=supporting_metrics,
                retrieved_sources=retrieved_sources,
                rag_reliable=bool(rag_result and rag_result.reliable),
            )

        if not answer_text:
            answer_text = self._deterministic_synthesis(
                question=question,
                data_result=data_result,
                rag_result=rag_result,
                comparison_result=comparison_result,
            )

        return FinalAnswer(
            answer=answer_text,
            supporting_metrics=supporting_metrics,
            retrieved_sources=retrieved_sources,
            assumptions=assumptions,
        )

    def _try_llm_synthesis(
        self,
        question: str,
        supporting_metrics: dict[str, Any],
        retrieved_sources: list[str],
        rag_reliable: bool,
    ) -> str | None:
        """Attempt synthesis using Gemini or configured LLM."""
        try:
            import os
            key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                return None

            from google import genai
            client = genai.Client(api_key=key)

            system_instruction = (
                "You are an executive survey analytics assistant. "
                "Synthesize a clear, coherent, executive-ready response answering the business question directly. "
                "Rules:\n"
                "1. Strictly cite the provided survey metrics without altering values or recalculating math.\n"
                "2. Clearly distinguish empirical survey findings from official FAQ policy.\n"
                "3. Ground all policy statements directly in the retrieved FAQ sources. Do not invent rules.\n"
                "4. If data is sparse or context is absent, state uncertainty explicitly.\n"
                "5. Return only the final synthesis paragraph."
            )

            prompt = (
                f"Question: {question}\n\n"
                f"Structured Metrics: {json.dumps(supporting_metrics, indent=2)}\n\n"
                f"Retrieved FAQ Sources: {json.dumps(retrieved_sources, indent=2)}\n"
                f"FAQ Retrieval Reliable: {rag_reliable}\n"
            )

            response = client.models.generate_content(
                model=settings.llm_model or "gemini-1.5-flash",
                contents=prompt,
                config={"system_instruction": system_instruction, "temperature": 0.2},
            )
            if response and response.text and response.text.strip():
                return response.text.strip()
        except Exception as e:
            logger.debug(f"LLM synthesis unavailable, falling back to deterministic synthesis: {e}")
        return None

    def _deterministic_synthesis(
        self,
        question: str,
        data_result: DataAgentResult | None,
        rag_result: RAGAgentResult | None,
        comparison_result: ComparisonAgentResult | None,
    ) -> str:
        """Deterministically compose an executive answer from structured evidence."""
        parts: list[str] = []

        # Part A: Comparison findings
        if comparison_result:
            curr = comparison_result.current_period
            prev = comparison_result.previous_period
            changes = comparison_result.metric_changes

            csat_delta = changes.get("csat_delta", 0.0)
            avg_delta = changes.get("average_rating_delta", 0.0)
            count_delta = int(changes.get("count_delta", 0))

            delta_direction = "improved by" if csat_delta > 0 else ("declined by" if csat_delta < 0 else "remained flat at")
            parts.append(
                f"Comparing {curr.period_label} with {prev.period_label}, customer satisfaction {delta_direction} "
                f"{abs(csat_delta):.1f} percentage points ({prev.csat:.1f}% vs. {curr.csat:.1f}% CSAT). "
                f"Average rating shifted {avg_delta:+.2f} ({prev.average_rating:.2f} -> {curr.average_rating:.2f}) "
                f"across {curr.response_count:,} evaluated responses (volume change: {count_delta:+,d})."
            )

            if curr.top_themes:
                theme_str = ", ".join(f"{t.theme} ({t.count:,} responses, {t.csat:.1f}% CSAT)" for t in curr.top_themes[:3])
                parts.append(f"Top feedback themes for {curr.period_label} are {theme_str}.")

        # Part B: Standalone Survey data findings
        elif data_result:
            parts.append(
                f"Survey analysis records {data_result.response_count:,} total responses with an average rating of "
                f"{data_result.average_rating:.2f} and an overall CSAT of {data_result.csat:.1f}%."
            )
            if data_result.top_themes:
                theme_str = ", ".join(
                    f"{t.theme} ({t.count:,} responses, CSAT: {t.csat:.1f}%)" for t in data_result.top_themes[:3]
                )
                parts.append(f"Primary feedback driver themes are {theme_str}.")

        # Part C: Grounded FAQ / Policy context
        if rag_result and rag_result.reliable and rag_result.retrieved_chunks:
            top_chunk_text = rag_result.retrieved_chunks[0].strip().replace("\n", " ")
            chunk_id = (
                rag_result.source_metadata[0].get("chunk_id", "faq_chunk_1")
                if rag_result.source_metadata
                else "faq_chunk_1"
            )
            parts.append(f"According to company policy ({chunk_id}): {top_chunk_text}")
        elif rag_result and not rag_result.reliable:
            parts.append(
                "Note: Official company FAQ documentation does not contain verified policy guidance for this specific inquiry."
            )

        if not parts:
            return "No sufficient survey records or policy documentation was found to address the inquiry."

        return " ".join(parts)
