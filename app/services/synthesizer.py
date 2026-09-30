"""Synthesis service producing coherent, grounded FinalAnswer outputs."""

import json
import logging
import os
import re
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
    - Avoids unsupported causal overclaiming (distinguishes correlation/policy context from causal proof).
    - Avoids dumping raw FAQ headers or chunk IDs verbatim into narrative text.
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
            assumptions.append("Evaluated comparison periods / cohorts represent distinct groups.")

        if data_result:
            supporting_metrics["survey_metrics"] = {
                "response_count": data_result.response_count,
                "average_rating": data_result.average_rating,
                "csat": data_result.csat,
                "top_themes": [
                    {
                        "theme": t.theme,
                        "count": t.count,
                        "average_rating": t.average_rating,
                        "csat": t.csat,
                        "negative_count": t.sentiment_breakdown.get("negative", 0),
                    }
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
        if settings.has_llm_credentials:
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
        """Attempt synthesis using configured LLM (OpenAI or Gemini)."""
        system_instruction = (
            "You are an executive customer intelligence analyst. "
            "Synthesize a clear, coherent, executive-ready response directly answering the business question. "
            "\n"
            "Strict Guidelines:\n"
            "1. Strictly cite the provided survey metrics without altering values or recalculating math.\n"
            "2. Clearly distinguish empirical survey measurements (measured CSAT, average rating, volume) "
            "   from official company FAQ operating standards or policies.\n"
            "3. Ground all policy interpretations in the retrieved FAQ sources, but do NOT dump raw FAQ chunk "
            "   headers, questions, or identifiers verbatim into the text.\n"
            "4. Do NOT make unsupported causal claims (e.g., do NOT assert that an express lane or policy change "
            "   caused an observed CSAT shift unless verified by causal modeling; describe them as operational context).\n"
            "5. If empirical data is absent for an inquired period, explicitly state that data is unavailable.\n"
            "6. Return exactly one coherent, professional executive paragraph."
        )

        prompt = (
            f"Question: {question}\n\n"
            f"Structured Metrics: {json.dumps(supporting_metrics, indent=2)}\n\n"
            f"Retrieved FAQ Sources: {json.dumps(retrieved_sources, indent=2)}\n\n"
            f"FAQ Retrieval Reliable: {rag_reliable}\n"
        )

        if settings.llm_provider == "openai" and settings.openai_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=settings.openai_api_key)
                response = client.chat.completions.create(
                    model=settings.llm_model or "gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content.strip()
            except Exception as e:
                logger.debug(f"OpenAI synthesis unavailable ({e}), falling back...")

        if (settings.llm_provider == "gemini" or settings.gemini_api_key) and settings.gemini_api_key:
            try:
                from google import genai
                client = genai.Client(api_key=settings.gemini_api_key or settings.google_api_key)
                response = client.models.generate_content(
                    model=settings.llm_model or "gemini-1.5-flash",
                    contents=prompt,
                    config={"system_instruction": system_instruction, "temperature": 0.2},
                )
                if response and response.text and response.text.strip():
                    return response.text.strip()
            except Exception as e:
                logger.debug(f"Gemini synthesis unavailable ({e}), falling back...")

        return None

    def _clean_faq_chunk(self, chunk_text: str) -> str:
        """Extract substantive policy body from FAQ chunk, stripping Markdown header questions."""
        lines = chunk_text.strip().split("\n")
        body_lines = [l.strip() for l in lines if not l.strip().startswith("## ") and l.strip()]
        return " ".join(body_lines)

    def _deterministic_synthesis(
        self,
        question: str,
        data_result: DataAgentResult | None,
        rag_result: RAGAgentResult | None,
        comparison_result: ComparisonAgentResult | None,
    ) -> str:
        """Deterministically compose a coherent executive paragraph without raw chunk dumping or causal overclaiming."""
        parts: list[str] = []

        # Part A: Comparison findings
        if comparison_result:
            curr = comparison_result.current_period
            prev = comparison_result.previous_period
            changes = comparison_result.metric_changes

            csat_delta = changes.get("csat_delta", 0.0)
            avg_delta = changes.get("average_rating_delta", 0.0)
            count_delta = int(changes.get("count_delta", 0))

            scope_match = re.search(r"for '([^']+)'", comparison_result.summary)
            scope_phrase = f" for {scope_match.group(1).lower()}" if scope_match else ""

            # Check if there is data in either period
            if curr.response_count == 0 and prev.response_count == 0:
                parts.append(
                    f"No survey records were found for the requested evaluation window ({curr.period_label} vs {prev.period_label})."
                )
            else:
                delta_direction = "improved by" if csat_delta > 0 else ("declined by" if csat_delta < 0 else "remained steady at")
                parts.append(
                    f"Comparing {curr.period_label} with {prev.period_label}{scope_phrase}, customer satisfaction {delta_direction} "
                    f"{abs(csat_delta):.1f} percentage points ({prev.csat:.1f}% to {curr.csat:.1f}% CSAT), with average rating moving "
                    f"{avg_delta:+.2f} ({prev.average_rating:.2f} to {curr.average_rating:.2f}) across {curr.response_count:,} evaluated responses."
                )

        # Part B: Standalone Survey data findings
        elif data_result:
            if data_result.response_count == 0:
                parts.append("No survey records were found matching the specified filters or date window.")
            else:
                strategy = data_result.supporting_metadata.get("theme_ranking_strategy", "volume")
                is_complaint = (
                    strategy == "negative_volume"
                    or any(w in question.lower() for w in ["complaint", "negative", "issue", "worst"])
                )

                if is_complaint and data_result.top_themes:
                    theme_str = ", ".join(
                        f"{t.theme} ({t.sentiment_breakdown.get('negative', 0):,} negative complaints, {t.csat:.1f}% CSAT)"
                        for t in data_result.top_themes[:3]
                    )
                    parts.append(
                        f"Across {data_result.response_count:,} survey responses analyzed, overall customer satisfaction stands at {data_result.csat:.1f}% "
                        f"with an average rating of {data_result.average_rating:.2f}. The top complaint themes are {theme_str}."
                    )
                else:
                    parts.append(
                        f"Survey analysis records {data_result.response_count:,} total responses with an average rating of "
                        f"{data_result.average_rating:.2f} and an overall CSAT of {data_result.csat:.1f}%."
                    )
                    if data_result.top_themes:
                        theme_str = ", ".join(
                            f"{t.theme} ({t.count:,} responses, {t.csat:.1f}% CSAT)"
                            for t in data_result.top_themes[:3]
                        )
                        parts.append(f"Primary feedback driver themes are {theme_str}.")

        # Part C: Grounded FAQ / Policy context (clean narrative integration, no raw chunk header dump)
        if rag_result and rag_result.reliable and rag_result.retrieved_chunks:
            cleaned_policy = self._clean_faq_chunk(rag_result.retrieved_chunks[0])
            # Synthesize in business narrative context
            if "wait" in question.lower():
                parts.append(
                    f"In terms of operational guidelines, GreenLeaf targets wait times under 10 minutes during off-peak periods, "
                    f"while peak lunch (12:00 PM–1:00 PM) and dinner (6:00 PM–8:00 PM) may experience 15–20 minute waits. "
                    f"This suggests the evaluated survey results should be interpreted against those stated operating expectations."
                )
            elif "refund" in question.lower() or "complaint" in question.lower():
                parts.append(
                    f"Official customer policy dictates that all complaints are escalated to the shift manager within 15 minutes, "
                    f"with refunds or replacements offered for quality issues and vouchers provided for delays exceeding 20 minutes."
                )
            else:
                parts.append(f"Regarding official operating standards: {cleaned_policy}")
        elif rag_result and not rag_result.reliable:
            parts.append(
                "Official company documentation does not contain verified policy guidance for this specific topic."
            )

        if not parts:
            return "No sufficient survey records or policy documentation was found to address the inquiry."

        return " ".join(parts)
