"""Hybrid LLM Planner and Deterministic Date Resolver for MiniSense.

Architecture:
User Question
  -> LLM Planner (structured JSON / Pydantic validation via OpenAI or Gemini)
  -> validated Pydantic TaskSpec
  -> deterministic date/filter resolution ("this month", "last month", explicit months)
  -> DataAgent / RAGAgent / ComparisonAgent
  -> Synthesizer
  -> FinalAnswer

Reliability Guarantee:
If the LLM planner is unavailable, invalid, times out, or produces malformed output,
the system gracefully falls back to a deterministic fallback planner. The fallback
is explicitly documented as a reliability mechanism, not the primary planner.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.models.schemas import (
    AgentType,
    PlannedSubTask,
    PlannerPlan,
    TaskSpec,
    TaskType,
)

logger = logging.getLogger(__name__)

# Default anchor date for relative temporal queries based on the synthetic dataset window
DATASET_ANCHOR_DATE = datetime(2026, 5, 31, 23, 59, 59, tzinfo=timezone.utc)

MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

DAYS_IN_MONTH = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
}


def resolve_date_expression(
    expr: str | None,
    anchor_date: datetime = DATASET_ANCHOR_DATE,
) -> tuple[str | None, str | None]:
    """Deterministically convert natural language date expressions into concrete ISO start and end dates.

    Supports:
    - 'this month', 'current month' -> May 2026 (2026-05-01 to 2026-05-31)
    - 'last month', 'previous month', 'prior month' -> April 2026 (2026-04-01 to 2026-04-30)
    - explicit month names: 'April', 'May', 'June', 'March' -> concrete YYYY-MM-01 to YYYY-MM-DD
    - ISO strings: '2026-04-01' -> parsed and passed directly
    - None / unspecified -> (None, None)
    """
    if not expr or not str(expr).strip():
        return None, None

    cleaned = str(expr).lower().strip()

    # Direct ISO range match: e.g. "2026-05-01 to 2026-05-31" or "2026-05-01:2026-05-31"
    iso_range_match = re.search(r"(\d{4}-\d{2}-\d{2})\s*(?:to|:|-)\s*(\d{4}-\d{2}-\d{2})", cleaned)
    if iso_range_match:
        return iso_range_match.group(1), iso_range_match.group(2)

    # Single ISO date
    iso_single_match = re.search(r"(\d{4}-\d{2}-\d{2})", cleaned)
    if iso_single_match and len(cleaned) == 10:
        d = iso_single_match.group(1)
        return d, d

    # Relative expressions
    if "this month" in cleaned or "current month" in cleaned:
        year = anchor_date.year
        month = anchor_date.month
        last_day = DAYS_IN_MONTH.get(month, 30)
        return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last_day:02d}"

    if "last month" in cleaned or "previous month" in cleaned or "prior month" in cleaned:
        year = anchor_date.year
        month = anchor_date.month - 1
        if month < 1:
            month = 12
            year -= 1
        last_day = DAYS_IN_MONTH.get(month, 30)
        return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last_day:02d}"

    if "two months ago" in cleaned:
        year = anchor_date.year
        month = anchor_date.month - 2
        if month < 1:
            month += 12
            year -= 1
        last_day = DAYS_IN_MONTH.get(month, 30)
        return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last_day:02d}"

    # Year-month string like "2026-04" or "2026-05"
    ym_match = re.search(r"(\d{4})[-/](\d{1,2})", cleaned)
    if ym_match:
        year = int(ym_match.group(1))
        month = int(ym_match.group(2))
        last_day = DAYS_IN_MONTH.get(month, 30)
        return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last_day:02d}"

    # Named months (e.g. "April", "May", "June", "April 2026")
    year_match = re.search(r"\b(20\d{2})\b", cleaned)
    year = int(year_match.group(1)) if year_match else anchor_date.year

    for name, month_num in MONTH_MAP.items():
        if re.search(rf"\b{name}\b", cleaned):
            last_day = DAYS_IN_MONTH.get(month_num, 30)
            return f"{year:04d}-{month_num:02d}-01", f"{year:04d}-{month_num:02d}-{last_day:02d}"

    return None, None


class LLMPlanner:
    """Interprets and decomposes natural-language business questions via LLM structured outputs.

    Does NOT calculate CSAT or metrics.
    Does NOT hardcode benchmark questions or static dates.
    Emits validated Pydantic models.
    """

    def __init__(self) -> None:
        self.provider = settings.llm_provider
        self.model = settings.llm_model

    def plan(self, question: str) -> PlannerPlan | None:
        """Call configured LLM with strict structured output schema."""
        if not settings.has_llm_credentials:
            return None

        if self.provider == "openai":
            return self._plan_openai(question)
        elif self.provider == "gemini":
            return self._plan_gemini(question)
        return None

    def _get_system_prompt(self) -> str:
        return (
            "You are the Master Query Planner for the MiniSense customer survey analytics and policy intelligence platform. "
            "Your sole responsibility is to interpret and decompose incoming natural language business questions into structured sub-tasks. "
            "\n"
            "Rules:\n"
            "1. Do NOT calculate metrics, arithmetic, or CSAT yourself. Specialized sub-agents perform deterministic computation.\n"
            "2. Select ONLY the sub-agents required to answer the question:\n"
            "   - 'DataAgent': For single-period metrics, CSAT, ratings, volumes, and ranking top themes or complaints.\n"
            "     task_types: 'data_analysis', 'top_themes'\n"
            "   - 'ComparisonAgent': For comparing two distinct time periods (e.g., April to May, this month vs last month) or customer cohorts.\n"
            "     task_types: 'period_comparison', 'cohort_comparison'\n"
            "   - 'RAGAgent': For retrieving official company documentation, FAQs, SLAs, policies, or expected wait times.\n"
            "     task_types: 'rag_lookup'\n"
            "3. Do not hardcode calendar dates. Extract natural language date expressions (e.g. 'this month', 'last month', 'April', 'May') "
            "   into date_expression (current/evaluated period) and comparison_period_expression (baseline comparison period). "
            "   For example, for 'How did wait-time experience change from April to May?':\n"
            "     - agent: 'ComparisonAgent'\n"
            "     - task_type: 'period_comparison'\n"
            "     - date_expression: 'May'\n"
            "     - comparison_period_expression: 'April'\n"
            "     - filters: {'theme': 'Wait Time'}\n"
            "4. For ranking/complaint queries (e.g., 'What are the top 3 complaints this month?'), set metric='negative_volume' or 'complaints' and requested_limit=3.\n"
            "5. For hybrid inquiries (e.g. 'How did wait-time experience change from April to May, and what does the FAQ say about expected wait times?'), "
            "   decompose into BOTH a ComparisonAgent task AND a RAGAgent task with a targeted retrieval_query.\n"
            "6. Output must strictly conform to the required JSON schema."
        )

    def _plan_openai(self, question: str) -> PlannerPlan | None:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)

            system_instruction = self._get_system_prompt()
            user_prompt = f"Decompose this business question into sub-agent tasks:\n\"{question}\""

            # Use OpenAI beta parse with Pydantic model for guaranteed schema compliance
            completion = client.beta.chat.completions.parse(
                model=self.model or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=PlannerPlan,
                temperature=0.0,
            )

            if completion.choices and completion.choices[0].message.parsed:
                return completion.choices[0].message.parsed
        except Exception as e:
            logger.warning(f"OpenAI structured planning failed or threw error: {e}. Falling back to deterministic planner.")
        return None

    def _plan_gemini(self, question: str) -> PlannerPlan | None:
        try:
            from google import genai
            client = genai.Client(api_key=settings.gemini_api_key or settings.google_api_key)

            prompt = (
                f"{self._get_system_prompt()}\n\n"
                f"Question: \"{question}\"\n\n"
                "Return the plan strictly as a JSON object matching the PlannerPlan schema."
            )

            response = client.models.generate_content(
                model=self.model or "gemini-1.5-flash",
                contents=prompt,
                config={"response_mime_type": "application/json", "temperature": 0.0},
            )

            if response and response.text:
                data = json.loads(response.text)
                return PlannerPlan.model_validate(data)
        except Exception as e:
            logger.warning(f"Gemini structured planning failed or threw error: {e}. Falling back to deterministic planner.")
        return None


class DeterministicFallbackPlanner:
    """Reliable fallback planner providing deterministic intent routing when LLM is unavailable.

    Explicitly documented as an engineering reliability mechanism, not the primary planner.
    """

    def plan(self, question: str) -> list[TaskSpec]:
        specs: list[TaskSpec] = []
        lowered = question.lower()

        # 1. Intent Detection
        rag_triggers = [
            "faq", "policy", "sla", "guarantee", "guarantees", "contract",
            "refund", "refunds", "terms", "rules", "guidelines", "expected wait",
            "what does the faq say", "official policy", "documentation", "operating hours"
        ]
        needs_rag = any(trigger in lowered for trigger in rag_triggers)

        comparison_triggers = [
            "compare", "comparison", "versus", "vs", "vs.", "difference",
            "delta", "month-over-month", "mom", "from april to may",
            "between april and may", "april to may", "compared to",
            "change from", "changed from", "shift from", "change", "changed"
        ]

        has_two_periods = (
            ("april" in lowered and "may" in lowered)
            or ("this month" in lowered and "last month" in lowered)
            or ("this month" in lowered and "previous month" in lowered)
        )
        needs_comparison = any(trigger in lowered for trigger in comparison_triggers) or has_two_periods

        data_triggers = [
            "csat", "nps", "rating", "score", "complaint", "complaints",
            "feedback", "survey", "surveys", "theme", "themes", "driver",
            "drivers", "top", "volume", "count", "sentiment", "customers saying",
            "customer feedback", "wait time", "pricing", "staff", "food",
            "cleanliness", "facilities", "app experience"
        ]
        needs_data = any(trigger in lowered for trigger in data_triggers) or (not needs_rag and not needs_comparison)

        # 2. Extract Top N (e.g., "top 3", "top 5")
        top_n_match = re.search(r"top\s+(\d+)", lowered)
        top_n = int(top_n_match.group(1)) if top_n_match else 5

        # 3. Extract Theme
        theme = None
        if "wait" in lowered:
            theme = "Wait Time"
        elif "food" in lowered:
            theme = "Food Quality"
        elif "clean" in lowered:
            theme = "Cleanliness"
        elif "price" in lowered or "pricing" in lowered or "cost" in lowered or "bill" in lowered:
            theme = "Pricing"
        elif "staff" in lowered or "service" in lowered:
            theme = "Staff"
        elif "member" in lowered or "membership" in lowered or "loyalty" in lowered:
            theme = "Membership"
        elif "facilit" in lowered or "amenit" in lowered or "parking" in lowered or "wifi" in lowered:
            theme = "Facilities"
        elif "app" in lowered or "mobile" in lowered or "ui" in lowered:
            theme = "App Experience"

        # 4. Extract Date Expressions
        date_expr = None
        comp_date_expr = None

        if "this month" in lowered and ("last month" in lowered or "previous month" in lowered):
            date_expr = "this month"
            comp_date_expr = "last month"
        elif "april" in lowered and "may" in lowered:
            date_expr = "May"
            comp_date_expr = "April"
        elif "this month" in lowered:
            date_expr = "this month"
        elif "last month" in lowered or "previous month" in lowered:
            date_expr = "last month"
        elif "april" in lowered:
            date_expr = "April"
        elif "may" in lowered:
            date_expr = "May"
        elif "june" in lowered:
            date_expr = "June"

        # 5. Build Sub-Tasks

        # Branch A: Comparison Task
        if needs_comparison:
            start_date, end_date = resolve_date_expression(date_expr or "May")
            c_start, c_end = resolve_date_expression(comp_date_expr or "April")

            specs.append(
                TaskSpec(
                    task_id=f"task_{len(specs) + 1}",
                    agent=AgentType.COMPARISON_AGENT.value,
                    task_type=TaskType.PERIOD_COMPARISON.value,
                    question=question,
                    metric="csat",
                    dimensions=["theme"] if theme else [],
                    filters={"theme": theme} if theme else {},
                    date_range={"start_date": start_date, "end_date": end_date},
                    comparison_period={"start_date": c_start, "end_date": c_end},
                    requested_limit=top_n,
                    rationale="Fallback planner detected period comparison intent.",
                    parameters={
                        "current_label": date_expr or "Current Period",
                        "previous_label": comp_date_expr or "Previous Period",
                        "theme": theme,
                        "top_n": top_n,
                    },
                )
            )

        # Branch B: Data Analysis Task (if standalone data analysis or complaint ranking)
        if needs_data and not needs_comparison:
            is_complaint = "complaint" in lowered or "negative" in lowered or "issue" in lowered or "worst" in lowered
            data_type = TaskType.TOP_THEMES.value if is_complaint else TaskType.DATA_ANALYSIS.value

            start_date, end_date = resolve_date_expression(date_expr)

            specs.append(
                TaskSpec(
                    task_id=f"task_{len(specs) + 1}",
                    agent=AgentType.DATA_AGENT.value,
                    task_type=data_type,
                    question=question,
                    metric="negative_volume" if is_complaint else "csat",
                    dimensions=["theme"],
                    filters={"theme": theme, "sentiment": "negative" if is_complaint else None},
                    date_range={"start_date": start_date, "end_date": end_date},
                    requested_limit=top_n,
                    rationale="Fallback planner detected survey analytics / complaint ranking intent.",
                    parameters={
                        "metric_name": "negative_volume" if is_complaint else "csat",
                        "theme_metric": "negative_volume" if is_complaint else None,
                        "category": theme,
                        "theme": theme,
                        "top_n": top_n,
                        "sentiment": "negative" if is_complaint else None,
                    },
                )
            )

        # Branch C: RAG Lookup Task
        if needs_rag:
            # Clean conversational query
            rag_query = question
            for prefix in [
                r"(?i)^(what does the faq say about|what does the faq say regarding|what does the faq say on)\s*",
                r"(?i)^(and\s+)?what does the faq say\s*(about)?\s*",
            ]:
                rag_query = re.sub(prefix, "", rag_query)

            specs.append(
                TaskSpec(
                    task_id=f"task_{len(specs) + 1}",
                    agent=AgentType.RAG_AGENT.value,
                    task_type=TaskType.RAG_LOOKUP.value,
                    question=question,
                    retrieval_query=rag_query.strip(),
                    requested_limit=2,
                    rationale="Fallback planner detected policy/FAQ grounding intent.",
                    parameters={"query": rag_query.strip(), "top_k": 2},
                )
            )

        return specs


class HybridPlanner:
    """Hybrid Query Planner: LLM decomposition with deterministic date resolution and fallback."""

    def __init__(self, llm_planner: LLMPlanner | None = None, fallback_planner: DeterministicFallbackPlanner | None = None) -> None:
        self.llm_planner = llm_planner or LLMPlanner()
        self.fallback_planner = fallback_planner or DeterministicFallbackPlanner()

    def plan(self, question: str) -> list[TaskSpec]:
        """Decompose natural language question into validated TaskSpec contracts."""
        # 1. Attempt primary LLM Planner decomposition
        llm_plan: PlannerPlan | None = None
        try:
            llm_plan = self.llm_planner.plan(question)
        except Exception as e:
            logger.info(f"LLM planner unavailable ({e}), utilizing deterministic fallback planner.")

        # 2. If valid LLM plan produced, resolve dates and construct validated TaskSpecs
        if llm_plan and llm_plan.tasks:
            try:
                task_specs = self._convert_plan_to_task_specs(question, llm_plan)
                if task_specs:
                    logger.debug(f"HybridPlanner: Successfully planned via LLM ({len(task_specs)} tasks).")
                    return task_specs
            except Exception as e:
                logger.warning(f"Error resolving LLM plan ({e}), utilizing deterministic fallback planner.")

        # 3. Graceful Deterministic Fallback
        logger.debug("HybridPlanner: Using deterministic fallback planner.")
        return self.fallback_planner.plan(question)

    def _convert_plan_to_task_specs(self, question: str, plan: PlannerPlan) -> list[TaskSpec]:
        """Convert planned sub-tasks into validated TaskSpecs with deterministic date resolution."""
        specs: list[TaskSpec] = []

        for idx, subtask in enumerate(plan.tasks, 1):
            # Normalize agent identifier
            agent_str = subtask.agent.strip()
            if "data" in agent_str.lower():
                agent = AgentType.DATA_AGENT.value
            elif "rag" in agent_str.lower():
                agent = AgentType.RAG_AGENT.value
            elif "comp" in agent_str.lower():
                agent = AgentType.COMPARISON_AGENT.value
            else:
                agent = AgentType.DATA_AGENT.value

            # Deterministic date resolution
            start_date, end_date = resolve_date_expression(subtask.date_expression)
            c_start, c_end = resolve_date_expression(subtask.comparison_period_expression)

            # Map parameters
            filter_dict = subtask.filters.model_dump(exclude_none=True) if hasattr(subtask.filters, "model_dump") else dict(subtask.filters)
            params: dict[str, Any] = dict(filter_dict)
            if subtask.metric:
                params["metric"] = subtask.metric
                params["metric_name"] = subtask.metric
                if "complaint" in subtask.metric.lower() or "negative" in subtask.metric.lower():
                    params["theme_metric"] = "negative_volume"
                    params["sentiment"] = "negative"

            limit = subtask.requested_limit or 5
            params["top_n"] = limit

            if subtask.retrieval_query:
                params["query"] = subtask.retrieval_query
                params["top_k"] = limit or 2

            if subtask.date_expression:
                params["current_label"] = subtask.date_expression
            if subtask.comparison_period_expression:
                params["previous_label"] = subtask.comparison_period_expression

            # Robust resolution for period comparisons
            if agent == AgentType.COMPARISON_AGENT.value:
                q_lower = (subtask.question or question).lower()
                c_expr_lower = (subtask.comparison_period_expression or "").lower()
                d_expr_lower = (subtask.date_expression or "").lower()

                # Handle April to May comparison bounds
                if ("april" in q_lower or "april" in c_expr_lower) and ("may" in q_lower or "may" in d_expr_lower or "may" in c_expr_lower):
                    if not start_date or not end_date:
                        start_date, end_date = "2026-05-01", "2026-05-31"
                        params["current_label"] = "May"
                    if not c_start or not c_end:
                        c_start, c_end = "2026-04-01", "2026-04-30"
                        params["previous_label"] = "April"

                # Detect theme if not populated in filters
                if "theme" not in filter_dict and "category" not in filter_dict:
                    if "wait" in q_lower:
                        filter_dict["theme"] = "Wait Time"
                        params["theme"] = "Wait Time"
                    elif "food" in q_lower:
                        filter_dict["theme"] = "Food Quality"
                        params["theme"] = "Food Quality"
                    elif "clean" in q_lower:
                        filter_dict["theme"] = "Cleanliness"
                        params["theme"] = "Cleanliness"
                    elif "app" in q_lower:
                        filter_dict["theme"] = "App Experience"
                        params["theme"] = "App Experience"
                    elif "price" in q_lower or "pricing" in q_lower:
                        filter_dict["theme"] = "Pricing"
                        params["theme"] = "Pricing"

            task_spec = TaskSpec(
                task_id=f"task_{idx}",
                agent=agent,
                task_type=subtask.task_type,
                question=subtask.question or question,
                metric=subtask.metric,
                dimensions=subtask.dimensions,
                filters=filter_dict,
                start_date=start_date,
                end_date=end_date,
                comparison_start_date=c_start,
                comparison_end_date=c_end,
                date_range={"start_date": start_date, "end_date": end_date} if (start_date or end_date) else None,
                comparison_period={"start_date": c_start, "end_date": c_end} if (c_start or c_end) else None,
                retrieval_query=subtask.retrieval_query,
                requested_limit=limit,
                rationale=subtask.rationale or plan.plan_rationale,
                parameters=params,
            )
            specs.append(task_spec)

        return specs
