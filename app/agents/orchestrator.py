"""LangGraph Orchestration Layer for MiniSense.

Coordinates multi-agent query execution:
START -> Planner -> Task Routing -> (DataAgent / RAGAgent / ComparisonAgent) -> Collect -> Synthesis -> FinalAnswer -> END
"""

import re
from typing import Any, TypedDict
from langgraph.graph import END, START, StateGraph

from app.agents.comparison_agent import ComparisonAgent
from app.agents.data_agent import DataAgent
from app.agents.rag_agent import RAGAgent
from app.models.schemas import (
    AgentType,
    ComparisonAgentResult,
    DataAgentResult,
    FinalAnswer,
    RAGAgentResult,
    TaskSpec,
    TaskType,
)
from app.services.synthesizer import Synthesizer


class AgentGraphState(TypedDict, total=False):
    """Structured state passed across nodes in the LangGraph orchestration flow."""
    question: str
    planned_tasks: list[TaskSpec]
    active_agent_types: list[str]
    data_result: DataAgentResult | None
    rag_result: RAGAgentResult | None
    comparison_result: ComparisonAgentResult | None
    collected_results: dict[str, Any]
    final_answer: FinalAnswer | None
    error: str | None


class OrchestratorAgent:
    """Orchestrator and Planner constructing and managing the LangGraph execution flow."""

    def __init__(
        self,
        data_agent: DataAgent | None = None,
        rag_agent: RAGAgent | None = None,
        comparison_agent: ComparisonAgent | None = None,
        synthesizer: Synthesizer | None = None,
    ) -> None:
        self.data_agent = data_agent or DataAgent()
        self.rag_agent = rag_agent or RAGAgent()
        self.comparison_agent = comparison_agent or ComparisonAgent(data_agent=self.data_agent)
        self.synthesizer = synthesizer or Synthesizer()
        self.graph = self._build_graph()

    def plan(self, question: str) -> list[TaskSpec]:
        """Decompose a natural language business question into structured TaskSpecs.

        Selectively routes only to agents required by the intent:
        - "What are the top 3 complaints in May?" -> DataAgent only
        - "How did CSAT change from April to May?" -> DataAgent + ComparisonAgent
        - "Why are wait-time complaints increasing and what does the FAQ say?" -> DataAgent + RAGAgent
        """
        specs: list[TaskSpec] = []
        lowered = question.lower()

        # 1. Intent Detection
        rag_triggers = [
            "faq", "policy", "sla", "guarantee", "guarantees", "contract",
            "refund", "refunds", "terms", "rules", "downgrade", "downgraded",
            "cancellation", "what does the faq say", "official policy", "documentation"
        ]
        needs_rag = any(trigger in lowered for trigger in rag_triggers)

        comparison_triggers = [
            "compare", "comparison", "versus", "vs", "vs.", "difference",
            "delta", "month-over-month", "mom", "from april to may",
            "between april and may", "april to may", "compared to",
            "change from", "changed from", "shift from"
        ]
        has_period_change = ("april" in lowered and "may" in lowered and ("change" in lowered or "shift" in lowered))
        needs_comparison = any(trigger in lowered for trigger in comparison_triggers) or has_period_change

        data_triggers = [
            "csat", "nps", "rating", "score", "complaint", "complaints",
            "feedback", "survey", "surveys", "theme", "themes", "driver",
            "drivers", "top", "volume", "count", "sentiment", "wait-time",
            "wait time", "pricing", "staff", "food", "cleanliness",
            "app experience", "facilities", "enterprise", "self-serve", "self_serve"
        ]
        needs_data = any(trigger in lowered for trigger in data_triggers) or (not needs_rag and not needs_comparison)

        # 2. Extract extraction helpers
        # Dates
        has_april = "april" in lowered or "2026-04" in lowered
        has_may = "may" in lowered or "2026-05" in lowered

        # Top N extraction (e.g., "top 3", "top 5")
        top_n_match = re.search(r"top\s+(\d+)", lowered)
        top_n = int(top_n_match.group(1)) if top_n_match else 5

        # Cohort extraction
        cohort = None
        if "enterprise" in lowered:
            cohort = "enterprise"
        elif "self-serve" in lowered or "self_serve" in lowered or "self serve" in lowered:
            cohort = "self_serve"

        # Theme extraction
        theme = None
        if "wait" in lowered or "wait-time" in lowered:
            theme = "Wait Time"
        elif "food" in lowered:
            theme = "Food Quality"
        elif "clean" in lowered:
            theme = "Cleanliness"
        elif "pricing" in lowered or "price" in lowered or "billing" in lowered or "refund" in lowered:
            theme = "Pricing"
        elif "staff" in lowered:
            theme = "Staff"
        elif "member" in lowered or "membership" in lowered:
            theme = "Membership"
        elif "facility" in lowered or "facilities" in lowered:
            theme = "Facilities"
        elif "app" in lowered:
            theme = "App Experience"

        # 3. Build TaskSpecs

        # Branch A: Comparison Task
        if needs_comparison:
            is_period = (has_april and has_may) or "month-over-month" in lowered or "mom" in lowered
            if is_period:
                specs.append(
                    TaskSpec(
                        task_id=f"task_{len(specs) + 1}",
                        agent=AgentType.COMPARISON_AGENT.value,
                        task_type=TaskType.PERIOD_COMPARISON.value,
                        question=question,
                        instruction="Perform deterministic month-over-month comparison between May 2026 and April 2026",
                        start_date="2026-05-01",
                        end_date="2026-05-31",
                        comparison_start_date="2026-04-01",
                        comparison_end_date="2026-04-30",
                        parameters={
                            "current_label": "May 2026",
                            "previous_label": "April 2026",
                            "theme": theme,
                            "top_n": top_n,
                        },
                    )
                )
            else:
                # Cohort comparison (e.g. enterprise vs self_serve)
                cohort_a = "enterprise" if "enterprise" in lowered else "enterprise"
                cohort_b = "self_serve" if ("self-serve" in lowered or "self_serve" in lowered) else "self_serve"
                specs.append(
                    TaskSpec(
                        task_id=f"task_{len(specs) + 1}",
                        agent=AgentType.COMPARISON_AGENT.value,
                        task_type=TaskType.COMPARISON.value,
                        question=question,
                        instruction=f"Compare metrics between {cohort_a} and {cohort_b} cohorts",
                        parameters={
                            "cohort_a": cohort_a,
                            "cohort_b": cohort_b,
                            "metric": "csat",
                            "metric_name": "csat",
                            "top_n": top_n,
                        },
                    )
                )

        # Branch B: Data Analysis Task
        if needs_data:
            # Determine specific data task type
            is_complaint_query = "complaint" in lowered or "negative" in lowered or "driver" in lowered
            data_task_type = TaskType.TOP_THEMES.value if is_complaint_query else TaskType.DATA_ANALYSIS.value

            data_start = "2026-05-01" if has_may and not has_april else ("2026-04-01" if has_april and not has_may else None)
            data_end = "2026-05-31" if has_may and not has_april else ("2026-04-30" if has_april and not has_may else None)

            specs.append(
                TaskSpec(
                    task_id=f"task_{len(specs) + 1}",
                    agent=AgentType.DATA_AGENT.value,
                    task_type=data_task_type,
                    question=question,
                    instruction="Compute exact deterministic survey metrics and theme drivers",
                    start_date=data_start,
                    end_date=data_end,
                    parameters={
                        "metric_name": "negative_volume" if is_complaint_query else "csat",
                        "theme_metric": "negative_volume" if is_complaint_query else None,
                        "cohort": cohort,
                        "category": theme,
                        "theme": theme,
                        "top_n": top_n,
                        "sentiment": "negative" if is_complaint_query else None,
                    },
                )
            )

        # Branch C: RAG Lookup Task
        if needs_rag:
            # Construct a clean targeted retrieval query
            rag_query = question
            if "faq" in lowered or "policy" in lowered:
                # Remove conversational prefix if query is asking 'what does the faq say'
                rag_query = re.sub(r"(?i)^(what does the faq say about|what does the faq say regarding|what does the faq say on)\s*", "", question)
            specs.append(
                TaskSpec(
                    task_id=f"task_{len(specs) + 1}",
                    agent=AgentType.RAG_AGENT.value,
                    task_type=TaskType.RAG_LOOKUP.value,
                    question=question,
                    instruction="Retrieve grounded policy context from verified FAQ store",
                    parameters={"query": rag_query.strip(), "top_k": 2},
                )
            )

        return specs

    def _build_graph(self) -> Any:
        """Construct the compiled LangGraph StateGraph."""
        builder = StateGraph(AgentGraphState)

        # Node 1: Planner
        def planner_node(state: AgentGraphState) -> dict[str, Any]:
            question = state.get("question", "")
            tasks = self.plan(question)
            active_types = list({t.agent for t in tasks})
            return {
                "planned_tasks": tasks,
                "active_agent_types": active_types,
            }

        # Router: Inspects active_agent_types and returns nodes to execute
        def route_tasks(state: AgentGraphState) -> list[str]:
            active = state.get("active_agent_types", [])
            nodes: list[str] = []
            if AgentType.DATA_AGENT.value in active:
                nodes.append("data_node")
            if AgentType.RAG_AGENT.value in active:
                nodes.append("rag_node")
            if AgentType.COMPARISON_AGENT.value in active:
                nodes.append("comparison_node")
            return nodes or ["collect_node"]

        # Sub-Agent Execution Nodes
        def data_node(state: AgentGraphState) -> dict[str, Any]:
            tasks = [t for t in state.get("planned_tasks", []) if t.agent == AgentType.DATA_AGENT.value]
            if not tasks:
                return {}
            # Execute primary data task
            result = self.data_agent.run(tasks[0])
            return {"data_result": result}

        def rag_node(state: AgentGraphState) -> dict[str, Any]:
            tasks = [t for t in state.get("planned_tasks", []) if t.agent == AgentType.RAG_AGENT.value]
            if not tasks:
                return {}
            # Execute primary RAG task
            result = self.rag_agent.run(tasks[0])
            return {"rag_result": result}

        def comparison_node(state: AgentGraphState) -> dict[str, Any]:
            tasks = [t for t in state.get("planned_tasks", []) if t.agent == AgentType.COMPARISON_AGENT.value]
            if not tasks:
                return {}
            # Execute primary comparison task
            result = self.comparison_agent.run(tasks[0])
            return {"comparison_result": result}

        # Collect Node: Aggregates structured agent results
        def collect_node(state: AgentGraphState) -> dict[str, Any]:
            collected: dict[str, Any] = {}
            if state.get("data_result"):
                collected["data"] = state["data_result"]
            if state.get("rag_result"):
                collected["rag"] = state["rag_result"]
            if state.get("comparison_result"):
                collected["comparison"] = state["comparison_result"]
            return {"collected_results": collected}

        # Synthesis Node: Generates FinalAnswer
        def synthesis_node(state: AgentGraphState) -> dict[str, Any]:
            question = state.get("question", "")
            data_res = state.get("data_result")
            rag_res = state.get("rag_result")
            comp_res = state.get("comparison_result")

            final_answer = self.synthesizer.synthesize(
                question=question,
                data_result=data_res,
                rag_result=rag_res,
                comparison_result=comp_res,
            )
            return {"final_answer": final_answer}

        # Add Nodes to Graph
        builder.add_node("planner", planner_node)
        builder.add_node("data_node", data_node)
        builder.add_node("rag_node", rag_node)
        builder.add_node("comparison_node", comparison_node)
        builder.add_node("collect_node", collect_node)
        builder.add_node("synthesis", synthesis_node)

        # Wire Edges
        builder.add_edge(START, "planner")
        builder.add_conditional_edges(
            "planner",
            route_tasks,
            ["data_node", "rag_node", "comparison_node", "collect_node"],
        )
        builder.add_edge("data_node", "collect_node")
        builder.add_edge("rag_node", "collect_node")
        builder.add_edge("comparison_node", "collect_node")
        builder.add_edge("collect_node", "synthesis")
        builder.add_edge("synthesis", END)

        return builder.compile()

    def run(self, question: str) -> FinalAnswer:
        """Execute the LangGraph orchestration flow and return the structured FinalAnswer."""
        initial_state: AgentGraphState = {
            "question": question,
            "planned_tasks": [],
            "active_agent_types": [],
            "collected_results": {},
        }
        final_state = self.graph.invoke(initial_state)
        answer = final_state.get("final_answer")
        if not answer:
            # Fallback guarantee
            answer = self.synthesizer.synthesize(question=question)
        return answer

    def invoke(self, state: dict[str, Any] | str) -> dict[str, Any]:
        """Convenience invocation wrapper supporting raw string or dictionary input."""
        if isinstance(state, str):
            payload: AgentGraphState = {"question": state, "planned_tasks": [], "active_agent_types": []}
        else:
            payload = state  # type: ignore[assignment]
        return self.graph.invoke(payload)  # type: ignore[no-any-return]
