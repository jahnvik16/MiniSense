"""LangGraph Orchestration Layer for MiniSense.

Coordinates multi-agent query execution:
START -> Planner -> Task Routing -> (DataAgent / RAGAgent / ComparisonAgent) -> Collect -> Synthesis -> FinalAnswer -> END
"""

import re
from typing import Any, TypedDict
from langgraph.graph import END, START, StateGraph

from app.agents.comparison_agent import ComparisonAgent
from app.agents.data_agent import DataAgent
from app.agents.planner import HybridPlanner
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
    """Orchestrator managing the LangGraph execution flow with Hybrid LLM planning."""

    def __init__(
        self,
        data_agent: DataAgent | None = None,
        rag_agent: RAGAgent | None = None,
        comparison_agent: ComparisonAgent | None = None,
        synthesizer: Synthesizer | None = None,
        planner: HybridPlanner | None = None,
    ) -> None:
        self.data_agent = data_agent or DataAgent()
        self.rag_agent = rag_agent or RAGAgent()
        self.comparison_agent = comparison_agent or ComparisonAgent(data_agent=self.data_agent)
        self.synthesizer = synthesizer or Synthesizer()
        self.planner = planner or HybridPlanner()
        self.graph = self._build_graph()

    def plan(self, question: str) -> list[TaskSpec]:
        """Decompose a natural language business question into structured TaskSpecs using HybridPlanner."""
        return self.planner.plan(question)

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
