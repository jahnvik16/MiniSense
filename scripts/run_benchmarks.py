"""Empirical benchmark evaluation script for MiniSense.

Executes the 3 benchmark scenarios and 5 unseen natural-language queries
through the LangGraph multi-agent pipeline and records the planner decomposition,
supporting deterministic metrics, retrieved FAQ chunks, and synthesized executive narrative.
"""

import json
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.agents.orchestrator import OrchestratorAgent


def run_all_evaluations() -> None:
    orchestrator = OrchestratorAgent()

    benchmark_questions = [
        # 3 Required Benchmark Scenarios
        ("Scenario 1 (Analytics / Ranking)", "What are the top 3 complaints this month?"),
        ("Scenario 2 (Longitudinal Comparison)", "How did wait-time experience change from April to May?"),
        ("Scenario 3 (Hybrid Analytics + RAG)", "How did wait-time experience change from April to May, and what does the FAQ say about expected wait times?"),
        
        # 5 Unseen Generalization Questions
        ("Unseen 1 (Theme Specific Analytics)", "What are customers saying about food quality this month?"),
        ("Unseen 2 (Root Cause & Month Query)", "Why did cleanliness satisfaction drop in May?"),
        ("Unseen 3 (Channel Comparison)", "Compare kiosk vs mobile channel satisfaction."),
        ("Unseen 4 (Pure RAG Policy Query)", "What does the FAQ say about handling customer complaints and refunds?"),
        ("Unseen 5 (Facility Complaint Themes)", "What are the main complaint themes across our facilities?"),

        # 3 Additional Paraphrased Scenarios
        ("Paraphrase 1 (Complaints Ranking)", "Which areas received the highest volume of customer complaints during the current month?"),
        ("Paraphrase 2 (Longitudinal Delta)", "What was the difference in customer satisfaction ratings for wait times between April 2026 and May 2026?"),
        ("Paraphrase 3 (Hybrid Policy & Analytics)", "How long should customers anticipate waiting for their orders according to policy, and how did recent pickup satisfaction perform?"),
    ]

    results = []

    for category, question in benchmark_questions:
        print("\n" + "=" * 80)
        print(f"[{category}]")
        print(f"Question: {question}")
        print("=" * 80)

        # 1. Inspect planner decomposition
        tasks = orchestrator.plan(question)
        print(f"Planner Tasks ({len(tasks)}):")
        for t in tasks:
            print(f"  -> Agent: {t.agent} | Type: {t.task_type} | Dates: {t.start_date} to {t.end_date} | Comp: {t.comparison_start_date} to {t.comparison_end_date} | Query: {t.retrieval_query}")

        # 2. Run full graph execution
        final_answer = orchestrator.run(question)

        print("\nFinal Synthesized Business Narrative:")
        print(f"  {final_answer.answer}")

        print("\nSupporting Metrics:")
        print(f"  {json.dumps(final_answer.supporting_metrics, indent=4)}")

        print("\nRetrieved Sources:")
        print(f"  {final_answer.retrieved_sources}")

        print("\nAssumptions:")
        print(f"  {final_answer.assumptions}")

        results.append({
            "category": category,
            "question": question,
            "tasks": [t.model_dump() for t in tasks],
            "answer": final_answer.answer,
            "supporting_metrics": final_answer.supporting_metrics,
            "retrieved_sources": final_answer.retrieved_sources,
            "assumptions": final_answer.assumptions,
        })

    with open("evaluation/benchmark_runs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n\nAll 8 evaluation scenarios completed successfully and saved to evaluation/benchmark_runs.json!")


if __name__ == "__main__":
    run_all_evaluations()
