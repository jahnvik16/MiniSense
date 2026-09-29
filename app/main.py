"""MiniSense CLI — Survey Feedback Multi-Agent Analysis System."""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

# Bootstrap workspace root into sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Suppress noisy external library warnings for clean CLI output unless in debug
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from app.config import settings
from app.services.survey_service import SurveyService
from app.rag.retriever import FAQRetriever
from app.agents.orchestrator import OrchestratorAgent


def format_metrics(metrics: dict) -> str:
    """Format supporting metrics into a clean, human-readable console summary."""
    lines = []
    if "comparison" in metrics:
        comp = metrics["comparison"]
        curr = comp.get("current_period", {})
        prev = comp.get("previous_period", {})
        changes = comp.get("metric_changes", {})

        lines.append("  [Period / Cohort Comparison]")
        lines.append(f"    * Baseline ({prev.get('label', 'Previous')}): CSAT = {prev.get('csat', 0.0):.1f}% | Avg Rating = {prev.get('average_rating', 0.0):.2f} | Volume = {prev.get('count', 0):,}")
        lines.append(f"    * Current  ({curr.get('label', 'Current')}):  CSAT = {curr.get('csat', 0.0):.1f}% | Avg Rating = {curr.get('average_rating', 0.0):.2f} | Volume = {curr.get('count', 0):,}")
        lines.append(f"    * Shifts:   CSAT Delta = {changes.get('csat_delta', 0.0):+.2f}% ({changes.get('csat_pct_change', 0.0):+.1f}%) | Rating Delta = {changes.get('average_rating_delta', 0.0):+.2f} | Count Delta = {int(changes.get('count_delta', 0)):+d}")

    if "survey_metrics" in metrics:
        sm = metrics["survey_metrics"]
        lines.append("  [Survey Dataset Metrics]")
        lines.append(f"    * Total Evaluated Responses : {sm.get('response_count', 0):,}")
        lines.append(f"    * Average Rating (1-5 scale): {sm.get('average_rating', 0.0):.2f}")
        lines.append(f"    * Overall CSAT (% >= 4)     : {sm.get('csat', 0.0):.1f}%")
        top_themes = sm.get("top_themes", [])
        if top_themes:
            lines.append("    * Top Driver Themes:")
            for t in top_themes:
                theme_name = t.get("theme", "Unknown")
                count = t.get("count", 0)
                csat = t.get("csat", 0.0)
                rating = t.get("average_rating", 0.0)
                lines.append(f"      - {theme_name:<16}: {count:,} responses | CSAT: {csat:.1f}% | Avg Rating: {rating:.2f}")

    return "\n".join(lines) if lines else "  No structured metrics recorded."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniSense — Survey Analysis Multi-Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.main "How did customer satisfaction change from April to May?"
  python -m app.main "What are the top 3 complaints in May?"
  python -m app.main "Why are wait-time complaints increasing and what does the FAQ say?"
  python -m app.main --debug "Compare CSAT between enterprise and self_serve"
        """,
    )
    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        default=None,
        help="Natural language business question (positional)",
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default=None,
        help="Natural language business question (flag)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: display planner decomposition, sub-agent dispatches, and intermediate state",
    )
    args = parser.parse_args()

    question = args.query or args.question or "How did customer satisfaction change from April to May?"

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # 1. Pre-flight initialization
    print("=" * 78)
    print("  MiniSense: Survey Feedback Multi-Agent System (LangGraph)")
    print("=" * 78)

    # Load configuration
    provider_str = settings.llm_provider
    print(f"  [Init] Configuration loaded (LLM Provider: {provider_str}, Model: {settings.llm_model})")

    # Load survey dataset
    try:
        survey_service = SurveyService()
        record_count = survey_service.get_response_count()
        print(f"  [Init] Survey dataset verified ({record_count:,} records loaded from {settings.surveys_file.name})")
    except Exception as e:
        print(f"  [Warning] Could not load surveys: {e}. Run 'python scripts/generate_data.py' to generate data.")

    # Load FAISS index
    try:
        retriever = FAQRetriever()
        index_status = "active" if retriever.index_path.exists() else "uninitialized"
        print(f"  [Init] Knowledge base verified (FAISS index {index_status}, model: {settings.embedding_model})")
    except Exception as e:
        print(f"  [Warning] Could not load FAISS index: {e}. Run 'python scripts/build_index.py' to build index.")

    logging.getLogger("google_genai").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    print(f"\n  [Question] \"{question}\"")

    # 2. Orchestration execution
    orchestrator = OrchestratorAgent()

    if args.debug:
        print("\n" + "-" * 78)
        print("  [DEBUG: Planner Decomposition]")
        tasks = orchestrator.plan(question)
        for idx, t in enumerate(tasks, 1):
            print(f"    Task {idx}: [{t.task_type}] -> {t.agent}")
            print(f"      Question: {t.question}")
            if t.start_date or t.end_date:
                print(f"      Date Range: {t.start_date} to {t.end_date}")
            if t.comparison_start_date or t.comparison_end_date:
                print(f"      Comparison Dates: {t.comparison_start_date} to {t.comparison_end_date}")
            print(f"      Parameters: {t.parameters}")
        print("-" * 78)

    # Execute LangGraph
    print("  [Executing] Dispatching to specialized agents via LangGraph...")
    initial_state = {
        "question": question,
        "planned_tasks": [],
        "active_agent_types": [],
        "collected_results": {},
    }
    final_state = orchestrator.graph.invoke(initial_state)
    final_answer = final_state.get("final_answer")

    if not final_answer:
        final_answer = orchestrator.synthesizer.synthesize(question=question)

    if args.debug:
        print("\n" + "-" * 78)
        print("  [DEBUG: Intermediate Agent Outputs]")
        if final_state.get("data_result"):
            print(f"    * DataAgent: {final_state['data_result'].summary}")
        if final_state.get("rag_result"):
            print(f"    * RAGAgent: {len(final_state['rag_result'].retrieved_chunks)} chunk(s) retrieved (reliable={final_state['rag_result'].reliable})")
        if final_state.get("comparison_result"):
            print(f"    * ComparisonAgent: {final_state['comparison_result'].summary}")
        print("-" * 78)

    # 3. Clean, readable executive display
    print("\n" + "=" * 78)
    print("  BUSINESS EXECUTIVE ANSWER")
    print("=" * 78)
    print(f"\n  {final_answer.answer}\n")

    if final_answer.supporting_metrics:
        print("-" * 78)
        print("  SUPPORTING METRICS (DETERMINISTIC PYTHON):")
        print(format_metrics(final_answer.supporting_metrics))

    if final_answer.retrieved_sources:
        print("-" * 78)
        print("  GROUNDING KNOWLEDGE BASE SOURCES:")
        for source in final_answer.retrieved_sources:
            lines = source.split("\n", 1)
            header = lines[0]
            body = lines[1].strip() if len(lines) > 1 else ""
            print(f"    * [{header}]")
            if body:
                print(f"      \"{body[:200]}...\"" if len(body) > 200 else f"      \"{body}\"")

    if final_answer.assumptions:
        print("-" * 78)
        print("  EXPLICIT ASSUMPTIONS:")
        for asm in final_answer.assumptions:
            print(f"    * {asm}")

    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
