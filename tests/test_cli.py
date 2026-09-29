"""Tests for MiniSense CLI entrypoint and command line execution."""

import subprocess
import sys


def test_cli_positional_query_execution() -> None:
    """Test CLI execution using positional question argument."""
    cmd = [
        sys.executable,
        "-m",
        "app.main",
        "How did customer satisfaction change from April to May?",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output = res.stdout

    assert "MiniSense: Survey Feedback Multi-Agent System (LangGraph)" in output
    assert "BUSINESS EXECUTIVE ANSWER" in output
    assert "SUPPORTING METRICS (DETERMINISTIC PYTHON)" in output
    assert "Period / Cohort Comparison" in output
    assert "CSAT Delta" in output


def test_cli_debug_mode_output() -> None:
    """Test CLI output with --debug flag showing planner decomposition."""
    cmd = [
        sys.executable,
        "-m",
        "app.main",
        "--debug",
        "What are the top 3 complaints in May?",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output = res.stdout

    assert "[DEBUG: Planner Decomposition]" in output
    assert "DataAgent" in output
    assert "BUSINESS EXECUTIVE ANSWER" in output
    assert "Pricing" in output or "CSAT" in output
