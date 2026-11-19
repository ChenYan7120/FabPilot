"""FabPilot evaluation framework.

Public surface:
  * :mod:`fabpilot.eval.tasks` — scenarios with ground truth + scripted routing.
  * :mod:`fabpilot.eval.scripted` — deterministic ``ScriptedLLM`` test double.
  * :mod:`fabpilot.eval.metrics` — nine metric categories computed from real outputs.
  * :mod:`fabpilot.eval.runner` — runs every (config, scenario) pair.
  * :mod:`fabpilot.eval.report` — renders JSON / Markdown reports.
"""
from fabpilot.eval.metrics import NUMERIC_METRICS, compute_metrics
from fabpilot.eval.report import render_json, render_markdown, write_report
from fabpilot.eval.runner import CONFIGS, EvalConfig, Harness, build_harness, run_evaluation, run_scenario
from fabpilot.eval.tasks import SCENARIOS, GroundTruth, Scenario

__all__ = [
    "CONFIGS",
    "EvalConfig",
    "GroundTruth",
    "Harness",
    "NUMERIC_METRICS",
    "SCENARIOS",
    "Scenario",
    "build_harness",
    "compute_metrics",
    "render_json",
    "render_markdown",
    "run_evaluation",
    "run_scenario",
    "write_report",
]
