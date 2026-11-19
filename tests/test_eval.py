"""Evaluation framework tests: metric computation and report rendering."""
from __future__ import annotations

from dataclasses import dataclass, field

from fabpilot.eval.metrics import (
    NUMERIC_METRICS,
    compute_metrics,
    task_success,
    tool_accuracy,
)
from fabpilot.eval.report import aggregate, render_markdown, render_json


@dataclass
class _Config:
    use_supervisor: bool = True
    use_diagnosis: bool = True
    use_rag: bool = True


@dataclass
class _GroundTruth:
    equipment_id: str = ""
    channel: str = ""
    root_cause: str | None = None
    relevant_sources: list[str] = field(default_factory=list)
    anomaly_scenario: str | None = None
    forecast_cut_fraction: float = 0.8


@dataclass
class _Scenario:
    expected_agents: list[str] = field(default_factory=list)
    ground_truth: _GroundTruth = field(default_factory=_GroundTruth)


def test_task_success_requires_completed_and_report():
    assert task_success({"status": "completed", "final_report": "ok"}) == 1.0
    assert task_success({"status": "running", "final_report": "ok"}) == 0.0
    assert task_success({"status": "completed", "final_report": ""}) == 0.0


def test_tool_accuracy_counts_ok():
    state = {"tool_history": [{"ok": True}, {"ok": False}, {"ok": True}]}
    out = tool_accuracy(state)
    assert out["tool_calls"] == 3
    assert out["tool_error_count"] == 1
    assert abs(out["tool_accuracy"] - 2 / 3) < 1e-9


def test_compute_metrics_returns_numeric_bundle():
    # A completed run with one delegation to the expected agent.
    state = {
        "status": "completed",
        "final_report": "root cause: mfc drift",
        "tool_history": [{"ok": True, "name": "sensor_data"}],
        "delegation_records": [{"agent": "diagnosis"}],
        "hypotheses": [{"id": "h1", "description": "mfc drift", "status": "supported", "priority": 0.9}],
        "evidence": [{"hypothesis_id": "h1"}],
        "trace": [{"decision": "x"}, {"decision": "y"}],
        "anomaly_results": [],
        "errors": [],
    }
    scenario = _Scenario(
        expected_agents=["diagnosis"],
        ground_truth=_GroundTruth(root_cause="mfc drift", relevant_sources=[]),
    )
    metrics = compute_metrics(state, scenario, _Config(), harness=None, wall_time_s=0.5)
    assert metrics["task_success"] == 1.0
    assert metrics["routing_accuracy"] == 1.0
    assert metrics["diagnosis_top_k"] == 1.0
    assert metrics["efficiency_tool_calls"] == 1
    assert metrics["robustness"] == 1.0


def test_report_aggregate_averages_numeric_metrics():
    payload = {
        "configs": ["full", "no_rag"],
        "scenarios": ["A"],
        "results": [
            {
                "config": "full",
                "scenario": "A",
                "metrics": {"task_success": 1.0, "tool_error_rate": 0.0},
                "summary": {"status": "completed", "final_report": "r1", "n_tool_calls": 2, "delegated_agents": ["diagnosis"], "n_hypotheses": 1},
            },
            {
                "config": "full",
                "scenario": "A",
                "metrics": {"task_success": 0.0, "tool_error_rate": 0.5},
                "summary": {"status": "failed", "final_report": "r2", "n_tool_calls": 1, "delegated_agents": [], "n_hypotheses": 0},
            },
            {
                "config": "no_rag",
                "scenario": "A",
                "metrics": {"task_success": 1.0, "tool_error_rate": 0.0},
                "summary": {"status": "completed", "final_report": "r3", "n_tool_calls": 1, "delegated_agents": [], "n_hypotheses": 0},
            },
        ],
    }
    agg = aggregate(payload)
    full = agg["per_config"]["full"]
    assert full["task_success"] == 0.5
    assert full["tool_error_rate"] == 0.25
    md = render_markdown(payload)
    assert "# FabPilot Evaluation Report" in md
    assert render_json(payload).startswith("{")
    assert "no_rag" in render_json(payload)


def test_numeric_metrics_is_complete():
    assert "task_success" in NUMERIC_METRICS
    assert "ts_anomaly_f1" in NUMERIC_METRICS
    assert "robustness" in NUMERIC_METRICS
