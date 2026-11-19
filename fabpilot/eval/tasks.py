"""Evaluation scenarios (business tasks) with ground truth and scripted routing.

Each scenario scripts the *routing decisions* (which agent to delegate to, which
tools to call and with which arguments) via a ``ScriptedLLM``. The tools themselves
run for real against the seeded database / forecast registry / knowledge base, so
metrics reflect real outputs. Four scenarios cover the system's core capabilities:

  A. forecast — pure regression (Time-Series Agent)
  B. anomaly — anomaly detection (Time-Series Agent)
  C. forecast-driven diagnosis — anomaly -> correlation -> hypothesis-evidence memory
  D. knowledge QA — hybrid RAG retrieval with citations (Knowledge Agent)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from fabpilot.eval.scripted import Step, ScriptContext, stop, tool

# Ground-truth root cause strings match fabpilot.data.generator.SCENARIOS.
MFC_DRIFT = "MFC mass flow controller drift / control valve stiction"


@dataclass(frozen=True)
class GroundTruth:
    equipment_id: str
    channel: str | None = None
    root_cause: str | None = None
    anomaly_scenario: str | None = None
    relevant_sources: tuple[str, ...] = ()
    forecast_cut_fraction: float = 0.9


@dataclass(frozen=True)
class Scenario:
    id: str
    name: str
    user_task: str
    expected_agents: tuple[str, ...]
    ground_truth: GroundTruth
    build_script: Callable[[], list[Step]] = field(repr=False, compare=False)
    build_baseline_script: Callable[[], list[Step]] = field(repr=False, compare=False, default=None)


# --------------------------------------------------------------------------- #
# A. Forecast
# --------------------------------------------------------------------------- #


def _scenario_a() -> list[Step]:
    return [
        tool("delegate_to_timeseries", {
            "task": "Forecast the gas_flow channel for the next 16 points and summarize the trend.",
            "equipment_id": "ETCH-01",
        }),
        tool("forecast", {"equipment_id": "ETCH-01", "target": "gas_flow", "horizon": 16}),
        stop("Forecast complete: gas_flow projected forward with the statistical baseline model."),
        tool("finish", {"report": "# Gas Flow Forecast\n\nA 16-point forward forecast of ETCH-01 gas_flow was produced by the statistical baseline model."}),
        stop("done"),
    ]


# --------------------------------------------------------------------------- #
# B. Anomaly detection
# --------------------------------------------------------------------------- #


def _scenario_b() -> list[Step]:
    return [
        tool("delegate_to_timeseries", {
            "task": "Detect anomalies in the gas_flow channel and report the flagged timestamps.",
            "equipment_id": "ETCH-01",
        }),
        tool("anomaly_detection", {
            "equipment_id": "ETCH-01",
            "channel": "gas_flow",
            "method": "rolling_zscore",
            "threshold": 3.0,
        }),
        stop("Anomaly detection complete: flagged points above the z-score threshold."),
        tool("finish", {"report": "# Anomaly Detection\n\nRolling z-score flagged anomalous gas_flow points on ETCH-01."}),
        stop("done"),
    ]


# --------------------------------------------------------------------------- #
# C. Forecast-driven diagnosis (hypothesis-evidence memory)
# --------------------------------------------------------------------------- #


def _scenario_c_evidence(ctx: ScriptContext):
    hyp_id = (ctx.data_for("add_hypothesis") or {})["hypothesis_id"]
    retrieval = ctx.data_for("document_retrieval") or {}
    refs = [d.get("source") for d in retrieval.get("results", []) if d.get("source")][:3]
    return tool("add_evidence", {
        "hypothesis_id": hyp_id,
        "source_tool": "correlation",
        "observation": "gas_flow and pressure are positively correlated during the anomaly window",
        "kind": "supporting",
        "confidence": 0.8,
        "query_params": {"channel_a": "gas_flow", "channel_b": "pressure", "method": "pearson"},
        "source_refs": refs,
    })


def _scenario_c_evaluate(ctx: ScriptContext):
    hyp_id = (ctx.data_for("add_hypothesis") or {})["hypothesis_id"]
    return tool("evaluate_hypothesis", {"hypothesis_id": hyp_id})


def _scenario_c() -> list[Step]:
    return [
        tool("delegate_to_timeseries", {
            "task": "Detect anomalies in gas_flow and measure its correlation with pressure.",
            "equipment_id": "ETCH-01",
        }),
        tool("anomaly_detection", {"equipment_id": "ETCH-01", "channel": "gas_flow", "threshold": 3.0}),
        tool("correlation", {
            "equipment_id": "ETCH-01",
            "channel_a": "gas_flow",
            "channel_b": "pressure",
            "method": "pearson",
        }),
        stop("gas_flow anomalies detected; gas_flow-pressure correlation computed."),
        tool("delegate_to_diagnosis", {
            "task": "Diagnose the root cause of the gas_flow anomaly on ETCH-01.",
            "equipment_id": "ETCH-01",
        }),
        tool("add_hypothesis", {
            "description": "MFC mass flow controller drift causing gas_flow to rise and chamber pressure to follow",
            "priority": 0.8,
        }),
        tool("document_retrieval", {
            "query": "MFC mass flow controller drift gas flow setpoint",
            "k": 5,
            "doc_type": "sop",
        }),
        tool("document_retrieval", {
            "query": "MFC mass flow controller drift gas flow setpoint",
            "k": 5,
            "doc_type": "fault_cases",
        }),
        _scenario_c_evidence,
        _scenario_c_evaluate,
        stop("Root cause: MFC mass flow controller drift / control valve stiction (supported)."),
        tool("finish", {
            "report": "# Diagnosis\n\nRoot cause: MFC mass flow controller drift / control valve stiction. "
            "Supported by the gas_flow-pressure correlation and matching fault-case precedent."
        }),
        stop("done"),
    ]


# --------------------------------------------------------------------------- #
# D. Knowledge QA
# --------------------------------------------------------------------------- #


def _scenario_d() -> list[Step]:
    return [
        tool("delegate_to_knowledge", {
            "task": "Retrieve the SOP and fault-case precedent for vacuum leak handling.",
            "equipment_id": "ETCH-02",
        }),
        tool("document_retrieval", {
            "query": "vacuum leak slow pump degradation chamber pressure rise",
            "k": 5,
        }),
        tool("similar_case_retrieval", {
            "query": "vacuum leak pressure rise",
            "k": 5,
        }),
        stop("Retrieved the vacuum-leak SOP and matching historical fault case with citations."),
        tool("finish", {
            "report": "# Knowledge Retrieval\n\nVacuum-leak SOP and fault-case precedent retrieved with source citations."
        }),
        stop("done"),
    ]


# --------------------------------------------------------------------------- #
# Baseline scripts (flat single-agent ReAct, no routing / memory / RAG)
# --------------------------------------------------------------------------- #


def _baseline_a() -> list[Step]:
    return [
        tool("forecast", {"equipment_id": "ETCH-01", "target": "gas_flow", "horizon": 16}),
        stop("Forecast complete: statistical baseline model projected ETCH-01 gas_flow 16 points ahead."),
    ]


def _baseline_b() -> list[Step]:
    return [
        tool("anomaly_detection", {
            "equipment_id": "ETCH-01", "channel": "gas_flow", "method": "rolling_zscore", "threshold": 3.0,
        }),
        stop("Anomaly detection complete: rolling z-score flagged anomalous gas_flow points."),
    ]


def _baseline_c() -> list[Step]:
    return [
        tool("anomaly_detection", {"equipment_id": "ETCH-01", "channel": "gas_flow", "threshold": 3.0}),
        tool("correlation", {
            "equipment_id": "ETCH-01", "channel_a": "gas_flow", "channel_b": "pressure", "method": "pearson",
        }),
        stop("Numerical analysis complete: gas_flow anomalies detected and gas_flow-pressure correlation computed (no root-cause claim)."),
    ]


def _baseline_d() -> list[Step]:
    return [
        tool("document_retrieval", {
            "query": "vacuum leak slow pump degradation chamber pressure rise", "k": 5,
        }),
        stop("Knowledge retrieval unavailable in the baseline (no knowledge base configured)."),
    ]


SCENARIOS: list[Scenario] = [
    Scenario(
        id="A_forecast",
        name="Forecast",
        user_task="Forecast gas_flow for ETCH-01.",
        expected_agents=("timeseries",),
        ground_truth=GroundTruth(equipment_id="ETCH-01", channel="gas_flow"),
        build_script=_scenario_a,
        build_baseline_script=_baseline_a,
    ),
    Scenario(
        id="B_anomaly",
        name="Anomaly detection",
        user_task="Detect anomalies in ETCH-01 gas_flow.",
        expected_agents=("timeseries",),
        ground_truth=GroundTruth(
            equipment_id="ETCH-01",
            channel="gas_flow",
            anomaly_scenario="mfc_flow_drift",
        ),
        build_script=_scenario_b,
        build_baseline_script=_baseline_b,
    ),
    Scenario(
        id="C_diagnosis",
        name="Forecast-driven diagnosis",
        user_task="Diagnose the root cause of the ETCH-01 gas_flow anomaly.",
        expected_agents=("timeseries", "diagnosis"),
        ground_truth=GroundTruth(
            equipment_id="ETCH-01",
            channel="gas_flow",
            root_cause=MFC_DRIFT,
            anomaly_scenario="mfc_flow_drift",
            relevant_sources=("sop/mfc_flow_drift.md", "fault_cases/mfc_flow_drift.md"),
        ),
        build_script=_scenario_c,
        build_baseline_script=_baseline_c,
    ),
    Scenario(
        id="D_knowledge",
        name="Knowledge QA",
        user_task="How should a vacuum leak on ETCH-02 be handled?",
        expected_agents=("knowledge",),
        ground_truth=GroundTruth(
            equipment_id="ETCH-02",
            relevant_sources=("sop/vacuum_leak.md", "fault_cases/vacuum_leak.md"),
        ),
        build_script=_scenario_d,
        build_baseline_script=_baseline_d,
    ),
]
