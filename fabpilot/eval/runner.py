"""Evaluation runner: wires three comparison configs against the four scenarios.

The harness seeds one temporary SQLite database and one knowledge base, then for
every (config, scenario) pair builds a *fresh* runtime (fresh ``ScriptedLLM``,
fresh ``HypothesisEvidenceMemory``, fresh graph) so runs never contaminate each
other. Metrics are computed from the real final state of each run.
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from fabpilot.config import Settings
from fabpilot.data.database import build_session_scope
from fabpilot.data.seed import seed_into
from fabpilot.eval.metrics import compute_metrics
from fabpilot.eval.scripted import ScriptedLLM
from fabpilot.eval.tasks import SCENARIOS, Scenario
from fabpilot.forecasting.registry import build_forecast_registry
from fabpilot.memory.checkpoint import build_checkpointer
from fabpilot.memory.hypothesis import HypothesisEvidenceMemory
from fabpilot.rag.pipeline import build_knowledge_base
from fabpilot.runtime import Runtime
from fabpilot.state import initial_state
from fabpilot.tools import build_tool_registry
from fabpilot.tools.registry import ToolContext
from fabpilot.utils import new_id


@dataclass(frozen=True)
class EvalConfig:
    name: str
    use_supervisor: bool
    use_diagnosis: bool
    use_rag: bool


CONFIGS: list[EvalConfig] = [
    EvalConfig("full", use_supervisor=True, use_diagnosis=True, use_rag=True),
    EvalConfig("no_rag", use_supervisor=True, use_diagnosis=True, use_rag=False),
    EvalConfig("baseline", use_supervisor=False, use_diagnosis=False, use_rag=False),
]


@dataclass
class Harness:
    settings: Settings
    session_scope: Callable
    forecast_registry: Any
    knowledge_base: Any  # KnowledgeBase | None
    tmpdir: str


# Tools the flat baseline agent may use (no memory, no delegation; RAG is absent
# because the baseline config disables the knowledge base).
BASELINE_TOOLS = [
    "sensor_data",
    "forecast",
    "anomaly_detection",
    "correlation",
    "lag_analysis",
    "statistical_analysis",
    "alarm_log",
    "maintenance_record",
    "fault_case",
    "equipment_status",
    "document_retrieval",
    "similar_case_retrieval",
]

BASELINE_SYSTEM_PROMPT = """\
You are a flat single-agent numerical analyst. You produce numbers and flags, not \
root causes. Use the provided tools to analyse the equipment, then summarise what \
you measured. Never invent a value or a root cause; if a tool errors, say so.
"""


def build_harness(seed: int = 42, **gen_kwargs: Any) -> Harness:
    tmpdir = tempfile.mkdtemp(prefix="fabpilot_eval_")
    db_path = Path(tmpdir) / "eval.db"
    settings = Settings(
        database_url=f"sqlite:///{db_path}",
        data_dir=str(Path(tmpdir)),
        knowledge_base_dir="knowledge_base",
        llm_provider="fake",
        embedding_provider="hash",
        vector_store="memory",
    )
    session_scope = build_session_scope(settings.database_url, create_tables=True)
    seed_into(session_scope, seed=seed, **gen_kwargs)
    forecast_registry = build_forecast_registry(settings)
    knowledge_base = build_knowledge_base(settings)
    return Harness(
        settings=settings,
        session_scope=session_scope,
        forecast_registry=forecast_registry,
        knowledge_base=knowledge_base,
        tmpdir=tmpdir,
    )


def _build_baseline_graph(llm, tool_registry, ctx, settings):
    from langgraph.graph import END, StateGraph

    from fabpilot.agents.base import run_react_loop
    from fabpilot.agents.common import collect_retrieved_docs, collect_tool_data
    from fabpilot.llm.schemas import ChatMessage
    from fabpilot.state import AgentState

    def node(state: dict[str, Any]) -> dict[str, Any]:
        result = run_react_loop(
            llm=llm,
            tool_registry=tool_registry,
            ctx=ctx,
            system_prompt=BASELINE_SYSTEM_PROMPT,
            messages=[ChatMessage(role="user", content=state.get("user_task", ""))],
            max_iterations=settings.agent_max_iterations,
            tool_budget=settings.tool_budget,
            loop_detection_window=settings.loop_detection_window,
            allowed_tools=BASELINE_TOOLS,
            agent_name="baseline",
        )
        return {
            "tool_history": result.tool_history,
            "trace": result.decisions,
            "forecast_results": collect_tool_data(result.tool_history, "forecast"),
            "anomaly_results": collect_tool_data(result.tool_history, "anomaly_detection"),
            "retrieved_docs": collect_retrieved_docs(result.tool_history),
            "messages": [{"role": "assistant", "name": "baseline", "content": result.final_content}],
            "final_report": result.final_content,
            "status": "completed",
        }

    g = StateGraph(AgentState)
    g.add_node("baseline", node)
    g.set_entry_point("baseline")
    g.add_edge("baseline", END)
    return g.compile()


def build_eval_runtime(config: EvalConfig, scenario: Scenario, harness: Harness) -> Runtime:
    settings = harness.settings
    script = scenario.build_baseline_script() if not config.use_supervisor else scenario.build_script()
    llm = ScriptedLLM(script())
    tool_registry = build_tool_registry()
    forecast_registry = harness.forecast_registry
    knowledge_base = harness.knowledge_base if config.use_rag else None
    memory = HypothesisEvidenceMemory() if config.use_diagnosis else None

    ctx = ToolContext(
        session_scope=harness.session_scope,
        forecast_registry=forecast_registry,
        knowledge_base=knowledge_base,
        memory=memory,
        settings=settings,
        chart_dir=str(settings.data_path / "charts"),
        report_dir=str(settings.data_path / "reports"),
    )

    if config.use_supervisor:
        from fabpilot.workflows import build_graph

        graph = build_graph(llm, tool_registry, ctx, settings, checkpointer=build_checkpointer(settings))
    else:
        graph = _build_baseline_graph(llm, tool_registry, ctx, settings)

    return Runtime(
        settings=settings,
        llm=llm,
        tool_registry=tool_registry,
        ctx=ctx,
        forecast_registry=forecast_registry,
        knowledge_base=knowledge_base,
        memory=memory,
        checkpointer=None,
        graph=graph,
    )


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    tool_history = state.get("tool_history") or []
    hypotheses = state.get("hypotheses") or []
    return {
        "status": state.get("status"),
        "n_tool_calls": len(tool_history),
        "n_tool_errors": sum(1 for h in tool_history if not h.get("ok")),
        "delegated_agents": [r.get("agent") for r in (state.get("delegation_records") or [])],
        "n_hypotheses": len(hypotheses),
        "n_evidence": len(state.get("evidence") or []),
        "final_report": (state.get("final_report") or "").strip()[:300],
    }


def run_scenario(config: EvalConfig, scenario: Scenario, harness: Harness) -> dict[str, Any]:
    runtime = build_eval_runtime(config, scenario, harness)
    state = initial_state(new_id("sess"), new_id("thread"), scenario.user_task)
    t0 = time.perf_counter()
    try:
        final = runtime.graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
        wall = time.perf_counter() - t0
    except Exception as exc:  # a run must never abort the whole evaluation
        final = dict(state)
        final["status"] = "failed"
        final["errors"] = [{"error": f"{type(exc).__name__}: {exc}"}]
        wall = time.perf_counter() - t0

    metrics = compute_metrics(final, scenario, config, harness, wall)
    return {
        "config": config.name,
        "scenario": scenario.id,
        "metrics": metrics,
        "summary": _state_summary(final),
    }


def run_evaluation(seed: int = 42, **gen_kwargs: Any) -> dict[str, Any]:
    harness = build_harness(seed=seed, **gen_kwargs)
    results: list[dict[str, Any]] = []
    for config in CONFIGS:
        for scenario in SCENARIOS:
            results.append(run_scenario(config, scenario, harness))
    return {
        "tmpdir": harness.tmpdir,
        "configs": [c.name for c in CONFIGS],
        "scenarios": [s.id for s in SCENARIOS],
        "results": results,
    }
