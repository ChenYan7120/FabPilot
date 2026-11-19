"""LangGraph assembly: three specialist subgraphs + a supervisor top graph.

The Supervisor node runs its own ReAct loop over delegation tools; each
delegation tool invokes the corresponding specialist subgraph synchronously and
returns the specialist's summary. State is accumulated with the same append-only
reducers so the final state carries the full audit trail.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from fabpilot.agents import (
    build_diagnosis_agent,
    build_knowledge_agent,
    build_supervisor_node,
    build_timeseries_agent,
)
from fabpilot.state import AgentState


def _make_subgraph(node):
    g = StateGraph(AgentState)
    g.add_node("agent", node)
    g.set_entry_point("agent")
    g.set_finish_point("agent")
    return g.compile()


def build_subgraphs(llm, tool_registry, ctx) -> dict[str, Any]:
    return {
        "timeseries": _make_subgraph(build_timeseries_agent(llm, tool_registry, ctx)),
        "diagnosis": _make_subgraph(build_diagnosis_agent(llm, tool_registry, ctx)),
        "knowledge": _make_subgraph(build_knowledge_agent(llm, tool_registry, ctx)),
    }


def build_graph(llm, tool_registry, ctx, settings, checkpointer=None):
    """Return the compiled top-level FabPilot graph."""
    subgraphs = build_subgraphs(llm, tool_registry, ctx)
    supervisor_node = build_supervisor_node(llm, subgraphs, ctx, settings)

    top = StateGraph(AgentState)
    top.add_node("supervisor", supervisor_node)
    top.set_entry_point("supervisor")
    top.add_edge("supervisor", END)
    return top.compile(checkpointer=checkpointer)
