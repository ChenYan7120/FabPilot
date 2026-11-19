"""LangGraph 装配：三个专职子图 + 一个 Supervisor 顶层图。

Supervisor 节点在委托工具之上运行自己的 ReAct 循环；每个委托工具同步调用
对应的专职子图，并返回该专家的总结。状态通过相同的只追加 reducer 累积，
因此最终状态携带完整的审计轨迹。
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
    """返回编译后的 FabPilot 顶层图。"""
    subgraphs = build_subgraphs(llm, tool_registry, ctx)
    supervisor_node = build_supervisor_node(llm, subgraphs, ctx, settings)

    top = StateGraph(AgentState)
    top.add_node("supervisor", supervisor_node)
    top.set_entry_point("supervisor")
    top.add_edge("supervisor", END)
    return top.compile(checkpointer=checkpointer)
