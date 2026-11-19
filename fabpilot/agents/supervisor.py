"""Supervisor Agent: hierarchical ReAct router over delegation tools.

The Supervisor does not analyse data itself. Its "tools" are delegation handles
that invoke the three specialist subgraphs synchronously; each returns the
specialist's summary as an observation. A final ``finish`` tool carries the
Markdown report. The whole delegation trail is recorded in shared state.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from fabpilot.agents.base import run_react_loop
from fabpilot.agents.prompts import SUPERVISOR_SYSTEM_PROMPT
from fabpilot.llm.schemas import ChatMessage
from fabpilot.state import diff_state
from fabpilot.tools.registry import Tool, ToolContext, ToolRegistry
from fabpilot.utils import new_id


class DelegateInput(BaseModel):
    task: str = Field(description="Self-contained instruction for the specialist agent.")
    equipment_id: str | None = Field(default=None, description="Equipment code if relevant.")


class FinishInput(BaseModel):
    report: str = Field(description="Final Markdown report citing only results the agents actually returned.")


AGENT_DESCRIPTIONS = {
    "timeseries": (
        "Delegate numerical analysis to the Time-Series Agent: query sensor data, "
        "forecast, anomaly detection, correlation, lag, statistics, charts."
    ),
    "diagnosis": (
        "Delegate causal root-cause analysis to the Diagnosis Agent (hypothesis-driven "
        "reasoning with a deterministic Hypothesis-Evidence Memory)."
    ),
    "knowledge": (
        "Delegate knowledge retrieval to the Knowledge Agent (SOPs, alarm guides, "
        "historical fault cases) with source citations."
    ),
}


def _delegate_tool(agent_name: str, subgraph, shared: dict[str, Any]) -> Tool:
    def func(ctx: ToolContext, task: str, equipment_id: str | None = None) -> dict[str, Any]:
        subtask_id = new_id("sub")
        s = dict(shared["state"])
        s["subtasks"] = list(s.get("subtasks", [])) + [
            {
                "id": subtask_id,
                "agent": agent_name,
                "task": task,
                "equipment_id": equipment_id,
                "status": "pending",
            }
        ]
        s["delegation_records"] = list(s.get("delegation_records", [])) + [
            {"subtask_id": subtask_id, "agent": agent_name, "task": task, "equipment_id": equipment_id}
        ]
        updated = subgraph.invoke(s)
        shared["state"] = updated
        summary = ""
        for sub in reversed(updated.get("subtasks", [])):
            if sub.get("id") == subtask_id:
                summary = sub.get("summary", "")
                break
        return {"agent": agent_name, "subtask_id": subtask_id, "status": "done", "summary": summary}

    return Tool(
        name=f"delegate_to_{agent_name}",
        description=AGENT_DESCRIPTIONS[agent_name],
        input_model=DelegateInput,
        func=func,
        tags=["delegation"],
    )


def _finish_tool(shared: dict[str, Any]) -> Tool:
    def func(ctx: ToolContext, report: str) -> dict[str, Any]:
        shared["final_report"] = report
        return {"status": "finished"}

    return Tool(
        name="finish",
        description="Deliver the final Markdown report and end the task.",
        input_model=FinishInput,
        func=func,
        tags=["delegation"],
    )


def build_supervisor_node(llm, subgraphs: dict[str, Any], ctx: ToolContext, settings):
    def node(state: dict[str, Any]) -> dict[str, Any]:
        shared: dict[str, Any] = {"state": dict(state)}

        registry = ToolRegistry()
        for agent_name, subgraph in subgraphs.items():
            registry.register(_delegate_tool(agent_name, subgraph, shared))
        registry.register(_finish_tool(shared))

        result = run_react_loop(
            llm=llm,
            tool_registry=registry,
            ctx=ctx,
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
            messages=[ChatMessage(role="user", content=state.get("user_task", ""))],
            max_iterations=settings.supervisor_max_iterations,
            tool_budget=settings.tool_budget,
            loop_detection_window=settings.loop_detection_window,
            agent_name="supervisor",
        )

        delta = diff_state(state, shared["state"])
        delta["tool_history"] = delta.get("tool_history", []) + result.tool_history
        delta["trace"] = delta.get("trace", []) + result.decisions
        delta["messages"] = delta.get("messages", []) + [
            {"role": "assistant", "name": "supervisor", "content": result.final_content}
        ]
        delta["final_report"] = shared.get("final_report") or result.final_content
        delta["status"] = "completed"
        delta["current_agent"] = "supervisor"
        return delta

    return node
