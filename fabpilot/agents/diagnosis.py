"""Diagnosis Agent: hypothesis-driven root-cause analysis backed by the
Hypothesis-Evidence Memory (real deterministic state machine, driven via tools).
"""
from __future__ import annotations

import json
from typing import Any

from fabpilot.agents.base import run_react_loop
from fabpilot.agents.common import (
    agent_summary_message,
    extract_equipment_id,
    last_subtask_for,
)
from fabpilot.agents.prompts import DIAGNOSIS_SYSTEM_PROMPT
from fabpilot.llm.schemas import ChatMessage
from fabpilot.tools import AGENT_TOOL_NAMES


def _context_block(state: dict[str, Any]) -> str:
    parts: list[str] = []
    if state.get("forecast_results"):
        parts.append("Forecast results:\n" + json.dumps(state["forecast_results"], ensure_ascii=False, default=str))
    if state.get("anomaly_results"):
        parts.append("Anomaly results:\n" + json.dumps(state["anomaly_results"], ensure_ascii=False, default=str))
    return "\n\n".join(parts)


def build_diagnosis_agent(llm, tool_registry, ctx):
    settings = ctx.settings
    allowed = AGENT_TOOL_NAMES["diagnosis"]

    def node(state: dict[str, Any]) -> dict[str, Any]:
        subtask = last_subtask_for(state, "diagnosis")
        equipment_id = (subtask or {}).get("equipment_id") or extract_equipment_id(state)
        task = (subtask or {}).get("task") or state.get("user_task", "")

        lines = []
        if equipment_id:
            lines.append(f"Equipment: {equipment_id}")
        lines.append(f"Fault to diagnose: {task}")
        context = _context_block(state)
        if context:
            lines.append("Context from prior analysis:\n" + context)
        user_text = "\n\n".join(lines)

        result = run_react_loop(
            llm=llm,
            tool_registry=tool_registry,
            ctx=ctx,
            system_prompt=DIAGNOSIS_SYSTEM_PROMPT,
            messages=[ChatMessage(role="user", content=user_text)],
            max_iterations=settings.agent_max_iterations,
            tool_budget=settings.tool_budget,
            loop_detection_window=settings.loop_detection_window,
            allowed_tools=allowed,
            agent_name="diagnosis",
        )

        delta: dict[str, Any] = {
            "tool_history": result.tool_history,
            "trace": result.decisions,
            "messages": [agent_summary_message("diagnosis", result.final_content)],
        }

        if ctx.memory is not None:
            snapshot = ctx.memory.snapshot()
            delta["hypotheses"] = snapshot["hypotheses"]
            delta["evidence"] = snapshot["evidence"]
            pending: list[dict[str, Any]] = []
            for h in snapshot["hypotheses"]:
                pending.extend(h.get("pending_checks", []))
            delta["pending_checks"] = pending

        if subtask:
            delta["subtasks"] = [
                {
                    "id": subtask["id"],
                    "agent": "diagnosis",
                    "task": subtask.get("task"),
                    "status": "done",
                    "summary": result.final_content,
                }
            ]
        return delta

    return node
