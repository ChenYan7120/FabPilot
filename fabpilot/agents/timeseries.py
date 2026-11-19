"""时序 Agent：数值分析（查询、预测、异常、相关性、滞后、统计、图表）。
它产出的是数值与异常标记，而不是根因。
"""
from __future__ import annotations

from typing import Any

from fabpilot.agents.base import run_react_loop
from fabpilot.agents.common import (
    agent_summary_message,
    collect_tool_data,
    extract_equipment_id,
    last_subtask_for,
)
from fabpilot.agents.prompts import TIMESERIES_SYSTEM_PROMPT
from fabpilot.llm.schemas import ChatMessage
from fabpilot.tools import AGENT_TOOL_NAMES


def build_timeseries_agent(llm, tool_registry, ctx):
    settings = ctx.settings
    allowed = AGENT_TOOL_NAMES["timeseries"]

    def node(state: dict[str, Any]) -> dict[str, Any]:
        subtask = last_subtask_for(state, "timeseries")
        equipment_id = (subtask or {}).get("equipment_id") or extract_equipment_id(state)
        task = (subtask or {}).get("task") or state.get("user_task", "")

        user_text = f"Equipment: {equipment_id}\n\nTask: {task}" if equipment_id else f"Task: {task}"

        result = run_react_loop(
            llm=llm,
            tool_registry=tool_registry,
            ctx=ctx,
            system_prompt=TIMESERIES_SYSTEM_PROMPT,
            messages=[ChatMessage(role="user", content=user_text)],
            max_iterations=settings.agent_max_iterations,
            tool_budget=settings.tool_budget,
            loop_detection_window=settings.loop_detection_window,
            allowed_tools=allowed,
            agent_name="timeseries",
        )

        delta: dict[str, Any] = {
            "tool_history": result.tool_history,
            "trace": result.decisions,
            "forecast_results": collect_tool_data(result.tool_history, "forecast"),
            "anomaly_results": collect_tool_data(result.tool_history, "anomaly_detection"),
            "messages": [agent_summary_message("timeseries", result.final_content)],
        }
        if subtask:
            delta["subtasks"] = [
                {
                    "id": subtask["id"],
                    "agent": "timeseries",
                    "task": subtask.get("task"),
                    "status": "done",
                    "summary": result.final_content,
                }
            ]
        return delta

    return node
