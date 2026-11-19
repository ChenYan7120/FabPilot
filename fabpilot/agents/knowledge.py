"""知识 Agent：基于知识库的混合检索，并附带来源引用。"""
from __future__ import annotations

from typing import Any

from fabpilot.agents.base import run_react_loop
from fabpilot.agents.common import (
    agent_summary_message,
    collect_retrieved_docs,
    extract_equipment_id,
    last_subtask_for,
)
from fabpilot.agents.prompts import KNOWLEDGE_SYSTEM_PROMPT
from fabpilot.llm.schemas import ChatMessage
from fabpilot.tools import AGENT_TOOL_NAMES


def build_knowledge_agent(llm, tool_registry, ctx):
    settings = ctx.settings
    allowed = AGENT_TOOL_NAMES["knowledge"]

    def node(state: dict[str, Any]) -> dict[str, Any]:
        subtask = last_subtask_for(state, "knowledge")
        equipment_id = (subtask or {}).get("equipment_id") or extract_equipment_id(state)
        task = (subtask or {}).get("task") or state.get("user_task", "")

        user_text = f"Equipment: {equipment_id}\n\nQuestion: {task}" if equipment_id else f"Question: {task}"

        result = run_react_loop(
            llm=llm,
            tool_registry=tool_registry,
            ctx=ctx,
            system_prompt=KNOWLEDGE_SYSTEM_PROMPT,
            messages=[ChatMessage(role="user", content=user_text)],
            max_iterations=settings.agent_max_iterations,
            tool_budget=settings.tool_budget,
            loop_detection_window=settings.loop_detection_window,
            allowed_tools=allowed,
            agent_name="knowledge",
        )

        delta: dict[str, Any] = {
            "tool_history": result.tool_history,
            "trace": result.decisions,
            "retrieved_docs": collect_retrieved_docs(result.tool_history),
            "messages": [agent_summary_message("knowledge", result.final_content)],
        }
        if subtask:
            delta["subtasks"] = [
                {
                    "id": subtask["id"],
                    "agent": "knowledge",
                    "task": subtask.get("task"),
                    "status": "done",
                    "summary": result.final_content,
                }
            ]
        return delta

    return node
