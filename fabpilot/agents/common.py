"""Small helpers shared by the agent node factories."""
from __future__ import annotations

import re
from typing import Any

_EQUIP_RE = re.compile(r"\b(?:ETCH|CVD|CMP|PVD|LITH|IMP|DIFF)-[0-9]{2,3}\b", re.IGNORECASE)


def last_subtask_for(state: dict[str, Any], agent: str) -> dict[str, Any] | None:
    subtasks = state.get("subtasks") or []
    for s in reversed(subtasks):
        if s.get("agent") == agent and s.get("status") in ("pending", None):
            return s
    return None


def extract_equipment_id(state: dict[str, Any]) -> str | None:
    for s in reversed(state.get("subtasks") or []):
        if s.get("equipment_id"):
            return s["equipment_id"]
    m = _EQUIP_RE.search(state.get("user_task", ""))
    return m.group(0).upper() if m else None


def collect_tool_data(tool_history: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [h["data"] for h in tool_history if h.get("name") == name and h.get("ok") and h.get("data")]


def collect_retrieved_docs(tool_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for h in tool_history:
        if h.get("name") in ("document_retrieval", "similar_case_retrieval") and h.get("ok"):
            docs.extend(h.get("data", {}).get("results", []))
    return docs


def agent_summary_message(agent: str, content: str) -> dict[str, Any]:
    return {"role": "assistant", "name": agent, "content": content}
