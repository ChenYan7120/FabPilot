"""Agent layer: Supervisor + Time-Series + Diagnosis + Knowledge + shared ReAct loop."""
from __future__ import annotations

from fabpilot.agents.base import ReactResult, run_react_loop
from fabpilot.agents.diagnosis import build_diagnosis_agent
from fabpilot.agents.knowledge import build_knowledge_agent
from fabpilot.agents.supervisor import build_supervisor_node
from fabpilot.agents.timeseries import build_timeseries_agent

__all__ = [
    "ReactResult",
    "run_react_loop",
    "build_supervisor_node",
    "build_timeseries_agent",
    "build_diagnosis_agent",
    "build_knowledge_agent",
]
