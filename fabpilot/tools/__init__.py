"""Tool layer: registry + the full catalog of data / time-series / knowledge / general tools."""
from __future__ import annotations

from fabpilot.tools import data_tools, general_tools, knowledge_tools, memory_tools, timeseries_tools
from fabpilot.tools.registry import Tool, ToolContext, ToolRegistry, ToolResult

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolContext",
    "ToolResult",
    "build_tool_registry",
    "AGENT_TOOL_NAMES",
    "MEMORY_TOOL_NAMES",
]

MEMORY_TOOL_NAMES = memory_tools.MEMORY_TOOL_NAMES

# Every tool is a ``Tool`` instance created by the ``@tool`` decorator.
_ALL_TOOLS: list[Tool] = [
    # data
    data_tools.sql_query,
    data_tools.equipment_status,
    data_tools.sensor_data,
    data_tools.alarm_log,
    data_tools.maintenance_record,
    data_tools.fault_case,
    # time-series
    timeseries_tools.forecast,
    timeseries_tools.anomaly_detection,
    timeseries_tools.correlation,
    timeseries_tools.lag_analysis,
    timeseries_tools.statistical_analysis,
    # knowledge
    knowledge_tools.document_retrieval,
    knowledge_tools.similar_case_retrieval,
    # general
    general_tools.python_analysis,
    general_tools.chart,
    general_tools.report,
    # hypothesis-evidence memory (internal to the Diagnosis Agent)
    memory_tools.add_hypothesis,
    memory_tools.add_evidence,
    memory_tools.add_pending_check,
    memory_tools.complete_pending_check,
    memory_tools.evaluate_hypothesis,
    memory_tools.list_hypotheses,
]

# Tool subsets exposed to each specialised agent. The Supervisor uses dedicated
# delegation tools rather than these domain tools.
AGENT_TOOL_NAMES: dict[str, list[str]] = {
    "timeseries": [
        "sensor_data",
        "forecast",
        "anomaly_detection",
        "correlation",
        "lag_analysis",
        "statistical_analysis",
        "chart",
        "python_analysis",
    ],
    "diagnosis": [
        "sensor_data",
        "alarm_log",
        "maintenance_record",
        "fault_case",
        "anomaly_detection",
        "correlation",
        "lag_analysis",
        "statistical_analysis",
        "document_retrieval",
        "similar_case_retrieval",
        "python_analysis",
        *MEMORY_TOOL_NAMES,
    ],
    "knowledge": [
        "document_retrieval",
        "similar_case_retrieval",
    ],
}


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for t in _ALL_TOOLS:
        registry.register(t)
    return registry
