"""工具层：registry 以及 data / time-series / knowledge / general 工具的完整目录。"""
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

# 每个工具都是由 ``@tool`` 装饰器创建的 ``Tool`` 实例。
_ALL_TOOLS: list[Tool] = [
    # 数据
    data_tools.sql_query,
    data_tools.equipment_status,
    data_tools.sensor_data,
    data_tools.alarm_log,
    data_tools.maintenance_record,
    data_tools.fault_case,
    # 时间序列
    timeseries_tools.forecast,
    timeseries_tools.anomaly_detection,
    timeseries_tools.correlation,
    timeseries_tools.lag_analysis,
    timeseries_tools.statistical_analysis,
    # 知识
    knowledge_tools.document_retrieval,
    knowledge_tools.similar_case_retrieval,
    # 通用
    general_tools.python_analysis,
    general_tools.chart,
    general_tools.report,
    # 假设-证据 memory（Diagnosis Agent 内部使用）
    memory_tools.add_hypothesis,
    memory_tools.add_evidence,
    memory_tools.add_pending_check,
    memory_tools.complete_pending_check,
    memory_tools.evaluate_hypothesis,
    memory_tools.list_hypotheses,
]

# 暴露给各专职 agent 的工具子集。Supervisor 使用专门的
# 委派工具，而不是这些领域工具。
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
