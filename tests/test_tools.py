"""tool 注册表测试：目录完整性与 agent 的 tool 子集。"""
from __future__ import annotations

from fabpilot.tools import AGENT_TOOL_NAMES, MEMORY_TOOL_NAMES, build_tool_registry


def test_registry_contains_all_expected_tools():
    registry = build_tool_registry()
    names = set(registry.names())
    for expected in [
        "sensor_data",
        "forecast",
        "anomaly_detection",
        "correlation",
        "lag_analysis",
        "statistical_analysis",
        "document_retrieval",
        "similar_case_retrieval",
        "equipment_status",
        "alarm_log",
        "maintenance_record",
        "fault_case",
        "python_analysis",
        "chart",
        "report",
    ]:
        assert expected in names
    for expected in MEMORY_TOOL_NAMES:
        assert expected in names


def test_agent_subsets_are_valid():
    registry = build_tool_registry()
    for agent, names in AGENT_TOOL_NAMES.items():
        for name in names:
            assert registry.get(name).name == name


def test_diagnosis_agent_has_memory_tools():
    assert all(name in AGENT_TOOL_NAMES["diagnosis"] for name in MEMORY_TOOL_NAMES)


def test_tool_openai_schema():
    registry = build_tool_registry()
    schema = registry.get("sensor_data").to_openai_schema()
    assert schema["function"]["name"] == "sensor_data"
    assert "properties" in schema["function"]["parameters"]
