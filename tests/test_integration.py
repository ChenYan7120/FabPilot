"""集成测试：将真实 tool 接入种子化的 SQLite 数据库。

这些测试验证规格所关注的那条边界——tool 连接的是真实内部实现
（数据库、预测注册表、知识库），而非占位符。
"""
from __future__ import annotations

from fabpilot.tools import build_tool_registry
from fabpilot.tools.registry import ToolContext


def _ctx(seeded_scope, forecast_registry, knowledge_base, settings):
    return ToolContext(
        session_scope=seeded_scope,
        forecast_registry=forecast_registry,
        knowledge_base=knowledge_base,
        memory=None,
        settings=settings,
        chart_dir=str(settings.data_path / "charts"),
        report_dir=str(settings.data_path / "reports"),
    )


def test_equipment_status_lists_seeded_equipment(seeded_scope, forecast_registry, knowledge_base, settings):
    registry = build_tool_registry()
    ctx = _ctx(seeded_scope, forecast_registry, knowledge_base, settings)
    result = registry.invoke(ctx, {"equipment_id": None})
    assert result.ok, result.error
    codes = [e["code"] for e in result.data["equipment"]]
    assert "ETCH-01" in codes


def test_sensor_data_returns_real_rows(seeded_scope, forecast_registry, knowledge_base, settings):
    registry = build_tool_registry()
    ctx = _ctx(seeded_scope, forecast_registry, knowledge_base, settings)
    result = registry.invoke(ctx, {"equipment_id": "ETCH-01", "channel": "gas_flow", "limit": 50})
    assert result.ok, result.error
    assert result.data["n_points"] > 0
    assert "gas_flow" in result.data["channels"]


def test_forecast_returns_points(seeded_scope, forecast_registry, knowledge_base, settings):
    registry = build_tool_registry()
    ctx = _ctx(seeded_scope, forecast_registry, knowledge_base, settings)
    result = registry.invoke(ctx, {"equipment_id": "ETCH-01", "target": "gas_flow", "model": "statistical_baseline"})
    assert result.ok, result.error
    assert result.data["model_name"] == "statistical_baseline"
    assert len(result.data["forecast"]) > 0


def test_anomaly_detection_runs(seeded_scope, forecast_registry, knowledge_base, settings):
    registry = build_tool_registry()
    ctx = _ctx(seeded_scope, forecast_registry, knowledge_base, settings)
    result = registry.invoke(
        ctx, {"equipment_id": "ETCH-01", "channel": "gas_flow", "method": "rolling_zscore", "window": 24, "threshold": 3.0}
    )
    assert result.ok, result.error
    assert result.data["n_points"] > 0


def test_unknown_equipment_fails_cleanly(seeded_scope, forecast_registry, knowledge_base, settings):
    registry = build_tool_registry()
    ctx = _ctx(seeded_scope, forecast_registry, knowledge_base, settings)
    result = registry.invoke(ctx, {"equipment_id": "DOES-NOT-EXIST"})
    assert not result.ok


def test_document_retrieval_cites_sources(seeded_scope, forecast_registry, knowledge_base, settings):
    registry = build_tool_registry()
    ctx = _ctx(seeded_scope, forecast_registry, knowledge_base, settings)
    result = registry.invoke(ctx, {"query": "gas flow drift", "k": 5})
    assert result.ok, result.error
    assert result.data["n_results"] > 0
    assert all(d["source"] for d in result.data["results"])
