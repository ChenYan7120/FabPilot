"""应用服务：真实的数据/分析 tool + 完整的 agent 运行时。

数据与分析接口不需要 LLM，因此该服务会基于真实的数据库 / 预测注册表 /
知识库直接构建 ``ToolContext``。任务接口在首次使用时惰性构建完整的
层级式运行时（LLM + LangGraph），因此启动 API 时无需 API key。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from fabpilot.config import Settings, get_settings
from fabpilot.data.database import build_session_scope
from fabpilot.forecasting.registry import build_forecast_registry
from fabpilot.memory.checkpoint import build_session_store
from fabpilot.rag.pipeline import build_knowledge_base
from fabpilot.runtime import Runtime, build_runtime, run_task
from fabpilot.tools import build_tool_registry
from fabpilot.tools.registry import ToolContext


class FabPilotServiceError(Exception):
    """当某项 service 操作失败时抛出（由路由映射为 HTTP 错误）。"""


class FabPilotService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.session_scope = build_session_scope(self.settings.database_url)
        self.forecast_registry = build_forecast_registry(self.settings)
        self.knowledge_base = build_knowledge_base(self.settings)
        self.tool_registry = build_tool_registry()
        self._ctx = ToolContext(
            session_scope=self.session_scope,
            forecast_registry=self.forecast_registry,
            knowledge_base=self.knowledge_base,
            memory=None,
            settings=self.settings,
            chart_dir=str(self.settings.data_path / "charts"),
            report_dir=str(self.settings.data_path / "reports"),
        )
        self._store = build_session_store(self.settings)
        self._runtime: Runtime | None = None

    # ---- tool 分发 ---------------------------------------------------- #

    def _call(self, name: str, **kwargs: Any) -> dict[str, Any]:
        result = self.tool_registry.invoke(self._ctx, kwargs)
        if not result.ok:
            raise FabPilotServiceError(result.error or f"tool '{name}' failed")
        return result.data

    # ---- MES/FDC 数据 ----------------------------------------------------- #

    def list_equipment(self) -> list[dict[str, Any]]:
        return self._call("equipment_status").get("equipment", [])

    def get_equipment(self, equipment_id: str) -> dict[str, Any]:
        return self._call("equipment_status", equipment_id=equipment_id)

    def sensor_data(
        self,
        equipment_id: str,
        channel: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        return self._call(
            "sensor_data",
            equipment_id=equipment_id,
            channel=channel,
            start=start,
            end=end,
            limit=limit,
        )

    def alarms(
        self,
        equipment_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        return self._call("alarm_log", equipment_id=equipment_id, start=start, end=end, limit=limit)

    def maintenance(
        self,
        equipment_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        return self._call("maintenance_record", equipment_id=equipment_id, start=start, end=end, limit=limit)

    def fault_cases(self, equipment_id: str | None = None, limit: int = 50) -> dict[str, Any]:
        return self._call("fault_case", equipment_id=equipment_id, limit=limit)

    # ---- 分析 -------------------------------------------------------- #

    def forecast(
        self,
        equipment_id: str,
        target: str = "gas_flow",
        horizon: int | None = None,
        lookback: int | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        return self._call(
            "forecast",
            equipment_id=equipment_id,
            target=target,
            horizon=horizon,
            lookback=lookback,
            model=model,
        )

    def detect_anomalies(
        self,
        equipment_id: str,
        channel: str = "gas_flow",
        method: str = "rolling_zscore",
        window: int = 24,
        threshold: float = 3.0,
    ) -> dict[str, Any]:
        return self._call(
            "anomaly_detection",
            equipment_id=equipment_id,
            channel=channel,
            method=method,
            window=window,
            threshold=threshold,
        )

    def search_knowledge(self, query: str, k: int = 5, doc_type: str | None = None) -> dict[str, Any]:
        return self._call("document_retrieval", query=query, k=k, doc_type=doc_type)

    # ---- agent 任务 ------------------------------------------------------ #

    @property
    def runtime(self) -> Runtime:
        if self._runtime is None:
            self._runtime = build_runtime(
                self.settings,
                session_scope=self.session_scope,
                forecast_registry=self.forecast_registry,
                knowledge_base=self.knowledge_base,
                tool_registry=self.tool_registry,
            )
        return self._runtime

    def run_task(self, user_task: str, session_id: str | None = None, thread_id: str | None = None) -> dict[str, Any]:
        state = run_task(self.runtime, user_task, session_id=session_id, thread_id=thread_id)
        record = {
            "thread_id": state["thread_id"],
            "session_id": state["session_id"],
            "user_task": user_task,
            "status": state.get("status"),
            "final_report": state.get("final_report", ""),
            "delegated_agents": [r.get("agent") for r in state.get("delegation_records", [])],
            "n_tool_calls": len(state.get("tool_history", [])),
            "hypotheses": state.get("hypotheses", []),
            "errors": state.get("errors", []),
        }
        self._store.save(record)
        return state

    def get_task(self, thread_id: str) -> dict[str, Any] | None:
        return self._store.get(thread_id)

    def list_tasks(self) -> list[dict[str, Any]]:
        return self._store.list()
