"""FabPilot REST API（FastAPI）。

对外暴露两类接口：
  * MES/FDC 数据接口 — 设备、传感器读数、告警、维护、
    故障案例、预测与异常检测（真实 tool，不依赖 LLM）。
  * 任务接口 — 运行完整的层级式 agent 系统并获取结果。

应用工厂见 ``fabpilot/api.app``。
"""
from fabpilot.api.app import create_app
from fabpilot.api.schemas import (
    AnomalyRequest,
    ForecastRequest,
    SearchRequest,
    TaskRequest,
    TaskResponse,
)
from fabpilot.api.service import FabPilotService, FabPilotServiceError

__all__ = [
    "create_app",
    "AnomalyRequest",
    "ForecastRequest",
    "SearchRequest",
    "TaskRequest",
    "TaskResponse",
    "FabPilotService",
    "FabPilotServiceError",
]
