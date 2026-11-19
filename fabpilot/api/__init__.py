"""FabPilot REST API (FastAPI).

Exposes two surfaces:
  * MES/FDC data endpoints — equipment, sensor readings, alarms, maintenance,
    fault cases, forecasting and anomaly detection (real tools, no LLM).
  * Task endpoints — run the full hierarchical agent system and fetch results.

See ``fabpilot/api.app`` for the app factory.
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
