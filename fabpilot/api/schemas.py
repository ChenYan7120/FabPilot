"""FabPilot API 的 Pydantic 请求/响应模型。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    user_task: str = Field(description="Natural-language task for the agent system.")
    session_id: str | None = Field(default=None, description="Optional session id; generated if omitted.")
    thread_id: str | None = Field(default=None, description="Optional LangGraph thread id.")


class TaskResponse(BaseModel):
    thread_id: str
    session_id: str
    status: str
    final_report: str
    user_task: str | None = Field(default=None, description="Echoed task text (set when listing history).")
    delegated_agents: list[str] = Field(default_factory=list)
    n_tool_calls: int = 0
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)


class ForecastRequest(BaseModel):
    equipment_id: str = Field(description="Equipment code, e.g. 'ETCH-01'.")
    target: str = Field(default="gas_flow", description="Channel to forecast.")
    horizon: int | None = Field(default=None, description="Forecast horizon in points.")
    lookback: int | None = Field(default=None, description="Lookback window in points.")
    model: str | None = Field(default=None, description="Forecast model name.")


class AnomalyRequest(BaseModel):
    equipment_id: str = Field(description="Equipment code, e.g. 'ETCH-01'.")
    channel: str = Field(default="gas_flow", description="Channel to scan.")
    method: str = Field(default="rolling_zscore", description="rolling_zscore | residual_zscore")
    window: int = Field(default=24, description="Rolling window size.")
    threshold: float = Field(default=3.0, description="Z-score threshold.")


class SearchRequest(BaseModel):
    query: str = Field(description="Natural-language query.")
    k: int = Field(default=5, description="Number of results.")
    doc_type: str | None = Field(default=None, description="'sop' | 'alarm' | 'fault_cases'")


class ErrorResponse(BaseModel):
    detail: str
