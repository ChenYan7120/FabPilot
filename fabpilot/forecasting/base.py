"""Forecasting model contracts.

The forecasting model is a *pure regressor*: given a lookback window of one or
more channels it returns future values for a single target channel. It performs
no anomaly detection and no diagnosis — those live in the Time-Series Agent.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class ForecastModelNotAvailable(Exception):
    """Raised when a registered model cannot run (no weights / missing dep)."""


class ForecastRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    equipment_id: str
    target: str
    lookback: int
    horizon: int
    series: dict[str, Any] = Field(default_factory=dict)  # channel -> 1-D history (len >= lookback)
    timestamps: list[datetime] = Field(default_factory=list)  # aligned with series history
    metadata: dict[str, Any] = Field(default_factory=dict)


class ForecastPoint(BaseModel):
    ts: datetime
    value: float


class ForecastResult(BaseModel):
    model_name: str
    equipment_id: str
    target: str
    lookback: int
    horizon: int
    forecast: list[ForecastPoint]
    lower: list[float] | None = None  # prediction-interval lower bound
    upper: list[float] | None = None
    in_sample_residuals: list[float] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class ForecastModel(Protocol):
    name: str

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        ...
