"""预测模型契约。

预测模型是一个 *纯回归器*：给定一个或多个 channel 的 lookback 窗口，返回单个目标 channel 的未来值。它不做任何异常检测，也不做诊断——这些功能位于 Time-Series Agent 中。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class ForecastModelNotAvailable(Exception):
    """当已注册的模型无法运行时抛出（无权重 / 缺少依赖）。"""


class ForecastRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    equipment_id: str
    target: str
    lookback: int
    horizon: int
    series: dict[str, Any] = Field(default_factory=dict)  # channel -> 一维历史数据（长度 >= lookback）
    timestamps: list[datetime] = Field(default_factory=list)  # 与 series 历史对齐
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
    lower: list[float] | None = None  # 预测区间下界
    upper: list[float] | None = None
    in_sample_residuals: list[float] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class ForecastModel(Protocol):
    name: str

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        ...
