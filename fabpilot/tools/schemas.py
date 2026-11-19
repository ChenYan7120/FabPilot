"""工具返回的结构化输出模型。

工具注册表会自动将这些模型序列化为 JSON；保持类型化可以让评测层和
API 基于真实字段进行断言，而不是面对不透明的字符串。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EquipmentInfo(BaseModel):
    code: str
    name: str
    equipment_type: str
    status: str
    location: str


class SensorDataResult(BaseModel):
    equipment_id: str
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    n_points: int = 0
    channels: list[str] = Field(default_factory=list)
    summary: dict[str, dict[str, float]] = Field(default_factory=dict)  # channel -> {mean,std,min,max}（均值、标准差、最小值、最大值）
    last_10_points: list[dict[str, Any]] = Field(default_factory=list)


class AlarmLogResult(BaseModel):
    equipment_id: str
    n_alarms: int = 0
    alarms: list[dict[str, Any]] = Field(default_factory=list)


class MaintenanceResult(BaseModel):
    equipment_id: str
    n_records: int = 0
    records: list[dict[str, Any]] = Field(default_factory=list)


class FaultCaseResult(BaseModel):
    n_cases: int = 0
    cases: list[dict[str, Any]] = Field(default_factory=list)


class ForecastPointOut(BaseModel):
    ts: datetime
    value: float
    lower: float | None = None
    upper: float | None = None


class ForecastToolResult(BaseModel):
    equipment_id: str
    target: str
    model_name: str
    horizon: int
    forecast: list[ForecastPointOut]
    in_sample_metrics: dict[str, float] = Field(default_factory=dict)


class AnomalyPoint(BaseModel):
    ts: datetime
    value: float
    zscore: float | None = None
    residual: float | None = None
    score: float
    is_anomaly: bool


class AnomalyResult(BaseModel):
    equipment_id: str
    channel: str
    method: str
    threshold: float
    n_points: int = 0
    n_anomalies: int = 0
    anomalies: list[AnomalyPoint] = Field(default_factory=list)


class CorrelationResult(BaseModel):
    channel_a: str
    channel_b: str
    method: str
    coefficient: float
    p_value: float | None = None
    n: int = 0
    note: str = ""


class LagResult(BaseModel):
    channel_a: str
    channel_b: str
    best_lag: int
    max_abs_corr: float
    correlations: list[dict[str, Any]] = Field(default_factory=list)


class StatsResult(BaseModel):
    channel: str
    n: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    p25: float
    p75: float
    skewness: float | None = None
    kurtosis: float | None = None


class RetrievedDoc(BaseModel):
    chunk_id: str
    doc_id: str
    doc_type: str
    title: str
    text: str
    score: float
    source: str


class RetrievalResult(BaseModel):
    query: str
    n_results: int = 0
    results: list[RetrievedDoc] = Field(default_factory=list)


class ChartResult(BaseModel):
    kind: str
    file_path: str
    title: str
    n_series: int = 0


class AnalysisResult(BaseModel):
    expression: str
    result: Any = None
    note: str = ""
