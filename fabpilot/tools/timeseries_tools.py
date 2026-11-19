"""时间序列分析工具。

边界说明：forecasting 是 *纯回归*（委派给 forecast
registry）；所有数值计算——基于残差/统计量的 anomaly detection、
correlation、lag、描述性统计——都放在这里，由 Time-Series Agent 完成。
这些工具都不会产出根因，它们产出的是数值 + 异常。
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from fabpilot.data import queries
from fabpilot.forecasting.base import ForecastModelNotAvailable, ForecastRequest
from fabpilot.tools.registry import ToolContext, tool
from fabpilot.tools.schemas import (
    AnomalyPoint,
    AnomalyResult,
    CorrelationResult,
    ForecastPointOut,
    ForecastToolResult,
    LagResult,
    StatsResult,
)
from fabpilot.utils import ToolValidationError


def _load_series(
    ctx: ToolContext,
    equipment_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    with ctx.session_scope() as session:  # type: ignore[misc]
        if queries.get_equipment(session, equipment_id) is None:
            raise ToolValidationError(f"equipment '{equipment_id}' not found")
        return queries.query_sensor_readings(session, equipment_id, start, end)


# --------------------------------------------------------------------------- #
# 预测
# --------------------------------------------------------------------------- #


class ForecastInput(BaseModel):
    equipment_id: str = Field(description="设备编码，如 'ETCH-01'。")
    target: str = Field(description="要预测的通道，如 'gas_flow'。")
    lookback: int | None = Field(default=None, description="回看窗口（以点数为单位）；默认取自配置。")
    horizon: int | None = Field(default=None, description="预测步长（以点数为单位）；默认取自配置。")
    model: str | None = Field(default=None, description="模型名称；默认取自配置中的默认值。")
    end: datetime | None = Field(default=None, description="锚点：仅使用截至该时间戳的数据。")


@tool(
    name="forecast",
    description=(
        "使用已注册的预测模型预测未来某个 sensor 通道。"
        "纯回归：返回未来值 + 预测区间，不做任何异常"
        "解读。预测风险/异常需要另行评估。"
    ),
    input_model=ForecastInput,
    tags=["timeseries"],
)
def forecast(
    ctx: ToolContext,
    equipment_id: str,
    target: str,
    lookback: int | None = None,
    horizon: int | None = None,
    model: str | None = None,
    end: datetime | None = None,
) -> ForecastToolResult:
    settings = ctx.settings
    lookback = lookback or settings.forecast_lookback
    horizon = horizon or settings.forecast_horizon
    model = model or settings.default_forecast_model

    df = _load_series(ctx, equipment_id, end=end)
    if target not in df.columns:
        raise ToolValidationError(f"unknown channel '{target}'")
    if len(df) < lookback:
        raise ToolValidationError(
            f"insufficient data: need {lookback} points, have {len(df)}"
        )

    series = {c: df[c].astype(float).tolist() for c in df.columns if c != "ts"}
    timestamps = df["ts"].tolist()

    request = ForecastRequest(
        equipment_id=equipment_id,
        target=target,
        lookback=lookback,
        horizon=horizon,
        series=series,
        timestamps=timestamps,
    )
    try:
        result = ctx.forecast_registry.forecast(model, request)
    except ForecastModelNotAvailable as exc:
        raise ToolValidationError(f"forecast model '{model}' unavailable: {exc}")

    return ForecastToolResult(
        equipment_id=equipment_id,
        target=target,
        model_name=result.model_name,
        horizon=result.horizon,
        forecast=[
            ForecastPointOut(ts=p.ts, value=p.value, lower=lo, upper=hi)
            for p, lo, hi in zip(result.forecast, result.lower or [], result.upper or [])
        ],
        in_sample_metrics=result.metrics,
    )


# --------------------------------------------------------------------------- #
# 异常检测
# --------------------------------------------------------------------------- #


class AnomalyInput(BaseModel):
    equipment_id: str = Field(description="设备编码，如 'ETCH-01'。")
    channel: str = Field(description="要扫描的通道，如 'gas_flow'。")
    method: str = Field(
        default="rolling_zscore",
        description="'rolling_zscore' 或 'residual_zscore' 之一。",
    )
    window: int = Field(default=24, description="滚动窗口大小（以点数为单位）。")
    threshold: float = Field(default=3.0, description="用于标记异常的 Z-score 阈值。")
    start: datetime | None = None
    end: datetime | None = None


@tool(
    name="anomaly_detection",
    description=(
        "通过滚动 / 残差 z-score 检测 sensor 通道中的异常点。"
        "返回带有 z-score 的被标记时间戳；它不会推断根因。"
    ),
    input_model=AnomalyInput,
    tags=["timeseries"],
)
def anomaly_detection(
    ctx: ToolContext,
    equipment_id: str,
    channel: str,
    method: str = "rolling_zscore",
    window: int = 24,
    threshold: float = 3.0,
    start: datetime | None = None,
    end: datetime | None = None,
) -> AnomalyResult:
    if method not in ("rolling_zscore", "residual_zscore"):
        raise ToolValidationError(f"unknown method '{method}'")
    df = _load_series(ctx, equipment_id, start, end)
    if channel not in df.columns:
        raise ToolValidationError(f"unknown channel '{channel}'")

    y = df[channel].astype(float).to_numpy()
    ts = df["ts"].to_list()
    n = len(y)

    if method == "residual_zscore":
        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (slope * x + intercept)
        rstd = pd.Series(residual).rolling(window, min_periods=max(2, window // 2)).std().to_numpy()
        z = residual / np.where(rstd > 0, rstd, np.nan)
        score = np.abs(z)
        is_anomaly = score > threshold
    else:  # rolling_zscore
        s = pd.Series(y)
        mean = s.rolling(window, min_periods=max(2, window // 2)).mean().to_numpy()
        std = s.rolling(window, min_periods=max(2, window // 2)).std().to_numpy()
        z = (y - mean) / np.where(std > 0, std, np.nan)
        score = np.abs(z)
        residual = None
        is_anomaly = score > threshold

    anomalies = [
        AnomalyPoint(
            ts=ts[i],
            value=float(y[i]),
            zscore=None if np.isnan(z[i]) else float(z[i]),
            residual=None if residual is None or np.isnan(residual[i]) else float(residual[i]),
            score=0.0 if np.isnan(score[i]) else float(score[i]),
            is_anomaly=bool(is_anomaly[i]),
        )
        for i in range(n)
        if bool(is_anomaly[i])
    ]

    return AnomalyResult(
        equipment_id=equipment_id,
        channel=channel,
        method=method,
        threshold=threshold,
        n_points=n,
        n_anomalies=len(anomalies),
        anomalies=anomalies,
    )


# --------------------------------------------------------------------------- #
# 相关性
# --------------------------------------------------------------------------- #


class CorrelationInput(BaseModel):
    equipment_id: str = Field(description="设备编码，如 'ETCH-01'。")
    channel_a: str = Field(description="第一个通道。")
    channel_b: str = Field(description="第二个通道。")
    method: str = Field(default="pearson", description="'pearson' 或 'spearman'。")
    start: datetime | None = None
    end: datetime | None = None


@tool(
    name="correlation",
    description=(
        "计算两个 sensor 通道之间的相关性。相关性不等于"
        "因果性——在下结论前应结合 lag_analysis 和领域推理。"
    ),
    input_model=CorrelationInput,
    tags=["timeseries"],
)
def correlation(
    ctx: ToolContext,
    equipment_id: str,
    channel_a: str,
    channel_b: str,
    method: str = "pearson",
    start: datetime | None = None,
    end: datetime | None = None,
) -> CorrelationResult:
    if method not in ("pearson", "spearman"):
        raise ToolValidationError(f"unknown method '{method}'")
    df = _load_series(ctx, equipment_id, start, end)
    for c in (channel_a, channel_b):
        if c not in df.columns:
            raise ToolValidationError(f"unknown channel '{c}'")

    sub = df[[channel_a, channel_b]].dropna()
    coeff = float(sub[channel_a].corr(sub[channel_b], method=method))
    return CorrelationResult(
        channel_a=channel_a,
        channel_b=channel_b,
        method=method,
        coefficient=coeff,
        n=len(sub),
        note="correlation does not imply causation",
    )


# --------------------------------------------------------------------------- #
# 滞后分析
# --------------------------------------------------------------------------- #


class LagInput(BaseModel):
    equipment_id: str = Field(description="设备编码，如 'ETCH-01'。")
    channel_a: str = Field(description="领先通道（被平移）。")
    channel_b: str = Field(description="响应通道。")
    max_lag: int = Field(default=24, description="要扫描的最大滞后（以点数为单位，双向）。")
    start: datetime | None = None
    end: datetime | None = None


@tool(
    name="lag_analysis",
    description=(
        "扫描 channel_a 相对 channel_b 平移后的互相关，滞后范围"
        "[-max_lag, max_lag]。best_lag 为正表示 channel_a 领先 channel_b"
        "相应点数（这只是因果方向的线索，而非证明）。"
    ),
    input_model=LagInput,
    tags=["timeseries"],
)
def lag_analysis(
    ctx: ToolContext,
    equipment_id: str,
    channel_a: str,
    channel_b: str,
    max_lag: int = 24,
    start: datetime | None = None,
    end: datetime | None = None,
) -> LagResult:
    df = _load_series(ctx, equipment_id, start, end)
    for c in (channel_a, channel_b):
        if c not in df.columns:
            raise ToolValidationError(f"unknown channel '{c}'")
    a = df[channel_a].astype(float).to_numpy()
    b = df[channel_b].astype(float).to_numpy()

    corrs: list[dict] = []
    best_lag = 0
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            x, y = a[lag:], b[:-lag]
        elif lag < 0:
            x, y = a[:lag], b[-lag:]
        else:
            x, y = a, b
        if len(x) < 3:
            continue
        c = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
        corrs.append({"lag": lag, "correlation": c})
        if abs(c) > best:
            best = abs(c)
            best_lag = lag

    return LagResult(
        channel_a=channel_a,
        channel_b=channel_b,
        best_lag=best_lag,
        max_abs_corr=best,
        correlations=corrs,
    )


# --------------------------------------------------------------------------- #
# 统计分析
# --------------------------------------------------------------------------- #


class StatsInput(BaseModel):
    equipment_id: str = Field(description="设备编码，如 'ETCH-01'。")
    channel: str = Field(description="要汇总的通道。")
    start: datetime | None = None
    end: datetime | None = None


@tool(
    name="statistical_analysis",
    description="某通道的描述性统计（mean、std、分位数、skewness、kurtosis）。",
    input_model=StatsInput,
    tags=["timeseries"],
)
def statistical_analysis(
    ctx: ToolContext, equipment_id: str, channel: str,
    start: datetime | None = None, end: datetime | None = None,
) -> StatsResult:
    df = _load_series(ctx, equipment_id, start, end)
    if channel not in df.columns:
        raise ToolValidationError(f"unknown channel '{channel}'")
    s = df[channel].astype(float).dropna()
    return StatsResult(
        channel=channel,
        n=len(s),
        mean=float(s.mean()),
        std=float(s.std()),
        min=float(s.min()),
        max=float(s.max()),
        median=float(s.median()),
        p25=float(s.quantile(0.25)),
        p75=float(s.quantile(0.75)),
        skewness=float(s.skew()),
        kurtosis=float(s.kurt()),
    )
