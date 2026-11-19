"""确定性的、依赖轻量的预测基线。

这是一个 *真实* 模型（不是占位符）：它在 lookback 窗口上拟合简单趋势并外推，因此整个 agent 流水线、评估指标和演示场景无需 PyTorch 即可端到端产出真实数值。真实工业模型（PatchTST、iTransformer 等）通过同一注册表替换它。
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np

from fabpilot.forecasting.base import ForecastModel, ForecastPoint, ForecastRequest, ForecastResult


class StatisticalBaselineModel:
    name = "statistical_baseline"

    def __init__(self, method: str = "linear_trend") -> None:
        if method not in {"linear_trend", "persistence", "mean"}:
            raise ValueError(f"Unknown baseline method: {method}")
        self.method = method

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        y = np.asarray(request.series[request.target], dtype=float)[-request.lookback :]
        if y.size < 2:
            raise ValueError("lookback window too small for forecasting")

        x = np.arange(y.size, dtype=float)
        if self.method == "linear_trend":
            slope, intercept = np.polyfit(x, y, 1)
            fit = slope * x + intercept
            future = slope * (x[-1] + 1 + np.arange(request.horizon)) + intercept
        elif self.method == "persistence":
            fit = np.full_like(y, y[-1])
            future = np.full(request.horizon, y[-1])
        else:  # mean
            fit = np.full_like(y, y.mean())
            future = np.full(request.horizon, y.mean())

        residuals = y - fit
        sigma = float(residuals.std()) if residuals.size > 1 else 0.0
        z = 1.96  # 正态性假设下的约 95% 区间
        lower = (future - z * sigma).tolist()
        upper = (future + z * sigma).tolist()

        interval = _infer_interval(request.timestamps)
        if not request.timestamps:
            raise ValueError("forecast request requires timestamps")
        last_ts = request.timestamps[-1]
        points = [
            ForecastPoint(ts=last_ts + timedelta(seconds=interval * (i + 1)), value=float(v))
            for i, v in enumerate(future)
        ]

        return ForecastResult(
            model_name=self.name,
            equipment_id=request.equipment_id,
            target=request.target,
            lookback=request.lookback,
            horizon=request.horizon,
            forecast=points,
            lower=lower,
            upper=upper,
            in_sample_residuals=residuals.tolist(),
            metrics={
                "method": self.method,
                "in_sample_mse": float((residuals**2).mean()),
                "in_sample_mae": float(np.abs(residuals).mean()),
                "residual_std": sigma,
            },
        )


def _infer_interval(timestamps: list) -> int:
    if len(timestamps) >= 2:
        delta = (timestamps[-1] - timestamps[-2]).total_seconds()
        if delta > 0:
            return int(delta)
    return 300  # 默认 5 分钟
