"""预测模型注册表测试：真实统计基线与适配器骨架的对比。"""
from __future__ import annotations

import numpy as np
import pytest

from fabpilot.forecasting.base import (
    ForecastModelNotAvailable,
    ForecastRequest,
)
from fabpilot.forecasting.baseline import StatisticalBaselineModel


def _request(**kw) -> ForecastRequest:
    ts = [__import__("datetime").datetime(2026, 1, 1, 0, i) for i in range(20)]
    series = {"gas_flow": list(map(float, range(20)))}
    return ForecastRequest(
        equipment_id="ETCH-01", target="gas_flow", lookback=20, horizon=5, series=series, timestamps=ts, **kw
    )


def test_baseline_model_returns_forecast(forecast_registry):
    req = _request()
    result = forecast_registry.forecast("statistical_baseline", req)
    assert result.model_name == "statistical_baseline"
    assert len(result.forecast) == req.horizon
    assert all(np.isfinite(p.value) for p in result.forecast)


def test_baseline_linear_trend_extrapolates():
    model = StatisticalBaselineModel()
    req = _request()
    result = model.forecast(req)
    # 完全线性的递增序列 → 预测结果继续递增。
    assert result.forecast[-1].value > result.forecast[0].value


def test_adapters_are_interface_only_and_raise(forecast_registry):
    req = _request()
    for name in ("patchtst", "itransformer"):
        with pytest.raises(ForecastModelNotAvailable):
            forecast_registry.forecast(name, req)


def test_unknown_model_raises(forecast_registry):
    from fabpilot.utils import ConfigurationError

    with pytest.raises(ConfigurationError):
        forecast_registry.forecast("does_not_exist", _request())
