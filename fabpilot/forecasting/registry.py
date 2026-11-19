"""统一的预测模型注册表。

PatchTST、iTransformer 或任意自定义模型都通过统一的 ``ForecastModel`` 接口接入，因此时序 Agent 从不依赖具体实现。
"""
from __future__ import annotations

from fabpilot.config import Settings, get_settings
from fabpilot.forecasting.base import ForecastModel, ForecastModelNotAvailable
from fabpilot.forecasting.baseline import StatisticalBaselineModel
from fabpilot.utils import ConfigurationError


class ForecastModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, ForecastModel] = {}

    def register(self, model: ForecastModel) -> None:
        self._models[model.name] = model

    def get(self, name: str) -> ForecastModel:
        if name not in self._models:
            raise ConfigurationError(
                f"Unknown forecast model '{name}'. Available: {sorted(self._models)}"
            )
        return self._models[name]

    def names(self) -> list[str]:
        return sorted(self._models)

    def forecast(self, name: str, request):
        model = self.get(name)
        try:
            return model.forecast(request)
        except ForecastModelNotAvailable:
            raise


def build_forecast_registry(settings: Settings | None = None) -> ForecastModelRegistry:
    """包含已注册的确定性统计基线与适配器骨架的注册表（适配器仅在权重加载后才能运行）。"""
    settings = settings or get_settings()
    registry = ForecastModelRegistry()
    registry.register(StatisticalBaselineModel())

    from fabpilot.forecasting.adapters import ITRANSFORMER_ADAPTER, PATCHTST_ADAPTER

    registry.register(PATCHTST_ADAPTER)
    registry.register(ITRANSFORMER_ADAPTER)
    return registry


def get_default_model_name(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    return settings.default_forecast_model
