"""Unified Forecasting Model Registry.

PatchTST, iTransformer or any custom model plug in through a single
``ForecastModel`` interface so the Time-Series Agent never depends on a concrete
implementation.
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
    """Registry with the deterministic statistical baseline and the adapter
    skeletons registered (adapters only run once weights are loaded)."""
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
