"""Adapter skeletons for real deep time-series models (PatchTST, iTransformer).

These adapters define *how* a trained model plugs into the registry. They are
interface-only: invoking ``forecast`` before a checkpoint is loaded raises
``ForecastModelNotAvailable`` rather than returning fake numbers. PyTorch is an
optional dependency imported lazily so the rest of FabPilot runs without it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fabpilot.forecasting.base import (
    ForecastModel,
    ForecastModelNotAvailable,
    ForecastPoint,
    ForecastRequest,
    ForecastResult,
)


class _TorchForecastAdapter:
    """Shared scaffolding for checkpoint-backed models."""

    model_arch: str = "base"
    name: str = "base"

    def __init__(self) -> None:
        self._checkpoint_path: Path | None = None
        self._model: Any = None

    def load_checkpoint(self, path: str | Path, **model_kwargs: Any) -> None:
        try:
            import torch  # noqa: F401  (lazy import)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ForecastModelNotAvailable(
                "PyTorch is not installed. Install the 'forecast-torch' extra."
            ) from exc
        self._checkpoint_path = Path(path)
        if not self._checkpoint_path.exists():
            raise ForecastModelNotAvailable(f"Checkpoint not found: {path}")
        # In a real deployment this reconstructs the architecture, loads state_dict
        # and moves the model to eval mode. The exact keys are architecture-specific.
        self._model = self._build_model(**model_kwargs)
        self._model.load_state_dict(torch.load(self._checkpoint_path, map_location="cpu"))
        self._model.eval()

    def _build_model(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        if self._model is None:
            raise ForecastModelNotAvailable(
                f"{self.name} is registered but no checkpoint is loaded; "
                "call load_checkpoint() first or switch the default forecast model."
            )
        # Inference would convert request.series into a tensor, run the model and
        # return ForecastResult. Kept explicit so no placeholder values are emitted.
        raise ForecastModelNotAvailable(f"{self.name} inference not wired to a live model.")


class PatchTSTAdapter(_TorchForecastAdapter):
    name = "patchtst"
    model_arch = "PatchTST"

    def _build_model(self, **kwargs: Any) -> Any:
        raise NotImplementedError("PatchTST architecture reconstruction is implementation-specific.")


class iTransformerAdapter(_TorchForecastAdapter):
    name = "itransformer"
    model_arch = "iTransformer"

    def _build_model(self, **kwargs: Any) -> Any:
        raise NotImplementedError("iTransformer architecture reconstruction is implementation-specific.")


PATCHTST_ADAPTER: ForecastModel = PatchTSTAdapter()
ITRANSFORMER_ADAPTER: ForecastModel = iTransformerAdapter()
