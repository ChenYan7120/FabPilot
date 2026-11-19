"""用于真实深度时序模型（PatchTST、iTransformer）的 adapter 骨架。

这些 adapter 定义训练好的模型 *如何* 接入 registry。它们仅提供接口：在 checkpoint 加载之前调用 ``forecast`` 会抛出 ``ForecastModelNotAvailable``，而不是返回虚假数值。PyTorch 是可选依赖，采用惰性导入，因此 FabPilot 的其余部分无需它即可运行。
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
    """基于 checkpoint 的模型所共用的脚手架。"""

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
        # 在真实部署中，这里会重建架构、加载 state_dict 并将模型切换到 eval 模式。具体的键因架构而异。
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
        # 推理会将 request.series 转换为 tensor、运行模型并返回 ForecastResult。这里保持显式，以免输出任何占位值。
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
