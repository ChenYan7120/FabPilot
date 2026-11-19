"""共享工具：日志、异常与标识符生成。"""
from __future__ import annotations

import logging
import uuid
from typing import Any


class FabPilotError(Exception):
    """所有 FabPilot 错误的基类。"""


class ConfigurationError(FabPilotError):
    """当应用配置错误时抛出。"""


class DataAccessError(FabPilotError):
    """当数据源无法查询或数据不一致时抛出。"""


class ToolExecutionError(FabPilotError):
    """当某个 tool 执行失败时抛出。"""

    def __init__(self, tool_name: str, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.tool_name = tool_name
        self.retryable = retryable


class ToolValidationError(ToolExecutionError):
    """当 tool 参数校验失败时抛出。"""


class AgentLoopError(FabPilotError):
    """当 agent 的 ReAct 循环触发硬停止（预算 / 循环）时抛出。"""


_configured = False


def setup_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def new_id(prefix: str) -> str:
    """确定性友好的唯一 id，例如 ``hyp_3f2a...``。"""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def safe_json(obj: Any) -> Any:
    """尽最大努力把值转换为可 JSON 序列化的结构。"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [safe_json(v) for v in obj]
    return str(obj)
