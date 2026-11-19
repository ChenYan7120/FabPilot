"""Shared utilities: logging, exceptions and identifier generation."""
from __future__ import annotations

import logging
import uuid
from typing import Any


class FabPilotError(Exception):
    """Base class for all FabPilot errors."""


class ConfigurationError(FabPilotError):
    """Raised when the application is misconfigured."""


class DataAccessError(FabPilotError):
    """Raised when a data source cannot be queried or is inconsistent."""


class ToolExecutionError(FabPilotError):
    """Raised when a tool fails to execute."""

    def __init__(self, tool_name: str, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.tool_name = tool_name
        self.retryable = retryable


class ToolValidationError(ToolExecutionError):
    """Raised when tool arguments fail validation."""


class AgentLoopError(FabPilotError):
    """Raised when an agent ReAct loop hits a hard stop (budget / loop)."""


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
    """Deterministic-friendly unique id, e.g. ``hyp_3f2a...``."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def safe_json(obj: Any) -> Any:
    """Best-effort conversion of a value to a JSON-serialisable structure."""
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
