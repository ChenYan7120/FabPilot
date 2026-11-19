"""Tool Registry: the single place tools are defined, validated and executed.

Every tool exposes structured Pydantic input, validates its arguments, runs
against a real implementation and returns a structured ``ToolResult`` envelope.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from fabpilot.utils import ToolExecutionError, ToolValidationError, get_logger, safe_json

logger = get_logger(__name__)


@dataclass
class ToolContext:
    """Shared services injected into every tool call."""

    session_scope: Callable | None = None  # context-manager yielding a DB Session
    forecast_registry: Any = None
    knowledge_base: Any = None
    memory: Any = None  # HypothesisEvidenceMemory (used by the Diagnosis Agent)
    settings: Any = None
    chart_dir: str = "charts"
    report_dir: str = "reports"


class ToolResult(BaseModel):
    ok: bool
    tool: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_message(self) -> str:
        if self.ok:
            import json

            return json.dumps(self.data, ensure_ascii=False, default=str)
        return f"ERROR: {self.error}"


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        func: Callable[..., Any],
        tags: list[str] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.input_model = input_model
        self.func = func
        self.tags = tags or []

    def to_openai_schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        }

    def invoke(self, ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        try:
            validated = self.input_model.model_validate(arguments or {})
        except ValidationError as exc:
            return ToolResult(ok=False, tool=self.name, error=f"invalid arguments: {exc.errors()}")
        try:
            output = self.func(ctx, **validated.model_dump())
        except ToolValidationError as exc:
            return ToolResult(ok=False, tool=self.name, error=str(exc))
        except Exception as exc:  # defensive: a tool must never crash the agent loop
            logger.exception("tool %s failed", self.name)
            return ToolResult(ok=False, tool=self.name, error=f"{type(exc).__name__}: {exc}")

        if isinstance(output, BaseModel):
            data = output.model_dump(mode="json")
        elif isinstance(output, dict):
            data = safe_json(output)
        else:
            data = {"value": safe_json(output)}
        return ToolResult(ok=True, tool=self.name, data=data)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolExecutionError(name, f"unknown tool '{name}'", retryable=False)
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        return [self._tools[n] for n in sorted(self._tools)]

    def subset(self, names: list[str]) -> list[Tool]:
        return [self.get(n) for n in names]

    def schemas(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        tools = self.subset(names) if names else self.all()
        return [t.to_openai_schema() for t in tools]


def tool(name: str, description: str, input_model: type[BaseModel], tags: list[str] | None = None):
    """Decorator that registers a plain function as a ``Tool``."""

    def decorator(func: Callable[..., Any]) -> Tool:
        return Tool(name=name, description=description, input_model=input_model, func=func, tags=tags)

    return decorator
