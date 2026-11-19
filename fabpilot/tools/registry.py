"""工具注册表：工具的定义、校验与执行的唯一入口。

每个工具暴露结构化的 Pydantic 输入，校验其参数，运行真实的
实现，并返回结构化的 ``ToolResult`` 封装。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from fabpilot.utils import ToolExecutionError, ToolValidationError, get_logger, safe_json

logger = get_logger(__name__)


@dataclass
class ToolContext:
    """注入到每个工具调用中的共享服务。"""

    session_scope: Callable | None = None  # 生成 DB Session 的上下文管理器
    forecast_registry: Any = None
    knowledge_base: Any = None
    memory: Any = None  # 假设-证据记忆（诊断 Agent 使用）
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
        except Exception as exc:  # 防御性处理：工具绝不能使 agent 循环崩溃
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
    """将普通函数注册为 ``Tool`` 的装饰器。"""

    def decorator(func: Callable[..., Any]) -> Tool:
        return Tool(name=name, description=description, input_model=input_model, func=func, tags=tags)

    return decorator
