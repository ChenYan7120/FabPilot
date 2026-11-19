"""LLM 消息 / 响应 schema。

我们有意使用自己定义的小型消息类型，而不是把 agent 循环耦合到特定厂商的 SDK。
OpenAI 兼容 provider 会把这些映射为线上传输格式；FakeLLM provider 则直接生成它们。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list["ToolCall"] = Field(default_factory=list)

    def to_openai(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        return msg


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    def to_openai(self) -> dict[str, Any]:
        import json

        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments, ensure_ascii=False)},
        }


class AssistantResponse(BaseModel):
    """LLM 一轮回复中可见、可审计的部分。

    ``content`` 是模型对外暴露的决策摘要；隐藏的 chain-of-thought 永远不会被记录或呈现。
    """

    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)
