"""LLM message / response schemas.

We intentionally use our own small message types instead of coupling the agent
loop to a specific vendor's SDK. The OpenAI-compatible provider maps these to
the wire format; the FakeLLM provider produces them directly.
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
    """The visible, auditable part of an LLM turn.

    ``content`` is the decision summary the model exposes; hidden chain-of-thought
    is never recorded or surfaced.
    """

    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)
