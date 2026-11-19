"""Deterministic FakeLLM for offline / integration testing.

This provider replays a caller-supplied script of (content, tool_calls) turns.
It is *only* a test double: it makes no attempt to reason, and it is never used
to masquerade pre-scripted routing as autonomous LLM behaviour. When the script
is exhausted it returns an empty stop response so graphs always terminate.
"""
from __future__ import annotations

from collections import deque

from fabpilot.llm.schemas import AssistantResponse, ChatMessage, ToolCall
from fabpilot.utils import new_id


class FakeLLM:
    def __init__(self, script: list[AssistantResponse] | None = None) -> None:
        self._script: deque[AssistantResponse] = deque(script or [])
        self.calls: list[dict] = []  # audit of everything it was asked to complete

    def complete(self, messages: list[ChatMessage], tools: list[dict] | None = None) -> AssistantResponse:
        self.calls.append(
            {
                "n_messages": len(messages),
                "last_role": messages[-1].role if messages else None,
                "tools": [t["function"]["name"] for t in (tools or [])],
            }
        )
        if self._script:
            return self._script.popleft()
        return AssistantResponse(content="", tool_calls=[], finish_reason="stop")

    # ---- convenience constructors used by tests ----
    @staticmethod
    def tool(name: str, arguments: dict | None = None, content: str = "") -> AssistantResponse:
        return AssistantResponse(
            content=content,
            tool_calls=[ToolCall(id=new_id("call"), name=name, arguments=arguments or {})],
        )

    @staticmethod
    def stop(content: str = "done") -> AssistantResponse:
        return AssistantResponse(content=content, tool_calls=[], finish_reason="stop")
