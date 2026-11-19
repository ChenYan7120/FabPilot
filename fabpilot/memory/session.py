"""Session memory: per-session message history (short-term working memory)."""
from __future__ import annotations

from typing import Any

from fabpilot.llm.schemas import ChatMessage
from fabpilot.utils import new_id


class SessionMemory:
    def __init__(self, session_id: str | None = None, thread_id: str | None = None) -> None:
        self.session_id = session_id or new_id("sess")
        self.thread_id = thread_id or self.session_id
        self.messages: list[ChatMessage] = []

    def add(self, role: str, content: str, **kwargs: Any) -> None:
        self.messages.append(ChatMessage(role=role, content=content, **kwargs))  # type: ignore[arg-type]

    def recent(self, n: int = 12) -> list[ChatMessage]:
        return self.messages[-n:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "messages": [m.model_dump() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMemory":
        mem = cls(session_id=data["session_id"], thread_id=data.get("thread_id"))
        mem.messages = [ChatMessage.model_validate(m) for m in data.get("messages", [])]
        return mem
