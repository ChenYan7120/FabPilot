"""Deterministic scripted LLM for evaluation.

``FakeLLM`` replays a flat list of responses. That is enough for routing, but the
Diagnosis Agent's memory tools are *chained*: ``add_evidence`` needs the id that
``add_hypothesis`` returned on the previous turn. ``ScriptedLLM`` closes that gap
by letting a script step be either a static ``AssistantResponse`` or a callable
that receives a ``ScriptContext`` of the *real* tool results observed so far.

This is still a test double: it makes no autonomous decisions. Every tool it
asks the agent loop to run is a real tool over the real database / forecast
registry / knowledge base, so the metrics computed downstream reflect real
outputs, not scripted strings.
"""
from __future__ import annotations

import json
from collections import deque
from typing import Any, Callable, Union

from fabpilot.llm.schemas import AssistantResponse, ChatMessage


class ScriptContext:
    """A view of the tool results produced so far, populated from the tool
    messages the ReAct loop feeds back."""

    def __init__(self) -> None:
        self.tool_results: list[dict[str, Any]] = []

    def data_for(self, name: str) -> dict[str, Any] | None:
        for r in reversed(self.tool_results):
            if r.get("name") == name and r.get("ok"):
                return r.get("data")
        return None

    @property
    def last(self) -> dict[str, Any] | None:
        return self.tool_results[-1] if self.tool_results else None


Step = Union[AssistantResponse, Callable[[ScriptContext], AssistantResponse]]


class ScriptedLLM:
    def __init__(self, steps: list[Step]) -> None:
        self._steps: deque[Step] = deque(steps)
        self.ctx = ScriptContext()
        self.calls: list[dict[str, Any]] = []
        # Track how many tool messages we have already ingested *per messages list*,
        # because the supervisor and each sub-agent run their own ReAct loop with
        # their own growing ``messages`` list (same object across iterations).
        self._seen_by_list: dict[int, int] = {}

    def complete(self, messages: list[ChatMessage], tools: list[dict] | None = None) -> AssistantResponse:
        self.calls.append(
            {
                "n_messages": len(messages),
                "last_role": messages[-1].role if messages else None,
                "tools": [t["function"]["name"] for t in (tools or [])],
            }
        )

        # Ingest the new tool-result messages (observations from the previous turn).
        list_id = id(messages)
        tool_msgs = [m for m in messages if m.role == "tool"]
        seen = self._seen_by_list.get(list_id, 0)
        for m in tool_msgs[seen:]:
            self.ctx.tool_results.append(_parse_tool_message(m))
        self._seen_by_list[list_id] = len(tool_msgs)

        if not self._steps:
            return AssistantResponse(content="", tool_calls=[], finish_reason="stop")
        step = self._steps.popleft()
        return step(self.ctx) if callable(step) else step


def _parse_tool_message(m: ChatMessage) -> dict[str, Any]:
    name = m.name or ""
    content = m.content or ""
    ok = not content.startswith("ERROR:")
    data: dict[str, Any] = {}
    error: str | None = None
    if ok:
        try:
            parsed = json.loads(content)
            data = parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            data = {"raw": content}
    else:
        error = content[len("ERROR: ") :] if content.startswith("ERROR: ") else content
    return {"name": name, "ok": ok, "data": data, "error": error}


# Convenience constructors so task scripts read like the FakeLLM API.
def tool(name: str, arguments: dict | None = None, content: str = "") -> AssistantResponse:
    from fabpilot.llm.fake import FakeLLM

    return FakeLLM.tool(name, arguments, content)


def stop(content: str = "done") -> AssistantResponse:
    from fabpilot.llm.fake import FakeLLM

    return FakeLLM.stop(content)
