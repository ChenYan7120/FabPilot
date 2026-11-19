"""用于评测的确定性脚本化 LLM。

``FakeLLM`` 只是重放一个扁平的响应列表，对路由而言已足够，但
诊断 Agent 的记忆工具是*链式*的：``add_evidence`` 需要上一轮
``add_hypothesis`` 返回的 id。``ScriptedLLM`` 弥补了这一缺口——
它允许脚本步骤既可以是静态的 ``AssistantResponse``，也可以是接收
``ScriptContext``（含迄今观测到的*真实* tool 结果）的可调用对象。

它仍然是测试替身：不做出任何自主决策。它要求 agent 循环运行的每个 tool
都是作用于真实数据库 / 预测注册表 / 知识库之上的真实 tool，因此下游计算的
指标反映的是真实输出，而非脚本字符串。
"""
from __future__ import annotations

import json
from collections import deque
from typing import Any, Callable, Union

from fabpilot.llm.schemas import AssistantResponse, ChatMessage


class ScriptContext:
    """对迄今产生的 tool 结果的视图，由 ReAct 循环回传的
    tool 消息填充而成。"""

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
        # 记录*每个 messages 列表*中已摄入的 tool 消息数量，
        # 因为 supervisor 与每个 sub-agent 各自运行自己的 ReAct 循环，
        # 且各自维护不断增长的 ``messages`` 列表（跨迭代复用同一对象）。
        self._seen_by_list: dict[int, int] = {}

    def complete(self, messages: list[ChatMessage], tools: list[dict] | None = None) -> AssistantResponse:
        self.calls.append(
            {
                "n_messages": len(messages),
                "last_role": messages[-1].role if messages else None,
                "tools": [t["function"]["name"] for t in (tools or [])],
            }
        )

        # 摄入新的 tool 结果消息（上一轮的观察）。
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


# 便捷构造器，让任务脚本读起来像 FakeLLM API。
def tool(name: str, arguments: dict | None = None, content: str = "") -> AssistantResponse:
    from fabpilot.llm.fake import FakeLLM

    return FakeLLM.tool(name, arguments, content)


def stop(content: str = "done") -> AssistantResponse:
    from fabpilot.llm.fake import FakeLLM

    return FakeLLM.stop(content)
