"""用于离线 / 集成测试的确定性 FakeLLM。

该 provider 会按调用方提供的脚本重放一组 (content, tool_calls) 轮次。它 *只是* 一个
测试替身：它不会尝试推理，也绝不会把预先编排好的路由伪装成自主的 LLM 行为。
当脚本耗尽时，它会返回空的 stop 响应，从而保证图总能终止。
"""
from __future__ import annotations

from collections import deque

from fabpilot.llm.schemas import AssistantResponse, ChatMessage, ToolCall
from fabpilot.utils import new_id


class FakeLLM:
    def __init__(self, script: list[AssistantResponse] | None = None) -> None:
        self._script: deque[AssistantResponse] = deque(script or [])
        self.calls: list[dict] = []  # 记录它被要求 complete 的每次调用

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

    # ---- 供测试使用的便捷构造方法 ----
    @staticmethod
    def tool(name: str, arguments: dict | None = None, content: str = "") -> AssistantResponse:
        return AssistantResponse(
            content=content,
            tool_calls=[ToolCall(id=new_id("call"), name=name, arguments=arguments or {})],
        )

    @staticmethod
    def stop(content: str = "done") -> AssistantResponse:
        return AssistantResponse(content=content, tool_calls=[], finish_reason="stop")
