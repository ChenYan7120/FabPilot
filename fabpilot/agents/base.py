"""Supervisor 与每个专职 agent 共用的 ReAct 循环。

该循环与工具无关：先让 LLM 做决策，再在真实注册表上执行所请求的工具，把观察结果
反馈回去，直到 LLM 停止调用工具（或触发预算/循环保护）时结束。每一轮都会记录
可审计的决策摘要 + 工具调用 + 结果——绝不隐藏思维链。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from fabpilot.llm.schemas import ChatMessage, ToolCall
from fabpilot.tools.registry import ToolContext, ToolRegistry, ToolResult
from fabpilot.utils import AgentLoopError, get_logger, new_id

logger = get_logger(__name__)


@dataclass
class ReactResult:
    final_content: str
    messages: list[ChatMessage] = field(default_factory=list)
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    stopped_by: str = "llm_stop"  # llm_stop | max_iterations | tool_budget | loop


def _stable_key(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, sort_keys=True, default=str)


def _detect_loop(history: list[dict[str, Any]], window: int) -> bool:
    if window < 2 or len(history) < window:
        return False
    tail = history[-window:]
    key = (tail[0]["name"], _stable_key(tail[0]["arguments"]))
    return all((h["name"], _stable_key(h["arguments"])) == key for h in tail[1:])


def run_react_loop(
    llm,
    tool_registry: ToolRegistry,
    ctx: ToolContext,
    system_prompt: str,
    messages: list[ChatMessage],
    max_iterations: int,
    tool_budget: int,
    loop_detection_window: int,
    allowed_tools: list[str] | None = None,
    agent_name: str = "agent",
) -> ReactResult:
    allowed = set(allowed_tools) if allowed_tools is not None else None
    tool_schemas = tool_registry.schemas(allowed_tools)
    full: list[ChatMessage] = [ChatMessage(role="system", content=system_prompt), *messages]

    tool_history: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    tool_calls_made = 0

    for step in range(max_iterations):
        resp = llm.complete(full, tool_schemas)

        if not resp.wants_tools:
            full.append(ChatMessage(role="assistant", content=resp.content))
            return ReactResult(
                final_content=resp.content or "",
                messages=full,
                tool_history=tool_history,
                decisions=decisions,
                stopped_by="llm_stop",
            )

        assistant_msg = ChatMessage(role="assistant", content=resp.content or "", tool_calls=resp.tool_calls)
        full.append(assistant_msg)

        step_results: list[dict[str, Any]] = []
        for tc in resp.tool_calls:
            if tool_calls_made >= tool_budget:
                step_results.append({"tool": tc.name, "ok": False, "error": "tool budget exhausted"})
                continue
            result = _execute(tool_registry, ctx, tc, allowed, agent_name)
            tool_calls_made += 1
            tool_history.append(
                {
                    "agent": agent_name,
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "ok": result.ok,
                    "error": result.error,
                    "data": result.data,
                }
            )
            full.append(
                ChatMessage(
                    role="tool",
                    content=result.as_message(),
                    tool_call_id=tc.id,
                    name=tc.name,
                )
            )
            step_results.append({"tool": tc.name, "ok": result.ok, "error": result.error})

        decisions.append(
            {
                "step": step,
                "agent": agent_name,
                "decision": resp.content or "",
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in resp.tool_calls
                ],
                "results": step_results,
            }
        )

        if tool_calls_made >= tool_budget:
            return ReactResult(
                final_content=resp.content or "tool budget exhausted",
                messages=full,
                tool_history=tool_history,
                decisions=decisions,
                stopped_by="tool_budget",
            )
        if _detect_loop(tool_history, loop_detection_window):
            return ReactResult(
                final_content=resp.content or "loop detected",
                messages=full,
                tool_history=tool_history,
                decisions=decisions,
                stopped_by="loop",
            )

    return ReactResult(
        final_content="maximum iterations reached",
        messages=full,
        tool_history=tool_history,
        decisions=decisions,
        stopped_by="max_iterations",
    )


def _execute(
    tool_registry: ToolRegistry,
    ctx: ToolContext,
    tc: ToolCall,
    allowed: set[str] | None,
    agent_name: str,
) -> ToolResult:
    if not tc.name:
        return ToolResult(ok=False, tool="", error="empty tool name")
    if allowed is not None and tc.name not in allowed:
        return ToolResult(ok=False, tool=tc.name, error=f"tool '{tc.name}' not permitted for {agent_name}")
    return tool_registry.invoke(ctx, tc.arguments or {})
