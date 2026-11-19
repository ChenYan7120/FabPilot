"""共享的 LangGraph state。

共享 state 是 Supervisor 与各专职 agent 之间传递的唯一事实来源。追加式日志字段使用
``operator.add`` 作为 reducer；可替换字段（hypotheses / evidence / pending checks）会
被最后一次更新它们的 agent 整体覆盖。
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    # 会话标识
    session_id: str
    thread_id: str

    # 任务
    user_task: str
    status: str  # pending | running | completed | failed
    current_agent: str

    # 追加式日志
    messages: Annotated[list[dict[str, Any]], operator.add]
    tool_history: Annotated[list[dict[str, Any]], operator.add]
    delegation_records: Annotated[list[dict[str, Any]], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[dict[str, Any]], operator.add]

    # 累加式结果
    subtasks: Annotated[list[dict[str, Any]], operator.add]
    forecast_results: Annotated[list[dict[str, Any]], operator.add]
    anomaly_results: Annotated[list[dict[str, Any]], operator.add]
    retrieved_docs: Annotated[list[dict[str, Any]], operator.add]

    # 可替换 memory（Hypothesis-Evidence）
    hypotheses: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    pending_checks: list[dict[str, Any]]

    # 最终输出
    final_report: str

    # 预算 / 记账
    step_counts: dict[str, int]


# 通过 operator.add 累加的字段（追加式）。
LIST_FIELDS: frozenset[str] = frozenset(
    {
        "messages",
        "tool_history",
        "delegation_records",
        "trace",
        "errors",
        "subtasks",
        "forecast_results",
        "anomaly_results",
        "retrieved_docs",
    }
)


def initial_state(session_id: str, thread_id: str, user_task: str) -> AgentState:
    return {
        "session_id": session_id,
        "thread_id": thread_id,
        "user_task": user_task,
        "status": "pending",
        "current_agent": "supervisor",
        "messages": [],
        "tool_history": [],
        "delegation_records": [],
        "trace": [],
        "errors": [],
        "subtasks": [],
        "forecast_results": [],
        "anomaly_results": [],
        "retrieved_docs": [],
        "hypotheses": [],
        "evidence": [],
        "pending_checks": [],
        "final_report": "",
        "step_counts": {"supervisor": 0, "timeseries": 0, "diagnosis": 0, "knowledge": 0},
    }


def diff_state(before: AgentState, after: AgentState) -> AgentState:
    """返回将 ``before`` 变为 ``after`` 的局部差异。

    追加式字段只包含 *新增* 的条目（这样 LangGraph 节点返回该差异时不会重复追加）；
    可替换字段仅在其发生变化时返回。
    """
    delta: AgentState = {}
    for key in after:
        if key in LIST_FIELDS:
            added = after[key][len(before.get(key, [])) :]
            if added:
                delta[key] = added  # type: ignore[literal-required]
        else:
            if before.get(key) != after[key]:
                delta[key] = after[key]  # type: ignore[literal-required]
    return delta
