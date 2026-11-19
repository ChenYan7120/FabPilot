"""Shared LangGraph state.

The shared state is the single source of truth passed between the Supervisor and
the specialised agents. Append-only log fields use ``operator.add`` reducers;
replaceable fields (hypotheses / evidence / pending checks) are overwritten in
full by whichever agent last updated them.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    # session identity
    session_id: str
    thread_id: str

    # task
    user_task: str
    status: str  # pending | running | completed | failed
    current_agent: str

    # append-only logs
    messages: Annotated[list[dict[str, Any]], operator.add]
    tool_history: Annotated[list[dict[str, Any]], operator.add]
    delegation_records: Annotated[list[dict[str, Any]], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[dict[str, Any]], operator.add]

    # additive results
    subtasks: Annotated[list[dict[str, Any]], operator.add]
    forecast_results: Annotated[list[dict[str, Any]], operator.add]
    anomaly_results: Annotated[list[dict[str, Any]], operator.add]
    retrieved_docs: Annotated[list[dict[str, Any]], operator.add]

    # replaceable memory (Hypothesis-Evidence)
    hypotheses: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    pending_checks: list[dict[str, Any]]

    # final output
    final_report: str

    # budgets / bookkeeping
    step_counts: dict[str, int]


# Fields that accumulate with operator.add (append-only).
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
    """Return the partial delta that turns ``before`` into ``after``.

    Append-only fields become the *added* items only (so a LangGraph node can
    return this delta without double-appending); replaceable fields are returned
    when they changed.
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
