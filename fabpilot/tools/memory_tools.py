"""Hypothesis-Evidence Memory 工具。

这些是 Diagnosis Agent 的 *内部* 工具：它们修改确定性的
``HypothesisEvidenceMemory``（而不是数据库）。把它们做成工具，可以让 ReAct
循环以显式、可审计的步骤驱动 Observation -> Hypothesis -> Evidence -> Re-evaluation 循环，
而不是隐藏在 prompt 行为里。
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from fabpilot.memory.hypothesis import EvidenceKind, HypothesisEvidenceMemory
from fabpilot.tools.registry import ToolContext, tool
from fabpilot.utils import ToolValidationError

MEMORY_TOOL_NAMES = [
    "add_hypothesis",
    "add_evidence",
    "add_pending_check",
    "complete_pending_check",
    "evaluate_hypothesis",
    "list_hypotheses",
]


def _memory(ctx: ToolContext) -> HypothesisEvidenceMemory:
    if ctx.memory is None:
        raise ToolValidationError("hypothesis-evidence memory is not initialised")
    return ctx.memory


class AddHypothesisInput(BaseModel):
    description: str = Field(description="该 hypothesis，表述为关于根因的可证伪断言。")
    priority: float = Field(default=0.5, description="优先级，取值 [0,1]；数值越大越先调查。")


@tool(
    name="add_hypothesis",
    description="在工作 memory 中记录一个新的可证伪根因 hypothesis。",
    input_model=AddHypothesisInput,
    tags=["memory"],
)
def add_hypothesis(ctx: ToolContext, description: str, priority: float = 0.5) -> dict:
    hyp = _memory(ctx).add_hypothesis(description, priority=priority)
    return {"hypothesis_id": hyp.id, "description": hyp.description, "status": hyp.status.value}


class AddEvidenceInput(BaseModel):
    hypothesis_id: str = Field(description="该 evidence 所支撑的 hypothesis。")
    source_tool: str = Field(description="产生该观测的 tool（如 'correlation'）。")
    observation: str = Field(description="简洁的事实性观测，如 'corr = 0.82'。")
    kind: str = Field(default="neutral", description="supporting | refuting | neutral")
    confidence: float = Field(default=0.5, description="置信度，取值 [0,1]。")
    query_params: dict[str, Any] = Field(default_factory=dict, description="所使用的 tool 参数。")
    source_refs: list[str] = Field(default_factory=list, description="文档 / 案例引用。")


@tool(
    name="add_evidence",
    description=(
        "将一条 evidence 观测附加到某个 hypothesis。correlation 只会被记录为"
        "evidence，而不会被当作因果——状态由 memory 的确定性评估决定。"
    ),
    input_model=AddEvidenceInput,
    tags=["memory"],
)
def add_evidence(
    ctx: ToolContext,
    hypothesis_id: str,
    source_tool: str,
    observation: str,
    kind: str = "neutral",
    confidence: float = 0.5,
    query_params: dict[str, Any] | None = None,
    source_refs: list[str] | None = None,
) -> dict:
    mem = _memory(ctx)
    if hypothesis_id not in mem.hypotheses:
        raise ToolValidationError(f"unknown hypothesis '{hypothesis_id}'")
    try:
        kind_enum = EvidenceKind(kind)
    except ValueError:
        raise ToolValidationError(f"unknown evidence kind '{kind}'")
    ev = mem.add_evidence(
        hypothesis_id=hypothesis_id,
        source_tool=source_tool,
        query_params=query_params or {},
        observation=observation,
        kind=kind_enum,
        source_refs=source_refs or [],
        confidence=confidence,
    )
    return {"evidence_id": ev.id, "kind": ev.kind.value, "observation": ev.observation}


class AddPendingCheckInput(BaseModel):
    hypothesis_id: str = Field(description="要验证的 hypothesis。")
    description: str = Field(description="还需要检查的内容。")
    tool: str = Field(description="为该项检查运行的 tool。")
    query: dict[str, Any] = Field(default_factory=dict, description="该 tool 的参数。")


@tool(
    name="add_pending_check",
    description="注册一个验证步骤，必须在其运行后 hypothesis 才能被支持。",
    input_model=AddPendingCheckInput,
    tags=["memory"],
)
def add_pending_check(
    ctx: ToolContext, hypothesis_id: str, description: str, tool: str, query: dict[str, Any] | None = None
) -> dict:
    mem = _memory(ctx)
    if hypothesis_id not in mem.hypotheses:
        raise ToolValidationError(f"unknown hypothesis '{hypothesis_id}'")
    check = mem.add_pending_check(hypothesis_id, description, tool, query or {})
    return {"check_id": check.id, "description": check.description}


class CompleteCheckInput(BaseModel):
    hypothesis_id: str = Field(description="拥有该项检查的 hypothesis。")
    check_id: str = Field(description="待处理检查的 id。")
    evidence_ids: list[str] = Field(default_factory=list, description="该项检查产生的 evidence。")


@tool(
    name="complete_pending_check",
    description="将某项待处理的验证检查标记为已完成，并关联其产生的 evidence。",
    input_model=CompleteCheckInput,
    tags=["memory"],
)
def complete_pending_check(
    ctx: ToolContext, hypothesis_id: str, check_id: str, evidence_ids: list[str] | None = None
) -> dict:
    _memory(ctx).complete_pending_check(hypothesis_id, check_id, evidence_ids or [])
    return {"hypothesis_id": hypothesis_id, "check_id": check_id, "status": "done"}


class EvaluateHypothesisInput(BaseModel):
    hypothesis_id: str = Field(description="要根据其 evidence 重新评估的 hypothesis。")


@tool(
    name="evaluate_hypothesis",
    description="根据其 evidence 和待处理检查，确定性地重新评估某个 hypothesis 的状态。",
    input_model=EvaluateHypothesisInput,
    tags=["memory"],
)
def evaluate_hypothesis(ctx: ToolContext, hypothesis_id: str) -> dict:
    status = _memory(ctx).evaluate_hypothesis(hypothesis_id)
    return {"hypothesis_id": hypothesis_id, "status": status.value}


class ListHypothesesInput(BaseModel):
    pass


@tool(
    name="list_hypotheses",
    description="从工作 memory 中返回当前的 hypotheses 及其状态。",
    input_model=ListHypothesesInput,
    tags=["memory"],
)
def list_hypotheses(ctx: ToolContext) -> dict:
    return _memory(ctx).snapshot()
