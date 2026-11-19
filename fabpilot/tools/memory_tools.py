"""Hypothesis-Evidence Memory tools.

These are the Diagnosis Agent's *internal* tools: they mutate the deterministic
``HypothesisEvidenceMemory`` (not the database). Making them tools lets the ReAct
loop drive the Observation -> Hypothesis -> Evidence -> Re-evaluation cycle as
explicit, auditable steps instead of hidden prompt behaviour.
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
    description: str = Field(description="The hypothesis, stated as a falsifiable claim about a root cause.")
    priority: float = Field(default=0.5, description="Priority in [0,1]; higher = investigate first.")


@tool(
    name="add_hypothesis",
    description="Record a new falsifiable root-cause hypothesis in the working memory.",
    input_model=AddHypothesisInput,
    tags=["memory"],
)
def add_hypothesis(ctx: ToolContext, description: str, priority: float = 0.5) -> dict:
    hyp = _memory(ctx).add_hypothesis(description, priority=priority)
    return {"hypothesis_id": hyp.id, "description": hyp.description, "status": hyp.status.value}


class AddEvidenceInput(BaseModel):
    hypothesis_id: str = Field(description="Hypothesis this evidence bears on.")
    source_tool: str = Field(description="Tool that produced the observation (e.g. 'correlation').")
    observation: str = Field(description="Concise factual observation, e.g. 'corr = 0.82'.")
    kind: str = Field(default="neutral", description="supporting | refuting | neutral")
    confidence: float = Field(default=0.5, description="Confidence in [0,1].")
    query_params: dict[str, Any] = Field(default_factory=dict, description="Tool arguments used.")
    source_refs: list[str] = Field(default_factory=list, description="Doc / case citations.")


@tool(
    name="add_evidence",
    description=(
        "Attach an evidence observation to a hypothesis. A correlation is recorded as "
        "evidence, never as causation — the memory's deterministic evaluation decides status."
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
    hypothesis_id: str = Field(description="Hypothesis to verify.")
    description: str = Field(description="What still needs to be checked.")
    tool: str = Field(description="Tool to run for this check.")
    query: dict[str, Any] = Field(default_factory=dict, description="Arguments for that tool.")


@tool(
    name="add_pending_check",
    description="Register a verification step that must run before the hypothesis can be supported.",
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
    hypothesis_id: str = Field(description="Hypothesis owning the check.")
    check_id: str = Field(description="Pending check id.")
    evidence_ids: list[str] = Field(default_factory=list, description="Evidence produced by the check.")


@tool(
    name="complete_pending_check",
    description="Mark a pending verification check as done and link the evidence it produced.",
    input_model=CompleteCheckInput,
    tags=["memory"],
)
def complete_pending_check(
    ctx: ToolContext, hypothesis_id: str, check_id: str, evidence_ids: list[str] | None = None
) -> dict:
    _memory(ctx).complete_pending_check(hypothesis_id, check_id, evidence_ids or [])
    return {"hypothesis_id": hypothesis_id, "check_id": check_id, "status": "done"}


class EvaluateHypothesisInput(BaseModel):
    hypothesis_id: str = Field(description="Hypothesis to re-evaluate from its evidence.")


@tool(
    name="evaluate_hypothesis",
    description="Deterministically re-evaluate a hypothesis's status from its evidence and pending checks.",
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
    description="Return the current hypotheses and their statuses from working memory.",
    input_model=ListHypothesesInput,
    tags=["memory"],
)
def list_hypotheses(ctx: ToolContext) -> dict:
    return _memory(ctx).snapshot()
