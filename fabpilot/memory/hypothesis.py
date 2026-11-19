"""Hypothesis-Evidence Memory.

The Diagnosis Agent's core state machine. This is real code — status transitions,
evidence linking and termination decisions are deterministic, not prompt claims:

  Observation -> Hypothesis -> Additional Analysis -> Re-evaluation

A statistical correlation is recorded as *evidence*, never as confirmed causation:
the kind/status transitions below only treat tool results as supporting or
refuting material for an explicitly stated hypothesis.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from fabpilot.utils import new_id


def _now() -> datetime:
    return datetime.now(timezone.utc)


class HypothesisStatus(str, Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"
    DEPRIORITIZED = "deprioritized"


class EvidenceKind(str, Enum):
    SUPPORTING = "supporting"
    REFUTING = "refuting"
    NEUTRAL = "neutral"


class PendingCheck(BaseModel):
    id: str
    description: str
    tool: str
    query: dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"  # pending | done | skipped
    evidence_ids: list[str] = Field(default_factory=list)


class Hypothesis(BaseModel):
    id: str
    description: str
    status: HypothesisStatus = HypothesisStatus.CANDIDATE
    priority: float = 0.5
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    refuting_evidence_ids: list[str] = Field(default_factory=list)
    pending_checks: list[PendingCheck] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @property
    def is_resolved(self) -> bool:
        return self.status in {HypothesisStatus.SUPPORTED, HypothesisStatus.REFUTED, HypothesisStatus.INCONCLUSIVE}


class Evidence(BaseModel):
    id: str
    hypothesis_id: str
    source_tool: str
    query_params: dict[str, Any] = Field(default_factory=dict)
    observation: str
    kind: EvidenceKind = EvidenceKind.NEUTRAL
    time_range: tuple[datetime, datetime] | None = None
    source_refs: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    created_at: datetime = Field(default_factory=_now)


class HypothesisEvidenceMemory:
    def __init__(self) -> None:
        self.hypotheses: dict[str, Hypothesis] = {}
        self.evidence: dict[str, Evidence] = {}

    # ---- hypothesis lifecycle ----
    def add_hypothesis(self, description: str, priority: float = 0.5) -> Hypothesis:
        hyp = Hypothesis(id=new_id("hyp"), description=description, priority=priority)
        self.hypotheses[hyp.id] = hyp
        return hyp

    def get_hypothesis(self, hyp_id: str) -> Hypothesis:
        return self.hypotheses[hyp_id]

    def set_status(self, hyp_id: str, status: HypothesisStatus) -> None:
        self.hypotheses[hyp_id].status = status
        self.hypotheses[hyp_id].updated_at = _now()

    def set_priority(self, hyp_id: str, priority: float) -> None:
        self.hypotheses[hyp_id].priority = priority
        self.hypotheses[hyp_id].updated_at = _now()

    def add_pending_check(self, hyp_id: str, description: str, tool: str, query: dict[str, Any]) -> PendingCheck:
        check = PendingCheck(id=new_id("chk"), description=description, tool=tool, query=query)
        self.hypotheses[hyp_id].pending_checks.append(check)
        self.hypotheses[hyp_id].updated_at = _now()
        return check

    def complete_pending_check(self, hyp_id: str, check_id: str, evidence_ids: list[str]) -> None:
        for check in self.hypotheses[hyp_id].pending_checks:
            if check.id == check_id:
                check.status = "done"
                check.evidence_ids = evidence_ids
                self.hypotheses[hyp_id].updated_at = _now()
                return
        raise KeyError(f"unknown pending check {check_id}")

    def skip_pending_check(self, hyp_id: str, check_id: str) -> None:
        for check in self.hypotheses[hyp_id].pending_checks:
            if check.id == check_id:
                check.status = "skipped"
                self.hypotheses[hyp_id].updated_at = _now()
                return
        raise KeyError(f"unknown pending check {check_id}")

    # ---- evidence lifecycle ----
    def add_evidence(
        self,
        hypothesis_id: str,
        source_tool: str,
        query_params: dict[str, Any],
        observation: str,
        kind: EvidenceKind = EvidenceKind.NEUTRAL,
        time_range: tuple[datetime, datetime] | None = None,
        source_refs: list[str] | None = None,
        confidence: float = 0.5,
    ) -> Evidence:
        ev = Evidence(
            id=new_id("evd"),
            hypothesis_id=hypothesis_id,
            source_tool=source_tool,
            query_params=query_params,
            observation=observation,
            kind=kind,
            time_range=time_range,
            source_refs=source_refs or [],
            confidence=confidence,
        )
        self.evidence[ev.id] = ev
        self._link_evidence(hypothesis_id, ev.id, kind)
        return ev

    def _link_evidence(self, hyp_id: str, evidence_id: str, kind: EvidenceKind) -> None:
        hyp = self.hypotheses[hyp_id]
        if kind == EvidenceKind.SUPPORTING:
            if evidence_id not in hyp.supporting_evidence_ids:
                hyp.supporting_evidence_ids.append(evidence_id)
        elif kind == EvidenceKind.REFUTING:
            if evidence_id not in hyp.refuting_evidence_ids:
                hyp.refuting_evidence_ids.append(evidence_id)
        hyp.updated_at = _now()

    def set_evidence_kind(self, evidence_id: str, kind: EvidenceKind) -> None:
        ev = self.evidence[evidence_id]
        ev.kind = kind
        hyp = self.hypotheses[ev.hypothesis_id]
        # re-link cleanly
        hyp.supporting_evidence_ids = [e for e in hyp.supporting_evidence_ids if e != evidence_id]
        hyp.refuting_evidence_ids = [e for e in hyp.refuting_evidence_ids if e != evidence_id]
        self._link_evidence(ev.hypothesis_id, evidence_id, kind)

    # ---- deterministic re-evaluation ----
    def evaluate_hypothesis(self, hyp_id: str, support_threshold: float = 0.5) -> HypothesisStatus:
        """Re-evaluate one hypothesis from its evidence and pending checks."""
        hyp = self.hypotheses[hyp_id]
        refuting = [eid for eid in hyp.refuting_evidence_ids if self.evidence[eid].confidence >= support_threshold]
        supporting = [eid for eid in hyp.supporting_evidence_ids if self.evidence[eid].confidence >= support_threshold]
        open_checks = [c for c in hyp.pending_checks if c.status == "pending"]

        if refuting:
            new_status = HypothesisStatus.REFUTED
        elif supporting and not open_checks:
            new_status = HypothesisStatus.SUPPORTED
        elif supporting and open_checks:
            new_status = HypothesisStatus.ACTIVE
        elif open_checks:
            new_status = HypothesisStatus.ACTIVE
        else:
            new_status = HypothesisStatus.INCONCLUSIVE

        self.set_status(hyp_id, new_status)
        return new_status

    def evaluate_all(self, support_threshold: float = 0.5) -> dict[str, HypothesisStatus]:
        return {hid: self.evaluate_hypothesis(hid, support_threshold) for hid in self.hypotheses}

    def termination_decision(self, remaining_budget: int) -> tuple[str, str]:
        """Decide whether to keep investigating or conclude.

        Returns (action, reason). ``action`` is one of "continue" | "stop".
        """
        if remaining_budget <= 0:
            return "stop", "tool/iteration budget exhausted"
        self.evaluate_all()
        if not self.hypotheses:
            return "continue", "no hypotheses generated yet"

        any_supported = any(h.status == HypothesisStatus.SUPPORTED for h in self.hypotheses.values())
        any_active = any(h.status == HypothesisStatus.ACTIVE for h in self.hypotheses.values())
        any_open = any(c.status == "pending" for h in self.hypotheses.values() for c in h.pending_checks)

        if any_supported and not any_open:
            return "stop", "at least one hypothesis supported and no pending checks remain"
        if not any_active and not any_open:
            # every hypothesis resolved (refuted / inconclusive) -> nothing more to verify
            return "stop", "all hypotheses resolved without remaining evidence to collect"
        return "continue", "active hypotheses still have pending checks"

    def best_supported(self) -> Hypothesis | None:
        supported = [h for h in self.hypotheses.values() if h.status == HypothesisStatus.SUPPORTED]
        if not supported:
            return None
        return max(supported, key=lambda h: h.priority)

    # ---- serialisation ----
    def snapshot(self) -> dict[str, Any]:
        return {
            "hypotheses": [h.model_dump(mode="json") for h in self.hypotheses.values()],
            "evidence": [e.model_dump(mode="json") for e in self.evidence.values()],
        }

    @staticmethod
    def from_snapshot(data: dict[str, Any]) -> "HypothesisEvidenceMemory":
        mem = HypothesisEvidenceMemory()
        for h in data.get("hypotheses", []):
            mem.hypotheses[h["id"]] = Hypothesis.model_validate(h)
        for e in data.get("evidence", []):
            mem.evidence[e["id"]] = Evidence.model_validate(e)
        return mem
