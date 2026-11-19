"""Hypothesis-Evidence Memory state machine tests.

These verify the *real* deterministic status transitions, not prompt behaviour.
"""
from __future__ import annotations

from fabpilot.memory.hypothesis import (
    EvidenceKind,
    HypothesisEvidenceMemory,
    HypothesisStatus,
)


def test_hypothesis_starts_candidate():
    mem = HypothesisEvidenceMemory()
    h = mem.add_hypothesis("MFC drift", priority=0.8)
    assert h.status == HypothesisStatus.CANDIDATE
    assert not h.is_resolved


def test_supporting_evidence_with_open_checks_keeps_active():
    mem = HypothesisEvidenceMemory()
    h = mem.add_hypothesis("MFC drift")
    mem.add_pending_check(h.id, "cross-check pressure", "correlation", {"a": 1})
    ev = mem.add_evidence(
        h.id, "lag_analysis", {"a": 1}, "flow leads pressure", kind=EvidenceKind.SUPPORTING, confidence=0.9
    )
    assert ev.id in h.supporting_evidence_ids
    assert mem.evaluate_hypothesis(h.id) == HypothesisStatus.ACTIVE


def test_supporting_evidence_without_open_checks_is_supported():
    mem = HypothesisEvidenceMemory()
    h = mem.add_hypothesis("MFC drift")
    mem.add_evidence(h.id, "lag_analysis", {}, "flow leads pressure", kind=EvidenceKind.SUPPORTING, confidence=0.9)
    assert mem.evaluate_hypothesis(h.id) == HypothesisStatus.SUPPORTED
    assert mem.best_supported() is h


def test_refuting_evidence_refutes():
    mem = HypothesisEvidenceMemory()
    h = mem.add_hypothesis("MFC drift")
    mem.add_evidence(h.id, "sensor_data", {}, "no drift", kind=EvidenceKind.REFUTING, confidence=0.9)
    assert mem.evaluate_hypothesis(h.id) == HypothesisStatus.REFUTED


def test_no_evidence_is_inconclusive():
    mem = HypothesisEvidenceMemory()
    h = mem.add_hypothesis("MFC drift")
    assert mem.evaluate_hypothesis(h.id) == HypothesisStatus.INCONCLUSIVE


def test_low_confidence_evidence_does_not_resolve():
    mem = HypothesisEvidenceMemory()
    h = mem.add_hypothesis("MFC drift")
    mem.add_evidence(h.id, "sensor_data", {}, "weak", kind=EvidenceKind.SUPPORTING, confidence=0.2)
    assert mem.evaluate_hypothesis(h.id, support_threshold=0.5) == HypothesisStatus.INCONCLUSIVE


def test_termination_decision_stops_when_supported_and_clean():
    mem = HypothesisEvidenceMemory()
    h = mem.add_hypothesis("MFC drift")
    mem.add_evidence(h.id, "lag_analysis", {}, "leads", kind=EvidenceKind.SUPPORTING, confidence=0.9)
    mem.evaluate_all()
    action, _reason = mem.termination_decision(remaining_budget=5)
    assert action == "stop"


def test_termination_decision_stops_when_budget_exhausted():
    mem = HypothesisEvidenceMemory()
    mem.add_hypothesis("MFC drift")
    action, _reason = mem.termination_decision(remaining_budget=0)
    assert action == "stop"


def test_snapshot_roundtrip():
    mem = HypothesisEvidenceMemory()
    h = mem.add_hypothesis("MFC drift")
    mem.add_evidence(h.id, "lag_analysis", {}, "leads", kind=EvidenceKind.SUPPORTING, confidence=0.9)
    clone = HypothesisEvidenceMemory.from_snapshot(mem.snapshot())
    assert set(clone.hypotheses) == set(mem.hypotheses)
    assert clone.hypotheses[h.id].status == mem.hypotheses[h.id].status


def test_correlation_alone_does_not_support_causal_claim():
    """A correlation can be *supporting evidence* for a hypothesis, but it never
    resolves the hypothesis by itself while a causal check is still pending."""
    mem = HypothesisEvidenceMemory()
    h = mem.add_hypothesis("pressure drop causes flow drift")
    mem.add_evidence(h.id, "correlation", {"a": "flow", "b": "pressure"}, "r=0.9", kind=EvidenceKind.SUPPORTING, confidence=0.9)
    # A causal confirmation is still an open, pending check.
    mem.add_pending_check(h.id, "verify lead/lag ordering", "lag_analysis", {"a": "flow", "b": "pressure"})
    # Correlation alone → still ACTIVE (not SUPPORTED), because causation is unproven.
    assert mem.evaluate_hypothesis(h.id) == HypothesisStatus.ACTIVE
