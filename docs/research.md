# Research: Samsung Semiconductor Defect Agent

This note records the design inspiration and where FabPilot diverges. It is based
on **public descriptions** of Samsung's semiconductor defect-classification work;
FabPilot contains no Samsung source code and reproduces no proprietary model.

## What we learned from the public accounts

Samsung's semiconductor defect agent (reported in connection with high-bandwidth
memory / chip fabrication) is described as combining:

1. **Per-image and per-wafer segmentation + defect classification** of defect
   micrographs, used to prioritise which wafers need review.
2. **Hypothesis-driven workflow** — an operator states a hypothesis, and the
   system gathers evidence to confirm or refute it, rather than blindly ranking
   every defect.
3. **A feedback loop** from found defects back into the classification / review
   pipeline, so the system improves as labelled examples accumulate.

The recurring theme is *hypothesis-first triage*: reduce the search space by
naming a hypothesis, then let evidence decide.

## What FabPilot adopts (conceptually)

FabPilot ports the **hypothesis-driven reasoning pattern** — not the vision model —
to industrial time-series diagnosis:

- A user task names a symptom (e.g. "gas_flow drift on ETCH-01").
- The **Diagnosis Agent** generates explicit hypotheses and collects evidence
  through tools (lag analysis, correlation, maintenance history, similar cases).
- A deterministic **Hypothesis-Evidence Memory** transitions each hypothesis
  through `candidate → active → supported / refuted / inconclusive`.
- Evidence is *linked* to hypotheses and *re-evaluated* — the same
  confirm/refute loop, applied to telemetry instead of micrographs.

## Where FabPilot deliberately diverges

| Samsung (public descriptions) | FabPilot |
|---|---|
| Vision: defect micrograph classification | Time-series: sensor telemetry, alarms, maintenance |
| Operator-authored hypothesis entry | Supervisor + Diagnosis Agent co-generate hypotheses |
| Defect-priority triage over wafers | Root-cause ranking over hypotheses |
| Proprietary vision models | Real statistical baseline + interface-only PatchTST/iTransformer adapters |
| (Not publicly detailed) | Full open, auditable ReAct trail in shared state |

## Why "hypothesis-first" matters for diagnosis

A naive agent that dumps every statistic on a symptom can drown in correlated
channels. Hypothesis-first triage forces a candidate causal story to be named
*first*, then verified with targeted evidence. That is why FabPilot treats a
correlation as **evidence for** a hypothesis, never as causation by itself — the
same discipline that keeps Samsung's reported system from over-trusting a
classifier's confidence.

## References

- Public reporting on Samsung's semiconductor AI agent (HBM defect
  classification and priority review) — see tech-press coverage of the
  Samsung/ASML/Applied Materials fab-automation announcements.
- [ragent](https://github.com/infiniflow/ragent) — source of the hybrid
  dense+sparse RRF retrieval design pattern (reimplemented in Python in
  `fabpilot/rag/`).
