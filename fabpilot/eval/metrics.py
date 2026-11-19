"""评测指标。

共九类，每类都基于*真实*运行产物（tool 历史、共享状态、真值标签、
留出的预测残差）计算，而非基于任何脚本字符串：

  task_success、tool_accuracy、routing_accuracy、diagnosis_top_k、
  evidence_consistency、rag_recall_at_k、ts_metrics（MAE/MSE/F1）、
  efficiency、robustness。

对某场景 / config 不适用的指标（无诊断 agent、未执行检索等）返回 ``None``，
并从聚合中排除。
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import numpy as np

from fabpilot.data import queries
from fabpilot.forecasting.base import ForecastRequest

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _token_overlap(candidate: str, target: str) -> float:
    ct = _tokens(candidate)
    tt = _tokens(target)
    if not tt:
        return 0.0
    return len(ct & tt) / len(tt)


def task_success(state: dict[str, Any]) -> float:
    ok = state.get("status") == "completed" and bool((state.get("final_report") or "").strip())
    return 1.0 if ok else 0.0


def tool_accuracy(state: dict[str, Any]) -> dict[str, Any]:
    entries = state.get("tool_history") or []
    ok = sum(1 for e in entries if e.get("ok"))
    total = len(entries)
    errors = total - ok
    return {
        "tool_accuracy": _safe_div(ok, total),
        "tool_error_count": errors,
        "tool_error_rate": _safe_div(errors, total),
        "tool_calls": total,
    }


def routing_accuracy(state: dict[str, Any], expected_agents: list[str]) -> dict[str, Any]:
    records = state.get("delegation_records") or []
    delegated = [r.get("agent") for r in records if r.get("agent")]
    expected = set(expected_agents)
    if not expected:
        return {"routing_accuracy": None, "delegated_agents": delegated, "expected_agents": sorted(expected)}
    recalled = [a for a in expected if a in delegated]
    return {
        "routing_accuracy": _safe_div(len(recalled), len(expected)),
        "delegated_agents": delegated,
        "expected_agents": sorted(expected),
        "expected_agents_recalled": sorted(recalled),
    }


def diagnosis_top_k(state: dict[str, Any], root_cause: str | None, k: int = 3) -> dict[str, Any]:
    hypotheses = state.get("hypotheses") or []
    if not root_cause or not hypotheses:
        return {"diagnosis_top_k": None, "n_hypotheses": len(hypotheses), "top_status": None}
    ranked = sorted(
        hypotheses,
        key=lambda h: (h.get("status") == "supported", float(h.get("priority", 0.5))),
        reverse=True,
    )
    top = ranked[:k]
    best_overlap = max(_token_overlap(h.get("description", ""), root_cause) for h in top)
    hit = 1.0 if best_overlap >= 0.5 else 0.0
    return {
        "diagnosis_top_k": hit,
        "n_hypotheses": len(hypotheses),
        "top_status": top[0].get("status") if top else None,
        "best_root_cause_overlap": round(best_overlap, 4),
    }


def evidence_consistency(state: dict[str, Any]) -> dict[str, Any]:
    evidence = state.get("evidence") or []
    hypotheses = state.get("hypotheses") or []
    hyp_ids = {h.get("id") for h in hypotheses}
    valid_statuses = {"candidate", "active", "supported", "refuted", "inconclusive", "deprioritized"}
    if not evidence:
        return {"evidence_consistency": None, "n_evidence": 0, "all_statuses_valid": None}
    linked = sum(1 for e in evidence if e.get("hypothesis_id") in hyp_ids)
    statuses_valid = all(h.get("status") in valid_statuses for h in hypotheses)
    resolved = sum(1 for h in hypotheses if h.get("status") in {"supported", "refuted", "inconclusive"})
    return {
        "evidence_consistency": _safe_div(linked, len(evidence)),
        "n_evidence": len(evidence),
        "n_resolved_hypotheses": resolved,
        "all_statuses_valid": statuses_valid,
    }


def _collect_retrieved_docs(state: dict[str, Any]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for h in state.get("tool_history") or []:
        if h.get("name") in ("document_retrieval", "similar_case_retrieval") and h.get("ok"):
            docs.extend(h.get("data", {}).get("results", []))
    docs.extend(state.get("retrieved_docs") or [])
    return docs


def rag_recall_at_k(state: dict[str, Any], relevant_sources: list[str]) -> dict[str, Any]:
    if not relevant_sources:
        return {"rag_recall_at_k": None, "n_retrieved_docs": 0}
    docs = _collect_retrieved_docs(state)
    retrieved_sources = {d.get("source") for d in docs if d.get("source")}
    hits = [s for s in relevant_sources if s in retrieved_sources]
    return {
        "rag_recall_at_k": _safe_div(len(hits), len(relevant_sources)),
        "n_retrieved_docs": len(docs),
        "relevant_sources_hit": hits,
        "relevant_sources_expected": list(relevant_sources),
    }


def _heldout_forecast_metrics(harness: Any, scenario: Any) -> dict[str, Any]:
    """已注册预测模型在场景切分上的留出准确率。

    这是由 harness 计算的模型级指标（在序列前 ``forecast_cut_fraction`` 部分上训练，
    预测接下来 ``horizon`` 个点，并与真实留出值比较）。它不依赖脚本化 LLM。
    """
    gt = scenario.ground_truth
    if not (gt.equipment_id and gt.channel):
        return {"ts_forecast_mae": None, "ts_forecast_mse": None}
    with harness.session_scope() as session:
        df = queries.query_sensor_readings(session, gt.equipment_id)
    if df.empty or gt.channel not in df.columns:
        return {"ts_forecast_mae": None, "ts_forecast_mse": None}

    n = len(df)
    cut = int(n * gt.forecast_cut_fraction)
    horizon = harness.settings.forecast_horizon
    actual = df.iloc[cut : cut + horizon][gt.channel].astype(float).to_numpy()
    if len(actual) < 2:
        return {"ts_forecast_mae": None, "ts_forecast_mse": None}

    train = df.iloc[:cut]
    request = ForecastRequest(
        equipment_id=gt.equipment_id,
        target=gt.channel,
        lookback=min(harness.settings.forecast_lookback, len(train)),
        horizon=len(actual),
        series={c: train[c].astype(float).tolist() for c in df.columns if c != "ts"},
        timestamps=train["ts"].tolist(),
    )
    result = harness.forecast_registry.forecast(harness.settings.default_forecast_model, request)
    pred = np.asarray([p.value for p in result.forecast[: len(actual)]], dtype=float)
    if pred.shape != actual.shape:
        return {"ts_forecast_mae": None, "ts_forecast_mse": None}
    return {
        "ts_forecast_mae": round(float(np.mean(np.abs(pred - actual))), 6),
        "ts_forecast_mse": round(float(np.mean((pred - actual) ** 2)), 6),
    }


def _anomaly_f1(harness: Any, scenario: Any, state: dict[str, Any]) -> dict[str, Any]:
    """针对模拟器注入的真值窗口计算的异常检测 F1。"""
    gt = scenario.ground_truth
    if not (gt.equipment_id and gt.channel and gt.anomaly_scenario):
        return {"ts_anomaly_precision": None, "ts_anomaly_recall": None, "ts_anomaly_f1": None}

    with harness.session_scope() as session:
        labels = queries.query_anomaly_labels(session, gt.equipment_id)
        df = queries.query_sensor_readings(session, gt.equipment_id)

    windows: list[tuple[datetime, datetime]] = []
    for lab in labels:
        if lab.scenario == gt.anomaly_scenario and gt.channel in (lab.affected_channels or []):
            windows.append((lab.start_ts, lab.end_ts))
    if not windows or df.empty:
        return {"ts_anomaly_precision": None, "ts_anomaly_recall": None, "ts_anomaly_f1": None}

    def _in_window(ts: datetime) -> bool:
        return any(s <= ts <= e for s, e in windows)

    flagged: list[datetime] = []
    for r in state.get("anomaly_results") or []:
        if r.get("channel") != gt.channel or r.get("equipment_id") != gt.equipment_id:
            continue
        for a in r.get("anomalies", []):
            if not a.get("is_anomaly"):
                continue
            try:
                flagged.append(datetime.fromisoformat(a["ts"]))
            except (TypeError, ValueError, KeyError):
                continue

    tp = sum(1 for ts in flagged if _in_window(ts))
    actual_count = int(df["ts"].apply(_in_window).sum())
    precision = _safe_div(tp, len(flagged)) if flagged else 0.0
    recall = _safe_div(tp, actual_count)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "ts_anomaly_precision": round(precision, 6),
        "ts_anomaly_recall": round(recall, 6),
        "ts_anomaly_f1": round(f1, 6),
        "n_flagged": len(flagged),
        "n_ground_truth_points": actual_count,
    }


def efficiency(state: dict[str, Any], wall_time_s: float) -> dict[str, Any]:
    decisions = state.get("trace") or []
    return {
        "efficiency_tool_calls": len(state.get("tool_history") or []),
        "efficiency_agent_steps": len(decisions),
        "efficiency_wall_time_s": round(wall_time_s, 4),
    }


def robustness(state: dict[str, Any]) -> dict[str, Any]:
    clean = state.get("status") == "completed" and not (state.get("errors") or [])
    return {"robustness": 1.0 if clean else 0.0, "status": state.get("status")}


def compute_metrics(
    state: dict[str, Any],
    scenario: Any,
    config: Any,
    harness: Any,
    wall_time_s: float,
) -> dict[str, Any]:
    """为一次 (scenario, config) 运行计算完整指标集合。"""
    m: dict[str, Any] = {}
    m["task_success"] = task_success(state)
    m.update(tool_accuracy(state))

    if config.use_supervisor:
        m.update(routing_accuracy(state, scenario.expected_agents))
    else:
        m["routing_accuracy"] = None

    if config.use_diagnosis:
        m.update(diagnosis_top_k(state, scenario.ground_truth.root_cause))
        m.update(evidence_consistency(state))
    else:
        m["diagnosis_top_k"] = None
        m["evidence_consistency"] = None

    if config.use_rag:
        m.update(rag_recall_at_k(state, scenario.ground_truth.relevant_sources))
    else:
        m["rag_recall_at_k"] = None

    m.update(_heldout_forecast_metrics(harness, scenario))
    m.update(_anomaly_f1(harness, scenario, state))
    m.update(efficiency(state, wall_time_s))
    m.update(robustness(state))
    return m


# 在跨场景聚合时应取平均的指标键。
NUMERIC_METRICS = [
    "task_success",
    "tool_accuracy",
    "tool_error_rate",
    "routing_accuracy",
    "diagnosis_top_k",
    "evidence_consistency",
    "rag_recall_at_k",
    "ts_forecast_mae",
    "ts_forecast_mse",
    "ts_anomaly_f1",
    "efficiency_tool_calls",
    "efficiency_agent_steps",
    "efficiency_wall_time_s",
    "robustness",
]
