"""Renders evaluation output to JSON and Markdown.

The runner returns raw per-run metrics; this module aggregates them per config
and produces a human-readable comparison table. Metrics that do not apply to a
scenario / config are ``None`` and excluded from aggregates.
"""
from __future__ import annotations

import json
from typing import Any

from fabpilot.eval.metrics import NUMERIC_METRICS

_METRIC_LABELS = {
    "task_success": "Task success",
    "tool_accuracy": "Tool accuracy",
    "tool_error_rate": "Tool error rate",
    "routing_accuracy": "Routing accuracy",
    "diagnosis_top_k": "Diagnosis top-k",
    "evidence_consistency": "Evidence consistency",
    "rag_recall_at_k": "RAG recall@k",
    "ts_forecast_mae": "Forecast MAE",
    "ts_forecast_mse": "Forecast MSE",
    "ts_anomaly_f1": "Anomaly F1",
    "efficiency_tool_calls": "Tool calls (efficiency)",
    "efficiency_agent_steps": "Agent steps (efficiency)",
    "efficiency_wall_time_s": "Wall time (s)",
    "robustness": "Robustness",
}


def _avg(values: list[float]) -> float | None:
    nums = [v for v in values if v is not None]
    return round(sum(nums) / len(nums), 4) if nums else None


def aggregate(payload: dict[str, Any]) -> dict[str, Any]:
    configs: list[str] = payload["configs"]
    results: list[dict[str, Any]] = payload["results"]

    per_config: dict[str, dict[str, float | None]] = {}
    for config in configs:
        cfg_results = [r for r in results if r["config"] == config]
        per_config[config] = {
            metric: _avg([r["metrics"].get(metric) for r in cfg_results])
            for metric in NUMERIC_METRICS
        }

    # Configs that have a value for every scenario (task success / robustness are
    # always defined; here we just keep everything).
    return {
        "configs": configs,
        "scenarios": payload["scenarios"],
        "per_config": per_config,
        "results": results,
    }


def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def render_markdown(payload: dict[str, Any]) -> str:
    agg = aggregate(payload)
    configs = agg["configs"]
    per_config = agg["per_config"]

    lines: list[str] = []
    lines.append("# FabPilot Evaluation Report")
    lines.append("")
    lines.append(
        "Metrics are computed from real tool outputs over a seeded database, "
        "forecast registry and knowledge base. A `—` means the metric does not "
        "apply to that config (no supervisor, no diagnosis agent, or no RAG)."
    )
    lines.append("")

    lines.append("## Aggregate metrics by config")
    lines.append("")
    header = "| Metric | " + " | ".join(configs) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(configs) + 1))
    for metric in NUMERIC_METRICS:
        label = _METRIC_LABELS.get(metric, metric)
        cells = [_fmt(per_config[c].get(metric)) for c in configs]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Per-run detail")
    lines.append("")
    lines.append("| Config | Scenario | Status | Tool calls | Delegated | Hypotheses | Final report |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in agg["results"]:
        s = r["summary"]
        report = (s.get("final_report") or "").replace("\n", " ").replace("|", "\\|")
        lines.append(
            f"| {r['config']} | {r['scenario']} | {s.get('status')} | {s.get('n_tool_calls')} "
            f"| {','.join(s.get('delegated_agents') or []) or '—'} | {s.get('n_hypotheses')} | {report} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_json(payload: dict[str, Any]) -> str:
    return json.dumps(aggregate(payload), ensure_ascii=False, indent=2, default=str)


def write_report(payload: dict[str, Any], markdown_path: str, json_path: str | None = None) -> None:
    with open(markdown_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(payload))
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            fh.write(render_json(payload))
