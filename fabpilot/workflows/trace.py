"""已完成 FabPilot 运行的审计轨迹格式化。"""
from __future__ import annotations

from typing import Any


def render_trace(state: dict[str, Any]) -> str:
    """将决策 / 委托 / 工具调用轨迹渲染为 Markdown。"""
    lines: list[str] = ["## Execution trace", ""]

    for rec in state.get("delegation_records", []):
        eq = rec.get("equipment_id") or "-"
        lines.append(f"- delegate → **{rec.get('agent')}** (equipment {eq}): {rec.get('task')}")

    lines.append("")
    lines.append("### Agent decisions & tool calls")
    for dec in state.get("trace", []):
        agent = dec.get("agent", "?")
        decision = (dec.get("decision") or "").strip()
        lines.append(f"- **{agent}** step {dec.get('step')}: {decision}" if decision else f"- **{agent}** step {dec.get('step')}")
        for tc in dec.get("tool_calls", []):
            lines.append(f"  - `{tc.get('name')}` {tc.get('arguments')}")

    lines.append("")
    lines.append("### Tool history")
    for h in state.get("tool_history", []):
        status = "ok" if h.get("ok") else f"error: {h.get('error')}"
        lines.append(f"- `{h.get('name')}` ({h.get('agent')}) → {status}")

    return "\n".join(lines)


def render_final_report(state: dict[str, Any]) -> str:
    report = state.get("final_report") or "No final report produced."
    trace = render_trace(state)
    return f"{report}\n\n{trace}\n"
