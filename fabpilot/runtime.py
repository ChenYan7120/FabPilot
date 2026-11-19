"""运行时装配：把配置、LLM、工具、预测、RAG、记忆 以及
LangGraph 图组装成一个 CLI / API / 评估层共用的单一对象。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fabpilot.config import Settings, get_settings
from fabpilot.forecasting.registry import build_forecast_registry
from fabpilot.llm.provider import build_llm
from fabpilot.memory.checkpoint import build_checkpointer
from fabpilot.memory.hypothesis import HypothesisEvidenceMemory
from fabpilot.rag.pipeline import build_knowledge_base
from fabpilot.state import initial_state
from fabpilot.tools import build_tool_registry
from fabpilot.tools.registry import ToolContext
from fabpilot.utils import new_id
from fabpilot.workflows import build_graph


@dataclass
class Runtime:
    settings: Settings
    llm: Any
    tool_registry: Any
    ctx: ToolContext
    forecast_registry: Any
    knowledge_base: Any
    memory: HypothesisEvidenceMemory
    checkpointer: Any
    graph: Any


def build_runtime(settings: Settings | None = None, **overrides: Any) -> Runtime:
    settings = settings or get_settings()

    llm = overrides.get("llm") or build_llm(settings)
    tool_registry = overrides.get("tool_registry") or build_tool_registry()
    forecast_registry = overrides.get("forecast_registry") or build_forecast_registry(settings)
    knowledge_base = overrides.get("knowledge_base") or build_knowledge_base(settings)
    memory = overrides.get("memory") or HypothesisEvidenceMemory()
    checkpointer = overrides.get("checkpointer") or build_checkpointer(settings)

    from fabpilot.data.database import build_session_scope

    session_scope = overrides.get("session_scope") or build_session_scope(settings.database_url)

    ctx = ToolContext(
        session_scope=session_scope,
        forecast_registry=forecast_registry,
        knowledge_base=knowledge_base,
        memory=memory,
        settings=settings,
        chart_dir=str(settings.data_path / "charts"),
        report_dir=str(settings.data_path / "reports"),
    )

    graph = build_graph(llm, tool_registry, ctx, settings, checkpointer=checkpointer)
    return Runtime(
        settings=settings,
        llm=llm,
        tool_registry=tool_registry,
        ctx=ctx,
        forecast_registry=forecast_registry,
        knowledge_base=knowledge_base,
        memory=memory,
        checkpointer=checkpointer,
        graph=graph,
    )


def run_task(runtime: Runtime, user_task: str, session_id: str | None = None, thread_id: str | None = None) -> dict[str, Any]:
    session_id = session_id or new_id("sess")
    thread_id = thread_id or session_id
    state = initial_state(session_id, thread_id, user_task)
    config = {"configurable": {"thread_id": thread_id}}
    return runtime.graph.invoke(state, config=config)
