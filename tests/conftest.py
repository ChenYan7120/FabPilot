"""共享的 pytest fixture。

需要数据库的测试使用临时 SQLite 文件和 ``hash`` embedder /
内存向量存储，因此无需任何外部服务（Postgres、LLM、embeddings API）。
"""
from __future__ import annotations

import pytest

from fabpilot.config import Settings
from fabpilot.data.database import build_session_scope
from fabpilot.data.seed import seed_into
from fabpilot.forecasting.registry import build_forecast_registry
from fabpilot.rag.pipeline import build_knowledge_base


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        llm_provider="fake",
        embedding_provider="hash",
        vector_store="memory",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        knowledge_base_dir=str(tmp_path / "kb"),
        data_dir=str(tmp_path / "data"),
    )


@pytest.fixture
def session_scope(settings):
    return build_session_scope(settings.database_url, create_tables=True)


@pytest.fixture
def seeded_scope(session_scope):
    seed_into(session_scope, seed=42, n_equipment=3, days=2)
    return session_scope


@pytest.fixture
def forecast_registry(settings):
    return build_forecast_registry(settings)


@pytest.fixture
def knowledge_base(settings, tmp_path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir(exist_ok=True)
    (kb_dir / "sop").mkdir(exist_ok=True)
    (kb_dir / "fault_cases").mkdir(exist_ok=True)
    (kb_dir / "sop" / "mfc.md").write_text(
        "# MFC drift\nGas flow drift is caused by MFC ageing and shows a monotonic decrease.",
        encoding="utf-8",
    )
    (kb_dir / "fault_cases" / "case.md").write_text(
        "# Case\nETCH-01 gas flow drifted below setpoint; pressure lagged behind.",
        encoding="utf-8",
    )
    return build_knowledge_base(settings)
