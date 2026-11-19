"""Checkpointing and persistent session store.

Two distinct concerns:
  * LangGraph checkpointer — saves the *graph execution* state so a run can be
    resumed / inspected by ``thread_id``.
  * SessionStore — persists session summaries + final reports for the API's
    "list history / fetch report" endpoints.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fabpilot.config import Settings, get_settings
from fabpilot.utils import get_logger, safe_json

logger = get_logger(__name__)


def build_checkpointer(settings: Settings | None = None):
    """Return a LangGraph checkpointer (in-memory by default, SQLite optional)."""
    from langgraph.checkpoint.memory import MemorySaver

    settings = settings or get_settings()
    if settings.checkpoint_sqlite:
        from langgraph.checkpoint.sqlite import SqliteSaver

        return SqliteSaver.from_conn_string(settings.checkpoint_sqlite)
    return MemorySaver()


class SessionStore:
    """JSON-file-backed session store used by the API layer."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def save(self, record: dict[str, Any]) -> None:
        data = self._load()
        thread_id = record.get("thread_id") or record.get("session_id")
        data[thread_id] = safe_json(record)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, thread_id: str) -> dict[str, Any] | None:
        return self._load().get(thread_id)

    def list(self) -> list[dict[str, Any]]:
        return list(self._load().values())


def build_session_store(settings: Settings | None = None) -> SessionStore:
    settings = settings or get_settings()
    return SessionStore(settings.data_path / "sessions.json")
