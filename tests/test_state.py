"""Shared LangGraph state tests."""
from __future__ import annotations

from fabpilot.state import LIST_FIELDS, diff_state, initial_state


def test_initial_state_has_expected_keys():
    s = initial_state("s1", "t1", "do something")
    assert s["session_id"] == "s1"
    assert s["thread_id"] == "t1"
    assert s["status"] == "pending"
    assert s["messages"] == []
    assert s["step_counts"]["supervisor"] == 0


def test_diff_state_returns_only_added_list_items():
    before = initial_state("s1", "t1", "task")
    after = initial_state("s1", "t1", "task")
    after["tool_history"] = [{"name": "sensor_data", "ok": True}]
    after["final_report"] = "done"
    after["status"] = "completed"

    delta = diff_state(before, after)
    assert delta["tool_history"] == [{"name": "sensor_data", "ok": True}]
    assert delta["final_report"] == "done"
    assert delta["status"] == "completed"


def test_diff_state_empty_when_unchanged():
    before = initial_state("s1", "t1", "task")
    after = initial_state("s1", "t1", "task")
    assert diff_state(before, after) == {}


def test_list_fields_declared():
    assert "tool_history" in LIST_FIELDS
    assert "messages" in LIST_FIELDS
    assert "final_report" not in LIST_FIELDS
