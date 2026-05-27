"""Unit tests for Trajectory / Step pydantic schemas."""
from __future__ import annotations
import pytest
from src.labeler.trajectory_schema import Trajectory, Step


def _minimal_raw() -> dict:
    return {
        "task_id": "django__django-12345",
        "task_type": "swe-bench-lite",
        "trajectory": [
            {
                "step": 0,
                "role": "assistant",
                "thought": "let me read the file",
                "tool": "read_file",
                "tool_args": {"path": "foo.py"},
                "tool_result": "def f(): pass",
            }
        ],
        "outcome": 1,
        "policy_model": "claude-sonnet-4-5",
        "timestamp": "2026-05-27T10:00:00Z",
    }


def test_minimal_trajectory_parses() -> None:
    t = Trajectory(**_minimal_raw())
    assert t.task_id == "django__django-12345"
    assert t.task_type == "swe-bench-lite"
    assert len(t.trajectory) == 1
    assert t.trajectory[0].tool == "read_file"
    assert t.trajectory[0].mc_label is None


def test_outcome_must_be_0_or_1() -> None:
    raw = _minimal_raw()
    raw["outcome"] = 2
    with pytest.raises(ValueError):
        Trajectory(**raw)


def test_step_mc_label_bounds() -> None:
    raw = _minimal_raw()
    raw["trajectory"][0]["mc_label"] = 1.5
    with pytest.raises(ValueError):
        Trajectory(**raw)


def test_step_mc_label_negative_rejected() -> None:
    raw = _minimal_raw()
    raw["trajectory"][0]["mc_label"] = -0.1
    with pytest.raises(ValueError):
        Trajectory(**raw)


def test_pure_thought_step_allowed() -> None:
    """A step with tool=None should be valid (pure thought)."""
    raw = _minimal_raw()
    raw["trajectory"].append({
        "step": 1,
        "role": "assistant",
        "thought": "now I will think",
        "tool": None,
        "tool_args": {},
        "tool_result": "",
    })
    t = Trajectory(**raw)
    assert t.trajectory[1].tool is None


def test_invalid_task_type_rejected() -> None:
    raw = _minimal_raw()
    raw["task_type"] = "made-up-set"
    with pytest.raises(ValueError):
        Trajectory(**raw)
