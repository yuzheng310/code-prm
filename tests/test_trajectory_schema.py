"""Unit tests for Trajectory / Step pydantic schemas."""
from __future__ import annotations
import pytest
from src.labeler.trajectory_schema import (
    Step,
    TestResult,
    TokenUsage,
    Trajectory,
)


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
    assert t.trajectory[0].step_label is None


def test_outcome_must_be_0_or_1() -> None:
    raw = _minimal_raw()
    raw["outcome"] = 2
    with pytest.raises(ValueError):
        Trajectory(**raw)


def test_step_label_bounds_upper() -> None:
    raw = _minimal_raw()
    raw["trajectory"][0]["step_label"] = 1.5
    with pytest.raises(ValueError):
        Trajectory(**raw)


def test_step_label_negative_rejected() -> None:
    raw = _minimal_raw()
    raw["trajectory"][0]["step_label"] = -0.1
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


# --- New schema fields (Phase 1 deviation: replay + cost + labeling metadata) ---


def test_default_rollout_id_is_zero() -> None:
    t = Trajectory(**_minimal_raw())
    assert t.rollout_id == 0


def test_rollout_id_must_be_non_negative() -> None:
    raw = _minimal_raw()
    raw["rollout_id"] = -1
    with pytest.raises(ValueError):
        Trajectory(**raw)


def test_optional_replay_fields_default_to_none() -> None:
    t = Trajectory(**_minimal_raw())
    assert t.run_id is None
    assert t.repo is None
    assert t.base_commit is None
    assert t.final_diff is None
    assert t.test_result is None
    assert t.token_usage is None
    assert t.label_method is None


def test_token_usage_nested_parse() -> None:
    raw = _minimal_raw()
    raw["token_usage"] = {
        "input_tokens": 25000,
        "output_tokens": 5000,
        "cache_read_tokens": 1000,
        "cache_creation_tokens": 500,
        "cost_usd": 0.15,
    }
    t = Trajectory(**raw)
    assert isinstance(t.token_usage, TokenUsage)
    assert t.token_usage.cost_usd == 0.15
    assert t.token_usage.input_tokens == 25000


def test_test_result_nested_parse() -> None:
    raw = _minimal_raw()
    raw["test_result"] = {
        "passed": True,
        "command": "pytest tests/",
        "exit_code": 0,
        "stdout_tail": "5 passed",
        "stderr_tail": "",
        "duration_sec": 12.5,
    }
    t = Trajectory(**raw)
    assert isinstance(t.test_result, TestResult)
    assert t.test_result.passed is True
    assert t.test_result.duration_sec == 12.5


def test_label_method_literal_validates() -> None:
    raw = _minimal_raw()
    raw["label_method"] = "llm_judge"
    t = Trajectory(**raw)
    assert t.label_method == "llm_judge"

    raw["label_method"] = "made_up_method"
    with pytest.raises(ValueError):
        Trajectory(**raw)


def test_step_object_constructs_directly() -> None:
    """Sanity check on the Step class (not just nested under Trajectory)."""
    s = Step(step=3, tool="bash", tool_args={"cmd": "ls"}, tool_result="a b c")
    assert s.step == 3
    assert s.role == "assistant"
    assert s.step_label is None


# --- task_prompt / task_metadata fields ---


def test_task_prompt_field_present_and_defaults_none() -> None:
    t = Trajectory(**_minimal_raw())
    assert t.task_prompt is None
    assert t.task_metadata == {}


def test_task_prompt_round_trips() -> None:
    raw = _minimal_raw()
    raw["task_prompt"] = "Fix the pagination bug."
    raw["task_metadata"] = {"difficulty": "medium"}
    t = Trajectory(**raw)
    assert t.task_prompt == "Fix the pagination bug."
    assert t.task_metadata["difficulty"] == "medium"


# --- outcome vs test_result consistency (model_validator) ---


def test_outcome_matches_test_result_passed() -> None:
    raw = _minimal_raw()
    raw["outcome"] = 1
    raw["test_result"] = {"passed": True, "command": "pytest", "exit_code": 0}
    t = Trajectory(**raw)
    assert t.outcome == 1
    assert t.test_result.passed is True


def test_outcome_disagrees_with_test_result_raises() -> None:
    raw = _minimal_raw()
    raw["outcome"] = 1
    raw["test_result"] = {"passed": False, "command": "pytest", "exit_code": 1}
    with pytest.raises(ValueError, match="disagrees with"):
        Trajectory(**raw)


def test_test_result_absent_allows_any_outcome() -> None:
    """The consistency check only fires when test_result is present."""
    raw = _minimal_raw()
    raw["outcome"] = 0
    # test_result not set
    t = Trajectory(**raw)
    assert t.outcome == 0
    assert t.test_result is None
