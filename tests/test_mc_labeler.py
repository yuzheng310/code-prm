"""Unit tests for the MC labeler (pure functions that don't hit the API)."""
from __future__ import annotations
import json
from pathlib import Path

from src.labeler.mc_labeler import (
    _build_continuation_prompt,
    _parses_as_successful,
    label_trajectory_simplified,
)
from src.labeler.trajectory_schema import Trajectory

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_trajectory.json"


def _load_fixture() -> Trajectory:
    return Trajectory(**json.loads(FIXTURE.read_text()))


def test_outcome_zero_gets_all_zero_labels() -> None:
    t = _load_fixture()
    t.outcome = 0
    labeled = label_trajectory_simplified(t)
    assert all(s.mc_label == 0.0 for s in labeled.trajectory)


def test_outcome_one_left_unchanged_by_simplified_labeler() -> None:
    """outcome=1 path should NOT get labels from the simplified labeler.
    Real labels must come from mc_rollout_for_step()."""
    t = _load_fixture()
    assert t.outcome == 1
    labeled = label_trajectory_simplified(t)
    assert all(s.mc_label is None for s in labeled.trajectory)


def test_continuation_prompt_includes_task_id() -> None:
    t = _load_fixture()
    prompt = _build_continuation_prompt(t.trajectory[:2], t.task_id)
    assert "synth-001" in prompt


def test_continuation_prompt_includes_tool_names() -> None:
    t = _load_fixture()
    prompt = _build_continuation_prompt(t.trajectory[:2], t.task_id)
    assert "read_file" in prompt
    assert "edit" in prompt


def test_continuation_prompt_requests_outcome_line() -> None:
    t = _load_fixture()
    prompt = _build_continuation_prompt(t.trajectory[:1], t.task_id)
    assert "OUTCOME: PASS" in prompt
    assert "OUTCOME: FAIL" in prompt


def test_parses_pass_marker() -> None:
    assert _parses_as_successful("OUTCOME: PASS\nLooks good.")
    assert _parses_as_successful("outcome: pass\n...")  # case-insensitive
    assert not _parses_as_successful("OUTCOME: FAIL\nNope.")
    assert not _parses_as_successful("the trajectory seems fine")


def test_label_trajectory_simplified_returns_same_object() -> None:
    """Mutation in place — caller can use return value or original interchangeably."""
    t = _load_fixture()
    t.outcome = 0
    result = label_trajectory_simplified(t)
    assert result is t
