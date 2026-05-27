"""Unit tests for the step labeler (pure functions that don't hit the API)."""
from __future__ import annotations
import json
from pathlib import Path

from src.labeler.step_labeler import (
    _build_continuation_prompt,
    _parses_as_successful,
    label_trajectory_simplified,
)
from src.labeler.trajectory_schema import Step, Trajectory

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_trajectory.json"


def _load_fixture() -> Trajectory:
    return Trajectory(**json.loads(FIXTURE.read_text()))


# --- label_trajectory_simplified ---


def test_outcome_zero_gets_zero_labels_only_on_tool_steps() -> None:
    """outcome=0: tool steps get 0, pure-thought steps stay None.

    This symmetry with the outcome=1 path is essential — otherwise Phase 2
    training would silently treat 'pure-thought in failure' differently from
    'pure-thought in success'.
    """
    t = _load_fixture()
    t.outcome = 0
    # Inject a pure-thought step
    t.trajectory.append(Step(step=99, tool=None, thought="reflecting"))
    labeled = label_trajectory_simplified(t, only_tool_steps=True)
    for s in labeled.trajectory:
        if s.tool is None:
            assert s.step_label is None
        else:
            assert s.step_label == 0.0


def test_outcome_zero_with_only_tool_steps_false_labels_everything() -> None:
    t = _load_fixture()
    t.outcome = 0
    t.trajectory.append(Step(step=99, tool=None, thought="reflecting"))
    labeled = label_trajectory_simplified(t, only_tool_steps=False)
    assert all(s.step_label == 0.0 for s in labeled.trajectory)


def test_outcome_one_left_unchanged_by_simplified_labeler() -> None:
    """outcome=1 path should NOT get labels from the simplified labeler.
    Real labels must come from llm_judge_score_step()."""
    t = _load_fixture()
    assert t.outcome == 1
    labeled = label_trajectory_simplified(t)
    assert all(s.step_label is None for s in labeled.trajectory)


def test_label_trajectory_simplified_returns_same_object() -> None:
    """Mutation in place — caller can use return value or original interchangeably."""
    t = _load_fixture()
    t.outcome = 0
    result = label_trajectory_simplified(t)
    assert result is t


# --- _build_continuation_prompt ---


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


def test_continuation_prompt_includes_task_prompt_when_given() -> None:
    t = _load_fixture()
    prompt = _build_continuation_prompt(
        t.trajectory[:1],
        task_id=t.task_id,
        task_prompt="Fix the bug in query.py that breaks pagination.",
        task_type="swe-bench-lite",
    )
    assert "Fix the bug in query.py" in prompt
    assert "swe-bench-lite" in prompt
    assert "Problem statement:" in prompt


def test_continuation_prompt_truncates_long_task_prompt() -> None:
    t = _load_fixture()
    very_long = "x" * 5000
    prompt = _build_continuation_prompt(
        t.trajectory[:1],
        task_id=t.task_id,
        task_prompt=very_long,
    )
    assert "TRUNC" in prompt
    # Truncated prompt should be substantially shorter than 5000 chars
    assert prompt.count("x") < 5000


def test_continuation_prompt_omits_task_prompt_section_when_absent() -> None:
    t = _load_fixture()
    prompt = _build_continuation_prompt(t.trajectory[:1], task_id=t.task_id)
    assert "Problem statement:" not in prompt


# --- _parses_as_successful (line-anchored regex) ---


def test_parses_pass_when_at_line_start() -> None:
    assert _parses_as_successful("OUTCOME: PASS\nLooks good.")
    assert _parses_as_successful("outcome: pass\nReasoning...")  # case-insensitive
    assert _parses_as_successful("  OUTCOME: PASS  \nWith spaces")  # leading whitespace


def test_parses_fail_correctly() -> None:
    assert not _parses_as_successful("OUTCOME: FAIL\nNope.")
    assert not _parses_as_successful("the trajectory seems fine")


def test_does_NOT_parse_pass_when_embedded_in_narrative() -> None:
    """Regression: old `in` substring match would incorrectly mark these as PASS."""
    assert not _parses_as_successful("The result is not OUTCOME: PASS at this step.")
    assert not _parses_as_successful("Although it looks like OUTCOME: PASS it actually fails")
    assert not _parses_as_successful("xOUTCOME: PASS")  # not word-boundary OK
    # The above three would have matched under the old `"OUTCOME: PASS" in text.upper()`


def test_parses_pass_with_justification_on_same_line() -> None:
    """Justification appended after the verdict on the same line is still PASS."""
    assert _parses_as_successful("OUTCOME: PASS — the agent correctly identified the bug.")


# --- label_file integration test (offline; uses fake client, no API) ---


class _FakeClient:
    """Stand-in for RateLimitedClient that returns canned responses."""

    def __init__(self, response_text: str = "OUTCOME: PASS") -> None:
        self._text = response_text
        self.calls = 0

    def complete(self, messages, max_tokens=2048, temperature=0.9):
        self.calls += 1
        return self._text, 0, 0


def test_label_file_outcome_one_uses_llm_judge(tmp_path) -> None:
    """outcome=1 trajectory: should call the judge and stamp label_method=llm_judge."""
    from src.labeler.step_labeler import label_file
    from src.utils.jsonl_io import write_trajectories, read_trajectories

    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    traj = _load_fixture()  # outcome=1
    write_trajectories(src, [traj])

    client = _FakeClient(response_text="OUTCOME: PASS")
    label_file(src, dst, client, K=2)

    out = list(read_trajectories(dst))
    assert len(out) == 1
    assert out[0].label_method == "llm_judge"
    # tool steps got labeled
    tool_labels = [s.step_label for s in out[0].trajectory if s.tool is not None]
    assert all(l is not None for l in tool_labels)
    # Judge always said PASS, so labels are 1.0
    assert all(l == 1.0 for l in tool_labels)


def test_label_file_outcome_zero_uses_simplification_method(tmp_path) -> None:
    """outcome=0 trajectory: should NOT call judge; stamp outcome_zero_simplification."""
    from src.labeler.step_labeler import label_file
    from src.utils.jsonl_io import write_trajectories, read_trajectories

    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    traj = _load_fixture()
    traj.outcome = 0
    write_trajectories(src, [traj])

    client = _FakeClient(response_text="OUTCOME: PASS")
    label_file(src, dst, client, K=2)

    out = list(read_trajectories(dst))
    assert len(out) == 1
    assert out[0].label_method == "outcome_zero_simplification"
    # Judge must not have been called for outcome=0 path
    assert client.calls == 0


def test_label_file_mixed_outcomes_stamp_correctly(tmp_path) -> None:
    """A file with both outcome=0 and outcome=1 should stamp each correctly."""
    from src.labeler.step_labeler import label_file
    from src.utils.jsonl_io import write_trajectories, read_trajectories

    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    t1 = _load_fixture()
    t2 = _load_fixture()
    t2.outcome = 0
    t2.task_id = "synth-002-fail"
    write_trajectories(src, [t1, t2])

    client = _FakeClient(response_text="OUTCOME: FAIL")
    label_file(src, dst, client, K=2)

    out = list(read_trajectories(dst))
    by_id = {t.task_id: t for t in out}
    assert by_id["synth-001"].label_method == "llm_judge"
    assert by_id["synth-002-fail"].label_method == "outcome_zero_simplification"
