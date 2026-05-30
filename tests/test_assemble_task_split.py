"""Tests for task-grouped, stratified splitting in scripts/30_assemble_dataset.py.

The Phase-1 split shuffled at the *trajectory* level, scattering a task's
rollouts across train/val/test -> 100% task-level leakage and no clean
held-out Best-of-N candidate set. These tests pin the corrected behavior:
a task's rollouts always stay together, splits are disjoint by task,
mixed-outcome tasks are allocated by explicit counts, and non-mixed tasks
are stratified by pass-count bucket.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import pytest

from src.labeler.trajectory_schema import Step, Trajectory

# Load the numerically-prefixed script as a module (same trick as
# test_assemble_dataset.py). The module MUST be registered in sys.modules
# *before* exec_module, otherwise dataclasses with `from __future__ import
# annotations` fail to resolve their module and raise AttributeError.
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "30_assemble_dataset.py"
_spec = importlib.util.spec_from_file_location("assemble_dataset_tasksplit", _SCRIPT)
assemble = importlib.util.module_from_spec(_spec)
sys.modules["assemble_dataset_tasksplit"] = assemble
_spec.loader.exec_module(assemble)


def _traj(task_id: str, rollout_id: int, outcome: int) -> Trajectory:
    return Trajectory(
        task_id=task_id,
        task_type="bigcodebench-hard",
        rollout_id=rollout_id,
        trajectory=[Step(step=0, tool="bash", step_label=(1.0 if outcome else 0.0))],
        outcome=outcome,
        policy_model="test-model",
        timestamp="2026-01-01T00:00:00Z",
        task_prompt="solve it",
        label_method="llm_judge" if outcome else "outcome_zero_simplification",
    )


def _task(task_id: str, pass_count: int, rollouts: int = 4) -> list[Trajectory]:
    """A task with `pass_count` passing rollouts and the rest failing."""
    return [_traj(task_id, r, 1 if r < pass_count else 0) for r in range(rollouts)]


def _dataset(n_fail: int, n_pass: int, mixed_passcounts: list[int]) -> list[Trajectory]:
    """n_fail all-fail tasks, n_pass all-pass tasks, plus one mixed task per
    entry in mixed_passcounts (its value = #passing rollouts)."""
    trajs: list[Trajectory] = []
    i = 0
    for _ in range(n_fail):
        trajs += _task(f"f{i}", 0)
        i += 1
    for _ in range(n_pass):
        trajs += _task(f"p{i}", 4)
        i += 1
    for pc in mixed_passcounts:
        trajs += _task(f"m{i}", pc)
        i += 1
    return trajs


def _task_to_splits(train, val, test) -> dict[str, set[str]]:
    loc: dict[str, set[str]] = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        for t in part:
            loc.setdefault(t.task_id, set()).add(name)
    return loc


def _mixed_tasks(part) -> set[str]:
    by: dict[str, set[int]] = {}
    for t in part:
        by.setdefault(t.task_id, set()).add(t.outcome)
    return {tid for tid, oc in by.items() if len(oc) > 1}


def test_all_rollouts_of_a_task_stay_in_one_split():
    trajs = _dataset(10, 5, [1, 2, 3] * 4)  # 12 mixed
    train, val, test = assemble.split_by_task(
        trajs, mixed_alloc=(8, 2, 2), nonmixed_fracs=(0.6, 0.2, 0.2), seed=1
    )
    for tid, splits in _task_to_splits(train, val, test).items():
        assert len(splits) == 1, f"{tid} spread across {splits}"


def test_splits_are_disjoint_by_task():
    trajs = _dataset(10, 5, [1, 2, 3] * 4)
    train, val, test = assemble.split_by_task(
        trajs, mixed_alloc=(8, 2, 2), nonmixed_fracs=(0.6, 0.2, 0.2), seed=1
    )
    def task_ids(part):
        return {t.task_id for t in part}

    assert task_ids(train) & task_ids(val) == set()
    assert task_ids(train) & task_ids(test) == set()
    assert task_ids(val) & task_ids(test) == set()


def test_conserves_all_trajectories_no_dup_no_loss():
    trajs = _dataset(10, 5, [1, 2, 3] * 4)
    train, val, test = assemble.split_by_task(
        trajs, mixed_alloc=(8, 2, 2), nonmixed_fracs=(0.6, 0.2, 0.2), seed=1
    )
    ids = [(t.task_id, t.rollout_id) for t in train + val + test]
    assert len(ids) == len(set(ids)), "duplicate trajectories across splits"
    assert set(ids) == {(t.task_id, t.rollout_id) for t in trajs}, "lost trajectories"


def test_mixed_tasks_allocated_by_explicit_counts():
    trajs = _dataset(20, 10, [1, 2, 3] * 6)  # 18 mixed
    train, val, test = assemble.split_by_task(
        trajs, mixed_alloc=(10, 4, 4), nonmixed_fracs=(0.8, 0.1, 0.1), seed=3
    )
    assert len(_mixed_tasks(train)) == 10
    assert len(_mixed_tasks(val)) == 4
    assert len(_mixed_tasks(test)) == 4


def test_mixed_allocation_preserves_pass_count_buckets():
    # Mirrors the observed Phase 1 mixed histogram shape: 10 tasks at 1/4,
    # 11 tasks at 2/4, 7 tasks at 3/4. Val/test must not receive only the
    # easier mixed buckets.
    trajs = _dataset(0, 0, [1] * 10 + [2] * 11 + [3] * 7)
    train, val, test = assemble.split_by_task(
        trajs, mixed_alloc=(18, 4, 6), nonmixed_fracs=(0.8, 0.1, 0.1), seed=3
    )

    def mixed_pass_counts(part):
        counts = set()
        for task_id in _mixed_tasks(part):
            counts.add(sum(t.outcome for t in part if t.task_id == task_id))
        return counts

    assert len(_mixed_tasks(train)) == 18
    assert len(_mixed_tasks(val)) == 4
    assert len(_mixed_tasks(test)) == 6
    assert mixed_pass_counts(val) == {1, 2, 3}
    assert mixed_pass_counts(test) == {1, 2, 3}


def test_test_split_mixed_tasks_have_both_pass_and_fail():
    trajs = _dataset(20, 10, [1, 2, 3] * 6)
    _, _, test = assemble.split_by_task(
        trajs, mixed_alloc=(10, 4, 4), nonmixed_fracs=(0.8, 0.1, 0.1), seed=3
    )
    by: dict[str, list[int]] = {}
    for t in test:
        by.setdefault(t.task_id, []).append(t.outcome)
    mixed = [tid for tid, oc in by.items() if len(set(oc)) > 1]
    assert mixed, "test split must contain mixed-outcome tasks for Best-of-N"
    for tid in mixed:
        assert any(o == 1 for o in by[tid]) and any(o == 0 for o in by[tid])


def test_nonmixed_tasks_stratified_by_fraction():
    trajs = _dataset(10, 5, [])  # 10 all-fail, 5 all-pass, no mixed
    train, val, test = assemble.split_by_task(
        trajs, mixed_alloc=(0, 0, 0), nonmixed_fracs=(0.6, 0.2, 0.2), seed=1
    )

    def by_outcome(part, want: set[int]) -> set[str]:
        by: dict[str, set[int]] = {}
        for t in part:
            by.setdefault(t.task_id, set()).add(t.outcome)
        return {tid for tid, oc in by.items() if oc == want}

    # 10 all-fail -> 6/2/2, 5 all-pass -> 3/1/1
    assert (len(by_outcome(train, {0})), len(by_outcome(val, {0})), len(by_outcome(test, {0}))) == (6, 2, 2)
    assert (len(by_outcome(train, {1})), len(by_outcome(val, {1})), len(by_outcome(test, {1}))) == (3, 1, 1)


def test_deterministic_with_same_seed():
    trajs = _dataset(20, 10, [1, 2, 3] * 6)
    a = assemble.split_by_task(trajs, mixed_alloc=(10, 4, 4), seed=5)
    b = assemble.split_by_task(trajs, mixed_alloc=(10, 4, 4), seed=5)
    def task_ids(part):
        return sorted({t.task_id for t in part})

    assert task_ids(a[0]) == task_ids(b[0])
    assert task_ids(a[1]) == task_ids(b[1])
    assert task_ids(a[2]) == task_ids(b[2])


def test_raises_when_mixed_alloc_mismatches_mixed_count():
    trajs = _dataset(5, 5, [1, 2, 3])  # 3 mixed
    with pytest.raises(ValueError):
        assemble.split_by_task(trajs, mixed_alloc=(10, 4, 4), seed=1)  # 18 != 3


# --------------------------------------------------------------------------
# Split-level hard checks
# --------------------------------------------------------------------------
def test_check_no_task_overlap_passes_on_disjoint_splits():
    trajs = _dataset(10, 5, [1, 2, 3] * 4)
    train, val, test = assemble.split_by_task(
        trajs, mixed_alloc=(8, 2, 2), nonmixed_fracs=(0.6, 0.2, 0.2), seed=1
    )
    assert assemble.check_no_task_overlap(train, val, test).passed


def test_check_no_task_overlap_fails_when_task_in_two_splits():
    train = _task("x", 0)
    val = _task("x", 0)  # same task id in two splits
    res = assemble.check_no_task_overlap(train, val, [])
    assert not res.passed


def test_check_rollout_completeness_passes_on_full_tasks():
    trajs = _task("a", 2) + _task("b", 0)
    assert assemble.check_rollout_completeness(trajs, rollouts_per_task=4).passed


def test_check_rollout_completeness_fails_on_missing_rollout():
    trajs = _task("a", 2)[:3]  # only 3 of 4 rollouts
    assert not assemble.check_rollout_completeness(trajs, rollouts_per_task=4).passed


def test_check_rollout_completeness_fails_on_duplicate_rollout():
    trajs = _task("a", 2) + [_traj("a", 0, 0)]  # rollout_id 0 twice
    assert not assemble.check_rollout_completeness(trajs, rollouts_per_task=4).passed


def test_check_conservation_passes_when_nothing_lost():
    trajs = _dataset(10, 5, [1, 2, 3] * 4)
    train, val, test = assemble.split_by_task(
        trajs, mixed_alloc=(8, 2, 2), nonmixed_fracs=(0.6, 0.2, 0.2), seed=1
    )
    assert assemble.check_conservation(trajs, train, val, test).passed


def test_check_conservation_fails_on_loss():
    trajs = _dataset(10, 5, [1, 2, 3] * 4)
    train, val, test = assemble.split_by_task(
        trajs, mixed_alloc=(8, 2, 2), nonmixed_fracs=(0.6, 0.2, 0.2), seed=1
    )
    assert not assemble.check_conservation(trajs, train[1:], val, test).passed


def test_check_min_mixed_tasks_pass_and_fail():
    part = _task("m1", 1) + _task("m2", 2)  # 2 mixed tasks
    assert assemble.check_min_mixed_tasks(part, min_count=2, split_name="test").passed
    assert not assemble.check_min_mixed_tasks(part, min_count=3, split_name="test").passed


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------
def test_build_split_manifest_structure_and_values():
    trajs = _dataset(20, 10, [1, 2, 3] * 6)  # 18 mixed
    train, val, test = assemble.split_by_task(
        trajs, mixed_alloc=(10, 4, 4), nonmixed_fracs=(0.8, 0.1, 0.1), seed=3
    )
    checks = assemble.run_split_checks(
        trajs,
        train,
        val,
        test,
        rollouts_per_task=4,
        min_val_mixed_tasks=4,
        min_test_mixed_tasks=4,
    )
    m = assemble.build_split_manifest(
        train, val, test,
        seed=3,
        mixed_alloc=(10, 4, 4),
        nonmixed_fracs=(0.8, 0.1, 0.1),
        input_dirs=["data/labeled/bigcodebench-hard"],
        checks=checks,
    )
    assert m["seed"] == 3
    assert m["strategy"] == "task_grouped_stratified"
    assert m["mixed_alloc"] == [10, 4, 4]
    assert m["nonmixed_fracs"] == [0.8, 0.1, 0.1]
    assert m["input_dirs"] == ["data/labeled/bigcodebench-hard"]
    assert m["checks"] == [
        {"name": c.name, "passed": c.passed, "detail": c.detail}
        for c in checks
    ]

    # totals conserve the whole dataset
    assert m["totals"]["n_trajectories"] == len(trajs)
    assert m["totals"]["n_tasks"] == 20 + 10 + 18

    for name in ("train", "val", "test"):
        s = m["splits"][name]
        for key in (
            "n_tasks", "n_trajectories", "n_pass", "pass_rate",
            "n_mixed_tasks", "mixed_task_ids", "pass_count_histogram",
        ):
            assert key in s, f"{name}.{key} missing"
        assert s["mixed_task_ids"] == sorted(s["mixed_task_ids"])
        assert sum(s["pass_count_histogram"].values()) == s["n_tasks"]

    assert m["splits"]["train"]["n_mixed_tasks"] == 10
    assert m["splits"]["val"]["n_mixed_tasks"] == 4
    assert m["splits"]["test"]["n_mixed_tasks"] == 4


def test_build_split_manifest_is_json_serializable():
    trajs = _dataset(10, 5, [1, 2, 3] * 4)
    train, val, test = assemble.split_by_task(
        trajs, mixed_alloc=(8, 2, 2), nonmixed_fracs=(0.6, 0.2, 0.2), seed=1
    )
    m = assemble.build_split_manifest(
        train, val, test,
        seed=1, mixed_alloc=(8, 2, 2), nonmixed_fracs=(0.6, 0.2, 0.2),
        input_dirs=["d"],
        checks=[],
    )
    import json as _json
    _json.dumps(m)  # must not raise
