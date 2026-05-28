"""Integration tests for collect_batch.collect() orchestration.

We monkeypatch `_load_tasks()` and `run_task_with_codeagent` so these tests
run in milliseconds without any subprocess / HF dataset / Anthropic API.
"""
from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Any

import pytest

from src.eval import collect_batch
from src.eval.swebench_runner import TaskRunStatus  # noqa: F401 (type alias check)


# --- fixtures ---


@pytest.fixture
def fake_ts_repo(tmp_path: Path) -> Path:
    """Create a stub TS repo so preflight (dist/cli.js exists) passes."""
    repo = tmp_path / "ts_repo"
    (repo / "dist").mkdir(parents=True)
    (repo / "dist" / "cli.js").write_text("// stub")
    return repo


@pytest.fixture
def env(monkeypatch, fake_ts_repo: Path):
    monkeypatch.setenv("TS_REPO_PATH", str(fake_ts_repo))
    return monkeypatch


def _fake_tasks(n: int) -> list[dict[str, Any]]:
    return [{"instance_id": f"task-{i}", "problem_statement": "x"} for i in range(n)]


def _patch_loader_and_runner(
    monkeypatch,
    n_tasks: int,
    status_for: callable,
):
    """Replace _load_tasks (return n_tasks) and run_task_with_codeagent
    (return status_for(task_index, rollout_k))."""
    monkeypatch.setattr(collect_batch, "_load_tasks", lambda task_set: _fake_tasks(n_tasks))

    call_log: list[dict] = []

    def fake_runner(task, ts_repo, log_dir, timeout_sec=600, extra_env=None):
        idx = int(task["instance_id"].split("-")[1])
        k = int((extra_env or {}).get("CODE_PRM_ROLLOUT_ID", "0"))
        call_log.append({"task_idx": idx, "rollout": k})
        return status_for(idx, k)

    monkeypatch.setattr(collect_batch, "run_task_with_codeagent", fake_runner)
    return call_log


# --- tests ---


def test_limit_slices_tasks(env, tmp_path) -> None:
    """--limit 10 should run exactly 10 tasks, not all 100."""
    log = _patch_loader_and_runner(env, n_tasks=100, status_for=lambda i, k: "ok")

    tracker, stats = asyncio.run(collect_batch.collect(
        task_set="swebench-lite",
        num_rollouts=1,
        concurrency=2,
        log_dir=tmp_path / "logs",
        budget_usd=1000,
        limit=10,
        max_initial_failed_attempts=999,  # disable for this test
        allow_low_jsonl_success_ratio=True,  # fake runner doesn't write jsonl
    ))
    assert len(log) == 10
    assert stats.succeeded == 10
    assert stats.total == 10


def test_non_empty_log_dir_aborts_without_clean(env, tmp_path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "stale.jsonl").write_text('{"x": 1}\n')

    _patch_loader_and_runner(env, n_tasks=1, status_for=lambda i, k: "ok")

    with pytest.raises(RuntimeError, match="non-empty"):
        asyncio.run(collect_batch.collect(
            task_set="swebench-lite",
            num_rollouts=1, concurrency=1,
            log_dir=log_dir,
            budget_usd=1000,
        ))


def test_clean_does_not_rmtree_sibling_dirs(env, tmp_path) -> None:
    """Foot-gun guard: --clean must NOT rmtree sibling directories.

    Scenario: user sets log_dir to a PARENT of multiple task-set collections.
    The OLD code did `shutil.rmtree(child)` on every subdir under log_dir,
    losing all non-jsonl artifacts in siblings. The new code only deletes
    *.jsonl files and rmdirs empty dirs — so non-jsonl artifacts survive
    and the sibling directory itself survives.
    """
    log_dir = tmp_path / "data" / "raw"
    log_dir.mkdir(parents=True)
    (log_dir / "old.jsonl").write_text('{"x": 1}\n')
    # A sibling directory with a non-jsonl artifact — MUST survive --clean.
    sibling = log_dir / "swebench-lite"
    sibling.mkdir()
    (sibling / "README.md").write_text("notes from prior run")

    _patch_loader_and_runner(env, n_tasks=1, status_for=lambda i, k: "ok")

    # We don't care what the run returns; we care about filesystem side-effects.
    try:
        asyncio.run(collect_batch.collect(
            task_set="swebench-lite",
            num_rollouts=1, concurrency=1,
            log_dir=log_dir,
            budget_usd=1000,
            clean=True,
            max_initial_failed_attempts=999,
            allow_low_jsonl_success_ratio=True,
        ))
    except SystemExit:
        pass  # ratio gate may fire; doesn't matter for this assertion

    # The directly-located stale jsonl IS removed
    assert not (log_dir / "old.jsonl").exists()
    # The sibling dir and its NON-jsonl artifacts MUST survive (old code's
    # shutil.rmtree would have nuked these).
    assert sibling.exists(), "sibling dir was rmtree'd — foot-gun re-introduced!"
    assert (sibling / "README.md").exists()


def test_non_empty_log_dir_cleaned_with_flag(env, tmp_path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "stale.jsonl").write_text('{"x": 1}\n')

    _patch_loader_and_runner(env, n_tasks=1, status_for=lambda i, k: "ok")

    asyncio.run(collect_batch.collect(
        task_set="swebench-lite",
        num_rollouts=1, concurrency=1,
        log_dir=log_dir,
        budget_usd=1000,
        clean=True,
        allow_low_jsonl_success_ratio=True,
        max_initial_failed_attempts=999,
    ))
    # stale.jsonl should have been removed
    assert not (log_dir / "stale.jsonl").exists()


def test_clean_and_allow_append_mutually_exclusive(env, tmp_path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "stale.jsonl").write_text('{}\n')

    _patch_loader_and_runner(env, n_tasks=1, status_for=lambda i, k: "ok")

    with pytest.raises(RuntimeError, match="mutually exclusive"):
        asyncio.run(collect_batch.collect(
            task_set="swebench-lite",
            num_rollouts=1, concurrency=1,
            log_dir=log_dir,
            budget_usd=1000,
            clean=True, allow_append=True,
        ))


def test_low_jsonl_ratio_aborts_by_default(env, tmp_path) -> None:
    """All subprocess succeed (returns 'ok') but no jsonl written → SystemExit(3)."""
    log_dir = tmp_path / "logs"
    _patch_loader_and_runner(env, n_tasks=5, status_for=lambda i, k: "ok")

    with pytest.raises(SystemExit) as exc:
        asyncio.run(collect_batch.collect(
            task_set="swebench-lite",
            num_rollouts=1, concurrency=1,
            log_dir=log_dir,
            budget_usd=1000,
            max_initial_failed_attempts=999,
        ))
    assert exc.value.code == 3


def test_low_jsonl_ratio_allowed_via_flag(env, tmp_path) -> None:
    log_dir = tmp_path / "logs"
    _patch_loader_and_runner(env, n_tasks=5, status_for=lambda i, k: "ok")

    # Should NOT raise SystemExit
    asyncio.run(collect_batch.collect(
        task_set="swebench-lite",
        num_rollouts=1, concurrency=1,
        log_dir=log_dir,
        budget_usd=1000,
        max_initial_failed_attempts=999,
        allow_low_jsonl_success_ratio=True,
    ))


def test_num_rollouts_multiplies(env, tmp_path) -> None:
    """3 tasks × 4 rollouts → 12 runs, each with distinct CODE_PRM_ROLLOUT_ID."""
    log = _patch_loader_and_runner(env, n_tasks=3, status_for=lambda i, k: "ok")

    tracker, stats = asyncio.run(collect_batch.collect(
        task_set="swebench-lite",
        num_rollouts=4,
        concurrency=2,
        log_dir=tmp_path / "logs",
        budget_usd=1000,
        max_initial_failed_attempts=999,
        allow_low_jsonl_success_ratio=True,
    ))
    assert stats.total == 12
    rollouts_seen = sorted({c["rollout"] for c in log})
    assert rollouts_seen == [0, 1, 2, 3]


def test_launch_error_aborts_with_exit_4(env, tmp_path) -> None:
    """First task returns 'launch_error' → batch aborts AND exits non-zero."""
    def status_for(i, k):
        return "launch_error" if i == 0 else "ok"

    _patch_loader_and_runner(env, n_tasks=50, status_for=status_for)

    with pytest.raises(SystemExit) as exc:
        asyncio.run(collect_batch.collect(
            task_set="swebench-lite",
            num_rollouts=1,
            concurrency=1,  # serial to make the abort deterministic
            log_dir=tmp_path / "logs",
            budget_usd=1000,
            max_initial_failed_attempts=999,
        ))
    assert exc.value.code == 4


def test_budget_exceeded_skips_remaining(env, tmp_path) -> None:
    """Once tracker.over_budget(), remaining tasks count as skipped."""
    log = _patch_loader_and_runner(env, n_tasks=20, status_for=lambda i, k: "ok")

    # Each ok costs ~$0.0825 (25k input @ $3/M + 5k output @ $15/M).
    # Budget $0.20 should let ~2 tasks through.
    tracker, stats = asyncio.run(collect_batch.collect(
        task_set="swebench-lite",
        num_rollouts=1,
        concurrency=1,
        log_dir=tmp_path / "logs",
        budget_usd=0.20,
        max_initial_failed_attempts=999,
        allow_low_jsonl_success_ratio=True,
    ))
    assert stats.succeeded >= 1
    assert stats.succeeded < 20
    assert stats.over_budget_skipped > 0
    assert stats.total == 20


def test_timeout_does_not_abort_gather(env, tmp_path) -> None:
    """A single 'timeout' must not crash asyncio.gather. Remaining tasks proceed."""
    def status_for(i, k):
        return "timeout" if i == 0 else "ok"

    log = _patch_loader_and_runner(env, n_tasks=5, status_for=status_for)

    tracker, stats = asyncio.run(collect_batch.collect(
        task_set="swebench-lite",
        num_rollouts=1,
        concurrency=1,
        log_dir=tmp_path / "logs",
        budget_usd=1000,
        max_initial_failed_attempts=999,
        allow_low_jsonl_success_ratio=True,
    ))
    # Task 0 timed out; tasks 1..4 should have succeeded (4 successes, 1 timeout).
    assert stats.timed_out == 1
    assert stats.succeeded == 4
    assert stats.total == 5


def test_max_initial_failed_attempts_exits_4(env, tmp_path) -> None:
    """If the first N rollout ATTEMPTS all fail, batch exits non-zero."""
    _patch_loader_and_runner(env, n_tasks=50, status_for=lambda i, k: "failed")

    with pytest.raises(SystemExit) as exc:
        asyncio.run(collect_batch.collect(
            task_set="swebench-lite",
            num_rollouts=1,
            concurrency=1,
            log_dir=tmp_path / "logs",
            budget_usd=1000,
            max_initial_failed_attempts=3,
        ))
    assert exc.value.code == 4


def test_one_success_disables_initial_failure_guard(env, tmp_path) -> None:
    """Once any task succeeds, the early-failure guard never fires."""
    def status_for(i, k):
        # Task 0 succeeds; tasks 1, 2, 3 fail; task 4+ succeed.
        if i == 0 or i >= 4:
            return "ok"
        return "failed"

    log = _patch_loader_and_runner(env, n_tasks=10, status_for=status_for)

    tracker, stats = asyncio.run(collect_batch.collect(
        task_set="swebench-lite",
        num_rollouts=1,
        concurrency=1,
        log_dir=tmp_path / "logs",
        budget_usd=1000,
        max_initial_failed_attempts=3,
        allow_low_jsonl_success_ratio=True,
    ))
    # Even with 3 failures, total should be 10 (no abort).
    assert stats.total == 10
    assert stats.failed == 3
    assert stats.succeeded == 7


def test_preflight_missing_ts_repo_path_raises(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TS_REPO_PATH", str(tmp_path / "does_not_exist"))
    monkeypatch.setattr(collect_batch, "_load_tasks", lambda ts: _fake_tasks(1))

    with pytest.raises(RuntimeError, match="TS_REPO_PATH"):
        asyncio.run(collect_batch.collect(
            task_set="swebench-lite",
            num_rollouts=1,
            concurrency=1,
            log_dir=tmp_path / "logs",
            budget_usd=10,
        ))


def test_preflight_missing_cli_js_raises(monkeypatch, tmp_path) -> None:
    """TS repo exists but dist/cli.js doesn't → loud error."""
    ts = tmp_path / "ts_repo"
    ts.mkdir()
    # No dist/cli.js
    monkeypatch.setenv("TS_REPO_PATH", str(ts))
    monkeypatch.setattr(collect_batch, "_load_tasks", lambda ts_set: _fake_tasks(1))

    with pytest.raises(RuntimeError, match="cli.js"):
        asyncio.run(collect_batch.collect(
            task_set="swebench-lite",
            num_rollouts=1,
            concurrency=1,
            log_dir=tmp_path / "logs",
            budget_usd=10,
        ))
