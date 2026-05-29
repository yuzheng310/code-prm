#!/usr/bin/env python3
"""Backfill missing BigCodeBench rollout trajectories without rerunning full collection."""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.swebench_runner import load_bigcodebench_hard, run_task_with_codeagent  # noqa: E402
from src.utils.jsonl_io import read_trajectories  # noqa: E402


@dataclass(frozen=True, order=True)
class MissingRollout:
    task_id: str
    rollout_id: int


def find_missing_rollouts(log_dir: Path, expected_rollouts: int) -> list[MissingRollout]:
    task_to_rollouts: dict[str, set[int]] = {}
    for path in sorted(log_dir.glob("*.jsonl")):
        for traj in read_trajectories(path):
            task_to_rollouts.setdefault(traj.task_id, set()).add(traj.rollout_id)

    missing: list[MissingRollout] = []
    for task_id in sorted(task_to_rollouts):
        present = task_to_rollouts[task_id]
        for rollout_id in range(expected_rollouts):
            if rollout_id not in present:
                missing.append(MissingRollout(task_id, rollout_id))
    return missing


def backfill_missing(
    missing: list[MissingRollout],
    *,
    task_by_id: dict[str, dict],
    ts_repo: Path,
    log_dir: Path,
    timeout_sec: int,
    stream_output: bool,
) -> None:
    for item in missing:
        if item.task_id not in task_by_id:
            raise KeyError(f"Missing task_id not found in BigCodeBench dataset: {item.task_id}")
        run_id = str(uuid.uuid4())
        print(
            f"[backfill start] task={item.task_id} rollout={item.rollout_id} run_id={run_id}",
            flush=True,
        )
        status = run_task_with_codeagent(
            task_by_id[item.task_id],
            ts_repo,
            log_dir,
            timeout_sec=timeout_sec,
            extra_env={
                "CODE_PRM_ROLLOUT_ID": str(item.rollout_id),
                "CODE_PRM_RUN_ID": run_id,
            },
            stream_output=stream_output,
        )
        print(
            f"[backfill end] task={item.task_id} rollout={item.rollout_id} status={status}",
            flush=True,
        )
        if status != "ok":
            raise SystemExit(
                f"Backfill failed: task={item.task_id} rollout={item.rollout_id} status={status}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=Path("data/raw/bigcodebench-hard"))
    parser.add_argument("--expected-rollouts", type=int, default=4)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--stream-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if "TS_REPO_PATH" not in os.environ:
        raise SystemExit("TS_REPO_PATH must be set")

    missing = find_missing_rollouts(args.log_dir, expected_rollouts=args.expected_rollouts)
    print(f"Missing rollout count: {len(missing)}")
    for item in missing:
        print(f"  {item.task_id} rollout={item.rollout_id}")
    if not missing:
        return
    if args.dry_run:
        return

    tasks = {task["task_id"]: task for task in load_bigcodebench_hard()}
    backfill_missing(
        missing,
        task_by_id=tasks,
        ts_repo=Path(os.environ["TS_REPO_PATH"]),
        log_dir=args.log_dir,
        timeout_sec=args.timeout_sec,
        stream_output=args.stream_output,
    )


if __name__ == "__main__":
    main()
