"""Batched trajectory collection with concurrency + estimated cost cap.

Runs the user's TypeScript codeAgent on every task in a chosen task set,
N times each, with a semaphore-bounded concurrency.

COST TRACKING NOTE: This driver does NOT see real API usage — it just
pre-checks an *estimated* budget. The TS codeAgent writes real
`token_usage` into each trajectory (see ts_logger_spec.md), and the real
cost should be aggregated AFTER collection via
`src.utils.cost_aggregator`. The `--budget_usd` here is a soft pre-flight
estimate, not a hard guarantee.

OUTPUT LAYOUT: All rollouts of all tasks write to a single flat directory
`--log_dir`. The TS side receives `CODE_PRM_ROLLOUT_ID=k` per invocation
and stamps the trajectory record's `rollout_id`. Downstream code can read
the entire dataset via `glob("*.jsonl")` without recursing.
"""
from __future__ import annotations
import argparse
import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.progress import Progress

from src.eval.swebench_runner import (
    load_bigcodebench_hard,
    load_swebench_lite,
    run_task_with_codeagent,
)
from src.utils.cost_tracker import CostTracker


# Per-trajectory cost estimates (very approximate; refine after pilot).
# These are intentionally pessimistic so the soft cap fires early.
EST_INPUT_TOK_PER_TRAJECTORY = 25_000
EST_OUTPUT_TOK_PER_TRAJECTORY = 5_000


def _load_tasks(task_set: str) -> list[dict[str, Any]]:
    if task_set == "swebench-lite":
        return load_swebench_lite()
    if task_set == "bigcodebench-hard":
        return load_bigcodebench_hard()
    raise ValueError(f"Unknown task_set: {task_set!r}")


@dataclass
class CollectionStats:
    """Per-batch outcome stats. Reported after collection completes."""
    succeeded: int = 0      # subprocess exit code 0
    failed: int = 0         # subprocess exit code != 0 (non-timeout)
    timed_out: int = 0      # subprocess.TimeoutExpired caught
    crashed: int = 0        # python-level exception (NOT subprocess exit)
    over_budget_skipped: int = 0
    total: int = 0

    def add_success(self) -> None:
        self.succeeded += 1
        self.total += 1

    def add_failure(self) -> None:
        self.failed += 1
        self.total += 1

    def add_timeout(self) -> None:
        self.timed_out += 1
        self.total += 1

    def add_crash(self) -> None:
        self.crashed += 1
        self.total += 1

    def add_skipped(self) -> None:
        self.over_budget_skipped += 1
        self.total += 1

    def __str__(self) -> str:
        return (
            f"Stats: total={self.total} | "
            f"succeeded={self.succeeded} | failed={self.failed} | "
            f"timed_out={self.timed_out} | crashed={self.crashed} | "
            f"over_budget_skipped={self.over_budget_skipped}"
        )


async def collect(
    task_set: str,
    num_rollouts: int,
    concurrency: int,
    log_dir: Path,
    budget_usd: float,
    policy_model: str = "claude-sonnet-4-5",
    timeout_sec: int = 600,
) -> tuple[CostTracker, CollectionStats]:
    """Drive end-to-end collection. Returns (tracker, stats).

    Per-task failures (timeout, non-zero exit, crash) DO NOT abort the
    batch. They are counted in stats; collection continues for remaining
    tasks. This matches the plan's "some tasks timed out, OK as long as
    >= N succeed" tolerance.
    """
    tracker = CostTracker(budget_usd=budget_usd)
    stats = CollectionStats()
    tasks = _load_tasks(task_set)
    ts_repo = Path(os.environ["TS_REPO_PATH"])
    log_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    total_runs = len(tasks) * num_rollouts

    print(f"Collecting {len(tasks)} tasks x {num_rollouts} rollouts = {total_runs} runs")
    print(f"Concurrency: {concurrency}, Estimated budget cap: ${budget_usd:.2f}")
    print(f"Per-task timeout: {timeout_sec}s, Output dir (flat): {log_dir}")
    print("NOTE: real cost will be in trajectory token_usage; aggregate afterward.")

    async def one_run(task: dict[str, Any], k: int) -> None:
        async with sem:
            if tracker.over_budget():
                stats.add_skipped()
                return
            run_id = str(uuid.uuid4())
            extra_env = {
                "CODE_PRM_ROLLOUT_ID": str(k),
                "CODE_PRM_RUN_ID": run_id,
            }
            loop = asyncio.get_running_loop()
            # run_task_with_codeagent already swallows TimeoutExpired and
            # returns False. Anything else raised here is a true crash.
            t0 = time.monotonic()
            try:
                ok = await loop.run_in_executor(
                    None,
                    run_task_with_codeagent,
                    task,
                    ts_repo,
                    log_dir,
                    timeout_sec,
                    extra_env,
                )
            except Exception as exc:  # noqa: BLE001 — log and continue
                elapsed = time.monotonic() - t0
                print(f"  [crash] task={task.get('instance_id', task.get('task_id', '?'))} "
                      f"rollout={k} elapsed={elapsed:.1f}s err={exc!r}")
                stats.add_crash()
                return

            elapsed = time.monotonic() - t0
            if ok:
                stats.add_success()
            elif elapsed >= timeout_sec - 1.0:
                stats.add_timeout()
            else:
                stats.add_failure()

            # Pessimistic estimate; real cost is in trajectory.token_usage.
            tracker.add(
                policy_model,
                input_tokens=EST_INPUT_TOK_PER_TRAJECTORY,
                output_tokens=EST_OUTPUT_TOK_PER_TRAJECTORY,
            )

    with Progress() as prog:
        bar = prog.add_task("collect", total=total_runs)

        async def wrapped(task: dict[str, Any], k: int) -> None:
            try:
                await one_run(task, k)
            except Exception as exc:  # noqa: BLE001
                # Top-level safety net so one task NEVER crashes the gather.
                print(f"  [outer-crash] task={task.get('instance_id', '?')} "
                      f"rollout={k} err={exc!r}")
                stats.add_crash()
            finally:
                prog.advance(bar)

        coros = [wrapped(t, k) for t in tasks for k in range(num_rollouts)]
        await asyncio.gather(*coros)

    print(tracker)
    print(stats)
    print(
        "\n[!] Cost above is ESTIMATED. Aggregate real cost via:\n"
        "    python -m src.utils.cost_aggregator --dir " + str(log_dir)
    )
    return tracker, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task_set",
        required=True,
        choices=["swebench-lite", "bigcodebench-hard"],
    )
    p.add_argument("--num_rollouts", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--log_dir", type=Path, required=True)
    p.add_argument("--budget_usd", type=float, required=True)
    p.add_argument("--policy_model", default="claude-sonnet-4-5")
    args = p.parse_args()

    asyncio.run(
        collect(
            task_set=args.task_set,
            num_rollouts=args.num_rollouts,
            concurrency=args.concurrency,
            log_dir=args.log_dir,
            budget_usd=args.budget_usd,
            policy_model=args.policy_model,
        )
    )


if __name__ == "__main__":
    main()
