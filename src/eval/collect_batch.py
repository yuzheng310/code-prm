"""Batched trajectory collection with concurrency + cost cap.

Runs the user's TypeScript codeAgent on every task in a chosen task set,
N times each, with a semaphore-bounded concurrency. Halts when the
shared CostTracker exceeds budget.

The TS side writes the actual trajectory jsonl (see ts_logger_spec.md).
This module only orchestrates and tracks an *estimated* cost (because
we don't get usage tokens back from the subprocess).
"""
from __future__ import annotations
import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

from rich.progress import Progress

from src.eval.swebench_runner import (
    load_bigcodebench_hard,
    load_swebench_lite,
    run_task_with_codeagent,
)
from src.utils.cost_tracker import CostTracker


# Per-trajectory cost estimates (very approximate; updated based on pilot).
EST_INPUT_TOK_PER_TRAJECTORY = 25_000
EST_OUTPUT_TOK_PER_TRAJECTORY = 5_000


def _load_tasks(task_set: str) -> list[dict[str, Any]]:
    if task_set == "swebench-lite":
        return load_swebench_lite()
    if task_set == "bigcodebench-hard":
        return load_bigcodebench_hard()
    raise ValueError(f"Unknown task_set: {task_set!r}")


async def collect(
    task_set: str,
    num_rollouts: int,
    concurrency: int,
    log_dir: Path,
    budget_usd: float,
    policy_model: str = "claude-sonnet-4-5",
) -> CostTracker:
    """Drive end-to-end collection. Returns the tracker for caller inspection."""
    tracker = CostTracker(budget_usd=budget_usd)
    tasks = _load_tasks(task_set)
    ts_repo = Path(os.environ["TS_REPO_PATH"])
    sem = asyncio.Semaphore(concurrency)
    total_runs = len(tasks) * num_rollouts

    print(f"Collecting {len(tasks)} tasks x {num_rollouts} rollouts = {total_runs} runs")
    print(f"Concurrency: {concurrency}, Budget: ${budget_usd:.2f}")

    async def one_run(task: dict[str, Any], k: int) -> None:
        async with sem:
            if tracker.over_budget():
                return
            loop = asyncio.get_running_loop()
            run_log_dir = log_dir / f"rollout_{k}"
            await loop.run_in_executor(
                None,
                run_task_with_codeagent,
                task,
                ts_repo,
                run_log_dir,
            )
            # Approximate cost (no usage info from subprocess).
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
            finally:
                prog.advance(bar)

        coros = [wrapped(t, k) for t in tasks for k in range(num_rollouts)]
        await asyncio.gather(*coros)

    print(tracker)
    return tracker


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
