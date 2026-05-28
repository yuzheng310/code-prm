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
and stamps the trajectory record's `rollout_id`. Downstream code uses
`rglob("*.jsonl")` (robust to any future nested layouts).
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
    limit: int | None = None,
    max_initial_failed_attempts: int = 5,
    clean: bool = False,
    allow_append: bool = False,
    min_jsonl_success_ratio: float = 0.8,
    allow_low_jsonl_success_ratio: bool = False,
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
    if limit is not None:
        tasks = tasks[:limit]
    ts_repo = Path(os.environ["TS_REPO_PATH"])

    # --- Preflight: catch config errors BEFORE wasting money ---
    if not ts_repo.is_dir():
        raise RuntimeError(
            f"TS_REPO_PATH does not exist or is not a directory: {ts_repo}. "
            "Check your .env / shell env."
        )
    cli_js = ts_repo / "dist" / "cli.js"
    if not cli_js.exists():
        raise RuntimeError(
            f"TS codeAgent CLI not found at {cli_js}. Did you run the TS "
            "build step in your codeAgent repo? "
            "If your CLI lives elsewhere, edit src/eval/swebench_runner.py "
            "`run_task_with_codeagent` to point to it."
        )

    # Log-dir hygiene: refuse to silently append onto a previous run.
    log_dir.mkdir(parents=True, exist_ok=True)
    existing = list(log_dir.rglob("*.jsonl"))
    if existing:
        if clean and allow_append:
            raise RuntimeError("--clean and --allow_append are mutually exclusive.")
        if clean:
            import shutil
            for p in existing:
                p.unlink()
            # Clean out any empty subdirs as well (rmtree + recreate is safer
            # but we keep the original directory to preserve permissions).
            for child in log_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
            print(f"Cleaned {len(existing)} stale jsonl file(s) from {log_dir}")
        elif allow_append:
            print(
                f"[!] log_dir {log_dir} contains {len(existing)} existing "
                "jsonl file(s); appending as requested. Downstream stats "
                "WILL mix old and new trajectories."
            )
        else:
            raise RuntimeError(
                f"log_dir {log_dir} is non-empty (found "
                f"{len(existing)} *.jsonl file(s)). Refusing to append "
                "silently. Pass --clean to wipe, or --allow_append to merge."
            )
    sem = asyncio.Semaphore(concurrency)
    total_runs = len(tasks) * num_rollouts

    print(f"Collecting {len(tasks)} tasks x {num_rollouts} rollouts = {total_runs} runs")
    print(f"Concurrency: {concurrency}, Estimated budget cap: ${budget_usd:.2f}")
    print(f"Per-task timeout: {timeout_sec}s, Output dir (flat): {log_dir}")
    print("NOTE: real cost will be in trajectory token_usage; aggregate afterward.")

    # Sentinel that, once set, makes remaining rollouts skip cheaply.
    abort_flag = {"set": False, "reason": ""}

    async def one_run(task: dict[str, Any], k: int) -> None:
        async with sem:
            if abort_flag["set"]:
                stats.add_skipped()
                return
            if tracker.over_budget():
                stats.add_skipped()
                return
            run_id = str(uuid.uuid4())
            extra_env = {
                "CODE_PRM_ROLLOUT_ID": str(k),
                "CODE_PRM_RUN_ID": run_id,
            }
            loop = asyncio.get_running_loop()
            try:
                status = await loop.run_in_executor(
                    None,
                    run_task_with_codeagent,
                    task,
                    ts_repo,
                    log_dir,
                    timeout_sec,
                    extra_env,
                )
            except Exception as exc:  # noqa: BLE001 — log and continue
                print(f"  [crash] task={task.get('instance_id', task.get('task_id', '?'))} "
                      f"rollout={k} err={exc!r}")
                stats.add_crash()
                # Crash before any success is suspicious — count as initial failure too.
                _maybe_set_abort_on_early_failures()
                return

            if status == "ok":
                stats.add_success()
            elif status == "timeout":
                stats.add_timeout()
                _maybe_set_abort_on_early_failures()
            elif status == "launch_error":
                # Config bug. Don't burn money on remaining tasks.
                print(
                    f"  [launch_error] task={task.get('instance_id', '?')} "
                    f"rollout={k}: subprocess failed to launch. "
                    "ABORTING batch — check `node` is installed and TS repo built."
                )
                stats.add_failure()
                abort_flag["set"] = True
                abort_flag["reason"] = "launch_error"
                return
            else:  # "failed"
                stats.add_failure()
                _maybe_set_abort_on_early_failures()

            # Pessimistic estimate; real cost is in trajectory.token_usage.
            tracker.add(
                policy_model,
                input_tokens=EST_INPUT_TOK_PER_TRAJECTORY,
                output_tokens=EST_OUTPUT_TOK_PER_TRAJECTORY,
            )

    def _maybe_set_abort_on_early_failures() -> None:
        """Abort if the FIRST `max_initial_failed_attempts` rollout attempts
        all failed without a single success.

        NOTE: this counts ATTEMPTS, not unique tasks. With num_rollouts=4 a
        single broken task contributes 4 to this counter, so the guard
        fires sooner with high num_rollouts — that is acceptable because
        the configuration is broken either way.
        """
        if abort_flag["set"]:
            return
        if stats.succeeded > 0:
            return  # Once we have any success, this guard never fires.
        initial_failed_attempts = stats.failed + stats.timed_out + stats.crashed
        if initial_failed_attempts >= max_initial_failed_attempts:
            abort_flag["set"] = True
            abort_flag["reason"] = "max_initial_failed_attempts"
            print(
                f"\n[!] First {initial_failed_attempts} rollout attempts all "
                "failed before any succeeded. ABORTING batch — likely a TS-side "
                "config bug (missing deps, agent crash on init, wrong CLI args).\n"
                "    Investigate one task manually:\n"
                f"      node $TS_REPO_PATH/dist/cli.js run --task-id <id> --task-type <T>\n"
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

    # Sanity-check that the TS side actually wrote jsonl lines.
    # `stats.succeeded` only counts subprocess exit code 0 — the TS logger
    # might still not have written anything (CODE_PRM_LOG_DIR ignored,
    # logger not wired, agent exited 0 without finalize, etc.).
    jsonl_lines = _count_jsonl_lines(log_dir)
    print(f"\nTS-side jsonl lines written: {jsonl_lines}")
    print(f"Subprocess successes:        {stats.succeeded}")
    if stats.succeeded > 0:
        ratio = jsonl_lines / stats.succeeded
        print(f"jsonl/success ratio: {ratio:.0%}  (threshold: {min_jsonl_success_ratio:.0%})")
        if ratio < min_jsonl_success_ratio:
            msg = (
                f"\n[FATAL] Only {jsonl_lines}/{stats.succeeded} successful "
                f"runs produced a jsonl line ({ratio:.0%} < "
                f"{min_jsonl_success_ratio:.0%}). The TS logger is probably\n"
                "not wired up correctly. Inspect data/raw and check that the\n"
                "TS side honors CODE_PRM_LOG_DIR per ts_logger_spec.md.\n"
                "Pass --allow_low_jsonl_success_ratio to proceed anyway."
            )
            if allow_low_jsonl_success_ratio:
                print(msg.replace("[FATAL]", "[WARNING (override active)]"))
            else:
                print(msg)
                raise SystemExit(3)
    print(
        "\n[!] Cost above is ESTIMATED. Aggregate real cost via:\n"
        "    python -m src.utils.cost_aggregator --dir " + str(log_dir)
    )
    return tracker, stats


def _count_jsonl_lines(directory: Path) -> int:
    """Count non-empty lines across all *.jsonl files under `directory`."""
    n = 0
    for f in directory.rglob("*.jsonl"):
        try:
            with open(f) as fh:
                n += sum(1 for line in fh if line.strip())
        except OSError:
            continue
    return n


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
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of tasks (after dataset load). Useful for pilots.",
    )
    p.add_argument(
        "--timeout_sec", type=int, default=600,
        help="Per-task subprocess timeout (default 10 min).",
    )
    p.add_argument(
        "--max_initial_failed_attempts", type=int, default=5,
        help="If the first N rollout ATTEMPTS all fail before any success, "
             "abort the batch. Counts rollouts, not unique tasks — with "
             "num_rollouts=4 a single broken task contributes 4 attempts. "
             "Catches config bugs early instead of burning the whole budget.",
    )
    p.add_argument(
        "--clean", action="store_true",
        help="Wipe any existing *.jsonl files in --log_dir before collecting. "
             "Default behavior is to ABORT if --log_dir is non-empty.",
    )
    p.add_argument(
        "--allow_append", action="store_true",
        help="Allow collection to merge into an existing non-empty --log_dir. "
             "Use carefully — downstream stats will mix old and new trajectories.",
    )
    p.add_argument(
        "--min_jsonl_success_ratio", type=float, default=0.8,
        help="Required ratio of (jsonl lines written) / (subprocess successes). "
             "Below this, collection exits non-zero. Default 0.80.",
    )
    p.add_argument(
        "--allow_low_jsonl_success_ratio", action="store_true",
        help="Proceed even if the jsonl/success ratio is below threshold. "
             "Default is to ABORT after collection finishes.",
    )
    args = p.parse_args()

    asyncio.run(
        collect(
            task_set=args.task_set,
            num_rollouts=args.num_rollouts,
            concurrency=args.concurrency,
            log_dir=args.log_dir,
            budget_usd=args.budget_usd,
            policy_model=args.policy_model,
            timeout_sec=args.timeout_sec,
            limit=args.limit,
            max_initial_failed_attempts=args.max_initial_failed_attempts,
            clean=args.clean,
            allow_append=args.allow_append,
            min_jsonl_success_ratio=args.min_jsonl_success_ratio,
            allow_low_jsonl_success_ratio=args.allow_low_jsonl_success_ratio,
        )
    )


if __name__ == "__main__":
    main()
