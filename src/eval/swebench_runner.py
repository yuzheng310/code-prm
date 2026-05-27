"""SWE-bench Lite / BigCodeBench-Hard task loaders + TS codeAgent launcher.

This module is NOT a full SWE-bench harness. It does TWO things:

1. Load task lists from the public benchmarks (via HuggingFace `datasets`).
2. Launch the user's TypeScript codeAgent as a subprocess on a single task,
   passing env vars so the TS side can:
       - emit a trajectory jsonl line (CODE_PRM_LOG_DIR)
       - tag the trajectory with rollout_id / run_id (CODE_PRM_*)
       - receive the task payload (SWEBENCH_TASK_JSON)

OUTCOME ATTRIBUTION (read this carefully):
   The pass/fail "outcome" stored in the trajectory is decided by the TS
   codeAgent, NOT by this Python module. The TS side is responsible for
   running the task's tests (e.g. via SWE-bench docker harness or pytest
   directly) and writing the `outcome` int into the jsonl.

   This module only reports whether the subprocess exited non-zero
   (process-level success), which is distinct from agent-task success.

For a full SWE-bench eval with docker grading, see Phase 3 work.
"""
from __future__ import annotations
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def load_swebench_lite() -> list[dict[str, Any]]:
    """Return the full SWE-bench Lite test split (300 tasks).

    Lazily imports `datasets` to avoid forcing the dep on consumers that
    only need the launcher stub.
    """
    from datasets import load_dataset  # local import: heavy
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    return [dict(row) for row in ds]


def load_bigcodebench_hard(limit: int = 300) -> list[dict[str, Any]]:
    """Return up to `limit` rows from BigCodeBench-Hard.

    BigCodeBench is harder than HumanEval/MBPP — closer to real-world tasks.
    See https://huggingface.co/datasets/bigcode/bigcodebench-hard.
    """
    from datasets import load_dataset  # local import: heavy
    ds = load_dataset("bigcode/bigcodebench-hard", split="v0.1.4")
    n = min(limit, len(ds))
    return [dict(row) for row in ds.select(range(n))]


def run_task_with_codeagent(
    task: dict[str, Any],
    ts_repo: Path,
    log_dir: Path,
    timeout_sec: int = 600,
    extra_env: dict[str, str] | None = None,
) -> bool:
    """Run the TS codeAgent on one task as a subprocess.

    Side effect: the TS side appends a trajectory line to
    `log_dir/<task_type>_<YYYYMMDD>.jsonl` (see `ts_logger_spec.md`).

    Args:
        task: One row from a task-set loader.
        ts_repo: Path to the user's TS codeAgent repo.
        log_dir: Where to write trajectory jsonl. Created if absent.
        timeout_sec: Subprocess wall-clock timeout.
        extra_env: Additional env vars (e.g. CODE_PRM_ROLLOUT_ID, CODE_PRM_RUN_ID)
            forwarded to the TS subprocess.

    Returns:
        True iff the subprocess exited 0. The agent-level pass/fail is in
        the jsonl, NOT in this return value. Returns False (instead of
        raising) on subprocess timeout — callers in batch mode should NOT
        crash the whole batch on one slow task.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # SWE-bench uses `instance_id`; BigCodeBench uses `task_id`.
    if "instance_id" in task:
        task_id = task["instance_id"]
        task_type = "swe-bench-lite"
    elif "task_id" in task:
        task_id = task["task_id"]
        task_type = "bigcodebench-hard"
    else:
        raise KeyError(
            f"Task dict has neither 'instance_id' nor 'task_id'. Keys: {list(task.keys())}"
        )

    env = os.environ.copy()
    env["CODE_PRM_LOG_DIR"] = str(log_dir)
    env["CODE_PRM_TASK_TYPE"] = task_type
    # Forward the full task payload so the TS side can extract problem statement etc.
    env["CODE_PRM_TASK_JSON"] = json.dumps(task)
    # Legacy alias — to be removed once TS code migrates to CODE_PRM_TASK_JSON.
    env["SWEBENCH_TASK_JSON"] = env["CODE_PRM_TASK_JSON"]
    if extra_env:
        env.update(extra_env)

    # TODO(user): Adjust this command to match your actual codeAgent CLI.
    # The placeholder assumes: `node <ts_repo>/dist/cli.js run --task-id X --task-type <T>`
    # If your CLI differs, change the argv here.
    cmd = [
        "node",
        str(ts_repo / "dist" / "cli.js"),
        "run",
        "--task-id", task_id,
        "--task-type", task_type,
    ]

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        # Don't crash the batch; just report process-level failure.
        return False
    except (FileNotFoundError, OSError):
        # node binary missing, or other OS-level launch failure.
        return False
    return result.returncode == 0


if __name__ == "__main__":
    # Smoke entry point: load tasks and print summary.
    tasks = load_swebench_lite()
    print(f"Loaded {len(tasks)} SWE-bench Lite tasks.")
    if tasks:
        first = tasks[0]
        print(f"First SWE-bench instance: {first.get('instance_id', '<missing>')}")
        print(f"Problem statement (first 200 chars):")
        print((first.get("problem_statement", "") or "")[:200])

    print()
    bc_tasks = load_bigcodebench_hard()
    print(f"Loaded {len(bc_tasks)} BigCodeBench-Hard tasks.")
    if bc_tasks:
        first_bc = bc_tasks[0]
        # BigCodeBench uses 'task_id' as identifier
        print(f"First BigCodeBench task: {first_bc.get('task_id', '<missing>')}")
