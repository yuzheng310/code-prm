"""SWE-bench Lite task loader + codeAgent invoker.

Two responsibilities:
1. Load the SWE-bench Lite task list (300 Python bug-fix tasks).
2. Invoke the user's TypeScript codeAgent on a single task as a subprocess,
   passing CODE_PRM_LOG_DIR so the TS side writes trajectory jsonl.

The actual pass/fail outcome is written by the TS side into the jsonl
(per `src/collector/ts_logger_spec.md`). This module's return value
indicates only whether the subprocess succeeded.
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
    only need the runner stub.
    """
    from datasets import load_dataset  # local import: heavy
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    return [dict(row) for row in ds]


def run_task_with_codeagent(
    task: dict[str, Any],
    ts_repo: Path,
    log_dir: Path,
    timeout_sec: int = 600,
) -> bool:
    """Run the TS codeAgent on one SWE-bench Lite task.

    Side effect: the TS side appends a trajectory line to
    `log_dir/<task_type>_<YYYYMMDD>.jsonl` (see `ts_logger_spec.md`).

    Args:
        task: One row from `load_swebench_lite()`.
        ts_repo: Path to the user's TS codeAgent repo.
        log_dir: Where to write trajectory jsonl. Will be created if absent.
        timeout_sec: Subprocess wall-clock timeout.

    Returns:
        True iff the subprocess exited with code 0. The actual test
        outcome lives inside the jsonl line — read that to know.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CODE_PRM_LOG_DIR"] = str(log_dir)
    env["SWEBENCH_TASK_JSON"] = json.dumps(task)

    # TODO(user): Adjust this command to match your actual codeAgent CLI.
    # The placeholder assumes: `node <ts_repo>/dist/cli.js run --task-id X --task-type swe-bench-lite`
    # If your CLI differs, change the argv here.
    cmd = [
        "node",
        str(ts_repo / "dist" / "cli.js"),
        "run",
        "--task-id", task["instance_id"],
        "--task-type", "swe-bench-lite",
    ]

    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=timeout_sec,
    )
    return result.returncode == 0


if __name__ == "__main__":
    # Smoke entry point: load tasks and print summary.
    tasks = load_swebench_lite()
    print(f"Loaded {len(tasks)} SWE-bench Lite tasks.")
    if tasks:
        first = tasks[0]
        print(f"First instance: {first.get('instance_id', '<missing>')}")
        print(f"Problem statement (first 200 chars):")
        print((first.get("problem_statement", "") or "")[:200])
