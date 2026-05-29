"""SWE-bench Lite / BigCodeBench-Hard task loaders + TS codeAgent launcher.

This module is NOT a full SWE-bench harness. It does TWO things:

1. Load task lists from the public benchmarks (via HuggingFace `datasets`).
2. Launch the user's TypeScript codeAgent as a subprocess on a single task,
   passing env vars so the TS side can:
       - emit a trajectory jsonl line (CODE_PRM_LOG_DIR)
       - tag the trajectory with rollout_id / run_id (CODE_PRM_*)
       - receive the task payload (CODE_PRM_TASK_JSON)

OUTCOME ATTRIBUTION (read this carefully):
   The pass/fail "outcome" stored in the trajectory is decided by the TS
   codeAgent, NOT by this Python module. The TS side is responsible for
   running the task's tests (e.g. via SWE-bench docker harness or pytest
   directly) and writing the `outcome` int into the jsonl.

   This module only reports a process-level status (`TaskRunStatus`)
   which is distinct from agent-task success.

For a full SWE-bench eval with docker grading, see Phase 3 work.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, IO, Literal


def _safe_path_component(value: Any) -> str:
    text = str(value)
    safe = "".join(ch if (ch.isalnum() or ch in {"_", "."}) else "_" for ch in text)
    return safe.strip("._") or "unknown"


def _bigcodebench_prompt(task: dict[str, Any], task_id: str) -> str:
    instruction = task.get("instruct_prompt") or task.get("prompt") or f"Solve task {task_id}"
    code_prompt = task.get("code_prompt")
    entry_point = task.get("entry_point")

    parts = [
        "IMPORTANT: Write your final solution to a file named `task.py` "
        "in the current working directory.",
    ]
    if isinstance(entry_point, str) and entry_point:
        parts.append(
            f"The grader imports your solution as `from task import {entry_point}`; "
            f"implement entry point `{entry_point}` exactly."
        )
    else:
        parts.append("The grader imports your solution from `task.py`.")
    parts.append("")
    parts.append(str(instruction))
    if isinstance(code_prompt, str) and code_prompt:
        parts.extend(["", "Use this exact function signature/stub:", "```python", code_prompt, "```"])
    return "\n".join(parts)


# Process-level launch outcome. NOT the same as agent-task pass/fail.
# - "ok"           — subprocess exited 0 (TS side ran to completion)
# - "failed"       — subprocess exited non-zero (TS side error)
# - "timeout"      — exceeded timeout_sec; subprocess was killed
# - "launch_error" — node binary missing / OS-level launch failure;
#                    this is a CONFIG bug, not a task failure, and
#                    callers should usually abort the batch.
TaskRunStatus = Literal["ok", "failed", "timeout", "launch_error"]


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
    stream_output: bool = False,
) -> TaskRunStatus:
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
        stream_output: If true, print the TS subprocess stdout/stderr as it runs.

    Returns:
        One of `TaskRunStatus`. The agent-level pass/fail is in the jsonl,
        NOT in this return. "timeout" is recoverable per task; "launch_error"
        signals a config bug that the caller should treat as a hard stop
        (e.g. `node` binary missing or TS repo not built).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_dir = log_dir.resolve()
    ts_repo = ts_repo.resolve()

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

    work_dir: Path | None = None
    if task_type == "bigcodebench-hard":
        rollout_id = env.get("CODE_PRM_ROLLOUT_ID", "0")
        run_id = env.get("CODE_PRM_RUN_ID", "run")
        work_dir = (
            log_dir
            / "_workdirs"
            / (
                f"{_safe_path_component(task_id)}"
                f"__rollout_{_safe_path_component(rollout_id)}"
                f"__{_safe_path_component(run_id)}"
            )
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        env["CODE_PRM_WORK_DIR"] = str(work_dir)
        problem_text = _bigcodebench_prompt(task, str(task_id))
    else:
        problem_text = (
            task.get("problem_statement")
            or task.get("prompt")
            or task.get("instruct_prompt")
            or f"Solve task {task_id}"
        )

    # The default target is pi (github.com/earendil-works/pi) — pi's CLI is
    # `node <pi-coding-agent>/dist/cli.js`. Set TS_REPO_PATH to point at
    # `<pi-clone>/packages/coding-agent`. For a different agent, override
    # this argv construction; the rest of the pipeline only depends on the
    # subprocess writing a trajectory line to $CODE_PRM_LOG_DIR per
    # ts_logger_spec.md.
    cmd = [
        "node",
        str(ts_repo / "dist" / "cli.js"),
        "-p", problem_text,           # pi: -p / --prompt for non-interactive single prompt
    ]

    try:
        if stream_output:
            rollout_id = env.get("CODE_PRM_ROLLOUT_ID", "0")
            return _run_streaming_subprocess(
                cmd=cmd,
                env=env,
                timeout_sec=timeout_sec,
                output_prefix=f"{task_id} rollout={rollout_id}",
                cwd=work_dir,
            )
        run_kwargs = dict(env=env, capture_output=True, text=True, timeout=timeout_sec)
        if work_dir is not None:
            run_kwargs["cwd"] = work_dir
        result = subprocess.run(cmd, **run_kwargs)
    except subprocess.TimeoutExpired:
        # Don't crash the batch; per-task timeout is recoverable.
        return "timeout"
    except (FileNotFoundError, PermissionError, OSError):
        # `node` binary missing, or other OS-level launch failure.
        # This is a CONFIG bug — caller should abort the batch, not paper over it.
        return "launch_error"
    return "ok" if result.returncode == 0 else "failed"


def _run_streaming_subprocess(
    cmd: list[str],
    env: dict[str, str],
    timeout_sec: int,
    output_prefix: str,
    cwd: Path | None = None,
) -> TaskRunStatus:
    """Run a subprocess while streaming stdout/stderr with task context."""

    def pump(pipe: IO[str] | None, stream: IO[str], channel: str) -> None:
        if pipe is None:
            return
        try:
            for line in iter(pipe.readline, ""):
                print(f"[{output_prefix} {channel}] {line}", end="", file=stream, flush=True)
        finally:
            pipe.close()

    try:
        popen_kwargs = dict(
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if cwd is not None:
            popen_kwargs["cwd"] = cwd
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except (FileNotFoundError, PermissionError, OSError):
        return "launch_error"

    threads = [
        threading.Thread(target=pump, args=(proc.stdout, sys.stdout, "stdout"), daemon=True),
        threading.Thread(target=pump, args=(proc.stderr, sys.stderr, "stderr"), daemon=True),
    ]
    for thread in threads:
        thread.start()

    started_at = time.monotonic()
    timed_out = False
    while proc.poll() is None:
        if time.monotonic() - started_at >= timeout_sec:
            timed_out = True
            proc.kill()
            break
        time.sleep(0.1)

    for thread in threads:
        thread.join(timeout=1)

    if timed_out:
        return "timeout"
    return "ok" if proc.returncode == 0 else "failed"


if __name__ == "__main__":
    # Smoke entry point: load tasks and print summary.
    tasks = load_swebench_lite()
    print(f"Loaded {len(tasks)} SWE-bench Lite tasks.")
    if tasks:
        first = tasks[0]
        print(f"First SWE-bench instance: {first.get('instance_id', '<missing>')}")
        print("Problem statement (first 200 chars):")
        print((first.get("problem_statement", "") or "")[:200])

    print()
    bc_tasks = load_bigcodebench_hard()
    print(f"Loaded {len(bc_tasks)} BigCodeBench-Hard tasks.")
    if bc_tasks:
        first_bc = bc_tasks[0]
        # BigCodeBench uses 'task_id' as identifier
        print(f"First BigCodeBench task: {first_bc.get('task_id', '<missing>')}")
