"""Step-level label generator (LLM-judge surrogate, Phase 1).

PHASE 1 LABELING SEMANTICS:

For each step in an outcome=1 trajectory, this module asks an LLM (Haiku by
default) to judge whether the partial trajectory looks likely to succeed.
It runs the judge K times at temperature > 0 and records the success
fraction as `step_label` in [0, 1].

THIS IS NOT MONTE-CARLO ROLLOUT. Real MC rollout would require:
  1. Restoring agent state at step i (repo snapshot, prior outputs)
  2. Re-running the agent from step i with stochastic decoding
  3. Running the real test suite K times
  4. Recording the empirical success rate

This module's surrogate is much cheaper but is a WEAK SUPERVISION signal
biased by the judge's calibration. We mark trained labels as
`label_method="llm_judge"` for downstream honesty.

For outcome=0 trajectories, simplification: all steps get `step_label = 0`
(Math-Shepherd outcome-only simplification — avoids noisy judge calls on
failure paths).

Phase 2 future work: upgrade to real MC rollout via repo replay + sandboxed
test execution. The Trajectory.replay fields (`repo`, `base_commit`,
`final_diff`) are already in the schema to support this.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

from src.labeler.anthropic_client import RateLimitedClient
from src.labeler.trajectory_schema import Step, Trajectory
from src.utils.jsonl_io import append_trajectory, read_trajectories


# Strict line-anchored match. Tolerates leading whitespace and any trailing text
# on the SAME line (justification) but requires the OUTCOME: PASS marker to be
# at the start of a line, not embedded in narrative text like
# "the trajectory is not OUTCOME: PASS".
_PASS_RE = re.compile(r"(?im)^\s*OUTCOME:\s*PASS\b")


def label_trajectory_simplified(
    traj: Trajectory,
    only_tool_steps: bool = True,
) -> Trajectory:
    """Apply outcome-only simplification for outcome=0; leave outcome=1 untouched.

    The `only_tool_steps` flag MUST match what the outcome=1 path uses
    (default True), otherwise pure-thought steps end up with asymmetric
    treatment between success and failure trajectories (failure: label=0,
    success: label=None) — which silently biases Phase 2 training.

    Returns the same Trajectory object, mutated in place. Does NOT mark
    label_method (caller must do that explicitly).
    """
    if traj.outcome == 0:
        for s in traj.trajectory:
            if only_tool_steps and s.tool is None:
                continue
            s.step_label = 0.0
    return traj


def _build_continuation_prompt(
    prefix: list[Step],
    task_id: str,
    task_prompt: str | None = None,
    task_type: str | None = None,
) -> str:
    """Render the partial trajectory + problem statement as a judge prompt.

    The task description (`task_prompt`) is crucial: without it the judge
    sees only tool calls and cannot tell whether the trajectory is solving
    the right problem. The TS logger is responsible for populating
    `Trajectory.task_prompt` from the benchmark's problem statement.
    """
    lines: list[str] = []
    if task_type:
        lines.append(f"Task type: {task_type}")
    lines.append(f"Task id: {task_id}")
    if task_prompt:
        # Keep the problem statement bounded to avoid prompt bloat.
        snippet = task_prompt if len(task_prompt) <= 3000 else (
            task_prompt[:1500] + "\n...[TRUNC]...\n" + task_prompt[-1500:]
        )
        lines.append("")
        lines.append("Problem statement:")
        lines.append(snippet)

    lines.append("")
    lines.append("Trajectory so far:")
    for s in prefix:
        tool_args_repr = json.dumps(s.tool_args)[:200]
        lines.append(
            f"  Step {s.step}: {s.tool}({tool_args_repr}) "
            f"-> {s.tool_result[:200]}"
        )

    lines.append("")
    lines.append(
        "Given this partial trajectory and the problem statement above, "
        "predict whether the trajectory will ultimately PASS the task's tests. "
        "Reply with exactly one line whose FIRST line is either:\n"
        "  OUTCOME: PASS\n"
        "or\n"
        "  OUTCOME: FAIL\n"
        "Optionally append a brief justification (1-2 sentences) AFTER the verdict line."
    )
    return "\n".join(lines)


def _parses_as_successful(text: str) -> bool:
    """Detect whether the LLM judged the partial trajectory as likely to PASS.

    Uses a line-anchored regex so that narrative text like
    "not OUTCOME: PASS at this step" does not register as PASS.
    """
    return _PASS_RE.search(text) is not None


def llm_judge_score_step(
    traj: Trajectory,
    step_idx: int,
    client: RateLimitedClient,
    K: int = 4,
) -> float:
    """Estimate step_label for one step via K LLM-judge calls (NOT real MC).

    Returns successes / K, a value in {0/K, 1/K, ..., K/K}.
    """
    prefix = traj.trajectory[: step_idx + 1]
    prompt = _build_continuation_prompt(
        prefix,
        task_id=traj.task_id,
        task_prompt=traj.task_prompt,
        task_type=traj.task_type,
    )
    successes = 0
    for _ in range(K):
        text, _, _ = client.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.9,
        )
        if _parses_as_successful(text):
            successes += 1
    return successes / K


def label_file(
    input_path: Path,
    output_path: Path,
    client: RateLimitedClient,
    K: int = 4,
    only_tool_steps: bool = True,
) -> None:
    """Label every trajectory in `input_path` jsonl, write to `output_path` jsonl.

    Stamps `label_method` per trajectory:
    - outcome=1 path  -> "llm_judge"          (real judge calls made)
    - outcome=0 path  -> "outcome_zero_simplification"

    Atomic write: writes to a `.tmp` sibling and replaces on success. If
    ANY exception is raised mid-stream (API error, budget exceeded, schema
    error), the `.tmp` is removed and `output_path` is left unchanged.
    Downstream readers (label_all, assembly) thus never see partial files.

    Args:
        input_path: Source jsonl of unlabeled trajectories.
        output_path: Destination jsonl. Overwrites atomically on success.
        client: Rate-limited client (typically wrapping Haiku for cost).
        K: Number of judge calls per tool step.
        only_tool_steps: If True, skip pure-thought steps (tool is None) in
            BOTH success and failure paths. They keep step_label = None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        for traj in read_trajectories(input_path):
            if traj.outcome == 0:
                label_trajectory_simplified(traj, only_tool_steps=only_tool_steps)
                traj.label_method = "outcome_zero_simplification"
            else:
                for i, step in enumerate(traj.trajectory):
                    if only_tool_steps and step.tool is None:
                        continue
                    step.step_label = llm_judge_score_step(traj, i, client, K=K)
                traj.label_method = "llm_judge"
            append_trajectory(tmp_path, traj)
    except BaseException:
        # Includes BudgetExceededError, KeyboardInterrupt, etc.
        # Drop the partial file so it never leaks into the dataset.
        tmp_path.unlink(missing_ok=True)
        raise

    # Atomic rename: either output_path is the new complete file, or
    # output_path is unchanged (we never write a partial output_path).
    tmp_path.replace(output_path)
