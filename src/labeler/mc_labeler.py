"""MC rollout label generator.

For each step in an outcome=1 trajectory, re-roll K times from that step
using Haiku, ask the LLM to predict the final outcome, count predicted
successes, and assign mc_i = successes / K.

For outcome=0 trajectories, simplification: set mc_i = 0 for all steps
(Math-Shepherd simplification — avoids noisy MC on failure paths).

The "rollout" here is a LIGHTWEIGHT LLM-judge surrogate, not literal tool
re-execution. Phase 2 may upgrade to real tool re-execution.
"""
from __future__ import annotations
import json
from pathlib import Path

from src.labeler.anthropic_client import RateLimitedClient
from src.labeler.trajectory_schema import Step, Trajectory
from src.utils.jsonl_io import append_trajectory, read_trajectories


def label_trajectory_simplified(traj: Trajectory) -> Trajectory:
    """Apply Math-Shepherd simplification for outcome=0; leave outcome=1 unchanged.

    Returns the same Trajectory object, mutated in place for convenience.
    """
    if traj.outcome == 0:
        for s in traj.trajectory:
            s.mc_label = 0.0
    return traj


def _build_continuation_prompt(prefix: list[Step], task_id: str) -> str:
    """Render the partial trajectory as a prompt asking the LLM to predict outcome."""
    lines = [f"Task: {task_id}", "Trajectory so far:"]
    for s in prefix:
        tool_args_repr = json.dumps(s.tool_args)[:200]
        lines.append(
            f"  Step {s.step}: {s.tool}({tool_args_repr}) "
            f"-> {s.tool_result[:200]}"
        )
    lines.append("")
    lines.append(
        "Given this partial trajectory, predict the final outcome. "
        "Reply with exactly one line: either 'OUTCOME: PASS' or 'OUTCOME: FAIL', "
        "followed by a brief justification (1-2 sentences)."
    )
    return "\n".join(lines)


def _parses_as_successful(text: str) -> bool:
    """Detect whether the LLM judged the partial trajectory as likely to PASS."""
    return "OUTCOME: PASS" in text.upper()


def mc_rollout_for_step(
    traj: Trajectory,
    step_idx: int,
    client: RateLimitedClient,
    K: int = 4,
) -> float:
    """Estimate mc_label for one step via K LLM-judge rollouts.

    Returns successes / K, a value in {0/K, 1/K, ..., K/K}.
    """
    prefix = traj.trajectory[: step_idx + 1]
    prompt = _build_continuation_prompt(prefix, traj.task_id)
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

    Args:
        input_path: Source jsonl of unlabeled trajectories.
        output_path: Destination jsonl. Overwrites if it already exists.
        client: Rate-limited client (typically wrapping Haiku for cost).
        K: Number of MC rollouts per tool step.
        only_tool_steps: If True, skip pure-thought steps (tool is None).
            They keep mc_label = None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    for traj in read_trajectories(input_path):
        if traj.outcome == 0:
            label_trajectory_simplified(traj)
        else:
            for i, step in enumerate(traj.trajectory):
                if only_tool_steps and step.tool is None:
                    continue
                step.mc_label = mc_rollout_for_step(traj, i, client, K=K)
        append_trajectory(output_path, traj)
