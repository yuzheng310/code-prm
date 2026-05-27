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
from pathlib import Path

from src.labeler.anthropic_client import RateLimitedClient
from src.labeler.trajectory_schema import Step, Trajectory
from src.utils.jsonl_io import append_trajectory, read_trajectories


def label_trajectory_simplified(traj: Trajectory) -> Trajectory:
    """Apply outcome-only simplification for outcome=0; leave outcome=1 untouched.

    Returns the same Trajectory object, mutated in place. Does NOT mark
    label_method (caller must do that explicitly).
    """
    if traj.outcome == 0:
        for s in traj.trajectory:
            s.step_label = 0.0
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

    Stamps `label_method = "llm_judge"` on every output trajectory for honesty.

    Args:
        input_path: Source jsonl of unlabeled trajectories.
        output_path: Destination jsonl. Overwrites if it already exists.
        client: Rate-limited client (typically wrapping Haiku for cost).
        K: Number of judge calls per tool step.
        only_tool_steps: If True, skip pure-thought steps (tool is None).
            They keep step_label = None.
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
                step.step_label = llm_judge_score_step(traj, i, client, K=K)
        traj.label_method = "llm_judge"
        append_trajectory(output_path, traj)
