#!/usr/bin/env python3
"""Resume a partially-written labeled jsonl tmp file and finalize it.

This exists because label_file writes atomically to <output>.tmp and only
promotes to the final jsonl after the whole source file finishes. If a long
labeling run dies late, we can reuse the completed prefix instead of paying to
relabel the entire raw file.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.labeler.anthropic_client import RateLimitedClient  # noqa: E402
from src.labeler.step_labeler import label_trajectory_simplified, llm_judge_score_step  # noqa: E402
from src.labeler.trajectory_schema import Trajectory  # noqa: E402
from src.utils.cost_tracker import CostTracker  # noqa: E402
from src.utils.jsonl_io import append_trajectory, read_trajectories  # noqa: E402


def _identity(traj: Trajectory) -> tuple[str, str | None, int, int]:
    return (traj.task_id, traj.run_id, traj.rollout_id, traj.outcome)


def resume_label_file(
    *,
    input_path: Path,
    output_path: Path,
    tmp_path: Path,
    client: Any,
    tracker: Any,
    K: int,
    model: str,
    only_tool_steps: bool = True,
) -> None:
    raw = list(read_trajectories(input_path))
    partial = list(read_trajectories(tmp_path))
    if len(partial) > len(raw):
        raise ValueError(f"tmp has {len(partial)} rows but raw has only {len(raw)}")

    for idx, partial_traj in enumerate(partial):
        raw_traj = raw[idx]
        if _identity(partial_traj) != _identity(raw_traj):
            raise ValueError(
                f"tmp row {idx} does not match raw prefix: tmp={_identity(partial_traj)} raw={_identity(raw_traj)}"
            )

    total_steps_to_label = 0
    for traj in raw[len(partial):]:
        if traj.outcome != 1:
            continue
        for step in traj.trajectory:
            if only_tool_steps and step.tool is None:
                continue
            total_steps_to_label += 1

    done_steps = 0
    for idx, traj in enumerate(raw[len(partial):], start=len(partial) + 1):
        if traj.outcome == 0:
            label_trajectory_simplified(traj, only_tool_steps=only_tool_steps)
            traj.label_method = "outcome_zero_simplification"
            print(
                f"[resume traj {idx}/{len(raw)}] task={traj.task_id} outcome=0 -> simplification",
                flush=True,
            )
        else:
            step_count = 0
            for step_idx, step in enumerate(traj.trajectory):
                if only_tool_steps and step.tool is None:
                    continue
                step.step_label = llm_judge_score_step(traj, step_idx, client, K=K)
                step_count += 1
                done_steps += 1
                pct = done_steps / max(total_steps_to_label, 1) * 100
                print(
                    f"  step {step_idx} (tool={step.tool}): label={step.step_label:.2f} "
                    f"[{done_steps}/{total_steps_to_label}={pct:.0f}%]",
                    flush=True,
                )
            traj.label_method = "llm_judge"
            print(
                f"[resume traj {idx}/{len(raw)}] task={traj.task_id} outcome=1 -> llm_judge {step_count} steps",
                flush=True,
            )
        append_trajectory(tmp_path, traj)

    tmp_path.replace(output_path)
    manifest = {
        "tool": "scripts.21_resume_label_file",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "K": K,
        "model": model,
        "resumed_rows": len(partial),
        "completed_rows": len(raw),
        "cost_per_model": getattr(tracker, "per_model", {}),
        "total_cost_usd": getattr(tracker, "total_usd", 0.0),
        "skipped_files": [],
        "processed_files": [{"input": str(input_path.resolve()), "output": str(output_path.resolve())}],
    }
    (output_path.parent / "labeling_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--model", default="claude-opus-4-7")
    args = parser.parse_args()

    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    if not tmp_path.exists():
        raise SystemExit(f"tmp file does not exist: {tmp_path}")
    tracker = CostTracker(budget_usd=1_000_000)
    client = RateLimitedClient(tracker, model=args.model)
    resume_label_file(
        input_path=args.input,
        output_path=args.output,
        tmp_path=tmp_path,
        client=client,
        tracker=tracker,
        K=args.K,
        model=args.model,
    )
    print(f"Resumed labeling complete: {args.output}")


if __name__ == "__main__":
    main()
