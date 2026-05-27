"""Drive step labeling across many jsonl files with per-file budget tracking.

Reads every *.jsonl under `--input_dir` (recursive), runs `label_file` on
each, writes labeled output to `--output_dir`. Halts when the shared
CostTracker exceeds budget. The actual API spend lives in
`RateLimitedClient` / `CostTracker`.

Output filenames preserve the input's relative subpath flattened with "__"
to avoid collisions when nested directories contain same-named files
(e.g. rollout_0/foo.jsonl + rollout_1/foo.jsonl).

Also performs a pre-flight `task_prompt` coverage check on the INPUT
trajectories. If too few have `task_prompt`, the LLM judge has no problem
statement to anchor on and label quality will be near-random. Default
threshold: 90% of outcome=1 trajectories must have a non-empty
`task_prompt`. Warns on stdout and continues (does not abort) so the user
can decide whether to proceed.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from src.labeler.anthropic_client import RateLimitedClient
from src.labeler.step_labeler import label_file
from src.utils.cost_tracker import CostTracker
from src.utils.jsonl_io import read_trajectories


def _flatten_relative_name(input_dir: Path, file: Path) -> str:
    """Return a unique flat filename: 'rollout_0__swe-bench-lite_20260527.jsonl'."""
    rel = file.relative_to(input_dir)
    return "__".join(rel.parts)


def task_prompt_coverage(files: list[Path]) -> tuple[int, int, int]:
    """Return (n_outcome_one, n_with_prompt, n_total) by scanning input jsonl.

    `task_prompt` is critical to LLM-judge label quality. Missing it means
    the judge has no problem statement to anchor on.
    """
    n_outcome_one = 0
    n_with_prompt = 0
    n_total = 0
    for f in files:
        for t in read_trajectories(f):
            n_total += 1
            if t.outcome == 1:
                n_outcome_one += 1
                if t.task_prompt and t.task_prompt.strip():
                    n_with_prompt += 1
    return n_outcome_one, n_with_prompt, n_total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--budget_usd", type=float, required=True)
    p.add_argument("--K", type=int, default=4, help="LLM-judge calls per step")
    p.add_argument(
        "--model",
        default="claude-haiku-4-5",
        help="Anthropic model used for the LLM-judge rollouts",
    )
    args = p.parse_args()

    tracker = CostTracker(budget_usd=args.budget_usd)
    client = RateLimitedClient(tracker, model=args.model)

    files = sorted(args.input_dir.rglob("*.jsonl"))
    if not files:
        print(f"WARN: no *.jsonl files (recursive) in {args.input_dir}")
        return

    # Pre-flight: task_prompt coverage on outcome=1 trajectories.
    n_one, n_prompt, n_total = task_prompt_coverage(files)
    if n_one > 0:
        coverage = n_prompt / n_one * 100.0
        print(f"task_prompt coverage on outcome=1 trajectories: "
              f"{n_prompt}/{n_one} = {coverage:.1f}%  (total trajectories: {n_total})")
        if coverage < 90.0:
            print(
                "\n[!] task_prompt coverage is LOW. The LLM judge will see only\n"
                "    tool traces, not the problem statement, and labels may be\n"
                "    near-random. Recommended: fix the TS logger to populate\n"
                "    task_prompt (see src/collector/ts_logger_spec.md) and\n"
                "    re-run collection. Continuing anyway in 5 seconds — Ctrl-C\n"
                "    to abort.\n"
            )
            import time
            time.sleep(5)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        out = args.output_dir / _flatten_relative_name(args.input_dir, f)
        print(f"Labeling {f}  ->  {out}")
        try:
            label_file(f, out, client, K=args.K)
        except Exception as e:
            print(f"  ERROR on {f}: {e}")
            if tracker.over_budget():
                print("Over budget — stopping.")
                break
            continue
        print(f"  cost so far: {tracker}")
        if tracker.over_budget():
            print("OVER BUDGET — stopping.")
            break

    print(f"\nFINAL: {tracker}")


if __name__ == "__main__":
    main()
