"""Drive MC labeling across many jsonl files with per-file budget tracking.

Reads every *.jsonl in `--input_dir`, runs `label_file` on each, writes
labeled output to `--output_dir`. Halts when the shared CostTracker
exceeds budget. The actual API spend lives in `RateLimitedClient` /
`CostTracker` — this module just orchestrates files.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from src.labeler.anthropic_client import RateLimitedClient
from src.labeler.mc_labeler import label_file
from src.utils.cost_tracker import CostTracker


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--budget_usd", type=float, required=True)
    p.add_argument("--K", type=int, default=4, help="MC rollouts per step")
    p.add_argument(
        "--model",
        default="claude-haiku-4-5",
        help="Anthropic model used for the LLM-judge rollouts",
    )
    args = p.parse_args()

    tracker = CostTracker(budget_usd=args.budget_usd)
    client = RateLimitedClient(tracker, model=args.model)

    files = sorted(args.input_dir.glob("*.jsonl"))
    if not files:
        print(f"WARN: no *.jsonl files in {args.input_dir}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        out = args.output_dir / f.name
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
