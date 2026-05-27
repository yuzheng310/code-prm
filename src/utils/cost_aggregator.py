"""Aggregate REAL cost from trajectory token_usage records.

The collection driver (`src.eval.collect_batch`) reports only an *estimated*
budget. The TS codeAgent records actual `token_usage` per trajectory. This
module sums those up to give the true spend after collection.

Usage:
    python -m src.utils.cost_aggregator --dir data/raw/swebench-lite
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path

from src.labeler.trajectory_schema import Trajectory
from src.utils.jsonl_io import read_trajectories


def aggregate(input_dir: Path) -> dict[str, float | int]:
    """Walk all *.jsonl under input_dir (recursive) and sum token usage."""
    total_in = 0
    total_out = 0
    total_cache_read = 0
    total_cache_create = 0
    total_cost = 0.0
    n_with_usage = 0
    n_total = 0
    by_model: Counter[str] = Counter()

    for f in sorted(input_dir.rglob("*.jsonl")):
        for t in read_trajectories(f):
            n_total += 1
            by_model[t.policy_model] += 1
            if t.token_usage is not None:
                n_with_usage += 1
                total_in += t.token_usage.input_tokens
                total_out += t.token_usage.output_tokens
                total_cache_read += t.token_usage.cache_read_tokens
                total_cache_create += t.token_usage.cache_creation_tokens
                total_cost += t.token_usage.cost_usd

    return {
        "n_trajectories": n_total,
        "n_with_usage": n_with_usage,
        "coverage_pct": (n_with_usage / n_total * 100.0) if n_total > 0 else 0.0,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_cache_read_tokens": total_cache_read,
        "total_cache_creation_tokens": total_cache_create,
        "total_cost_usd": round(total_cost, 4),
        "by_model": dict(by_model),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, required=True,
                   help="Root directory of *.jsonl files (searched recursively)")
    args = p.parse_args()

    if not args.dir.exists():
        print(f"ERROR: {args.dir} does not exist")
        raise SystemExit(1)

    summary = aggregate(args.dir)
    print(f"Real-cost aggregation for {args.dir}:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:30s} {v:>12,.4f}")
        elif isinstance(v, int):
            print(f"  {k:30s} {v:>12,}")
        else:
            print(f"  {k:30s} {v}")

    if summary["coverage_pct"] < 80:
        print(
            f"\n[!] Only {summary['coverage_pct']:.1f}% of trajectories had token_usage. "
            "Check that the TS logger is recording it (see ts_logger_spec.md)."
        )


if __name__ == "__main__":
    main()
