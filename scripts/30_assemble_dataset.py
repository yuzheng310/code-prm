#!/usr/bin/env python3
"""Combine all labeled trajectories into train/val/test split.

Reads from `data/labeled/{swebench-lite,bigcodebench-hard}/*.jsonl`,
deterministically shuffles, and writes to `data/code-trajectory-2.4k/`.

Reports per-split outcome balance to surface any drift between splits.

Run from project root:
    python scripts/30_assemble_dataset.py
"""
from __future__ import annotations
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.labeler.trajectory_schema import Trajectory  # noqa: E402
from src.utils.jsonl_io import read_trajectories, write_trajectories  # noqa: E402


def collect_all(input_dirs: list[Path]) -> list[Trajectory]:
    """Read every *.jsonl under each input dir (recursive) and concatenate."""
    all_trajs: list[Trajectory] = []
    for d in input_dirs:
        if not d.exists():
            print(f"  WARN: input dir {d} does not exist, skipping")
            continue
        files = sorted(d.rglob("*.jsonl"))
        for f in files:
            n_before = len(all_trajs)
            all_trajs.extend(read_trajectories(f))
            print(f"  + {f}: {len(all_trajs) - n_before} trajectories")
    return all_trajs


def split(
    trajectories: list[Trajectory],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[Trajectory], list[Trajectory], list[Trajectory]]:
    """Deterministic shuffle + slice into (train, val, test)."""
    rng = random.Random(seed)
    shuffled = list(trajectories)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return train, val, test


def report(split_name: str, data: list[Trajectory]) -> None:
    if not data:
        print(f"  {split_name}: EMPTY")
        return
    n_steps = sum(len(t.trajectory) for t in data)
    pass_rate = sum(t.outcome for t in data) / len(data)
    types = Counter(t.task_type for t in data)
    print(
        f"  {split_name}: {len(data)} trajectories, "
        f"{n_steps} steps, pass_rate={pass_rate:.2%}, types={dict(types)}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input_dirs",
        nargs="+",
        type=Path,
        default=[
            Path("data/labeled/swebench-lite"),
            Path("data/labeled/bigcodebench-hard"),
        ],
        help="Directories containing labeled *.jsonl files",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/code-trajectory-2.4k"),
    )
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--test_frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"Reading from: {[str(d) for d in args.input_dirs]}")
    all_trajs = collect_all(args.input_dirs)
    print(f"\nTotal collected: {len(all_trajs)} trajectories")

    if not all_trajs:
        print("ERROR: no trajectories found. Did labeling run?")
        sys.exit(1)

    train, val, test = split(all_trajs, args.val_frac, args.test_frac, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_trajectories(args.output_dir / "train.jsonl", train)
    write_trajectories(args.output_dir / "val.jsonl", val)
    write_trajectories(args.output_dir / "test.jsonl", test)

    print(f"\nWrote splits to {args.output_dir}/")
    print(f"  train.jsonl  ({len(train)})")
    print(f"  val.jsonl    ({len(val)})")
    print(f"  test.jsonl   ({len(test)})")
    print("\nPer-split stats:")
    for name, data in [("train", train), ("val", val), ("test", test)]:
        report(name, data)


if __name__ == "__main__":
    main()
