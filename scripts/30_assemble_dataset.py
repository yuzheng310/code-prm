#!/usr/bin/env python3
"""Combine all labeled trajectories into train/val/test split.

Reads from `data/labeled/{swebench-lite,bigcodebench-hard}/*.jsonl`,
deterministically shuffles, and writes to `data/code-trajectory-2.4k/`.

Before writing the final dataset, runs HARD CHECKS against Phase 1 exit
criteria. By default these fail fast — use `--skip_checks` to bypass
(useful for debugging or for the pilot stage where coverage is low).

Run from project root:
    python scripts/30_assemble_dataset.py
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.labeler.trajectory_schema import Trajectory  # noqa: E402
from src.utils.jsonl_io import read_trajectories, write_trajectories  # noqa: E402


# --- collection ---


def collect_all(input_dirs: list[Path]) -> list[Trajectory]:
    """Read every *.jsonl under each input dir (recursive) and concatenate.

    Excludes any file named `labeling_manifest.json` (not a jsonl).
    """
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


def inspect_manifests(
    input_dirs: list[Path],
    allow_skipped: bool = False,
) -> list[CheckResult]:
    """Read every `labeling_manifest.json` under each input dir, sanity-check.

    For each manifest:
    - print summary (when run, what model, K, task_prompt coverage achieved)
    - check that processed_files' outputs all exist on disk
    - check that skipped_files is empty (unless allow_skipped=True)
    - fail if no manifest is present in a labeled dir
    """
    results: list[CheckResult] = []
    for d in input_dirs:
        manifest_path = d / "labeling_manifest.json"
        if not manifest_path.exists():
            results.append(CheckResult(
                name=f"manifest present in {d.name}",
                passed=False,
                detail=f"no labeling_manifest.json — did label_all run "
                       f"successfully against {d}?",
            ))
            continue
        try:
            data = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            results.append(CheckResult(
                name=f"manifest valid JSON in {d.name}",
                passed=False,
                detail=f"manifest unreadable: {e!r}",
            ))
            continue
        # Print summary
        print(
            f"  Manifest in {d.name}: started={data.get('started_at')}, "
            f"K={data.get('K')}, model={data.get('model')}, "
            f"task_prompt_coverage={data.get('task_prompt_coverage')}, "
            f"total_cost_usd={data.get('total_cost_usd')}"
        )
        missing_outputs: list[str] = []
        for entry in data.get("processed_files", []):
            out_path = Path(entry["output"])
            if not out_path.exists():
                missing_outputs.append(str(out_path))
        results.append(CheckResult(
            name=f"manifest outputs exist in {d.name}",
            passed=len(missing_outputs) == 0,
            detail=(
                f"all {len(data.get('processed_files', []))} outputs present"
                if not missing_outputs
                else f"{len(missing_outputs)} output(s) missing (first: {missing_outputs[:3]})"
            ),
        ))
        # Hard check on skipped_files unless explicitly allowed.
        skipped = data.get("skipped_files", [])
        if skipped and not allow_skipped:
            sample = [s.get("input", "?") for s in skipped[:3]]
            results.append(CheckResult(
                name=f"no skipped files in {d.name}",
                passed=False,
                detail=(
                    f"{len(skipped)} input file(s) were skipped by label_all "
                    f"(first: {sample}). Re-run labeling on those inputs, "
                    "or pass --allow_skipped_in_manifest if you accept "
                    "incomplete labeling."
                ),
            ))
        elif skipped and allow_skipped:
            # Surface it but don't fail
            results.append(CheckResult(
                name=f"no skipped files in {d.name}",
                passed=True,
                detail=f"{len(skipped)} skipped — allowed via flag",
            ))
    return results


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


# --- exit-criteria checks ---


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def check_label_method_set(trajectories: list[Trajectory]) -> CheckResult:
    missing = [t.task_id for t in trajectories if t.label_method is None]
    return CheckResult(
        name="label_method set on every trajectory",
        passed=len(missing) == 0,
        detail=(
            "all trajectories tagged"
            if not missing
            else f"{len(missing)} trajectories missing label_method "
                 f"(first: {missing[:3]})"
        ),
    )


def check_task_prompt_coverage(
    trajectories: list[Trajectory],
    threshold: float = 0.95,
) -> CheckResult:
    outcome_one = [t for t in trajectories if t.outcome == 1]
    with_prompt = [
        t for t in outcome_one if t.task_prompt and t.task_prompt.strip()
    ]
    coverage = len(with_prompt) / max(len(outcome_one), 1)
    return CheckResult(
        name=f"task_prompt coverage on outcome=1 >= {threshold:.0%}",
        passed=coverage >= threshold,
        detail=f"{len(with_prompt)}/{len(outcome_one)} = {coverage:.1%}",
    )


def check_step_label_coverage(
    trajectories: list[Trajectory],
    threshold: float = 0.80,
) -> CheckResult:
    n_tool_steps = 0
    n_labeled = 0
    for t in trajectories:
        if t.outcome != 1:
            continue
        for s in t.trajectory:
            if s.tool is not None:
                n_tool_steps += 1
                if s.step_label is not None:
                    n_labeled += 1
    coverage = n_labeled / max(n_tool_steps, 1)
    return CheckResult(
        name=f"step_label coverage on outcome=1 tool steps >= {threshold:.0%}",
        passed=coverage >= threshold,
        detail=f"{n_labeled}/{n_tool_steps} = {coverage:.1%}",
    )


def check_token_usage_coverage(
    trajectories: list[Trajectory],
    threshold: float = 0.80,
) -> CheckResult:
    with_usage = [t for t in trajectories if t.token_usage is not None]
    coverage = len(with_usage) / max(len(trajectories), 1)
    return CheckResult(
        name=f"token_usage coverage >= {threshold:.0%}",
        passed=coverage >= threshold,
        detail=f"{len(with_usage)}/{len(trajectories)} = {coverage:.1%}",
    )


def check_outcome_zero_simplification_labels(
    trajectories: list[Trajectory],
) -> CheckResult:
    """For trajectories tagged `outcome_zero_simplification`, every TOOL step
    must have `step_label == 0.0`. Pure-thought steps may keep None.

    Catches the case where label_method was set but the step_label assignment
    didn't actually happen (e.g. only_tool_steps=False path bug or partial run).
    """
    bad: list[str] = []
    for t in trajectories:
        if t.label_method != "outcome_zero_simplification":
            continue
        for s in t.trajectory:
            if s.tool is not None and s.step_label != 0.0:
                bad.append(t.task_id)
                break
    return CheckResult(
        name="outcome_zero_simplification trajectories have step_label=0 on every tool step",
        passed=len(bad) == 0,
        detail=(
            "all clean"
            if not bad
            else f"{len(bad)} trajectories have non-zero tool step_label "
                 f"(first: {bad[:3]})"
        ),
    )


def check_step_label_distribution_non_degenerate(
    trajectories: list[Trajectory],
    low: float = 0.2,
    high: float = 0.8,
) -> CheckResult:
    labels: list[float] = []
    for t in trajectories:
        if t.label_method != "llm_judge":
            continue
        for s in t.trajectory:
            if s.step_label is not None:
                labels.append(s.step_label)
    if not labels:
        return CheckResult(
            name=f"step_label mean in [{low:.1f}, {high:.1f}]",
            passed=False,
            detail="no llm_judge step labels found",
        )
    mean = sum(labels) / len(labels)
    return CheckResult(
        name=f"step_label mean in [{low:.1f}, {high:.1f}]",
        passed=low <= mean <= high,
        detail=f"mean={mean:.3f} over {len(labels)} judge-labeled steps",
    )


def run_all_checks(trajectories: list[Trajectory]) -> list[CheckResult]:
    return [
        check_label_method_set(trajectories),
        check_task_prompt_coverage(trajectories, threshold=0.95),
        check_step_label_coverage(trajectories, threshold=0.80),
        check_outcome_zero_simplification_labels(trajectories),
        check_token_usage_coverage(trajectories, threshold=0.80),
        check_step_label_distribution_non_degenerate(trajectories),
    ]


# --- main ---


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
    p.add_argument(
        "--skip_checks", action="store_true",
        help="Skip Phase 1 exit-criteria checks. Use ONLY for debugging or "
             "for low-coverage pilots where you know the data isn't final.",
    )
    p.add_argument(
        "--allow_skipped_in_manifest", action="store_true",
        help="Allow manifest.skipped_files to be non-empty (still reports it). "
             "Default behavior is to FAIL if label_all skipped any input file.",
    )
    args = p.parse_args()

    print(f"Reading from: {[str(d) for d in args.input_dirs]}")
    all_trajs = collect_all(args.input_dirs)
    print(f"\nTotal collected: {len(all_trajs)} trajectories")

    if not all_trajs:
        print("ERROR: no trajectories found. Did labeling run?")
        sys.exit(1)

    if not args.skip_checks:
        print("\nInspecting labeling manifests...")
        manifest_results = inspect_manifests(
            args.input_dirs,
            allow_skipped=args.allow_skipped_in_manifest,
        )

        print("\nRunning Phase 1 exit-criteria checks...")
        criteria_results = run_all_checks(all_trajs)

        all_results = manifest_results + criteria_results
        for r in all_results:
            mark = "PASS" if r.passed else "FAIL"
            print(f"  [{mark}] {r.name}: {r.detail}")
        if any(not r.passed for r in all_results):
            print(
                "\n[FATAL] One or more pre-assembly checks failed.\n"
                "Fix the upstream pipeline (collection/labeling) and re-run.\n"
                "Or pass --skip_checks to assemble anyway (NOT recommended\n"
                "for final Phase 1 dataset).\n"
            )
            sys.exit(2)
        print("All checks passed.\n")
    else:
        print("[!] --skip_checks set; not validating exit criteria.\n")

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
