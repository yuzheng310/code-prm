#!/usr/bin/env python3
"""Combine labeled trajectories into leakage-safe train/val/test splits.

Reads labeled trajectory jsonl files, validates Phase 1 exit criteria, then
splits at the task level: all rollouts for one task stay in the same split.
This is required for held-out evaluation and Best-of-N, because different
rollouts of the same task share the same prompt, tests, and failure modes.

Run from project root:
    python scripts/30_assemble_dataset.py
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.labeler.trajectory_schema import Trajectory  # noqa: E402
from src.utils.jsonl_io import read_trajectories, write_trajectories  # noqa: E402


# --- shared check-result type (used by inspect_manifests + exit-criteria
#     checks; declared up front for readability so forward references go
#     away) ---


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


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
            relocated_path = d / out_path.name
            if not out_path.exists() and not relocated_path.exists():
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
    """Legacy trajectory-level split.

    Kept only for tests/debugging. Final Phase 1 data must use split_by_task()
    so that rollouts from one task never leak across train/val/test.
    """
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


def _task_key(t: Trajectory) -> tuple[str, int]:
    return (t.task_id, t.rollout_id)


def _group_by_task(trajectories: list[Trajectory]) -> dict[str, list[Trajectory]]:
    groups: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        groups[t.task_id].append(t)
    return dict(groups)


def _pass_count(group: list[Trajectory]) -> int:
    return sum(t.outcome for t in group)


def _flatten_groups(groups: list[list[Trajectory]]) -> list[Trajectory]:
    flattened: list[Trajectory] = []
    for group in groups:
        flattened.extend(sorted(group, key=lambda t: t.rollout_id))
    return flattened


def _split_groups_by_fracs(
    groups: list[list[Trajectory]],
    fracs: tuple[float, float, float],
) -> tuple[list[list[Trajectory]], list[list[Trajectory]], list[list[Trajectory]]]:
    n = len(groups)
    n_train = int(n * fracs[0])
    n_val = int(n * fracs[1])
    train = groups[:n_train]
    val = groups[n_train : n_train + n_val]
    test = groups[n_train + n_val :]
    return train, val, test


def _take_mixed_groups(
    by_pass_count: dict[int, list[list[Trajectory]]],
    mixed_alloc: tuple[int, int, int],
) -> tuple[list[list[Trajectory]], list[list[Trajectory]], list[list[Trajectory]]]:
    """Allocate mixed tasks while preserving 1/4, 2/4, 3/4 buckets.

    We fill train, then val, then test. Within each split, each next task is
    taken from the pass-count bucket with the largest remaining fraction. This
    keeps small held-out splits from being filled only by one mixed difficulty
    bucket while still honoring exact total mixed task counts.
    """
    total_mixed = sum(len(by_pass_count.get(pc, [])) for pc in (1, 2, 3))
    expected = sum(mixed_alloc)
    if total_mixed != expected:
        raise ValueError(
            f"mixed_alloc sums to {expected}, but dataset has {total_mixed} "
            "mixed-outcome tasks"
        )

    originals = {pc: len(by_pass_count.get(pc, [])) for pc in (1, 2, 3)}
    remaining = {pc: list(by_pass_count.get(pc, [])) for pc in (1, 2, 3)}
    splits: tuple[list[list[Trajectory]], list[list[Trajectory]], list[list[Trajectory]]] = (
        [],
        [],
        [],
    )
    for split_idx, target in enumerate(mixed_alloc):
        for _ in range(target):
            candidates = [pc for pc in (1, 2, 3) if remaining[pc]]
            if not candidates:
                raise ValueError("mixed allocation underfilled; this is a split bug")
            pc = max(
                candidates,
                key=lambda p: (len(remaining[p]) / originals[p], len(remaining[p])),
            )
            splits[split_idx].append(remaining[pc].pop())
    return splits


def split_by_task(
    trajectories: list[Trajectory],
    *,
    mixed_alloc: tuple[int, int, int] = (18, 4, 6),
    nonmixed_fracs: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list[Trajectory], list[Trajectory], list[Trajectory]]:
    """Split by task_id, keeping every task's rollouts together.

    mixed_alloc is (train, val, test) task counts for mixed-outcome tasks.
    Non-mixed tasks are stratified by pass-count buckets, so all-fail and
    all-pass tasks retain approximately similar proportions in every split.
    """
    rng = random.Random(seed)
    groups = _group_by_task(trajectories)

    by_pass_count: dict[int, list[list[Trajectory]]] = defaultdict(list)
    for group in groups.values():
        by_pass_count[_pass_count(group)].append(group)

    for bucket in by_pass_count.values():
        bucket.sort(key=lambda group: group[0].task_id)
        rng.shuffle(bucket)

    train_groups, val_groups, test_groups = _take_mixed_groups(by_pass_count, mixed_alloc)

    for pc, groups_in_bucket in sorted(by_pass_count.items()):
        if 0 < pc < len(groups_in_bucket[0]):
            continue
        train_b, val_b, test_b = _split_groups_by_fracs(groups_in_bucket, nonmixed_fracs)
        train_groups.extend(train_b)
        val_groups.extend(val_b)
        test_groups.extend(test_b)

    for bucket in (train_groups, val_groups, test_groups):
        bucket.sort(key=lambda group: group[0].task_id)

    return _flatten_groups(train_groups), _flatten_groups(val_groups), _flatten_groups(test_groups)


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


# --- exit-criteria checks (CheckResult defined above) ---


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


def _task_ids(data: list[Trajectory]) -> set[str]:
    return {t.task_id for t in data}


def _mixed_task_ids(data: list[Trajectory]) -> list[str]:
    mixed: list[str] = []
    for task_id, group in _group_by_task(data).items():
        outcomes = {t.outcome for t in group}
        if outcomes == {0, 1}:
            mixed.append(task_id)
    return sorted(mixed)


def _pass_count_histogram(data: list[Trajectory]) -> dict[str, int]:
    hist: Counter[str] = Counter()
    for group in _group_by_task(data).values():
        hist[f"{_pass_count(group)}/{len(group)}"] += 1
    return dict(sorted(hist.items()))


def check_no_task_overlap(
    train: list[Trajectory],
    val: list[Trajectory],
    test: list[Trajectory],
) -> CheckResult:
    train_ids = _task_ids(train)
    val_ids = _task_ids(val)
    test_ids = _task_ids(test)
    overlaps = {
        "train_val": sorted(train_ids & val_ids),
        "train_test": sorted(train_ids & test_ids),
        "val_test": sorted(val_ids & test_ids),
    }
    bad = {k: v[:5] for k, v in overlaps.items() if v}
    return CheckResult(
        name="task ids do not overlap across splits",
        passed=not bad,
        detail="all disjoint" if not bad else f"overlaps found: {bad}",
    )


def check_rollout_completeness(
    trajectories: list[Trajectory],
    rollouts_per_task: int = 4,
) -> CheckResult:
    expected = set(range(rollouts_per_task))
    bad: list[str] = []
    duplicate: list[str] = []
    for task_id, group in _group_by_task(trajectories).items():
        rollout_ids = [t.rollout_id for t in group]
        if len(rollout_ids) != len(set(rollout_ids)):
            duplicate.append(task_id)
            continue
        if set(rollout_ids) != expected:
            bad.append(f"{task_id}:{sorted(rollout_ids)}")
    ok = not bad and not duplicate
    return CheckResult(
        name=f"every task has rollout ids {sorted(expected)}",
        passed=ok,
        detail=(
            "all complete"
            if ok
            else f"missing/bad={bad[:5]}, duplicate={duplicate[:5]}"
        ),
    )


def check_conservation(
    original: list[Trajectory],
    train: list[Trajectory],
    val: list[Trajectory],
    test: list[Trajectory],
) -> CheckResult:
    original_ids = {_task_key(t) for t in original}
    split_ids = [_task_key(t) for t in train + val + test]
    duplicate_count = len(split_ids) - len(set(split_ids))
    missing = sorted(original_ids - set(split_ids))[:5]
    extra = sorted(set(split_ids) - original_ids)[:5]
    ok = not duplicate_count and not missing and not extra and len(split_ids) == len(original)
    return CheckResult(
        name="split conserves all trajectories without duplicates",
        passed=ok,
        detail=(
            f"{len(split_ids)}/{len(original)} conserved"
            if ok
            else (
                f"split={len(split_ids)}, original={len(original)}, "
                f"duplicates={duplicate_count}, missing={missing}, extra={extra}"
            )
        ),
    )


def check_min_mixed_tasks(
    data: list[Trajectory],
    min_count: int,
    split_name: str,
) -> CheckResult:
    mixed = _mixed_task_ids(data)
    return CheckResult(
        name=f"{split_name} has at least {min_count} mixed-outcome tasks",
        passed=len(mixed) >= min_count,
        detail=f"{len(mixed)} mixed tasks",
    )


def run_split_checks(
    original: list[Trajectory],
    train: list[Trajectory],
    val: list[Trajectory],
    test: list[Trajectory],
    *,
    rollouts_per_task: int,
    min_val_mixed_tasks: int,
    min_test_mixed_tasks: int,
) -> list[CheckResult]:
    return [
        check_rollout_completeness(original, rollouts_per_task=rollouts_per_task),
        check_no_task_overlap(train, val, test),
        check_conservation(original, train, val, test),
        check_min_mixed_tasks(val, min_val_mixed_tasks, "val"),
        check_min_mixed_tasks(test, min_test_mixed_tasks, "test"),
    ]


def _split_summary(data: list[Trajectory]) -> dict[str, object]:
    task_ids = _task_ids(data)
    n_pass = sum(t.outcome for t in data)
    return {
        "n_tasks": len(task_ids),
        "n_trajectories": len(data),
        "n_pass": n_pass,
        "pass_rate": round(n_pass / len(data), 4) if data else 0.0,
        "n_mixed_tasks": len(_mixed_task_ids(data)),
        "mixed_task_ids": _mixed_task_ids(data),
        "pass_count_histogram": _pass_count_histogram(data),
    }


def _check_results_for_manifest(checks: list[CheckResult]) -> list[dict[str, object]]:
    return [
        {"name": check.name, "passed": check.passed, "detail": check.detail}
        for check in checks
    ]


def build_split_manifest(
    train: list[Trajectory],
    val: list[Trajectory],
    test: list[Trajectory],
    *,
    seed: int,
    mixed_alloc: tuple[int, int, int],
    nonmixed_fracs: tuple[float, float, float],
    input_dirs: list[str],
    checks: list[CheckResult],
) -> dict[str, object]:
    all_data = train + val + test
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "strategy": "task_grouped_stratified",
        "seed": seed,
        "mixed_alloc": list(mixed_alloc),
        "nonmixed_fracs": list(nonmixed_fracs),
        "input_dirs": input_dirs,
        "checks": _check_results_for_manifest(checks),
        "totals": {
            "n_tasks": len(_task_ids(all_data)),
            "n_trajectories": len(all_data),
            "n_mixed_tasks": len(_mixed_task_ids(all_data)),
        },
        "splits": {
            "train": _split_summary(train),
            "val": _split_summary(val),
            "test": _split_summary(test),
        },
    }




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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input_dirs",
        nargs="+",
        type=Path,
        default=[Path("data/labeled/bigcodebench-hard")],
        help="Directories containing labeled *.jsonl files",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/code-trajectory-2.4k-tasksplit"),
    )
    p.add_argument(
        "--split_mode",
        choices=("task_grouped", "trajectory"),
        default="task_grouped",
        help="task_grouped is leakage-safe. trajectory is legacy/debug only.",
    )
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--test_frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--expected_rollouts_per_task", type=int, default=4)
    p.add_argument(
        "--mixed_alloc",
        nargs=3,
        type=int,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(18, 4, 6),
        help="Mixed-outcome task allocation for task_grouped split",
    )
    p.add_argument(
        "--nonmixed_fracs",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.8, 0.1, 0.1),
        help="Per-pass-count-bucket fractions for non-mixed tasks",
    )
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
    return p


def validate_args(args: argparse.Namespace) -> None:
    default_tasksplit_dir = Path("data/code-trajectory-2.4k-tasksplit")
    if args.split_mode == "trajectory" and args.output_dir == default_tasksplit_dir:
        raise SystemExit(
            "--split_mode trajectory is legacy/debug-only and must not write "
            "to the task-grouped output directory. Pass --output_dir to an "
            "explicit debug path."
        )


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)

    print(f"Reading from: {[str(d) for d in args.input_dirs]}")
    all_trajs = collect_all(args.input_dirs)
    print(f"\nTotal collected: {len(all_trajs)} trajectories")

    if not all_trajs:
        print("ERROR: no trajectories found. Did labeling run?")
        sys.exit(1)

    split_results: list[CheckResult] = []

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

    if args.split_mode == "trajectory":
        print("[!] Using legacy trajectory-level split; not valid for final Phase 2/3.")
        train, val, test = split(all_trajs, args.val_frac, args.test_frac, args.seed)
    else:
        mixed_alloc = tuple(args.mixed_alloc)
        nonmixed_fracs = tuple(args.nonmixed_fracs)
        train, val, test = split_by_task(
            all_trajs,
            mixed_alloc=mixed_alloc,
            nonmixed_fracs=nonmixed_fracs,
            seed=args.seed,
        )

        print("\nRunning task-grouped split checks...")
        split_results = run_split_checks(
            all_trajs,
            train,
            val,
            test,
            rollouts_per_task=args.expected_rollouts_per_task,
            min_val_mixed_tasks=mixed_alloc[1],
            min_test_mixed_tasks=mixed_alloc[2],
        )
        for r in split_results:
            mark = "PASS" if r.passed else "FAIL"
            print(f"  [{mark}] {r.name}: {r.detail}")
        if any(not r.passed for r in split_results):
            print("\n[FATAL] Task-grouped split checks failed.\n")
            sys.exit(2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_trajectories(args.output_dir / "train.jsonl", train)
    write_trajectories(args.output_dir / "val.jsonl", val)
    write_trajectories(args.output_dir / "test.jsonl", test)
    if args.split_mode == "task_grouped":
        manifest = build_split_manifest(
            train,
            val,
            test,
            seed=args.seed,
            mixed_alloc=tuple(args.mixed_alloc),
            nonmixed_fracs=tuple(args.nonmixed_fracs),
            input_dirs=[str(d) for d in args.input_dirs],
            checks=split_results,
        )
        (args.output_dir / "split_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

    print(f"\nWrote splits to {args.output_dir}/")
    print(f"  train.jsonl  ({len(train)})")
    print(f"  val.jsonl    ({len(val)})")
    print(f"  test.jsonl   ({len(test)})")
    print("\nPer-split stats:")
    for name, data in [("train", train), ("val", val), ("test", test)]:
        report(name, data)


if __name__ == "__main__":
    main()
