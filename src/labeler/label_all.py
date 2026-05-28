"""Drive step labeling across many jsonl files with per-file budget tracking.

Reads every *.jsonl under `--input_dir` (recursive), runs `label_file` on
each, writes labeled output to `--output_dir`. Halts when the shared
CostTracker exceeds budget. The actual API spend lives in
`RateLimitedClient` / `CostTracker`.

Output filenames preserve the input's relative subpath flattened with "__"
to avoid collisions when nested directories contain same-named files
(e.g. rollout_0/foo.jsonl + rollout_1/foo.jsonl).

Pre-flight `task_prompt` coverage check on the INPUT trajectories:
if too few have non-empty `task_prompt`, the LLM judge has no problem
statement to anchor on and labels will be near-random.

  - Default threshold: 95% of outcome=1 trajectories must have non-empty
    `task_prompt` (matches Phase 1 exit criterion).
  - Behavior on threshold violation: ABORT with SystemExit(2).
  - To override: pass `--allow_low_task_prompt_coverage`.
  - To raise/lower the threshold: pass `--min_task_prompt_coverage X.XX`.

After a successful run, writes a `labeling_manifest.json` to
`--output_dir` documenting: input/output paths, sizes, mtimes, K, model,
task_prompt coverage achieved, processed/skipped files, and total cost.
"""
from __future__ import annotations
import argparse
import datetime as _dt
import json
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
    # task_prompt is critical to LLM-judge label quality. Default threshold
    # matches the Phase 1 exit criterion (≥ 95%).
    p.add_argument(
        "--min_task_prompt_coverage",
        type=float, default=0.95,
        help="Required task_prompt coverage on outcome=1 trajectories "
             "(default 0.95 == 95%%, matching Phase 1 exit criteria)",
    )
    p.add_argument(
        "--allow_low_task_prompt_coverage",
        action="store_true",
        help="Proceed even when task_prompt coverage is below the threshold. "
             "Default behavior is to ABORT (don't burn API quota labeling junk).",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Wipe --output_dir before labeling. Prevents stale labels from "
             "a prior run leaking into the new output (which would silently "
             "mismatch the new --input_dir).",
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
        coverage = n_prompt / n_one
        coverage_pct = coverage * 100.0
        print(f"task_prompt coverage on outcome=1 trajectories: "
              f"{n_prompt}/{n_one} = {coverage_pct:.1f}%  (total trajectories: {n_total})"
              f"  (threshold: {args.min_task_prompt_coverage * 100:.1f}%)")
        if coverage < args.min_task_prompt_coverage:
            if args.allow_low_task_prompt_coverage:
                print(
                    "\n[!] task_prompt coverage is below threshold but "
                    "--allow_low_task_prompt_coverage was set. Continuing.\n"
                )
            else:
                print(
                    "\n[FATAL] task_prompt coverage below threshold "
                    f"({coverage_pct:.1f}% < {args.min_task_prompt_coverage * 100:.1f}%).\n"
                    "The LLM judge needs the problem statement to anchor on; "
                    "labels will be near-random.\n"
                    "Fix the TS logger (see src/collector/ts_logger_spec.md), "
                    "re-collect, then re-run.\n"
                    "To proceed anyway, pass --allow_low_task_prompt_coverage.\n"
                )
                raise SystemExit(2)

    if args.clean and args.output_dir.exists():
        import shutil
        print(f"Removing existing output dir: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Manifest accumulates input-file metadata for downstream provenance.
    started_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    processed: list[dict] = []
    skipped: list[dict] = []

    for f in files:
        out = args.output_dir / _flatten_relative_name(args.input_dir, f)
        print(f"Labeling {f}  ->  {out}")
        in_stat = f.stat()
        in_meta = {
            "input": str(f.resolve()),
            "input_size": in_stat.st_size,
            "input_mtime": in_stat.st_mtime,
            "output": str(out.resolve()),
        }
        try:
            label_file(f, out, client, K=args.K)
        except Exception as e:
            print(f"  ERROR on {f}: {e}")
            in_meta["error"] = repr(e)
            skipped.append(in_meta)
            if tracker.over_budget():
                print("Over budget — stopping.")
                break
            continue
        processed.append(in_meta)
        print(f"  cost so far: {tracker}")
        if tracker.over_budget():
            print("OVER BUDGET — stopping.")
            break

    finished_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    manifest = {
        "tool": "src.labeler.label_all",
        "started_at": started_at,
        "finished_at": finished_at,
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "K": args.K,
        "model": args.model,
        "min_task_prompt_coverage": args.min_task_prompt_coverage,
        "task_prompt_coverage": (n_prompt / n_one) if n_one > 0 else None,
        "processed_files": processed,
        "skipped_files": skipped,
        "cost_per_model": tracker.per_model,
        "total_cost_usd": tracker.total_usd,
    }
    manifest_path = args.output_dir / "labeling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nFINAL: {tracker}")
    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
