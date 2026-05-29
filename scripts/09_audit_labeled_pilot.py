#!/usr/bin/env python3
"""Audit labeled BigCodeBench pilot trajectories before full collection.

The raw pilot audit proves the grader is credible. This script proves the
labeler output preserves that contract and has useful step-label signal.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

from src.labeler.trajectory_schema import Trajectory  # noqa: E402


@dataclass(frozen=True)
class RowEvidence:
    task_id: str
    outcome: int
    label_method: str | None
    labels: list[float | None]
    n_tool_steps: int
    n_labeled_tool_steps: int


@dataclass(frozen=True)
class AuditResult:
    input_dir: Path
    files: list[Path]
    manifest_path: Path
    manifest: dict[str, Any] | None
    summary: dict[str, Any]
    rows: list[RowEvidence]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _read_json_records(input_dir: Path) -> tuple[list[tuple[Path, int, dict[str, Any]]], list[str]]:
    records: list[tuple[Path, int, dict[str, Any]]] = []
    errors: list[str] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path}:{line_no}: invalid JSON: {exc}")
                    continue
                if not isinstance(raw, dict):
                    errors.append(f"{path}:{line_no}: JSONL row is not an object")
                    continue
                records.append((path, line_no, raw))
    return records, errors


def _read_manifest(manifest_path: Path) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not manifest_path.exists():
        return None, [f"missing labeling_manifest.json at {manifest_path}"], warnings
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, [f"invalid labeling_manifest.json: {exc}"], warnings
    if not isinstance(raw, dict):
        return None, ["labeling_manifest.json is not an object"], warnings
    if raw.get("tool") != "src.labeler.label_all":
        warnings.append(f"manifest tool is {raw.get('tool')!r}, expected 'src.labeler.label_all'")
    if raw.get("K") != 4:
        warnings.append(f"manifest K is {raw.get('K')!r}, expected pilot K=4")
    if raw.get("skipped_files"):
        errors.append(f"manifest has skipped_files: {raw.get('skipped_files')}")
    if not raw.get("processed_files"):
        errors.append("manifest has no processed_files")
    return raw, errors, warnings


def _counter_dict(values: list[Any]) -> dict[Any, int]:
    return dict(sorted(Counter(values).items(), key=lambda item: str(item[0])))


def audit_dir(
    input_dir: Path,
    *,
    expected_count: int = 10,
    min_distinct_success_labels: int = 2,
) -> AuditResult:
    files = sorted(input_dir.glob("*.jsonl"))
    manifest_path = input_dir / "labeling_manifest.json"
    manifest, manifest_errors, manifest_warnings = _read_manifest(manifest_path)
    records, errors = _read_json_records(input_dir)
    errors.extend(manifest_errors)
    warnings = list(manifest_warnings)

    if not input_dir.exists():
        errors.append(f"Input dir does not exist: {input_dir}")
    if not files:
        errors.append(f"No *.jsonl files found in {input_dir}")
    if len(records) != expected_count:
        errors.append(f"Expected {expected_count} labeled trajectories, found {len(records)}")

    rows: list[RowEvidence] = []
    task_ids: list[str] = []
    run_ids: list[str | None] = []
    outcomes: list[int] = []
    methods: list[str | None] = []
    all_tool_labels: list[float] = []
    success_tool_labels: list[float] = []
    bad_outcome_zero_labels: list[tuple[str, int, float | None]] = []
    outcome_one_unlabeled: list[tuple[str, int]] = []
    outcome_one_total_tool_steps = 0
    outcome_one_labeled_tool_steps = 0
    total_tool_steps = 0
    total_labeled_tool_steps = 0

    for idx, (path, line_no, raw) in enumerate(records):
        task_id = str(raw.get("task_id", "<missing>"))
        task_ids.append(task_id)
        run_ids.append(raw.get("run_id"))
        try:
            traj = Trajectory(**raw)
        except ValidationError as exc:
            errors.append(f"{path}:{line_no} task={task_id}: schema validation failed: {exc.errors()}")
            continue

        outcomes.append(traj.outcome)
        methods.append(traj.label_method)
        tool_labels: list[float | None] = []
        n_tool_steps = 0
        n_labeled_tool_steps = 0
        for step in traj.trajectory:
            if step.tool is None:
                if step.step_label is not None:
                    warnings.append(
                        f"row {idx} task={traj.task_id}: pure-thought step {step.step} has label {step.step_label}"
                    )
                continue
            n_tool_steps += 1
            total_tool_steps += 1
            tool_labels.append(step.step_label)
            if step.step_label is not None:
                n_labeled_tool_steps += 1
                total_labeled_tool_steps += 1
                all_tool_labels.append(step.step_label)

            if traj.outcome == 0:
                if step.step_label not in (0, 0.0):
                    bad_outcome_zero_labels.append((traj.task_id, step.step, step.step_label))
            else:
                outcome_one_total_tool_steps += 1
                if step.step_label is None:
                    outcome_one_unlabeled.append((traj.task_id, step.step))
                else:
                    outcome_one_labeled_tool_steps += 1
                    success_tool_labels.append(step.step_label)

        rows.append(
            RowEvidence(
                task_id=traj.task_id,
                outcome=traj.outcome,
                label_method=traj.label_method,
                labels=tool_labels,
                n_tool_steps=n_tool_steps,
                n_labeled_tool_steps=n_labeled_tool_steps,
            )
        )

        if traj.outcome == 0 and traj.label_method != "outcome_zero_simplification":
            errors.append(
                f"row {idx} task={traj.task_id}: outcome=0 label_method={traj.label_method!r}, "
                "expected 'outcome_zero_simplification'"
            )
        if traj.outcome == 1 and traj.label_method != "llm_judge":
            errors.append(
                f"row {idx} task={traj.task_id}: outcome=1 label_method={traj.label_method!r}, "
                "expected 'llm_judge'"
            )

    duplicate_task_ids = sorted(task_id for task_id, count in Counter(task_ids).items() if count > 1)
    if duplicate_task_ids:
        errors.append(f"Duplicate task_id values: {duplicate_task_ids}")
    nonempty_run_ids = [run_id for run_id in run_ids if run_id]
    duplicate_run_ids = sorted(run_id for run_id, count in Counter(nonempty_run_ids).items() if count > 1)
    if duplicate_run_ids:
        errors.append(f"Duplicate run_id values: {duplicate_run_ids}")

    if bad_outcome_zero_labels:
        errors.append(f"outcome=0 has non-zero tool step labels: {bad_outcome_zero_labels[:20]}")
    if outcome_one_unlabeled:
        errors.append(f"outcome=1 has unlabeled tool steps: {outcome_one_unlabeled[:20]}")
    distinct_success_labels = sorted(set(success_tool_labels))
    if outcome_one_labeled_tool_steps > 0 and len(distinct_success_labels) < min_distinct_success_labels:
        errors.append(
            "success-path labels are degenerate: "
            f"distinct={distinct_success_labels}, required>={min_distinct_success_labels}"
        )
    if outcomes and 1 not in outcomes:
        errors.append("No outcome=1 trajectories; pilot cannot exercise llm_judge path")
    if outcomes and 0 not in outcomes:
        warnings.append("No outcome=0 trajectories; pilot did not exercise simplification path")

    success_mean = statistics.mean(success_tool_labels) if success_tool_labels else None
    success_median = statistics.median(success_tool_labels) if success_tool_labels else None
    summary = {
        "n_files": len(files),
        "files": [str(path) for path in files],
        "n_rows": len(records),
        "manifest_present": manifest is not None,
        "manifest_model": manifest.get("model") if manifest else None,
        "manifest_K": manifest.get("K") if manifest else None,
        "manifest_total_cost_usd": manifest.get("total_cost_usd") if manifest else None,
        "outcome": _counter_dict(outcomes),
        "label_method": _counter_dict(methods),
        "tool_steps": total_tool_steps,
        "labeled_tool_steps": total_labeled_tool_steps,
        "outcome_one_tool_steps": outcome_one_total_tool_steps,
        "outcome_one_labeled_tool_steps": outcome_one_labeled_tool_steps,
        "all_label_values": _counter_dict(all_tool_labels),
        "success_label_values": _counter_dict(success_tool_labels),
        "success_label_mean": round(success_mean, 4) if success_mean is not None else None,
        "success_label_median": round(success_median, 4) if success_median is not None else None,
        "bad_outcome_zero_labels": bad_outcome_zero_labels,
        "run_ids_unique": len(set(nonempty_run_ids)),
    }
    return AuditResult(
        input_dir=input_dir,
        files=files,
        manifest_path=manifest_path,
        manifest=manifest,
        summary=summary,
        rows=rows,
        errors=errors,
        warnings=warnings,
    )


def render_report(result: AuditResult) -> str:
    lines: list[str] = []
    lines.append(f"Labeled pilot audit for {result.input_dir}")
    lines.append(f"status: {'PASS' if result.ok else 'FAIL'}")
    lines.append("")
    lines.append("Summary:")
    for key, value in result.summary.items():
        lines.append(f"  {key}: {value}")

    if result.errors:
        lines.append("")
        lines.append("Errors:")
        for error in result.errors:
            lines.append(f"  - {error}")
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in result.warnings:
            lines.append(f"  - {warning}")

    lines.append("")
    lines.append("Rows:")
    for row in result.rows:
        lines.append(
            f"  {row.task_id} outcome={row.outcome} method={row.label_method} "
            f"labels={row.labels} "
            f"labeled_tool_steps={row.n_labeled_tool_steps}/{row.n_tool_steps}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("data/labeled/pilot"), help="Labeled pilot output directory")
    parser.add_argument("--expected-count", type=int, default=10, help="Expected labeled trajectory count")
    parser.add_argument(
        "--min-distinct-success-labels",
        type=int,
        default=2,
        help="Minimum distinct step_label values among outcome=1 tool steps",
    )
    args = parser.parse_args()

    result = audit_dir(
        args.dir,
        expected_count=args.expected_count,
        min_distinct_success_labels=args.min_distinct_success_labels,
    )
    print(render_report(result))
    raise SystemExit(0 if result.ok else 2)


if __name__ == "__main__":
    main()
