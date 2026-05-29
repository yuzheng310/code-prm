#!/usr/bin/env python3
"""Audit BigCodeBench pilot trajectories before labeling.

This script is intentionally evidence-heavy: it performs hard consistency checks
and prints compact per-row grader tails so the pilot can be reviewed without
ad-hoc heredocs.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Allow running as a script from project root or via an absolute path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

from src.labeler.trajectory_schema import Trajectory  # noqa: E402


@dataclass(frozen=True)
class RowEvidence:
    index: int
    file: str
    task_id: str
    outcome: Any
    passed: Any
    exit_code: Any
    command: str
    stderr_tail: str
    stdout_tail: str


@dataclass(frozen=True)
class AuditResult:
    input_dir: Path
    files: list[Path]
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


def _tail(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    return text[-max_chars:] if len(text) > max_chars else text


def _has_failure_signal(text: str) -> bool:
    signals = ("FAILED", "FAIL:", "ERROR:", "Traceback", "AssertionError", "ImportError")
    return any(signal in text for signal in signals)


def _has_grader_namespace_name_error(text: str) -> bool:
    return "_bcb_grader_" in text and "NameError: name " in text


def audit_dir(
    input_dir: Path,
    *,
    expected_count: int = 10,
    max_tail_chars: int = 800,
) -> AuditResult:
    files = sorted(input_dir.glob("*.jsonl"))
    records, errors = _read_json_records(input_dir)
    warnings: list[str] = []
    rows: list[RowEvidence] = []

    if not input_dir.exists():
        errors.append(f"Input dir does not exist: {input_dir}")
    if not files:
        errors.append(f"No *.jsonl files found in {input_dir}")

    trajectories: list[Trajectory] = []
    task_ids: list[str] = []
    run_ids: list[str | None] = []
    rollout_ids: list[Any] = []
    outcome_values: list[Any] = []
    passed_values: list[Any] = []
    policy_models: Counter[str] = Counter()
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_read_tokens = 0
    total_cache_creation_tokens = 0
    total_cost_usd = 0.0

    for idx, (path, line_no, raw) in enumerate(records):
        task_id = str(raw.get("task_id", "<missing>"))
        task_ids.append(task_id)
        run_ids.append(raw.get("run_id"))
        rollout_ids.append(raw.get("rollout_id"))
        outcome_values.append(raw.get("outcome"))
        policy_models[str(raw.get("policy_model", "<missing>"))] += 1

        test_result = raw.get("test_result")
        passed = test_result.get("passed") if isinstance(test_result, dict) else None
        passed_values.append(passed)
        stderr_tail = test_result.get("stderr_tail", "") if isinstance(test_result, dict) else ""
        stdout_tail = test_result.get("stdout_tail", "") if isinstance(test_result, dict) else ""
        combined_output = f"{stdout_tail}\n{stderr_tail}"
        rows.append(
            RowEvidence(
                index=idx,
                file=f"{path.name}:{line_no}",
                task_id=task_id,
                outcome=raw.get("outcome"),
                passed=passed,
                exit_code=test_result.get("exit_code") if isinstance(test_result, dict) else None,
                command=str(test_result.get("command", "")) if isinstance(test_result, dict) else "",
                stderr_tail=_tail(stderr_tail, max_tail_chars),
                stdout_tail=_tail(stdout_tail, max_tail_chars),
            )
        )

        try:
            traj = Trajectory(**raw)
        except ValidationError as exc:
            errors.append(f"{path}:{line_no} task={task_id}: schema validation failed: {exc.errors()}")
            continue
        trajectories.append(traj)

        if traj.task_type != "bigcodebench-hard":
            errors.append(f"row {idx} task={task_id}: task_type is {traj.task_type!r}, expected 'bigcodebench-hard'")
        if traj.test_result is None:
            errors.append(f"row {idx} task={task_id}: missing test_result")
        else:
            if not traj.test_result.command:
                errors.append(f"row {idx} task={task_id}: test_result.command is empty")
            if "Ran 0 tests" in combined_output or "Ran 0 test" in combined_output:
                errors.append(f"row {idx} task={task_id}: grader reported 0 tests")
            if _has_grader_namespace_name_error(combined_output):
                errors.append(
                    f"row {idx} task={task_id}: NameError in grader namespace; "
                    "BigCodeBench tests likely reference symbols that must be imported from task.py"
                )
            if traj.test_result.passed and "OK" not in combined_output:
                warnings.append(f"row {idx} task={task_id}: passed=True but unittest OK marker not found")
            if not traj.test_result.passed and not _has_failure_signal(combined_output):
                warnings.append(f"row {idx} task={task_id}: failed but no obvious failure signal in grader output")
        if not traj.task_prompt:
            errors.append(f"row {idx} task={task_id}: missing task_prompt")
        if not traj.run_id:
            errors.append(f"row {idx} task={task_id}: missing run_id")
        if traj.rollout_id != 0:
            warnings.append(f"row {idx} task={task_id}: rollout_id={traj.rollout_id}, expected pilot rollout_id=0")
        if traj.token_usage is None:
            errors.append(f"row {idx} task={task_id}: missing token_usage")
        else:
            total_input_tokens += traj.token_usage.input_tokens
            total_output_tokens += traj.token_usage.output_tokens
            total_cache_read_tokens += traj.token_usage.cache_read_tokens
            total_cache_creation_tokens += traj.token_usage.cache_creation_tokens
            total_cost_usd += traj.token_usage.cost_usd

    if len(records) != expected_count:
        errors.append(f"Expected {expected_count} trajectories, found {len(records)}")

    task_counts = Counter(task_ids)
    duplicate_task_ids = sorted(task_id for task_id, count in task_counts.items() if count > 1)
    if duplicate_task_ids:
        errors.append(f"Duplicate task_id values in one-rollout pilot: {duplicate_task_ids}")

    nonempty_run_ids = [run_id for run_id in run_ids if run_id]
    duplicate_run_ids = sorted(run_id for run_id, count in Counter(nonempty_run_ids).items() if count > 1)
    if duplicate_run_ids:
        errors.append(f"Duplicate run_id values: {duplicate_run_ids}")

    outcome_counts = Counter(outcome_values)
    if len(records) > 1 and len(outcome_counts) == 1:
        warnings.append(
            "Outcome distribution is degenerate; inspect grader tails before trusting labels"
        )
    if "unknown" in policy_models:
        warnings.append("Some trajectories have policy_model='unknown'; cost/model attribution may be stale")

    summary = {
        "n_files": len(files),
        "files": [str(path) for path in files],
        "n_rows": len(records),
        "schema_valid": len(trajectories),
        "outcome": dict(sorted(outcome_counts.items(), key=lambda item: str(item[0]))),
        "passed": dict(sorted(Counter(passed_values).items(), key=lambda item: str(item[0]))),
        "test_result_present": sum(isinstance(raw.get("test_result"), dict) for _, _, raw in records),
        "task_prompt_present": sum(bool(raw.get("task_prompt")) for _, _, raw in records),
        "token_usage_present": sum(raw.get("token_usage") is not None for _, _, raw in records),
        "run_ids_unique": len(set(nonempty_run_ids)),
        "rollout_ids": dict(sorted(Counter(rollout_ids).items(), key=lambda item: str(item[0]))),
        "policy_models": dict(policy_models),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cache_read_tokens": total_cache_read_tokens,
        "total_cache_creation_tokens": total_cache_creation_tokens,
        "total_cost_usd": round(total_cost_usd, 4),
    }
    return AuditResult(input_dir=input_dir, files=files, summary=summary, rows=rows, errors=errors, warnings=warnings)


def render_report(result: AuditResult) -> str:
    lines: list[str] = []
    lines.append(f"Pilot audit for {result.input_dir}")
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
            f"  [{row.index}] {row.task_id} file={row.file} "
            f"outcome={row.outcome} passed={row.passed} exit={row.exit_code} "
            f"command={row.command!r}"
        )
        if row.stderr_tail:
            lines.append("    stderr_tail:")
            lines.extend(f"      {line}" for line in row.stderr_tail.splitlines()[-12:])
        if row.stdout_tail:
            lines.append("    stdout_tail:")
            lines.extend(f"      {line}" for line in row.stdout_tail.splitlines()[-8:])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("data/raw/pilot"), help="Pilot raw jsonl directory")
    parser.add_argument("--expected-count", type=int, default=10, help="Expected pilot trajectory count")
    parser.add_argument("--max-tail-chars", type=int, default=800, help="Per-row stdout/stderr tail chars")
    args = parser.parse_args()

    result = audit_dir(args.dir, expected_count=args.expected_count, max_tail_chars=args.max_tail_chars)
    print(render_report(result))
    raise SystemExit(0 if result.ok else 2)


if __name__ == "__main__":
    main()
