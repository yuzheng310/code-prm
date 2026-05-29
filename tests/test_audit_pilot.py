"""Unit tests for the pilot audit script."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "08_audit_pilot.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_pilot", _SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


audit = _load_audit_module()


def _record(
    task_id: str,
    *,
    rollout_id: int = 0,
    run_id: str | None = None,
    outcome: int = 1,
    passed: bool = True,
    stderr_tail: str = "Ran 1 test in 0.001s\n\nOK\n",
    policy_model: str = "claude-sonnet-4-5",
) -> dict:
    return {
        "task_id": task_id,
        "task_type": "bigcodebench-hard",
        "run_id": run_id or f"run-{task_id}-{rollout_id}",
        "rollout_id": rollout_id,
        "task_prompt": "Implement add_one.",
        "task_metadata": {"entry_point": "add_one"},
        "trajectory": [
            {"step": 0, "tool": "write", "tool_args": {"path": "task.py"}, "tool_result": "ok"}
        ],
        "outcome": outcome,
        "test_result": {
            "passed": passed,
            "command": "python _bcb_grader.py",
            "exit_code": 0 if passed else 1,
            "stdout_tail": "",
            "stderr_tail": stderr_tail,
            "duration_sec": 0.1,
        },
        "policy_model": policy_model,
        "timestamp": "2026-05-29T00:00:00Z",
        "token_usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_read_tokens": 30,
            "cache_creation_tokens": 0,
            "cost_usd": 0.01,
        },
        "label_method": None,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_audit_passes_clean_pilot(tmp_path: Path) -> None:
    records = [_record(f"task-{i}") for i in range(10)]
    _write_jsonl(tmp_path / "bigcodebench-hard_20260529.jsonl", records)

    result = audit.audit_dir(tmp_path, expected_count=10, max_tail_chars=200)

    assert result.ok
    assert result.summary["n_rows"] == 10
    assert result.summary["test_result_present"] == 10
    assert result.summary["token_usage_present"] == 10
    assert not result.errors

def test_audit_fails_stale_or_wrong_pilot_count(tmp_path: Path) -> None:
    records = [_record(f"task-{i}") for i in range(11)]
    _write_jsonl(tmp_path / "bigcodebench-hard_20260529.jsonl", records)

    result = audit.audit_dir(tmp_path, expected_count=10, max_tail_chars=200)

    assert not result.ok
    assert any("Expected 10 trajectories, found 11" in error for error in result.errors)


def test_audit_passes_full_multi_rollout_layout(tmp_path: Path) -> None:
    records = []
    for task_id in ("task-a", "task-b"):
        for rollout_id in range(2):
            records.append(_record(task_id, rollout_id=rollout_id))
    _write_jsonl(tmp_path / "bigcodebench-hard_20260529.jsonl", records)

    result = audit.audit_dir(
        tmp_path,
        expected_count=4,
        expected_rollouts_per_task=2,
        max_tail_chars=200,
    )

    assert result.ok
    assert result.summary["unique_task_ids"] == 2
    assert result.summary["rollouts_per_task"] == {"task-a": [0, 1], "task-b": [0, 1]}


def test_audit_fails_full_multi_rollout_with_missing_rollout(tmp_path: Path) -> None:
    records = [
        _record("task-a", rollout_id=0),
        _record("task-a", rollout_id=0, run_id="other-run"),
    ]
    _write_jsonl(tmp_path / "bigcodebench-hard_20260529.jsonl", records)

    result = audit.audit_dir(
        tmp_path,
        expected_count=2,
        expected_rollouts_per_task=2,
        max_tail_chars=200,
    )

    assert not result.ok
    assert any("expected rollout ids [0, 1], got [0]" in error for error in result.errors)


def test_audit_fails_missing_test_result(tmp_path: Path) -> None:
    record = _record("task-0")
    record["test_result"] = None
    _write_jsonl(tmp_path / "bigcodebench-hard_20260529.jsonl", [record])

    result = audit.audit_dir(tmp_path, expected_count=1, max_tail_chars=200)

    assert not result.ok
    assert any("missing test_result" in error for error in result.errors)


def test_audit_flags_unittest_zero_tests_as_harness_error(tmp_path: Path) -> None:
    record = _record("task-0", stderr_tail="Ran 0 tests in 0.000s\n\nOK\n")
    _write_jsonl(tmp_path / "bigcodebench-hard_20260529.jsonl", [record])

    result = audit.audit_dir(tmp_path, expected_count=1, max_tail_chars=200)

    assert not result.ok
    assert any("0 tests" in error for error in result.errors)

def test_audit_flags_name_error_in_grader_namespace(tmp_path: Path) -> None:
    record = _record(
        "task-0",
        outcome=0,
        passed=False,
        stderr_tail=(
            "Traceback (most recent call last):\n"
            "  File \"/tmp/work/_bcb_grader_abc.py\", line 8, in test_valid_input\n"
            "    self.data = pd.DataFrame()\n"
            "                ^^\n"
            "NameError: name 'pd' is not defined\n"
            "Ran 7 tests in 0.003s\n\nFAILED (errors=7)\n"
        ),
    )
    _write_jsonl(tmp_path / "bigcodebench-hard_20260529.jsonl", [record])

    result = audit.audit_dir(tmp_path, expected_count=1, max_tail_chars=400)

    assert not result.ok
    assert any("NameError in grader namespace" in error for error in result.errors)


def test_render_report_contains_compact_row_evidence(tmp_path: Path) -> None:
    records = [
        _record("pass-task"),
        _record(
            "fail-task",
            outcome=0,
            passed=False,
            stderr_tail="FAIL: test_adds_one\nAssertionError: 1 != 2\nFAILED (failures=1)\n",
        ),
    ]
    _write_jsonl(tmp_path / "bigcodebench-hard_20260529.jsonl", records)

    result = audit.audit_dir(tmp_path, expected_count=2, max_tail_chars=80)
    report = audit.render_report(result)

    assert "n_rows: 2" in report
    assert "pass-task" in report
    assert "fail-task" in report
    assert "FAILED (failures=1)" in report

def test_render_report_can_limit_rows_for_full_runs(tmp_path: Path) -> None:
    records = [_record(f"task-{i}") for i in range(3)]
    _write_jsonl(tmp_path / "bigcodebench-hard_20260529.jsonl", records)

    result = audit.audit_dir(tmp_path, expected_count=3, max_tail_chars=80)
    report = audit.render_report(result, max_rows=1)
    rows_section = report.split("Rows:", 1)[1]
    assert "task-0" in rows_section
    assert "task-1" not in rows_section
    assert "... 2 more row(s) omitted" in rows_section
