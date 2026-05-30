"""Unit tests for the labeled-pilot audit script."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "09_audit_labeled_pilot.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_labeled_pilot", _SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


audit = _load_audit_module()


def _step(step: int, tool: str | None, label: float | None) -> dict:
    return {
        "step": step,
        "role": "assistant",
        "thought": "",
        "tool": tool,
        "tool_args": {"path": "task.py"} if tool else {},
        "tool_result": "ok" if tool else "",
        "step_label": label,
    }


def _record(
    task_id: str,
    *,
    run_id: str | None = None,
    rollout_id: int = 0,
    outcome: int,
    method: str,
    labels: list[float | None],
) -> dict:
    return {
        "task_id": task_id,
        "task_type": "bigcodebench-hard",
        "run_id": run_id or f"run-{task_id}-{rollout_id}",
        "rollout_id": rollout_id,
        "task_prompt": "Implement function.",
        "task_metadata": {"entry_point": "solve"},
        "trajectory": [_step(0, None, None)] + [_step(i + 1, "write", label) for i, label in enumerate(labels)],
        "outcome": outcome,
        "test_result": {
            "passed": bool(outcome),
            "command": "python _bcb_grader.py",
            "exit_code": 0 if outcome else 1,
            "stdout_tail": "",
            "stderr_tail": "OK" if outcome else "FAILED (failures=1)",
            "duration_sec": 0.1,
        },
        "policy_model": "claude-sonnet-4-5",
        "timestamp": "2026-05-29T00:00:00Z",
        "token_usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_read_tokens": 30,
            "cache_creation_tokens": 0,
            "cost_usd": 0.01,
        },
        "label_method": method,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _write_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "tool": "src.labeler.label_all",
                "K": 4,
                "model": "claude-opus-4-7",
                "processed_files": [{"input": "raw.jsonl", "output": "labeled.jsonl"}],
                "skipped_files": [],
                "total_cost_usd": 0.12,
            }
        ),
        encoding="utf-8",
    )


def test_audit_passes_clean_labeled_pilot(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "bigcodebench-hard_20260529.jsonl",
        [
            _record("pass-1", outcome=1, method="llm_judge", labels=[0.25, 0.5, 1.0]),
            _record("pass-2", outcome=1, method="llm_judge", labels=[0.0, 0.75]),
            _record("fail-1", outcome=0, method="outcome_zero_simplification", labels=[0.0, 0.0]),
        ],
    )
    _write_manifest(tmp_path / "labeling_manifest.json")

    result = audit.audit_dir(tmp_path, expected_count=3, min_distinct_success_labels=3)

    assert result.ok
    assert result.summary["n_rows"] == 3
    assert result.summary["label_method"] == {"llm_judge": 2, "outcome_zero_simplification": 1}
    assert result.summary["bad_outcome_zero_labels"] == []
    assert result.summary["success_label_values"] == {0.0: 1, 0.25: 1, 0.5: 1, 0.75: 1, 1.0: 1}


def test_audit_passes_full_multi_rollout_labeled_set(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "out.jsonl",
        [
            _record("task-a", rollout_id=0, outcome=1, method="llm_judge", labels=[0.25]),
            _record("task-a", rollout_id=1, outcome=0, method="outcome_zero_simplification", labels=[0.0]),
            _record("task-b", rollout_id=0, outcome=1, method="llm_judge", labels=[0.5]),
            _record("task-b", rollout_id=1, outcome=1, method="llm_judge", labels=[1.0]),
        ],
    )
    _write_manifest(tmp_path / "labeling_manifest.json")

    result = audit.audit_dir(
        tmp_path,
        expected_count=4,
        expected_rollouts_per_task=2,
        min_distinct_success_labels=2,
    )

    assert result.ok
    assert result.summary["unique_task_ids"] == 2
    assert result.summary["rollouts_per_task"] == {"task-a": [0, 1], "task-b": [0, 1]}


def test_audit_fails_full_multi_rollout_labeled_set_with_missing_rollout(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "out.jsonl",
        [
            _record("task-a", rollout_id=0, outcome=1, method="llm_judge", labels=[0.25]),
            _record("task-a", rollout_id=0, run_id="run-task-a-other", outcome=0, method="outcome_zero_simplification", labels=[0.0]),
        ],
    )
    _write_manifest(tmp_path / "labeling_manifest.json")

    result = audit.audit_dir(tmp_path, expected_count=2, expected_rollouts_per_task=2)

    assert not result.ok
    assert any("expected rollout ids [0, 1], got [0]" in error for error in result.errors)

def test_audit_fails_missing_manifest(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "out.jsonl", [_record("pass", outcome=1, method="llm_judge", labels=[0.5])])

    result = audit.audit_dir(tmp_path, expected_count=1)

    assert not result.ok
    assert any("missing labeling_manifest.json" in error for error in result.errors)


def test_audit_fails_stale_or_wrong_count(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "out.jsonl", [_record("pass", outcome=1, method="llm_judge", labels=[0.5])])
    _write_manifest(tmp_path / "labeling_manifest.json")

    result = audit.audit_dir(tmp_path, expected_count=10)

    assert not result.ok
    assert any("Expected 10 labeled trajectories, found 1" in error for error in result.errors)


def test_audit_fails_outcome_zero_nonzero_label(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "out.jsonl",
        [_record("fail", outcome=0, method="outcome_zero_simplification", labels=[0.0, 0.25])],
    )
    _write_manifest(tmp_path / "labeling_manifest.json")

    result = audit.audit_dir(tmp_path, expected_count=1)

    assert not result.ok
    assert any("outcome=0 has non-zero tool step labels" in error for error in result.errors)


def test_audit_fails_success_missing_llm_labels(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "out.jsonl", [_record("pass", outcome=1, method="llm_judge", labels=[None, None])])
    _write_manifest(tmp_path / "labeling_manifest.json")

    result = audit.audit_dir(tmp_path, expected_count=1)

    assert not result.ok
    assert any("outcome=1 has unlabeled tool steps" in error for error in result.errors)


def test_audit_fails_degenerate_success_labels(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "out.jsonl",
        [
            _record("pass-1", outcome=1, method="llm_judge", labels=[1.0, 1.0]),
            _record("pass-2", outcome=1, method="llm_judge", labels=[1.0]),
            _record("fail", outcome=0, method="outcome_zero_simplification", labels=[0.0]),
        ],
    )
    _write_manifest(tmp_path / "labeling_manifest.json")

    result = audit.audit_dir(tmp_path, expected_count=3, min_distinct_success_labels=2)

    assert not result.ok
    assert any("success-path labels are degenerate" in error for error in result.errors)


def test_render_report_contains_row_labels(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "out.jsonl",
        [
            _record("pass", outcome=1, method="llm_judge", labels=[0.25, 0.75]),
            _record("fail", outcome=0, method="outcome_zero_simplification", labels=[0.0]),
        ],
    )
    _write_manifest(tmp_path / "labeling_manifest.json")

    result = audit.audit_dir(tmp_path, expected_count=2)
    report = audit.render_report(result)

    assert "n_rows: 2" in report
    assert "pass outcome=1 method=llm_judge labels=[0.25, 0.75]" in report
    assert "fail outcome=0 method=outcome_zero_simplification labels=[0.0]" in report
