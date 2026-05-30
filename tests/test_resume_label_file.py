"""Unit tests for resuming a partially-written labeled jsonl tmp file."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "21_resume_label_file.py"


def _load_resume_module():
    spec = importlib.util.spec_from_file_location("resume_label_file", _SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


resume = _load_resume_module()


def _record(task_id: str, rollout_id: int, *, outcome: int, labels: list[float | None] | None = None) -> dict:
    labels = labels or [None, None]
    return {
        "task_id": task_id,
        "task_type": "bigcodebench-hard",
        "run_id": f"run-{task_id}-{rollout_id}",
        "rollout_id": rollout_id,
        "task_prompt": "Implement function.",
        "task_metadata": {"entry_point": "solve"},
        "trajectory": [
            {"step": 0, "tool": "write", "tool_args": {}, "tool_result": "ok", "step_label": labels[0]},
            {"step": 1, "tool": "bash", "tool_args": {}, "tool_result": "ok", "step_label": labels[1]},
        ],
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
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "cost_usd": 0.01,
        },
        "label_method": None if outcome else "outcome_zero_simplification",
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


class FakeClient:
    def __init__(self, outputs: list[bool]) -> None:
        self.outputs = list(outputs)

    def complete(self, messages, max_tokens=2048, temperature=0.9):
        passed = self.outputs.pop(0)
        text = "OUTCOME: PASS" if passed else "OUTCOME: FAIL"
        return text, 10, 5


def test_resume_labels_remaining_rows_and_promotes_tmp(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "out.jsonl"
    tmp_path_file = output_path.with_suffix(".jsonl.tmp")
    raw = [
        _record("task-a", 0, outcome=0, labels=[None, None]),
        _record("task-b", 1, outcome=1, labels=[None, None]),
        _record("task-c", 2, outcome=1, labels=[None, None]),
    ]
    partial = [
        _record("task-a", 0, outcome=0, labels=[0.0, 0.0]),
        _record("task-b", 1, outcome=1, labels=[0.25, 0.5]),
    ]
    _write_jsonl(input_path, raw)
    _write_jsonl(tmp_path_file, partial)

    tracker = type("Tracker", (), {"total_usd": 1.23, "per_model": {"claude-opus-4-7": 1.23}})()
    client = FakeClient([True, True, False, False])

    resume.resume_label_file(
        input_path=input_path,
        output_path=output_path,
        tmp_path=tmp_path_file,
        client=client,
        tracker=tracker,
        K=2,
        model="claude-opus-4-7",
    )

    assert output_path.exists()
    assert not tmp_path_file.exists()
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 3
    assert rows[0]["task_id"] == "task-a"
    assert rows[1]["trajectory"][0]["step_label"] == 0.25
    assert rows[2]["trajectory"][0]["step_label"] == 1.0
    assert rows[2]["trajectory"][1]["step_label"] == 0.0
    manifest = json.loads((output_path.parent / "labeling_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tool"] == "scripts.21_resume_label_file"
    assert manifest["resumed_rows"] == 2
    assert manifest["completed_rows"] == 3


def test_resume_rejects_prefix_mismatch(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "out.jsonl"
    tmp_path_file = output_path.with_suffix(".jsonl.tmp")
    raw = [_record("task-a", 0, outcome=0), _record("task-b", 1, outcome=1)]
    partial = [_record("task-x", 0, outcome=0, labels=[0.0, 0.0])]
    _write_jsonl(input_path, raw)
    _write_jsonl(tmp_path_file, partial)

    tracker = type("Tracker", (), {"total_usd": 0.0, "per_model": {}})()
    client = FakeClient([])

    try:
        resume.resume_label_file(
            input_path=input_path,
            output_path=output_path,
            tmp_path=tmp_path_file,
            client=client,
            tracker=tracker,
            K=2,
            model="claude-opus-4-7",
        )
    except ValueError as exc:
        assert "does not match raw prefix" in str(exc)
    else:
        raise AssertionError("expected ValueError")
