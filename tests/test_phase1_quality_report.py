"""Unit tests for Phase 1 quality report generation."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "12_report_phase1_quality.py"


def _load_report_module():
    spec = importlib.util.spec_from_file_location("phase1_quality_report", _SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


report = _load_report_module()


def _step(step: int, tool: str, label: float | None = None) -> dict:
    return {
        "step": step,
        "role": "assistant",
        "thought": "",
        "tool": tool,
        "tool_args": {"path": "task.py"},
        "tool_result": "ok",
        "step_label": label,
    }


def _record(
    task_id: str,
    rollout_id: int,
    *,
    outcome: int,
    tools: list[str],
    labels: list[float | None] | None = None,
    label_method: str | None = None,
) -> dict:
    labels = labels if labels is not None else [None] * len(tools)
    return {
        "task_id": task_id,
        "task_type": "bigcodebench-hard",
        "run_id": f"run-{task_id}-{rollout_id}",
        "rollout_id": rollout_id,
        "task_prompt": "Implement function.",
        "task_metadata": {"entry_point": "solve"},
        "repo": None,
        "base_commit": None,
        "final_diff": None,
        "trajectory": [_step(i, tool, labels[i]) for i, tool in enumerate(tools)],
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
            "input_tokens": 100 + rollout_id,
            "output_tokens": 50,
            "cache_read_tokens": 25,
            "cache_creation_tokens": 0,
            "cost_usd": 0.1 + rollout_id / 100,
        },
        "label_method": label_method,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_phase1_quality_report_generates_summary_markdown_and_svgs(tmp_path: Path) -> None:
    raw_records = [
        _record("task-a", 0, outcome=1, tools=["write", "bash"]),
        _record("task-a", 1, outcome=0, tools=["write"]),
        _record("task-b", 0, outcome=1, tools=["read", "write", "bash"]),
        _record("task-b", 1, outcome=1, tools=["write", "bash"]),
    ]
    labeled_records = [
        _record("task-a", 0, outcome=1, tools=["write", "bash"], labels=[0.25, 0.75], label_method="llm_judge"),
        _record("task-a", 1, outcome=0, tools=["write"], labels=[0.0], label_method="outcome_zero_simplification"),
        _record("task-b", 0, outcome=1, tools=["read", "write", "bash"], labels=[0.5, 1.0, 1.0], label_method="llm_judge"),
        _record("task-b", 1, outcome=1, tools=["write", "bash"], labels=[0.0, 0.25], label_method="llm_judge"),
    ]
    raw_dir = tmp_path / "raw"
    labeled_dir = tmp_path / "labeled"
    out_dir = tmp_path / "quality"
    _write_jsonl(raw_dir / "bigcodebench-hard_20260529.jsonl", raw_records)
    _write_jsonl(labeled_dir / "bigcodebench-hard_20260529.jsonl", labeled_records)

    summary = report.generate_report(raw_dir=raw_dir, labeled_dir=labeled_dir, out_dir=out_dir)

    assert summary["raw"]["n_trajectories"] == 4
    assert summary["raw"]["n_tasks"] == 2
    assert summary["raw"]["pass_rate"] == 0.75
    assert summary["raw"]["task_pass_count_histogram"] == {"1/2": 1, "2/2": 1}
    assert summary["raw"]["mixed_outcome_task_count"] == 1
    assert summary["raw"]["tool_usage"] == {"bash": 3, "read": 1, "write": 4}
    assert summary["labeled"]["label_method"] == {"llm_judge": 3, "outcome_zero_simplification": 1}
    assert summary["labeled"]["success_label_values"] == {"0.0": 1, "0.25": 2, "0.5": 1, "0.75": 1, "1.0": 2}
    assert summary["decision"]["status"] == "CAUTION"

    assert json.loads((out_dir / "summary.json").read_text(encoding="utf-8")) == summary
    report_md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "# Phase 1 Data Quality Report" in report_md
    assert "mixed_outcome_task_count" in report_md
    for name in [
        "01_outcome_distribution.svg",
        "02_pass_count_per_task.svg",
        "03_tool_steps_histogram.svg",
        "04_tool_usage_bar.svg",
        "05_token_cost_histogram.svg",
        "06_success_step_label_distribution.svg",
        "07_label_method_breakdown.svg",
    ]:
        svg = (out_dir / name).read_text(encoding="utf-8")
        assert svg.startswith("<svg")
        assert "</svg>" in svg


def test_phase1_quality_report_marks_extreme_pass_rate_as_stop(tmp_path: Path) -> None:
    raw_records = [
        _record(f"task-{i}", 0, outcome=0, tools=["write"])
        for i in range(20)
    ]
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "quality"
    _write_jsonl(raw_dir / "bigcodebench-hard_20260529.jsonl", raw_records)

    summary = report.generate_report(raw_dir=raw_dir, labeled_dir=None, out_dir=out_dir)

    assert summary["decision"]["status"] == "STOP"
    assert any("pass_rate" in reason for reason in summary["decision"]["reasons"])
    assert (out_dir / "01_outcome_distribution.svg").exists()
    assert not (out_dir / "06_success_step_label_distribution.svg").exists()
