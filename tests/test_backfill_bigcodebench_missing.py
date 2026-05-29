"""Unit tests for missing BigCodeBench rollout backfill."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "13_backfill_bigcodebench_missing.py"


def _load_backfill_module():
    spec = importlib.util.spec_from_file_location("backfill_bigcodebench_missing", _SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


backfill = _load_backfill_module()


def _record(task_id: str, rollout_id: int) -> dict:
    return {
        "task_id": task_id,
        "task_type": "bigcodebench-hard",
        "run_id": f"run-{task_id}-{rollout_id}",
        "rollout_id": rollout_id,
        "task_prompt": "Implement function.",
        "task_metadata": {"entry_point": "solve"},
        "trajectory": [{"step": 0, "tool": "write", "tool_args": {}, "tool_result": "ok"}],
        "outcome": 0,
        "test_result": {
            "passed": False,
            "command": "python _bcb_grader.py",
            "exit_code": 1,
            "stdout_tail": "",
            "stderr_tail": "FAILED (failures=1)",
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
        "label_method": None,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_find_missing_rollouts_detects_only_expected_gaps(tmp_path: Path) -> None:
    records = [
        _record("BigCodeBench/177", 0),
        _record("BigCodeBench/177", 2),
        _record("BigCodeBench/177", 3),
        _record("BigCodeBench/273", 0),
        _record("BigCodeBench/273", 1),
        _record("BigCodeBench/273", 3),
        _record("BigCodeBench/657", 0),
        _record("BigCodeBench/657", 1),
        _record("BigCodeBench/657", 3),
        _record("BigCodeBench/999", 0),
        _record("BigCodeBench/999", 1),
        _record("BigCodeBench/999", 2),
        _record("BigCodeBench/999", 3),
    ]
    _write_jsonl(tmp_path / "bigcodebench-hard_20260529.jsonl", records)

    missing = backfill.find_missing_rollouts(tmp_path, expected_rollouts=4)

    assert missing == [
        backfill.MissingRollout("BigCodeBench/177", 1),
        backfill.MissingRollout("BigCodeBench/273", 2),
        backfill.MissingRollout("BigCodeBench/657", 2),
    ]


def test_backfill_uses_loaded_task_and_rollout_env(tmp_path: Path, monkeypatch) -> None:
    missing = [backfill.MissingRollout("BigCodeBench/177", 1)]
    calls = []

    def fake_runner(task, ts_repo, log_dir, timeout_sec, extra_env, stream_output):
        calls.append((task, ts_repo, log_dir, timeout_sec, extra_env, stream_output))
        return "ok"

    monkeypatch.setattr(backfill, "run_task_with_codeagent", fake_runner)
    task_by_id = {"BigCodeBench/177": {"task_id": "BigCodeBench/177", "prompt": "x"}}

    backfill.backfill_missing(
        missing,
        task_by_id=task_by_id,
        ts_repo=Path("/tmp/pi"),
        log_dir=tmp_path,
        timeout_sec=900,
        stream_output=True,
    )

    assert len(calls) == 1
    task, ts_repo, log_dir, timeout_sec, extra_env, stream_output = calls[0]
    assert task == task_by_id["BigCodeBench/177"]
    assert ts_repo == Path("/tmp/pi")
    assert log_dir == tmp_path
    assert timeout_sec == 900
    assert extra_env["CODE_PRM_ROLLOUT_ID"] == "1"
    assert extra_env["CODE_PRM_RUN_ID"]
    assert stream_output is True


def test_backfill_raises_when_task_is_not_in_dataset(tmp_path: Path, monkeypatch) -> None:
    def fake_runner(*args, **kwargs):
        raise AssertionError("runner should not be called")

    monkeypatch.setattr(backfill, "run_task_with_codeagent", fake_runner)

    try:
        backfill.backfill_missing(
            [backfill.MissingRollout("BigCodeBench/missing", 0)],
            task_by_id={},
            ts_repo=Path("/tmp/pi"),
            log_dir=tmp_path,
            timeout_sec=600,
            stream_output=False,
        )
    except KeyError as exc:
        assert "BigCodeBench/missing" in str(exc)
    else:
        raise AssertionError("expected KeyError")
