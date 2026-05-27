"""Unit tests for swebench_runner.run_task_with_codeagent.

We don't actually invoke `node` here; we monkeypatch `subprocess.run` so the
test runs in seconds and works without the TS codeAgent installed.
"""
from __future__ import annotations
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.eval import swebench_runner


def _swe_task() -> dict:
    return {"instance_id": "django__django-1", "problem_statement": "fix bug"}


def _bigcode_task() -> dict:
    return {"task_id": "BigCodeBench/0", "prompt": "do thing"}


def test_uses_instance_id_for_swebench(tmp_path: Path, monkeypatch) -> None:
    """SWE-bench task is dispatched with --task-type swe-bench-lite."""
    captured = {}

    def fake_run(cmd, env=None, capture_output=None, text=None, timeout=None):
        captured["cmd"] = cmd
        captured["env"] = env
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok = swebench_runner.run_task_with_codeagent(
        _swe_task(), tmp_path / "ts_repo", tmp_path / "logs",
    )
    assert ok is True
    assert "--task-id" in captured["cmd"]
    idx = captured["cmd"].index("--task-id")
    assert captured["cmd"][idx + 1] == "django__django-1"
    type_idx = captured["cmd"].index("--task-type")
    assert captured["cmd"][type_idx + 1] == "swe-bench-lite"


def test_uses_task_id_for_bigcodebench(tmp_path: Path, monkeypatch) -> None:
    """BigCodeBench task is dispatched with --task-type bigcodebench-hard."""
    captured = {}

    def fake_run(cmd, env=None, capture_output=None, text=None, timeout=None):
        captured["cmd"] = cmd
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok = swebench_runner.run_task_with_codeagent(
        _bigcode_task(), tmp_path / "ts_repo", tmp_path / "logs",
    )
    assert ok is True
    idx = captured["cmd"].index("--task-id")
    assert captured["cmd"][idx + 1] == "BigCodeBench/0"
    type_idx = captured["cmd"].index("--task-type")
    assert captured["cmd"][type_idx + 1] == "bigcodebench-hard"


def test_raises_on_task_with_neither_id_field(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))
    with pytest.raises(KeyError):
        swebench_runner.run_task_with_codeagent(
            {"something_else": "x"}, tmp_path / "ts_repo", tmp_path / "logs",
        )


def test_timeout_returns_false_instead_of_raising(tmp_path: Path, monkeypatch) -> None:
    """One slow task must NOT crash the whole batch."""
    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    ok = swebench_runner.run_task_with_codeagent(
        _swe_task(), tmp_path / "ts_repo", tmp_path / "logs", timeout_sec=1,
    )
    assert ok is False  # no exception propagated


def test_filenotfound_returns_false(tmp_path: Path, monkeypatch) -> None:
    """Missing `node` binary should be reported as failure, not crash."""
    def raise_fnf(*args, **kwargs):
        raise FileNotFoundError("node not found")

    monkeypatch.setattr(subprocess, "run", raise_fnf)
    ok = swebench_runner.run_task_with_codeagent(
        _swe_task(), tmp_path / "ts_repo", tmp_path / "logs",
    )
    assert ok is False


def test_extra_env_forwarded(tmp_path: Path, monkeypatch) -> None:
    """Caller-supplied env vars (e.g. CODE_PRM_ROLLOUT_ID) must reach subprocess."""
    captured = {}

    def fake_run(cmd, env=None, capture_output=None, text=None, timeout=None):
        captured["env"] = env
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    swebench_runner.run_task_with_codeagent(
        _swe_task(), tmp_path / "ts_repo", tmp_path / "logs",
        extra_env={"CODE_PRM_ROLLOUT_ID": "3", "CODE_PRM_RUN_ID": "abc"},
    )
    assert captured["env"]["CODE_PRM_ROLLOUT_ID"] == "3"
    assert captured["env"]["CODE_PRM_RUN_ID"] == "abc"
    assert captured["env"]["CODE_PRM_LOG_DIR"] == str(tmp_path / "logs")
    assert captured["env"]["CODE_PRM_TASK_TYPE"] == "swe-bench-lite"
