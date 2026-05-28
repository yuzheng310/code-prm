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


def test_uses_pi_cli_with_prompt_for_swebench(tmp_path: Path, monkeypatch) -> None:
    """SWE-bench task is dispatched as `node cli.js -p <problem_statement>`."""
    captured: dict = {}

    def fake_run(cmd, env=None, capture_output=None, text=None, timeout=None):
        captured["cmd"] = cmd
        captured["env"] = env
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    status = swebench_runner.run_task_with_codeagent(
        _swe_task(), tmp_path / "ts_repo", tmp_path / "logs",
    )
    assert status == "ok"
    # pi CLI: node <cli.js> -p "<prompt>"
    assert captured["cmd"][0] == "node"
    assert "-p" in captured["cmd"]
    p_idx = captured["cmd"].index("-p")
    assert captured["cmd"][p_idx + 1] == "fix bug"  # the problem_statement
    # The env still carries the task_type so the extension can stamp it
    assert captured["env"]["CODE_PRM_TASK_TYPE"] == "swe-bench-lite"
    # And the full task payload so the extension can extract task_prompt
    assert "fix bug" in captured["env"]["CODE_PRM_TASK_JSON"]


def test_uses_pi_cli_for_bigcodebench(tmp_path: Path, monkeypatch) -> None:
    """BigCodeBench task: env still tagged bigcodebench-hard."""
    captured: dict = {}

    def fake_run(cmd, env=None, capture_output=None, text=None, timeout=None):
        captured["cmd"] = cmd
        captured["env"] = env
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    status = swebench_runner.run_task_with_codeagent(
        _bigcode_task(), tmp_path / "ts_repo", tmp_path / "logs",
    )
    assert status == "ok"
    p_idx = captured["cmd"].index("-p")
    assert captured["cmd"][p_idx + 1] == "do thing"  # the prompt
    assert captured["env"]["CODE_PRM_TASK_TYPE"] == "bigcodebench-hard"


def test_falls_back_to_placeholder_prompt_when_missing(tmp_path: Path, monkeypatch) -> None:
    """If task dict has neither problem_statement nor prompt, fall back to a
    'Solve task X' placeholder rather than crashing."""
    captured: dict = {}

    def fake_run(cmd, env=None, capture_output=None, text=None, timeout=None):
        captured["cmd"] = cmd
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    swebench_runner.run_task_with_codeagent(
        {"instance_id": "foo-1"},  # no problem_statement
        tmp_path / "ts_repo",
        tmp_path / "logs",
    )
    p_idx = captured["cmd"].index("-p")
    assert "Solve task foo-1" in captured["cmd"][p_idx + 1]


def test_returns_failed_on_nonzero_exit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: MagicMock(returncode=2))
    status = swebench_runner.run_task_with_codeagent(
        _swe_task(), tmp_path / "ts_repo", tmp_path / "logs",
    )
    assert status == "failed"


def test_raises_on_task_with_neither_id_field(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))
    with pytest.raises(KeyError):
        swebench_runner.run_task_with_codeagent(
            {"something_else": "x"}, tmp_path / "ts_repo", tmp_path / "logs",
        )


def test_timeout_returns_timeout_status(tmp_path: Path, monkeypatch) -> None:
    """One slow task must report 'timeout', NOT crash the batch."""
    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    status = swebench_runner.run_task_with_codeagent(
        _swe_task(), tmp_path / "ts_repo", tmp_path / "logs", timeout_sec=1,
    )
    assert status == "timeout"


def test_filenotfound_returns_launch_error(tmp_path: Path, monkeypatch) -> None:
    """Missing `node` binary should be reported as launch_error (config bug)."""
    def raise_fnf(*args, **kwargs):
        raise FileNotFoundError("node not found")

    monkeypatch.setattr(subprocess, "run", raise_fnf)
    status = swebench_runner.run_task_with_codeagent(
        _swe_task(), tmp_path / "ts_repo", tmp_path / "logs",
    )
    assert status == "launch_error"


def test_permission_error_returns_launch_error(tmp_path: Path, monkeypatch) -> None:
    """Permission errors are config bugs, not transient task failures."""
    def raise_perm(*args, **kwargs):
        raise PermissionError("can't exec node")

    monkeypatch.setattr(subprocess, "run", raise_perm)
    status = swebench_runner.run_task_with_codeagent(
        _swe_task(), tmp_path / "ts_repo", tmp_path / "logs",
    )
    assert status == "launch_error"


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
