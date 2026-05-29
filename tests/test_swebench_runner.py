"""Unit tests for swebench_runner.run_task_with_codeagent.

We don't actually invoke `node` here; we monkeypatch `subprocess.run` so the
test runs in seconds and works without the TS codeAgent installed.
"""
from __future__ import annotations
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.eval import swebench_runner


def _swe_task() -> dict:
    return {"instance_id": "django__django-1", "problem_statement": "fix bug"}


def _bigcode_task() -> dict:
    return {
        "task_id": "BigCodeBench/0",
        "instruct_prompt": "Return x plus one.",
        "code_prompt": "def add_one(x: int) -> int:\n    pass\n",
        "entry_point": "add_one",
        "test": "import unittest\n\nclass Test(unittest.TestCase):\n    pass\n",
    }


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

    def fake_run(cmd, env=None, capture_output=None, text=None, timeout=None, cwd=None):
        captured["cmd"] = cmd
        captured["env"] = env
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    status = swebench_runner.run_task_with_codeagent(
        _bigcode_task(), tmp_path / "ts_repo", tmp_path / "logs",
    )
    assert status == "ok"
    p_idx = captured["cmd"].index("-p")
    prompt = captured["cmd"][p_idx + 1]
    assert "task.py" in prompt
    assert "entry point `add_one`" in prompt
    assert captured["env"]["CODE_PRM_TASK_TYPE"] == "bigcodebench-hard"


def test_bigcodebench_prompt_includes_code_prompt_and_entry_point(
    tmp_path: Path, monkeypatch
) -> None:
    """BigCodeBench prompt must tell pi the exact stub and import target."""
    captured: dict = {}

    def fake_run(cmd, env=None, capture_output=None, text=None, timeout=None, cwd=None):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["cwd"] = cwd
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    swebench_runner.run_task_with_codeagent(
        _bigcode_task(),
        tmp_path / "ts_repo",
        tmp_path / "logs",
    )

    p_idx = captured["cmd"].index("-p")
    prompt = captured["cmd"][p_idx + 1]
    assert "task.py" in prompt
    assert "entry point `add_one`" in prompt
    assert "def add_one(x: int) -> int:" in prompt
    assert "Return x plus one." in prompt


def test_bigcodebench_subprocess_uses_rollout_specific_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    """Concurrent BigCodeBench rollouts must not share task.py or grader files."""
    captured: dict = {}

    def fake_run(cmd, env=None, capture_output=None, text=None, timeout=None, cwd=None):
        captured["cwd"] = cwd
        captured["env"] = env
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    swebench_runner.run_task_with_codeagent(
        _bigcode_task(),
        tmp_path / "ts_repo",
        tmp_path / "logs",
        extra_env={"CODE_PRM_ROLLOUT_ID": "7", "CODE_PRM_RUN_ID": "run-abc"},
    )

    cwd = Path(captured["cwd"])
    assert cwd.is_dir()
    assert cwd.parent.name == "_workdirs"
    assert "BigCodeBench_0" in cwd.name
    assert "rollout_7" in cwd.name
    assert "run_abc" in cwd.name
    assert captured["env"]["CODE_PRM_WORK_DIR"] == str(cwd)


def test_falls_back_to_placeholder_prompt_when_missing(tmp_path: Path, monkeypatch) -> None:
    """If task dict has neither problem_statement nor prompt, fall back to a
    'Solve task X' placeholder rather than crashing."""
    captured: dict = {}

    def fake_run(cmd, env=None, capture_output=None, text=None, timeout=None, cwd=None):
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


def test_stream_output_prints_subprocess_lines(tmp_path: Path, monkeypatch, capsys) -> None:
    """Streaming mode should expose TS subprocess output while it runs."""

    class FakePipe:
        def __init__(self, lines: list[str]) -> None:
            self._lines = iter(lines)

        def readline(self) -> str:
            return next(self._lines, "")

        def close(self) -> None:
            pass

    class FakeProcess:
        returncode = 0

        def __init__(self) -> None:
            self.stdout = FakePipe(["thinking\n"])
            self.stderr = FakePipe(["warning\n"])

        def poll(self) -> int:
            return self.returncode

        def kill(self) -> None:
            raise AssertionError("process should not be killed")

    captured: dict = {}

    def fake_popen(cmd, env=None, stdout=None, stderr=None, text=None, bufsize=None):
        captured["cmd"] = cmd
        captured["env"] = env
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    status = swebench_runner.run_task_with_codeagent(
        _swe_task(),
        tmp_path / "ts_repo",
        tmp_path / "logs",
        extra_env={"CODE_PRM_ROLLOUT_ID": "7"},
        stream_output=True,
    )

    assert status == "ok"
    assert captured["cmd"][0] == "node"
    out = capsys.readouterr()
    assert "[django__django-1 rollout=7 stdout] thinking" in out.out
    assert "[django__django-1 rollout=7 stderr] warning" in out.err
