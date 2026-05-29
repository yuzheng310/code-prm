"""Regression tests for the BigCodeBench grader harness contract."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


TEST_CODE = """import unittest

class TestAddOne(unittest.TestCase):
    def test_adds_one(self):
        self.assertEqual(task_func(1), 2)
"""


def _harness(test_code: str, entry_point: str) -> str:
    return (
        f"from task import {entry_point} as task_func\n"
        f"{test_code}\n"
        'if __name__ == "__main__":\n'
        "    unittest.main()\n"
    )


def _run_grader(tmp_path: Path, solution: str) -> subprocess.CompletedProcess[str]:
    (tmp_path / "task.py").write_text(solution, encoding="utf-8")
    (tmp_path / "grader.py").write_text(_harness(TEST_CODE, "add_one"), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "grader.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_bigcodebench_harness_fails_wrong_solution(tmp_path: Path) -> None:
    result = _run_grader(tmp_path, "def add_one(x):\n    return x\n")

    assert result.returncode != 0
    assert "FAILED" in result.stderr


def test_bigcodebench_harness_passes_correct_solution(tmp_path: Path) -> None:
    result = _run_grader(tmp_path, "def add_one(x):\n    return x + 1\n")

    assert result.returncode == 0
    assert "OK" in result.stderr


def test_trajectory_logger_builds_executable_bigcodebench_harness() -> None:
    source = Path("src/collector/trajectory_logger.ts").read_text(encoding="utf-8")

    assert "from task import ${entryPoint} as task_func" in source
    assert 'if __name__ == "__main__":' in source
    assert "unittest.main()" in source
