"""Guard tests for label_all's --clean foot-gun protection."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_ALL = ["python", "-m", "src.labeler.label_all"]


def _run(args: list[str], cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.labeler.label_all", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_clean_refuses_when_output_dir_equals_input_dir(tmp_path: Path) -> None:
    """The single most destructive foot-gun: --clean on same dir wipes inputs."""
    same = tmp_path / "data"
    same.mkdir()
    # Drop an input file so the labeler has something to scan.
    (same / "x.jsonl").write_text(
        '{"task_id":"t1","task_type":"swe-bench-lite","trajectory":[],'
        '"outcome":1,"policy_model":"claude-sonnet-4-5",'
        '"timestamp":"2026-05-27T10:00:00Z","task_prompt":"p"}\n'
    )

    result = _run([
        "--input_dir", str(same),
        "--output_dir", str(same),
        "--budget_usd", "1",
        "--clean",
    ])
    # Must NOT exit 0; input must survive.
    assert result.returncode != 0, f"Expected non-zero exit; got 0. stderr={result.stderr}"
    assert (same / "x.jsonl").exists(), (
        "Input file destroyed! --clean guard failed. Output: " + result.stdout + result.stderr
    )


def test_clean_refuses_when_output_dir_contains_input_dir(tmp_path: Path) -> None:
    """--input_dir nested under --output_dir is also destructive."""
    outer = tmp_path / "outer"
    inner = outer / "raw"
    inner.mkdir(parents=True)
    (inner / "x.jsonl").write_text(
        '{"task_id":"t1","task_type":"swe-bench-lite","trajectory":[],'
        '"outcome":1,"policy_model":"claude-sonnet-4-5",'
        '"timestamp":"2026-05-27T10:00:00Z","task_prompt":"p"}\n'
    )

    result = _run([
        "--input_dir", str(inner),
        "--output_dir", str(outer),
        "--budget_usd", "1",
        "--clean",
    ])
    assert result.returncode != 0
    assert (inner / "x.jsonl").exists()
