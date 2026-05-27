"""Unit tests for cost_aggregator."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from src.utils.cost_aggregator import aggregate


def _make_trajectory(
    *,
    task_id: str = "t1",
    task_type: str = "swe-bench-lite",
    outcome: int = 1,
    policy_model: str = "claude-sonnet-4-5",
    token_usage: dict | None = None,
) -> dict:
    record: dict = {
        "task_id": task_id,
        "task_type": task_type,
        "trajectory": [
            {"step": 0, "tool": "bash", "tool_args": {}, "tool_result": "ok"}
        ],
        "outcome": outcome,
        "policy_model": policy_model,
        "timestamp": "2026-05-27T10:00:00Z",
    }
    if token_usage is not None:
        record["token_usage"] = token_usage
    return record


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_aggregate_empty_dir(tmp_path: Path) -> None:
    out = aggregate(tmp_path)
    assert out["n_trajectories"] == 0
    assert out["n_with_usage"] == 0
    assert out["coverage_pct"] == 0.0
    assert out["total_cost_usd"] == 0.0


def test_aggregate_single_file_with_usage(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "a.jsonl",
        [
            _make_trajectory(
                task_id="x",
                token_usage={
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "cache_read_tokens": 0,
                    "cache_creation_tokens": 0,
                    "cost_usd": 0.10,
                },
            ),
        ],
    )
    out = aggregate(tmp_path)
    assert out["n_trajectories"] == 1
    assert out["n_with_usage"] == 1
    assert out["coverage_pct"] == 100.0
    assert out["total_input_tokens"] == 1000
    assert out["total_output_tokens"] == 500
    assert out["total_cost_usd"] == 0.10


def test_aggregate_mixed_coverage(tmp_path: Path) -> None:
    """Some trajectories have token_usage, some don't — coverage should reflect this."""
    _write_jsonl(
        tmp_path / "a.jsonl",
        [
            _make_trajectory(task_id="x1", token_usage={
                "input_tokens": 1000, "output_tokens": 0,
                "cache_read_tokens": 0, "cache_creation_tokens": 0,
                "cost_usd": 0.05,
            }),
            _make_trajectory(task_id="x2", token_usage=None),  # no usage
            _make_trajectory(task_id="x3", token_usage={
                "input_tokens": 2000, "output_tokens": 0,
                "cache_read_tokens": 0, "cache_creation_tokens": 0,
                "cost_usd": 0.10,
            }),
        ],
    )
    out = aggregate(tmp_path)
    assert out["n_trajectories"] == 3
    assert out["n_with_usage"] == 2
    assert abs(out["coverage_pct"] - 2 / 3 * 100) < 0.01
    assert out["total_cost_usd"] == 0.15


def test_aggregate_walks_nested_directories(tmp_path: Path) -> None:
    """rglob should pick up jsonl in subdirectories."""
    _write_jsonl(
        tmp_path / "a.jsonl",
        [_make_trajectory(task_id="t1", token_usage={
            "input_tokens": 1000, "output_tokens": 0,
            "cache_read_tokens": 0, "cache_creation_tokens": 0,
            "cost_usd": 0.05,
        })],
    )
    _write_jsonl(
        tmp_path / "sub" / "b.jsonl",
        [_make_trajectory(task_id="t2", token_usage={
            "input_tokens": 0, "output_tokens": 1000,
            "cache_read_tokens": 0, "cache_creation_tokens": 0,
            "cost_usd": 0.15,
        })],
    )
    out = aggregate(tmp_path)
    assert out["n_trajectories"] == 2
    assert out["total_cost_usd"] == 0.20


def test_aggregate_by_model_breakdown(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "a.jsonl",
        [
            _make_trajectory(task_id="t1", policy_model="claude-sonnet-4-5",
                             token_usage={"input_tokens": 100, "output_tokens": 50,
                                          "cache_read_tokens": 0, "cache_creation_tokens": 0,
                                          "cost_usd": 0.01}),
            _make_trajectory(task_id="t2", policy_model="claude-haiku-4-5",
                             token_usage={"input_tokens": 100, "output_tokens": 50,
                                          "cache_read_tokens": 0, "cache_creation_tokens": 0,
                                          "cost_usd": 0.001}),
            _make_trajectory(task_id="t3", policy_model="claude-sonnet-4-5",
                             token_usage=None),
        ],
    )
    out = aggregate(tmp_path)
    assert out["by_model"] == {"claude-sonnet-4-5": 2, "claude-haiku-4-5": 1}


def test_aggregate_sums_cache_tokens(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "a.jsonl",
        [
            _make_trajectory(task_id="t1", token_usage={
                "input_tokens": 100, "output_tokens": 50,
                "cache_read_tokens": 200, "cache_creation_tokens": 300,
                "cost_usd": 0.02,
            }),
            _make_trajectory(task_id="t2", token_usage={
                "input_tokens": 100, "output_tokens": 50,
                "cache_read_tokens": 400, "cache_creation_tokens": 600,
                "cost_usd": 0.03,
            }),
        ],
    )
    out = aggregate(tmp_path)
    assert out["total_cache_read_tokens"] == 600
    assert out["total_cache_creation_tokens"] == 900
