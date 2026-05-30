"""Unit tests for the assembly script's exit-criteria checks.

We don't shell out to the script; we import its functions directly.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

from src.labeler.trajectory_schema import Step, Trajectory


# --- import the script as a module ---


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "30_assemble_dataset.py"


def _load_assemble_module():
    spec = importlib.util.spec_from_file_location("assemble_dataset", _SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


assemble = _load_assemble_module()


# --- fixtures ---


def _traj(
    *,
    task_id: str,
    outcome: int,
    task_prompt: str | None,
    label_method: str | None,
    step_labels: list[float | None] | None = None,
    has_token_usage: bool = True,
) -> Trajectory:
    """Build a Trajectory with 3 tool steps + 1 thought step."""
    steps = [
        Step(step=0, tool="read_file", tool_args={"path": "x.py"}, tool_result="ok"),
        Step(step=1, tool="edit", tool_args={"diff": "..."}, tool_result="ok"),
        Step(step=2, tool="bash", tool_args={"cmd": "pytest"}, tool_result="ok"),
    ]
    if step_labels is not None:
        for step, label in zip(steps, step_labels):
            step.step_label = label
    raw: dict = {
        "task_id": task_id,
        "task_type": "swe-bench-lite",
        "trajectory": [s.model_dump() for s in steps],
        "outcome": outcome,
        "policy_model": "claude-sonnet-4-5",
        "timestamp": "2026-05-27T10:00:00Z",
        "task_prompt": task_prompt,
        "label_method": label_method,
    }
    if has_token_usage:
        raw["token_usage"] = {
            "input_tokens": 1000, "output_tokens": 500,
            "cache_read_tokens": 0, "cache_creation_tokens": 0,
            "cost_usd": 0.05,
        }
    return Trajectory(**raw)


# --- individual checks ---


def test_label_method_check_passes_when_all_set() -> None:
    trajs = [
        _traj(task_id="t1", outcome=1, task_prompt="p", label_method="llm_judge",
              step_labels=[0.5, 0.6, 0.7]),
    ]
    r = assemble.check_label_method_set(trajs)
    assert r.passed


def test_label_method_check_fails_when_missing() -> None:
    trajs = [
        _traj(task_id="t1", outcome=1, task_prompt="p", label_method=None),
    ]
    r = assemble.check_label_method_set(trajs)
    assert not r.passed


def test_task_prompt_coverage_passes_at_threshold() -> None:
    trajs = [
        _traj(task_id=f"t{i}", outcome=1, task_prompt="prompt",
              label_method="llm_judge", step_labels=[0.5, 0.5, 0.5])
        for i in range(19)
    ] + [
        _traj(task_id="t-no-prompt", outcome=1, task_prompt="",
              label_method="llm_judge", step_labels=[0.5, 0.5, 0.5])
    ]
    # 19/20 = 95% exactly
    r = assemble.check_task_prompt_coverage(trajs, threshold=0.95)
    assert r.passed


def test_task_prompt_coverage_fails_below_threshold() -> None:
    trajs = [
        _traj(task_id=f"t{i}", outcome=1, task_prompt="prompt",
              label_method="llm_judge", step_labels=[0.5, 0.5, 0.5])
        for i in range(9)
    ] + [
        _traj(task_id="t-no-prompt", outcome=1, task_prompt=None,
              label_method="llm_judge", step_labels=[0.5, 0.5, 0.5])
    ]
    # 9/10 = 90% < 95%
    r = assemble.check_task_prompt_coverage(trajs, threshold=0.95)
    assert not r.passed


def test_step_label_coverage_excludes_pure_thought_steps() -> None:
    """Pure-thought steps (tool=None) don't count toward coverage."""
    t = _traj(task_id="t1", outcome=1, task_prompt="p",
              label_method="llm_judge", step_labels=[0.5, 0.5, 0.5])
    # Add a thought-only step. It should NOT count for coverage on EITHER side.
    t.trajectory.append(Step(step=3, tool=None, thought="reflecting"))
    r = assemble.check_step_label_coverage([t], threshold=0.80)
    assert r.passed
    assert "3/3" in r.detail


def test_step_label_coverage_fails_when_under_threshold() -> None:
    t = _traj(task_id="t1", outcome=1, task_prompt="p",
              label_method="llm_judge", step_labels=[0.5, None, None])  # 1/3
    r = assemble.check_step_label_coverage([t], threshold=0.80)
    assert not r.passed


def test_step_label_coverage_ignores_outcome_zero() -> None:
    """outcome=0 trajectories should NOT count toward coverage stats."""
    t1 = _traj(task_id="t1", outcome=1, task_prompt="p",
               label_method="llm_judge", step_labels=[0.5, 0.5, 0.5])
    t2 = _traj(task_id="t2", outcome=0, task_prompt="p",
               label_method="outcome_zero_simplification",
               step_labels=[None, None, None])
    r = assemble.check_step_label_coverage([t1, t2], threshold=0.80)
    assert r.passed
    assert "3/3" in r.detail  # only t1's 3 tool steps counted


def test_token_usage_coverage() -> None:
    trajs = [
        _traj(task_id="t1", outcome=1, task_prompt="p", label_method="llm_judge",
              step_labels=[0.5, 0.5, 0.5], has_token_usage=True),
        _traj(task_id="t2", outcome=1, task_prompt="p", label_method="llm_judge",
              step_labels=[0.5, 0.5, 0.5], has_token_usage=False),
    ]
    r = assemble.check_token_usage_coverage(trajs, threshold=0.80)
    assert not r.passed   # 50% < 80%


def test_step_label_distribution_non_degenerate_passes() -> None:
    trajs = [
        _traj(task_id="t1", outcome=1, task_prompt="p", label_method="llm_judge",
              step_labels=[0.3, 0.5, 0.7]),
    ]
    r = assemble.check_step_label_distribution_non_degenerate(trajs)
    assert r.passed
    assert "mean=0.500" in r.detail


def test_step_label_distribution_fails_when_all_zero() -> None:
    trajs = [
        _traj(task_id="t1", outcome=1, task_prompt="p", label_method="llm_judge",
              step_labels=[0.0, 0.0, 0.0]),
    ]
    r = assemble.check_step_label_distribution_non_degenerate(trajs)
    assert not r.passed


def test_step_label_distribution_excludes_outcome_zero_simplification() -> None:
    """Distribution sanity check looks ONLY at llm_judge labels, not the
    forced-zero outcome=0 simplification."""
    t_judge = _traj(task_id="t1", outcome=1, task_prompt="p",
                    label_method="llm_judge", step_labels=[0.4, 0.5, 0.6])
    t_zero = _traj(task_id="t2", outcome=0, task_prompt="p",
                   label_method="outcome_zero_simplification",
                   step_labels=[0.0, 0.0, 0.0])
    r = assemble.check_step_label_distribution_non_degenerate([t_judge, t_zero])
    assert r.passed
    # If we accidentally included the outcome_zero labels, the mean would
    # crash to ~0.25 — but excluding them, mean is 0.5.
    assert "mean=0.500" in r.detail


def test_outcome_zero_simplification_check_passes_when_all_zero() -> None:
    t = _traj(task_id="t1", outcome=0, task_prompt="p",
              label_method="outcome_zero_simplification",
              step_labels=[0.0, 0.0, 0.0])
    r = assemble.check_outcome_zero_simplification_labels([t])
    assert r.passed


def test_outcome_zero_simplification_check_fails_when_nonzero_present() -> None:
    """outcome_zero_simplification + a tool step with step_label != 0 is a bug."""
    t = _traj(task_id="t1", outcome=0, task_prompt="p",
              label_method="outcome_zero_simplification",
              step_labels=[0.0, 0.5, 0.0])  # 0.5 is illegal here
    r = assemble.check_outcome_zero_simplification_labels([t])
    assert not r.passed
    assert "t1" in r.detail


def test_outcome_zero_simplification_check_ignores_thought_steps() -> None:
    """A pure-thought step (tool=None) with step_label=None is fine."""
    t = _traj(task_id="t1", outcome=0, task_prompt="p",
              label_method="outcome_zero_simplification",
              step_labels=[0.0, 0.0, 0.0])
    t.trajectory.append(Step(step=3, tool=None, thought="thinking"))  # None label
    r = assemble.check_outcome_zero_simplification_labels([t])
    assert r.passed


def test_outcome_zero_simplification_check_skips_llm_judge_trajectories() -> None:
    """The check only applies to outcome_zero_simplification, not llm_judge."""
    t = _traj(task_id="t1", outcome=1, task_prompt="p",
              label_method="llm_judge",
              step_labels=[0.5, 0.5, 0.5])  # non-zero is normal for judge
    r = assemble.check_outcome_zero_simplification_labels([t])
    assert r.passed


def test_run_all_checks_returns_six_results() -> None:
    trajs = [
        _traj(task_id="t1", outcome=1, task_prompt="p", label_method="llm_judge",
              step_labels=[0.5, 0.5, 0.5]),
    ]
    results = assemble.run_all_checks(trajs)
    assert len(results) == 6


# --- inspect_manifests (Fix 4 / round 7 + skipped_files in round 8) ---


def _write_manifest(
    dir_: Path,
    *,
    processed_outputs: list[Path],
    skipped: list[dict] | None = None,
) -> None:
    import json as _json
    manifest = {
        "tool": "src.labeler.label_all",
        "started_at": "2026-05-27T10:00:00Z",
        "finished_at": "2026-05-27T11:00:00Z",
        "input_dir": str(dir_),
        "output_dir": str(dir_),
        "K": 4,
        "model": "claude-haiku-4-5",
        "min_task_prompt_coverage": 0.95,
        "task_prompt_coverage": 0.99,
        "processed_files": [
            {"input": "in", "input_size": 1, "input_mtime": 0,
             "output": str(p)} for p in processed_outputs
        ],
        "skipped_files": skipped or [],
        "cost_per_model": {},
        "total_cost_usd": 0.0,
    }
    (dir_ / "labeling_manifest.json").write_text(_json.dumps(manifest))


def test_inspect_manifests_passes_when_clean(tmp_path: Path) -> None:
    out_file = tmp_path / "a.jsonl"
    out_file.write_text("{}\n")
    _write_manifest(tmp_path, processed_outputs=[out_file])
    results = assemble.inspect_manifests([tmp_path])
    assert all(r.passed for r in results), [r.detail for r in results]


def test_inspect_manifests_fails_when_manifest_missing(tmp_path: Path) -> None:
    # No manifest file at all
    results = assemble.inspect_manifests([tmp_path])
    assert any(not r.passed and "no labeling_manifest" in r.detail for r in results)


def test_inspect_manifests_fails_when_output_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    # Don't create the file
    _write_manifest(tmp_path, processed_outputs=[missing])
    results = assemble.inspect_manifests([tmp_path])
    assert any(not r.passed and "missing" in r.detail.lower() for r in results)


def test_inspect_manifests_fails_when_skipped_files_present(tmp_path: Path) -> None:
    out_file = tmp_path / "a.jsonl"
    out_file.write_text("{}\n")
    _write_manifest(
        tmp_path,
        processed_outputs=[out_file],
        skipped=[{"input": "bad.jsonl", "error": "API timeout"}],
    )
    results = assemble.inspect_manifests([tmp_path])
    assert any(
        not r.passed and "skipped" in r.name.lower() and "1" in r.detail
        for r in results
    )


def test_inspect_manifests_allows_skipped_with_flag(tmp_path: Path) -> None:
    out_file = tmp_path / "a.jsonl"
    out_file.write_text("{}\n")
    _write_manifest(
        tmp_path,
        processed_outputs=[out_file],
        skipped=[{"input": "bad.jsonl", "error": "API timeout"}],
    )
    results = assemble.inspect_manifests([tmp_path], allow_skipped=True)
    # All checks should pass under the override
    assert all(r.passed for r in results), [r.detail for r in results]

def test_arg_parser_defaults_to_bigcodebench_only() -> None:
    parser = assemble.build_arg_parser()
    args = parser.parse_args([])

    assert args.input_dirs == [Path("data/labeled/bigcodebench-hard")]
    assert args.output_dir == Path("data/code-trajectory-2.4k")
