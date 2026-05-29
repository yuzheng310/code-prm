#!/usr/bin/env python3
"""Generate Phase 1 data-quality summaries and SVG figures.

The hard audit scripts answer whether data is structurally safe to consume.
This report answers whether the collected/labeled data has useful PRM signal.
It uses only the Python standard library plus this project's schema models.
"""
from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.labeler.trajectory_schema import Trajectory  # noqa: E402
from src.utils.jsonl_io import read_trajectories  # noqa: E402


SVG_W = 900
SVG_H = 480
MARGIN_LEFT = 88
MARGIN_RIGHT = 30
MARGIN_TOP = 54
MARGIN_BOTTOM = 78


def _read_all(input_dir: Path | None) -> list[Trajectory]:
    if input_dir is None or not input_dir.exists():
        return []
    rows: list[Trajectory] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        rows.extend(read_trajectories(path))
    return rows


def _pct(n: int, d: int) -> float:
    return (n / d) if d else 0.0


def _round4(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


def _percentile(values: list[int | float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _tool_steps(traj: Trajectory) -> list[str]:
    return [step.tool for step in traj.trajectory if step.tool is not None]


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda item: str(item[0]))}


def _raw_summary(raw: list[Trajectory]) -> dict[str, Any]:
    n = len(raw)
    task_to_outcomes: dict[str, list[int]] = defaultdict(list)
    for traj in raw:
        task_to_outcomes[traj.task_id].append(traj.outcome)

    rollout_denoms = sorted({len(v) for v in task_to_outcomes.values()})
    default_denominator = max(rollout_denoms) if rollout_denoms else 0
    pass_count_hist: Counter[str] = Counter()
    mixed = 0
    for outcomes in task_to_outcomes.values():
        passes = sum(outcomes)
        denom = len(outcomes)
        pass_count_hist[f"{passes}/{denom}"] += 1
        if 0 < passes < denom:
            mixed += 1

    tool_counts = [len(_tool_steps(traj)) for traj in raw]
    tool_usage: Counter[str] = Counter()
    costs: list[float] = []
    token_usage_count = 0
    test_result_count = 0
    for traj in raw:
        tool_usage.update(_tool_steps(traj))
        if traj.token_usage is not None:
            token_usage_count += 1
            costs.append(traj.token_usage.cost_usd)
        if traj.test_result is not None:
            test_result_count += 1

    mean_tool_steps = statistics.mean(tool_counts) if tool_counts else 0.0
    median_tool_steps = statistics.median(tool_counts) if tool_counts else 0.0
    return {
        "n_trajectories": n,
        "n_tasks": len(task_to_outcomes),
        "expected_rollouts_per_task_observed": default_denominator,
        "pass_rate": _round4(_pct(sum(t.outcome for t in raw), n)),
        "task_pass_count_histogram": dict(sorted(pass_count_hist.items())),
        "mixed_outcome_task_count": mixed,
        "mixed_outcome_task_ratio": _round4(_pct(mixed, len(task_to_outcomes))),
        "tool_steps_mean": _round4(float(mean_tool_steps)),
        "tool_steps_median": _round4(float(median_tool_steps)),
        "tool_steps_p90": _round4(float(_percentile(tool_counts, 0.90) or 0.0)),
        "tool_steps_histogram": _counter_dict(Counter(tool_counts)),
        "tool_usage": _counter_dict(tool_usage),
        "token_usage_coverage": _round4(_pct(token_usage_count, n)),
        "test_result_coverage": _round4(_pct(test_result_count, n)),
        "total_cost_usd": _round4(sum(costs)),
        "cost_histogram": _histogram(costs, bins=8),
    }


def _labeled_summary(labeled: list[Trajectory]) -> dict[str, Any]:
    methods = Counter(traj.label_method for traj in labeled)
    success_labels: list[float] = []
    labeled_success_tool_steps = 0
    total_success_tool_steps = 0
    bad_zero = 0
    for traj in labeled:
        for step in traj.trajectory:
            if step.tool is None:
                continue
            if traj.outcome == 1:
                total_success_tool_steps += 1
                if step.step_label is not None:
                    labeled_success_tool_steps += 1
                    success_labels.append(step.step_label)
            elif step.step_label not in (0, 0.0):
                bad_zero += 1

    return {
        "n_trajectories": len(labeled),
        "label_method": _counter_dict(methods),
        "outcome_one_tool_step_label_coverage": _round4(
            _pct(labeled_success_tool_steps, total_success_tool_steps)
        ),
        "success_label_mean": _round4(statistics.mean(success_labels)) if success_labels else None,
        "success_label_median": _round4(statistics.median(success_labels)) if success_labels else None,
        "success_label_values": _counter_dict(Counter(success_labels)),
        "success_label_distinct": len(set(success_labels)),
        "bad_outcome_zero_label_count": bad_zero,
    }


def _histogram(values: list[float], bins: int) -> dict[str, int]:
    if not values:
        return {}
    lo = min(values)
    hi = max(values)
    if lo == hi:
        return {f"{lo:.3f}": len(values)}
    width = (hi - lo) / bins
    counts: Counter[str] = Counter()
    for value in values:
        idx = min(int((value - lo) / width), bins - 1)
        start = lo + idx * width
        end = start + width
        counts[f"{start:.3f}-{end:.3f}"] += 1
    return dict(counts)


def _decision(raw_summary: dict[str, Any], labeled_summary: dict[str, Any] | None) -> dict[str, Any]:
    reasons: list[str] = []
    cautions: list[str] = []
    pass_rate = raw_summary["pass_rate"] or 0.0
    mixed = raw_summary["mixed_outcome_task_count"]
    mean_steps = raw_summary["tool_steps_mean"] or 0.0
    if pass_rate < 0.05 or pass_rate > 0.85:
        reasons.append(f"pass_rate={pass_rate:.1%} outside hard bounds [5%, 85%]")
    elif pass_rate < 0.15 or pass_rate > 0.70:
        cautions.append(f"pass_rate={pass_rate:.1%} outside preferred range [15%, 70%]")
    if raw_summary["token_usage_coverage"] < 0.95:
        reasons.append(f"token_usage_coverage={raw_summary['token_usage_coverage']:.1%} < 95%")
    if raw_summary["test_result_coverage"] < 1.0:
        reasons.append(f"test_result_coverage={raw_summary['test_result_coverage']:.1%} < 100%")
    if mixed < 20 and raw_summary["n_tasks"] >= 50:
        cautions.append(f"mixed_outcome_task_count={mixed} < 20")
    if mean_steps < 2.0:
        cautions.append(f"tool_steps_mean={mean_steps:.2f} < 2")
    if labeled_summary:
        if labeled_summary["bad_outcome_zero_label_count"]:
            reasons.append("outcome=0 trajectories contain non-zero labels")
        if labeled_summary["success_label_distinct"] and labeled_summary["success_label_distinct"] < 3:
            reasons.append(
                f"success labels degenerate: distinct={labeled_summary['success_label_distinct']} < 3"
            )
        coverage = labeled_summary["outcome_one_tool_step_label_coverage"]
        if coverage is not None and coverage < 0.95:
            reasons.append(f"outcome=1 label coverage={coverage:.1%} < 95%")
    status = "STOP" if reasons else ("CAUTION" if cautions else "GO")
    return {"status": status, "reasons": reasons, "cautions": cautions}


def _bar_svg(title: str, data: dict[str, int | float], *, x_label: str = "", y_label: str = "count") -> str:
    items = list(data.items())
    plot_w = SVG_W - MARGIN_LEFT - MARGIN_RIGHT
    plot_h = SVG_H - MARGIN_TOP - MARGIN_BOTTOM
    max_v = max([float(v) for _, v in items], default=1.0) or 1.0
    bar_gap = 10
    bar_w = max(8, (plot_w - bar_gap * max(len(items) - 1, 0)) / max(len(items), 1))
    parts = [_svg_header(title), _axis(plot_w, plot_h, y_label)]
    for i, (label, value) in enumerate(items):
        v = float(value)
        h = plot_h * v / max_v
        x = MARGIN_LEFT + i * (bar_w + bar_gap)
        y = MARGIN_TOP + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#4C78A8"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="12">{v:g}</text>')
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{MARGIN_TOP + plot_h + 20}" '
            f'text-anchor="middle" font-size="11">{html.escape(str(label))}</text>'
        )
    if x_label:
        parts.append(f'<text x="{SVG_W / 2}" y="{SVG_H - 18}" text-anchor="middle" font-size="13">{html.escape(x_label)}</text>')
    parts.append("</svg>\n")
    return "\n".join(parts)


def _svg_header(title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_W}" height="{SVG_H}" viewBox="0 0 {SVG_W} {SVG_H}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f'<text x="{SVG_W / 2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">{html.escape(title)}</text>'
    )


def _axis(plot_w: int, plot_h: int, y_label: str) -> str:
    x0 = MARGIN_LEFT
    y0 = MARGIN_TOP + plot_h
    return (
        f'<line x1="{x0}" y1="{MARGIN_TOP}" x2="{x0}" y2="{y0}" stroke="#333"/>\n'
        f'<line x1="{x0}" y1="{y0}" x2="{x0 + plot_w}" y2="{y0}" stroke="#333"/>\n'
        f'<text x="22" y="{MARGIN_TOP + plot_h / 2}" transform="rotate(-90 22,{MARGIN_TOP + plot_h / 2})" '
        f'text-anchor="middle" font-size="13">{html.escape(y_label)}</text>'
    )


def _write_svg(path: Path, title: str, data: dict[str, int | float], x_label: str = "") -> None:
    path.write_text(_bar_svg(title, data, x_label=x_label), encoding="utf-8")


def _write_report_md(out_dir: Path, summary: dict[str, Any]) -> None:
    raw = summary["raw"]
    labeled = summary.get("labeled")
    decision = summary["decision"]
    lines = [
        "# Phase 1 Data Quality Report",
        "",
        f"Status: **{decision['status']}**",
        "",
        "## Raw Summary",
        "",
    ]
    for key in [
        "n_trajectories",
        "n_tasks",
        "pass_rate",
        "task_pass_count_histogram",
        "mixed_outcome_task_count",
        "mixed_outcome_task_ratio",
        "tool_steps_mean",
        "tool_steps_median",
        "tool_steps_p90",
        "token_usage_coverage",
        "test_result_coverage",
        "total_cost_usd",
    ]:
        lines.append(f"- `{key}`: `{raw.get(key)}`")
    if labeled:
        lines.extend(["", "## Labeled Summary", ""])
        for key in [
            "n_trajectories",
            "label_method",
            "outcome_one_tool_step_label_coverage",
            "success_label_mean",
            "success_label_median",
            "success_label_values",
            "bad_outcome_zero_label_count",
        ]:
            lines.append(f"- `{key}`: `{labeled.get(key)}`")
    lines.extend(["", "## Figures", ""])
    for svg in sorted(out_dir.glob("*.svg")):
        lines.append(f"![{svg.stem}]({svg.name})")
    if decision["reasons"] or decision["cautions"]:
        lines.extend(["", "## Decision Notes", ""])
        for reason in decision["reasons"]:
            lines.append(f"- STOP: {reason}")
        for caution in decision["cautions"]:
            lines.append(f"- CAUTION: {caution}")
    out_dir.joinpath("report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_report(raw_dir: Path, labeled_dir: Path | None, out_dir: Path) -> dict[str, Any]:
    raw = _read_all(raw_dir)
    labeled = _read_all(labeled_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.svg"):
        old.unlink()

    raw_summary = _raw_summary(raw)
    labeled_summary = _labeled_summary(labeled) if labeled else None
    decision = _decision(raw_summary, labeled_summary)
    summary: dict[str, Any] = {"raw": raw_summary, "decision": decision}
    if labeled_summary is not None:
        summary["labeled"] = labeled_summary

    _write_svg(
        out_dir / "01_outcome_distribution.svg",
        "Outcome Distribution",
        {"fail": len([t for t in raw if t.outcome == 0]), "pass": len([t for t in raw if t.outcome == 1])},
        x_label="outcome",
    )
    _write_svg(out_dir / "02_pass_count_per_task.svg", "Pass Count Per Task", raw_summary["task_pass_count_histogram"], x_label="passes / rollouts")
    _write_svg(out_dir / "03_tool_steps_histogram.svg", "Tool Steps Per Trajectory", raw_summary["tool_steps_histogram"], x_label="tool steps")
    _write_svg(out_dir / "04_tool_usage_bar.svg", "Tool Usage", raw_summary["tool_usage"], x_label="tool")
    _write_svg(out_dir / "05_token_cost_histogram.svg", "Cost Per Trajectory", raw_summary["cost_histogram"], x_label="cost USD")
    if labeled_summary is not None:
        _write_svg(out_dir / "06_success_step_label_distribution.svg", "Success Step Label Distribution", labeled_summary["success_label_values"], x_label="step_label")
        _write_svg(out_dir / "07_label_method_breakdown.svg", "Label Method Breakdown", labeled_summary["label_method"], x_label="label_method")

    out_dir.joinpath("summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_report_md(out_dir, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--labeled_dir", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("docs/phase1-quality"))
    args = parser.parse_args()
    summary = generate_report(args.raw_dir, args.labeled_dir, args.out_dir)
    print(f"Phase 1 quality report written to {args.out_dir}")
    print(f"status: {summary['decision']['status']}")


if __name__ == "__main__":
    main()
