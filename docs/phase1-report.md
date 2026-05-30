# Code-PRM Phase 1 Report

## Status: RAW/LABELED COMPLETE; ORIGINAL SPLIT OBSOLETE

The raw and labeled Phase 1 data pool is complete. The original assembled split at
`data/code-trajectory-2.4k/` used trajectory-level random splitting and has
task-level leakage, so it is obsolete for Phase 2/3. The valid Phase 2/3 input has
been regenerated from the labeled source with the task-grouped assembly script into
`data/code-trajectory-2.4k-tasksplit/` (seed=42, mixed_alloc=18/4/6) and
independently verified: zero task overlap, complete rollouts, 592 conserved.

## Deliverables
- Task-grouped trajectory dataset: `data/code-trajectory-2.4k-tasksplit/{train,val,test}.jsonl`
- Total trajectories: 592
- Raw trajectories: `data/raw/bigcodebench-hard/`
- Labeled trajectories: `data/labeled/bigcodebench-hard/`
- Labeling manifest: `data/labeled/bigcodebench-hard/labeling_manifest.json`
- Quality report: `docs/phase1-quality/report.md`
- Split manifest: `data/code-trajectory-2.4k-tasksplit/split_manifest.json` (records seed, strategy, input dirs, per-split stats, pass-count histograms, mixed task ids, and all hard-check results)
- Figures:
  - `docs/phase1-quality/01_outcome_distribution.svg`
  - `docs/phase1-quality/02_pass_count_per_task.svg`
  - `docs/phase1-quality/03_tool_steps_histogram.svg`
  - `docs/phase1-quality/04_tool_usage_bar.svg`
  - `docs/phase1-quality/05_token_cost_histogram.svg`
  - `docs/phase1-quality/06_success_step_label_distribution.svg`
  - `docs/phase1-quality/07_label_method_breakdown.svg`

## Verification
- Pytest/lint: `uv run pytest tests/ -q` → 166 passed; `uv run --with ruff ruff check ...` → All checks passed.
- Raw audit: PASS (`scripts/08_audit_pilot.py --dir data/raw/bigcodebench-hard --expected-count 592 --expected-rollouts-per-task 4`)
- Labeled audit: PASS (`scripts/09_audit_labeled_pilot.py --dir data/labeled/bigcodebench-hard --expected-count 592 --expected-rollouts-per-task 4`)
- Quality report: GO for raw/labeled structure (`scripts/12_report_phase1_quality.py`)
- Original assembly checks: PASS structurally, but the original trajectory-level split is obsolete due to task leakage.
- Task-grouped assembly: implemented in `scripts/30_assemble_dataset.py` and run locally from `data/labeled/bigcodebench-hard` to generate `data/code-trajectory-2.4k-tasksplit/`; split-level hard checks all PASS and are recorded in `split_manifest.json`.

## Original Split Statistics (Obsolete)
| Split | Trajectories | Steps | Pass rate | Avg steps / traj |
|---|---:|---:|---:|---:|
| train | 474 | 2070 | 29.96% | 4.37 |
| val | 59 | 231 | 25.42% | 3.92 |
| test | 59 | 239 | 33.90% | 4.05 |
| total | 592 | 2540 | 29.90% | 4.29 |

These files are not valid held-out splits because every val/test task also appears in train through another rollout.

## Task-Grouped Split Statistics
Default task-grouped assembly keeps all 4 rollouts of a task in one split and allocates mixed-outcome tasks as `train/val/test = 18/4/6`.

| Split | Tasks | Trajectories | Steps | Pass rate | Mixed tasks | Pass-count histogram |
|---|---:|---:|---:|---:|---:|---|
| train | 113 | 452 | 1825 | 28.98% | 18 | `0/4:71, 1/4:6, 2/4:7, 3/4:5, 4/4:24` |
| val | 15 | 60 | 320 | 31.67% | 4 | `0/4:8, 1/4:2, 2/4:1, 3/4:1, 4/4:3` |
| test | 20 | 80 | 395 | 33.75% | 6 | `0/4:10, 1/4:2, 2/4:3, 3/4:1, 4/4:4` |
| total | 148 | 592 | 2540 | 29.90% | 28 | `0/4:89, 1/4:10, 2/4:11, 3/4:7, 4/4:31` |

Task-grouped split checks passed: no task overlap, every task has rollout ids `[0,1,2,3]`, and all 592 `(task_id, rollout_id)` pairs are conserved. The BoN test set is intentionally documented as underpowered: 6 mixed tasks can show a direction, not a strong statistical conclusion.

## Raw Collection Quality
- Tasks: 148
- Rollouts per task: 4
- Outcome distribution:
  - fail: 415
  - pass: 177
- Pass rate: 29.90%
- Mixed-outcome tasks: 28 / 148 (18.92%)
- Tool-step statistics:
  - mean: 4.2905
  - median: 3.0
  - p90: 7.0
- Tool usage:
  - bash: 1432
  - write: 606
  - read: 413
  - edit: 89
- Token-usage coverage: 592 / 592 (100.0%)
- Test-result coverage: 592 / 592 (100.0%)
- Real raw-collection cost: $29.0379 (`python -m src.utils.cost_aggregator --dir data/raw/bigcodebench-hard`)

## Labeling Summary
- `label_method` breakdown:
  - `llm_judge`: 177 trajectories
  - `outcome_zero_simplification`: 415 trajectories
- outcome=1 tool-step label coverage: 673 / 673 (100.0%)
- outcome=0 non-zero label violations: 0
- success-path distinct label values: 5
- success-path mean `step_label`: 0.3536
- success-path median `step_label`: 0.25
- success-path `step_label` distribution:
  - 0.0: 286
  - 0.25: 137
  - 0.5: 65
  - 0.75: 55
  - 1.0: 130

## Cost Notes
- Raw collection cost is verified from trajectory `token_usage`: $29.0379.
- `data/labeled/bigcodebench-hard/labeling_manifest.json` under-reports total labeling spend because the run was resumed from a partial `.tmp` file; it only captures the resumed portion's tracked cost.
- The relay/dashboard billing record should be treated as the source of truth for final LLM-judge labeling cost.

## Interpretation
- The raw/labeled data pool is structurally complete: 148 tasks × 4 rollouts = 592 trajectories.
- The raw pass rate (29.9%) is neither too low nor too high, leaving enough successful and failed trajectories for exploratory PRM learning.
- The original trajectory-level split is invalid for Phase 2/3 because it leaks task identities across train/val/test.
- The task-grouped split fixes leakage and makes held-out BoN constructible, but the held-out BoN sample remains small under the default 6 mixed test tasks.
- The average trajectory has 4.29 tool steps, so this is not merely one-shot solution data; it contains enough multi-step behavior to test a training pipeline.
- The success-path judge labels are conservative but not degenerate: all five `step_label` buckets appear, and 250 / 673 success-path tool steps score at least 0.5.

## Known Limitations
- Phase 1 labels are LLM-judge surrogate labels, not real Monte Carlo rollout labels.
- `outcome=0` trajectories use outcome-only simplification (`step_label = 0` on tool steps).
- 84.8% of all tool-step labels are 0; Phase 2 must report zero-predictor and outcome-only ORM baselines, plus success-path / `llm_judge`-only metrics.
- Phase 1 uses BigCodeBench-Hard only; SWE-bench remains future work.
- Full labeling cost is not reconstructible from the final manifest alone because the run was resumed after interruption.
- Held-out BoN under the default split has only 6 mixed-outcome test tasks; conclusions are exploratory/qualitative, not definitive.

## Final Artifacts
- Raw: `data/raw/bigcodebench-hard/`
- Labeled: `data/labeled/bigcodebench-hard/`
- Obsolete trajectory-level split: `data/code-trajectory-2.4k/`
- Valid task-grouped split: `data/code-trajectory-2.4k-tasksplit/`
- Quality report: `docs/phase1-quality/`
