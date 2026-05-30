# Code-PRM Phase 1 Report

## Status: COMPLETE

## Deliverables
- Trajectory dataset: `data/code-trajectory-2.4k/{train,val,test}.jsonl`
- Total trajectories: 592
- Raw trajectories: `data/raw/bigcodebench-hard/`
- Labeled trajectories: `data/labeled/bigcodebench-hard/`
- Labeling manifest: `data/labeled/bigcodebench-hard/labeling_manifest.json`
- Quality report: `docs/phase1-quality/report.md`
- Figures:
  - `docs/phase1-quality/01_outcome_distribution.svg`
  - `docs/phase1-quality/02_pass_count_per_task.svg`
  - `docs/phase1-quality/03_tool_steps_histogram.svg`
  - `docs/phase1-quality/04_tool_usage_bar.svg`
  - `docs/phase1-quality/05_token_cost_histogram.svg`
  - `docs/phase1-quality/06_success_step_label_distribution.svg`
  - `docs/phase1-quality/07_label_method_breakdown.svg`

## Verification
- Pytest: all tests passed on the lab box.
- Raw audit: PASS (`scripts/08_audit_pilot.py --dir data/raw/bigcodebench-hard --expected-count 592 --expected-rollouts-per-task 4`)
- Labeled audit: PASS (`scripts/09_audit_labeled_pilot.py --dir data/labeled/bigcodebench-hard --expected-count 592 --expected-rollouts-per-task 4`)
- Quality report: GO (`scripts/12_report_phase1_quality.py`)
- Assembly checks: PASS (`scripts/30_assemble_dataset.py`)

## Dataset Statistics
| Split | Trajectories | Steps | Pass rate | Avg steps / traj |
|---|---:|---:|---:|---:|
| train | 474 | 2070 | 29.96% | 4.37 |
| val | 59 | 231 | 25.42% | 3.92 |
| test | 59 | 239 | 33.90% | 4.05 |
| total | 592 | 2540 | 29.90% | 4.29 |

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
- The raw dataset is structurally complete: 148 tasks × 4 rollouts = 592 trajectories.
- The raw pass rate (29.9%) is neither too low nor too high, leaving enough successful and failed trajectories for downstream ranking and PRM learning.
- There are 28 mixed-outcome tasks, which gives meaningful Best-of-N / trajectory-ranking signal.
- The average trajectory has 4.29 tool steps, so this is not merely one-shot solution data; it contains enough multi-step behavior to justify step-level supervision.
- The success-path judge labels are conservative but not degenerate: all five `step_label` buckets appear, and 250 / 673 success-path tool steps score at least 0.5.

## Known Limitations
- Phase 1 labels are LLM-judge surrogate labels, not real Monte Carlo rollout labels.
- `outcome=0` trajectories use outcome-only simplification (`step_label = 0` on tool steps).
- Phase 1 uses BigCodeBench-Hard only; SWE-bench remains future work.
- Full labeling cost is not reconstructible from the final manifest alone because the run was resumed after interruption.

## Final Artifacts
- Raw: `data/raw/bigcodebench-hard/`
- Labeled: `data/labeled/bigcodebench-hard/`
- Final dataset: `data/code-trajectory-2.4k/`
- Quality report: `docs/phase1-quality/`
