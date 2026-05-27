#!/usr/bin/env bash
# PILOT collection: 10 SWE-bench Lite tasks × 1 rollout = 10 trajectories.
# Wall-clock: ~30 min. Budget cap: $10 (soft estimate).
#
# Purpose: validate TS codeAgent integration + trajectory schema + token_usage
# coverage on a tiny sample BEFORE committing $250 to the full run.
#
# Run from project root, on the lab box.
set -euo pipefail

: "${ANTHROPIC_API_KEY:?must be set in env}"
: "${TS_REPO_PATH:?must be set in .env or shell}"

LOG_DIR="${CODE_PRM_LOG_DIR:-$PWD/data/raw/pilot}"
mkdir -p "$LOG_DIR"

python -m src.eval.collect_batch \
    --task_set swebench-lite \
    --num_rollouts 1 \
    --concurrency 1 \
    --limit 10 \
    --log_dir "$LOG_DIR" \
    --budget_usd 10

echo ""
echo "=== Pilot done. Inspect: ==="
echo "  ls -l $LOG_DIR/"
echo "  python -m src.utils.cost_aggregator --dir $LOG_DIR"
echo ""
echo "Verify in the jsonl files:"
echo "  - outcome distribution (not all 0 or all 1)"
echo "  - task_prompt populated"
echo "  - token_usage populated"
echo "  - rollout_id / run_id populated"
echo "Then proceed to scripts/10_collect_trajectories.sh"
