#!/usr/bin/env bash
# Collect trajectories on BigCodeBench-Hard v0.1.4 (148 tasks x 4 rollouts = 592 trajectories).
# Run from project root, inside tmux on the lab box. Do not label until the
# post-run raw audit below passes.
set -euo pipefail

: "${ANTHROPIC_API_KEY:?must be set in env}"
: "${TS_REPO_PATH:?must be set in .env or shell}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://api.deepseek.com/anthropic}"
echo "[env] ANTHROPIC_BASE_URL: $ANTHROPIC_BASE_URL"

LOG_DIR="${CODE_PRM_LOG_DIR:-$PWD/data/raw/bigcodebench-hard}"
mkdir -p "$LOG_DIR"

python -m src.eval.collect_batch \
    --task_set bigcodebench-hard \
    --num_rollouts 4 \
    --concurrency 4 \
    --log_dir "$LOG_DIR" \
    --budget_usd 1000000 \
    --clean

echo ""
echo "=== BigCodeBench collection done. Audit before labeling: ==="
echo "  python scripts/08_audit_pilot.py --dir $LOG_DIR --expected-count 592 --expected-rollouts-per-task 4 --max-rows 50"
echo "  python -m src.utils.cost_aggregator --dir $LOG_DIR"