#!/usr/bin/env bash
# Collect trajectories on BigCodeBench-Hard (300 tasks x 4 rollouts).
# Wall-clock: ~4-6 hours. Estimated cost: $80-100.
# Run from project root, inside tmux on the lab box.
set -euo pipefail

: "${ANTHROPIC_API_KEY:?must be set in env}"
: "${TS_REPO_PATH:?must be set in .env or shell}"

LOG_DIR="${CODE_PRM_LOG_DIR:-$PWD/data/raw/bigcodebench-hard}"
mkdir -p "$LOG_DIR"

python -m src.eval.collect_batch \
    --task_set bigcodebench-hard \
    --num_rollouts 4 \
    --concurrency 4 \
    --log_dir "$LOG_DIR" \
    --budget_usd 100
