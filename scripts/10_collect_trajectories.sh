#!/usr/bin/env bash
# Collect trajectories on SWE-bench Lite full set (300 tasks x 4 rollouts).
# Wall-clock: ~6-10 hours. Estimated cost: $200-250.
# Run from project root, inside tmux on the lab box.
set -euo pipefail

: "${ANTHROPIC_API_KEY:?must be set in env}"
: "${TS_REPO_PATH:?must be set in .env or shell}"

LOG_DIR="${CODE_PRM_LOG_DIR:-$PWD/data/raw/swebench-lite}"
mkdir -p "$LOG_DIR"

python -m src.eval.collect_batch \
    --task_set swebench-lite \
    --num_rollouts 4 \
    --concurrency 4 \
    --log_dir "$LOG_DIR" \
    --budget_usd 250
