#!/usr/bin/env bash
# PILOT collection: 10 BigCodeBench-Hard tasks × 1 rollout = 10 trajectories.
#
# Why BigCodeBench-Hard and not SWE-bench Lite for the pilot?
#   SWE-bench tasks are "fix this bug in django/sympy/etc." — they require
#   the agent to be inside the right repo at the right commit. Without
#   the SWE-bench docker harness (Phase 2 work), pi has no repo to read
#   and just spins until the 10-minute timeout.
#
#   BigCodeBench tasks are "write a Python function that does X" — fully
#   self-contained. Pi solves them in 1-3 minutes with no setup. Perfect
#   for validating the trajectory schema + extension pipeline before
#   committing real money.
#
# Wall-clock: ~15-30 min. Budget cap: $10 (soft estimate).
# Run from project root, on the lab box.
set -euo pipefail

: "${ANTHROPIC_API_KEY:?must be set in env}"
: "${TS_REPO_PATH:?must be set in .env or shell}"

# China-network mitigation: use HuggingFace mirror unless user explicitly set
# their own HF_ENDPOINT.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Anthropic relay default: DeepSeek's Anthropic-compatible endpoint.
# Users with direct Anthropic access can override:
#   ANTHROPIC_BASE_URL=https://api.anthropic.com bash scripts/...
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://api.deepseek.com/anthropic}"
echo "[env] ANTHROPIC_BASE_URL: $ANTHROPIC_BASE_URL"

LOG_DIR="${CODE_PRM_LOG_DIR:-$PWD/data/raw/pilot}"
mkdir -p "$LOG_DIR"

python -m src.eval.collect_batch \
    --task_set bigcodebench-hard \
    --num_rollouts 1 \
    --concurrency 1 \
    --limit 10 \
    --timeout_sec 300 \
    --log_dir "$LOG_DIR" \
    --budget_usd 1000000 \
    --clean \
    --stream_output

echo ""
echo "=== Pilot done. Inspect: ==="
echo "  ls -l $LOG_DIR/"
echo "  python -m src.utils.cost_aggregator --dir $LOG_DIR"
echo ""
echo "Verify in the jsonl files:"
echo "  - every trajectory has test_result"
echo "  - outcome == int(test_result.passed)"
echo "  - outcome distribution is non-degenerate enough to trust the grader"
echo "  - task_prompt, token_usage, rollout_id, and run_id are populated"
echo "  - failure stderr shows assertion/import/solution errors, not harness errors"
echo ""
echo "If schema + grader output look credible, run: bash scripts/06_label_pilot.sh"
