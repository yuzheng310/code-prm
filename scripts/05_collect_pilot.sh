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

LOG_DIR="${CODE_PRM_LOG_DIR:-$PWD/data/raw/pilot}"
mkdir -p "$LOG_DIR"

python -m src.eval.collect_batch \
    --task_set bigcodebench-hard \
    --num_rollouts 1 \
    --concurrency 1 \
    --limit 10 \
    --timeout_sec 300 \
    --log_dir "$LOG_DIR" \
    --budget_usd 10 \
    --clean

echo ""
echo "=== Pilot done. Inspect: ==="
echo "  ls -l $LOG_DIR/"
echo "  python -m src.utils.cost_aggregator --dir $LOG_DIR"
echo ""
echo "Verify in the jsonl files:"
echo "  - outcome distribution (will be all 0 — pilot doesn't run tests)"
echo "  - task_prompt populated"
echo "  - token_usage populated"
echo "  - rollout_id / run_id populated"
echo "  - trajectory has multiple steps with tool / tool_args / tool_result"
echo ""
echo "If schema OK + token_usage populated, the pipeline works end-to-end."
echo "SWE-bench docker grading is Phase 2 work."
