#!/usr/bin/env bash
# PILOT step labeling: LLM-judge label the 10 pilot trajectories.
#
# Purpose: validate that step_labeler runs end-to-end against real data,
# the manifest is written, and the step_label distribution is
# non-degenerate (not all 0 or all 1).
#
# Budget: $5 soft cap. Real cost typically ~$0.50-1.00 (Haiku at K=4
# over ~50-80 tool steps).
#
# Pilot data usually has low task_prompt coverage (the BigCodeBench
# tasks pass it via the prompt itself), so we pass
# --allow_low_task_prompt_coverage explicitly.

set -euo pipefail

: "${ANTHROPIC_API_KEY:?must be set in env}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://api.deepseek.com/anthropic}"
echo "[env] ANTHROPIC_BASE_URL: $ANTHROPIC_BASE_URL"

python -m src.labeler.label_all \
    --input_dir "${PILOT_RAW_DIR:-data/raw/pilot}" \
    --output_dir "${PILOT_LABELED_DIR:-data/labeled/pilot}" \
    --budget_usd 1000000 \
    --K 4 \
    --clean \
    --allow_low_task_prompt_coverage

echo ""
echo "=== Pilot labeling done. Audit: ==="
echo "  python scripts/09_audit_labeled_pilot.py --dir ${PILOT_LABELED_DIR:-data/labeled/pilot}"
