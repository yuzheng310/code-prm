#!/usr/bin/env bash
# Run LLM-judge step labeling on both swebench-lite and bigcodebench-hard
# collected data. (This is a weak-supervision surrogate, NOT Monte-Carlo
# rollout — see src/labeler/step_labeler.py docstring and spec §5.3.)
# Wall-clock: ~10-15 hours. Estimated cost: ~$120-140 (Haiku).
# Run from project root, inside tmux on the lab box.
set -euo pipefail

: "${ANTHROPIC_API_KEY:?must be set in env}"

python -m src.labeler.label_all \
    --input_dir data/raw/swebench-lite \
    --output_dir data/labeled/swebench-lite \
    --budget_usd 80 \
    --K 4 \
    --clean

python -m src.labeler.label_all \
    --input_dir data/raw/bigcodebench-hard \
    --output_dir data/labeled/bigcodebench-hard \
    --budget_usd 60 \
    --K 4 \
    --clean
