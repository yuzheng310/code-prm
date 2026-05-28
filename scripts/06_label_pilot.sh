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

python -m src.labeler.label_all \
    --input_dir "${PILOT_RAW_DIR:-data/raw/pilot}" \
    --output_dir "${PILOT_LABELED_DIR:-data/labeled/pilot}" \
    --budget_usd 5 \
    --K 4 \
    --clean \
    --allow_low_task_prompt_coverage

echo ""
echo "=== Pilot labeling done. Distribution check: ==="
python -c "
import json, statistics
from collections import Counter
labels = []
methods = Counter()
n_traj = 0
import glob
for f in glob.glob('data/labeled/pilot/*.jsonl'):
    for line in open(f):
        t = json.loads(line)
        n_traj += 1
        methods[t.get('label_method')] += 1
        for s in t['trajectory']:
            if s.get('step_label') is not None:
                labels.append(s['step_label'])
print(f'  trajectories: {n_traj}')
print(f'  label_method breakdown: {dict(methods)}')
print(f'  total labeled steps: {len(labels)}')
if labels:
    print(f'  mean step_label: {statistics.mean(labels):.3f}')
    print(f'  median: {statistics.median(labels):.3f}')
    bins = Counter()
    for x in labels:
        bins[round(x * 4) / 4] += 1
    print(f'  distribution (rounded to 0.25): {dict(sorted(bins.items()))}')
"
echo ""
echo "If mean ∈ [0.3, 0.7] and >= 2 distinct values, distribution is healthy."
echo "Next step: full collection (bash scripts/11_collect_bigcodebench.sh)"
