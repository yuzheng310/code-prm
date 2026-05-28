#!/usr/bin/env bash
# Force-pass pilot labeling: validates the LLM-judge code path.
#
# The normal pilot collection has outcome=0 for all trajectories
# (because we don't run a test command yet). Under our schema, outcome=0
# routes to "outcome_zero_simplification" (Math-Shepherd shortcut, no
# judge calls). That's correct behavior but means we never exercised
# llm_judge_score_step in the pilot.
#
# This script:
#   1. Copies the pilot raw jsonl into a side directory
#   2. Forces outcome=1 on every trajectory
#   3. Runs label_all on the modified copy
#   4. Asserts the step_label distribution is non-degenerate
#
# Cost: ~$0.50-1.00 (Haiku at K=4 over the same ~38 tool steps).

set -euo pipefail

: "${ANTHROPIC_API_KEY:?must be set in env}"
# ANTHROPIC_BASE_URL is honored automatically by anthropic_client.py if set.

RAW_DIR="data/raw/pilot"
FORCE_DIR="data/raw/pilot_force_pass"
LABEL_DIR="data/labeled/pilot_force_pass"

rm -rf "$FORCE_DIR"
mkdir -p "$FORCE_DIR"

python -c "
import json, glob, os
src_files = [f for f in glob.glob('${RAW_DIR}/*.jsonl') if not os.path.basename(f).startswith('.')]
n_traj = 0
for src in src_files:
    dst = os.path.join('${FORCE_DIR}', os.path.basename(src))
    with open(src) as fin, open(dst, 'w') as fout:
        for line in fin:
            t = json.loads(line)
            t['outcome'] = 1  # force pass to exercise LLM-judge path
            fout.write(json.dumps(t) + '\n')
            n_traj += 1
print(f'Force-passed {n_traj} trajectories into ${FORCE_DIR}/')
"

python -m src.labeler.label_all \
    --input_dir "$FORCE_DIR" \
    --output_dir "$LABEL_DIR" \
    --budget_usd 5 \
    --K 4 \
    --clean \
    --allow_low_task_prompt_coverage

echo ""
echo "=== LLM-judge distribution check ==="
python -c "
import json, statistics, glob
from collections import Counter
labels = []
methods = Counter()
for f in glob.glob('${LABEL_DIR}/*.jsonl'):
    for line in open(f):
        t = json.loads(line)
        methods[t.get('label_method')] += 1
        for s in t['trajectory']:
            if s.get('step_label') is not None:
                labels.append(s['step_label'])
print(f'  label_method breakdown: {dict(methods)}')
print(f'  total labeled steps: {len(labels)}')
if labels:
    print(f'  mean step_label: {statistics.mean(labels):.3f}')
    print(f'  median: {statistics.median(labels):.3f}')
    bins = Counter()
    for x in labels:
        bins[round(x * 4) / 4] += 1
    print(f'  distribution (rounded to 0.25): {dict(sorted(bins.items()))}')
    distinct = len(bins)
    healthy = (0.2 <= statistics.mean(labels) <= 0.8) and distinct >= 2
    print(f'  → {\"HEALTHY\" if healthy else \"DEGENERATE\"}: mean in [0.2,0.8] AND distinct>=2 = {healthy}')
else:
    print('  no labels — LLM-judge path was not exercised. Check label_method above.')
"
