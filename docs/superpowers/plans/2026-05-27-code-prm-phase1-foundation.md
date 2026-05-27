# Code-PRM Phase 1 Implementation Plan: Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## ⚠️ Plan revision log (2026-05-27, post-review)

This plan was reviewed mid-execution. Multiple revisions apply — read these
BEFORE running any task to avoid acting on stale text.

| # | Revision | Affected sections |
|---|---|---|
| R1 | OpenR baseline TRAINING dropped (spec §13 decision); only the OpenR clone is kept for reference and Best-of-N inference code | Task 6, 7 (now SKIPPED) |
| R2 | "MC labels" renamed to "step labels" everywhere. The Phase 1 labeler is an **LLM-judge surrogate**, not Monte-Carlo rollout (spec §5.3). Field `Step.mc_label` → `Step.step_label`. Module `mc_labeler.py` → `step_labeler.py`. Function `mc_rollout_for_step` → `llm_judge_score_step`. | Task 8, 16, 17, 18, 19 |
| R3 | Trajectory schema extended with `run_id`, `rollout_id`, `repo`, `base_commit`, `final_diff`, `test_result`, `token_usage`, `label_method`. Replay/cost/test fields are optional but recommended. | Task 8 |
| R4 | Collection output layout changed from nested `rollout_k/` subdirs to flat single directory. Trajectory carries `rollout_id` instead. TS logger must read `CODE_PRM_ROLLOUT_ID` and `CODE_PRM_RUN_ID` env vars. | Task 9, 13, 14, 19, 20 |
| R5 | `glob("*.jsonl")` → `rglob("*.jsonl")` in label_all.py and downstream readers, for robustness against any future nested layouts. Output filenames flatten relative subpath with "__" to avoid collisions. | Task 19, 20 |
| R6 | Cost tracking honesty: `collect_batch.py` does NOT enforce a hard budget cap (Python doesn't see real API spend). Soft pre-flight estimate only. Real cost via `src/utils/cost_aggregator.py` after collection, reading `Trajectory.token_usage`. | Task 12, 13, 14 |
| R7 | `swebench_runner.py` renamed in spirit: it's a "task loader + TS launcher", not a full SWE-bench harness. Outcome attribution clarified — TS side decides, not Python. | Task 10 |
| R8 | README checkbox initial state `[ ] Phase 1` (was `[x]` by mistake). | Task 1 |
| R9 | Python pinned to 3.12 + PyTorch 2.5 + CUDA 12.4 to match rental GPU image. | Task 2 |

The original task text below is preserved as history. Where it contradicts a
revision above, the revision wins. Specific tasks (8, 13, 16, 17, 19, 20)
should be cross-checked against the current source files before executing.

**Goal:** Build the data pipeline foundation — 2400 code-agent trajectories with step-level labels (LLM-judge surrogate) — so that Phase 2 (training) has clean inputs.

**Architecture:** Fork OpenR (reference only); instrument existing TypeScript codeAgent to emit `trajectory.jsonl`; build a Python step labeler that asks an LLM K=4 times per step whether the partial trajectory will succeed, recording the success fraction as `step_label ∈ [0, 1]` (LLM-judge surrogate; real MC deferred to Phase 2 future work).

**Tech Stack:** Python 3.12, PyTorch 2.5, Transformers, PEFT, OpenR (referenced), Anthropic SDK, SWE-bench, BigCodeBench, TypeScript (codeAgent integration).

**Deliverable at end of Phase 1:**
- `data/code-trajectory-2.4k/{train,val,test}.jsonl` with LLM-judge step labels
- Real cost (from token_usage aggregation) ≤ $500 (Phase 1 portion of $800 total)

**Out of scope (later phases):** PRM training (Phase 2), CodeProcessBench + end-to-end eval (Phase 3), real MC rollout upgrade (Future Work).

---

## File Structure

Files this plan will create (under `/Users/xiaoyuzheng/lightcode/agentrl/`):

```
agentrl/
├── pyproject.toml                          # Python project config (Task 1)
├── .gitignore                              # ignore data/, checkpoints/, .env (Task 1)
├── .env.example                            # API keys template (Task 1)
├── README.md                               # project root README (Task 1)
├── environment.yml                         # conda env spec (Task 2)
├── third_party/
│   └── openr/                              # fork submodule (Task 3)
├── src/
│   ├── __init__.py
│   ├── collector/
│   │   ├── __init__.py
│   │   └── ts_logger_spec.md               # TS-side integration contract (Task 8)
│   ├── labeler/
│   │   ├── __init__.py
│   │   ├── mc_labeler.py                   # MC rollout label generator (Task 16)
│   │   ├── anthropic_client.py             # rate-limited API wrapper (Task 15)
│   │   └── trajectory_schema.py            # pydantic schemas (Task 8)
│   ├── eval/
│   │   ├── __init__.py
│   │   └── swebench_runner.py              # SWE-bench task launcher (Task 10)
│   └── utils/
│       ├── __init__.py
│       ├── cost_tracker.py                 # token + $ tracking (Task 14)
│       └── jsonl_io.py                     # streaming read/write helpers (Task 8)
├── scripts/
│   ├── 00_setup_lab_box.sh                 # 3090 box bootstrap (Task 4)
│   ├── 01_train_openr_baseline.sh          # math PRM baseline (Task 6)
│   ├── 02_eval_openr_baseline.sh           # baseline eval (Task 7)
│   ├── 10_collect_trajectories.sh          # batched trajectory collection (Task 13)
│   ├── 20_label_mc.sh                      # MC labeling driver (Task 20)
│   └── 30_assemble_dataset.py              # train/val/test split (Task 23)
├── tests/
│   ├── __init__.py
│   ├── test_mc_labeler.py                  # unit tests for labeler (Task 17)
│   ├── test_trajectory_schema.py           # schema validation tests (Task 8)
│   └── fixtures/
│       └── synthetic_trajectory.json       # tiny trajectory for tests (Task 17)
├── data/                                   # gitignored
│   ├── raw/                                # raw collected trajectories
│   ├── labeled/                            # post-MC-labeling
│   └── code-trajectory-2.4k/               # final train/val/test
└── docs/
    ├── superpowers/
    │   ├── specs/
    │   │   └── 2026-05-27-code-prm-design.md  # (already exists)
    │   └── plans/
    │       └── 2026-05-27-code-prm-phase1-foundation.md  # (this file)
    └── phase1-report.md                    # phase 1 summary (Task 24)
```

TS codeAgent files (path: user's existing TS repo, abbreviated as `$TS_REPO`):

```
$TS_REPO/
└── src/
    └── hooks/
        └── trajectory_logger.ts            # new (Task 9)
```

---

## Prerequisites (do before Task 1)

You should have these ready BEFORE starting Task 1:

- [ ] **Anthropic API key with $400+ budget** — get from console.anthropic.com, save as `ANTHROPIC_API_KEY`
- [ ] **SSH access to the 3090 lab box** — confirm `ssh <labbox>` works; you'll set up there in Task 4
- [ ] **Disk space**: ≥ 100 GB free on the lab box (PRM800K + checkpoints + datasets)
- [ ] **Path to your existing TS codeAgent repo** — note it down, call it `$TS_REPO`. Will be referenced in Task 9.
- [ ] **Docker installed** on either local Mac or lab box (for SWE-bench harness in Task 10)

---

## Week 1: Environment Setup & OpenR Baseline

### Task 1: Bootstrap Python project layout

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `README.md`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "code-prm"
version = "0.1.0"
description = "Process Reward Model for Code Agent Trajectories"
requires-python = ">=3.11,<3.12"
dependencies = [
    "torch>=2.4,<2.6",
    "transformers>=4.45,<4.50",
    "peft>=0.13",
    "accelerate>=1.0",
    "anthropic>=0.39",
    "pydantic>=2.9",
    "datasets>=3.0",
    "pytest>=8.0",
    "tqdm>=4.66",
    "tenacity>=9.0",
    "fastapi>=0.115",
    "uvicorn>=0.32",
    "rich>=13.0",
]

[project.optional-dependencies]
dev = ["ruff>=0.7", "mypy>=1.13", "ipython>=8.0"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.egg-info/

# Envs
.env
.venv/
venv/

# Data & checkpoints (large)
data/
checkpoints/
wandb/
outputs/

# OS
.DS_Store
Thumbs.db

# Third party
third_party/openr/
```

- [ ] **Step 3: Write `.env.example`**

```bash
# Anthropic API
ANTHROPIC_API_KEY=sk-ant-...

# Lab box (for remote training)
LAB_BOX_SSH=user@10.x.x.x

# TS codeAgent repo path (for Task 9)
TS_REPO_PATH=/path/to/your/ts/codeagent

# Budget tracking
MAX_BUDGET_USD=800
```

- [ ] **Step 4: Write minimal `README.md`**

```markdown
# Code-PRM

Process Reward Model for Code Agent multi-turn trajectories.

See `docs/superpowers/specs/2026-05-27-code-prm-design.md` for full design.

## Quick Start

\`\`\`bash
conda env create -f environment.yml
conda activate code-prm
cp .env.example .env  # then fill in keys
pytest tests/         # smoke test
\`\`\`

## Status

- [x] Phase 1: Foundation (data pipeline, OpenR baseline)
- [ ] Phase 2: Training
- [ ] Phase 3: Eval & Ship
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore .env.example README.md
git commit -m "feat: bootstrap python project layout"
```

---

### Task 2: Set up local conda env

**Files:**
- Create: `environment.yml`

- [ ] **Step 1: Write `environment.yml`**

```yaml
name: code-prm
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pytorch=2.4
  - pytorch-cuda=12.1
  - pip:
      - -e .
```

- [ ] **Step 2: Create env locally**

Run: `conda env create -f environment.yml -n code-prm`
Expected: env created in ~5 min; ends with `Successfully installed code-prm-0.1.0`.

- [ ] **Step 3: Activate and smoke-test imports**

Run:
```bash
conda activate code-prm
python -c "import torch, transformers, peft, anthropic; print(torch.__version__, transformers.__version__)"
```
Expected: prints two versions, no ImportError.

- [ ] **Step 4: Commit**

```bash
git add environment.yml
git commit -m "feat: add conda environment spec"
```

---

### Task 3: Fork & vendor OpenR

**Files:**
- Add: `third_party/openr/` (git submodule)

- [ ] **Step 1: Fork OpenR on GitHub**

Browser: go to https://github.com/openreasoner/openr, click Fork → fork to your account.

- [ ] **Step 2: Add as submodule**

Run:
```bash
git submodule add https://github.com/<your-gh-user>/openr.git third_party/openr
git submodule update --init --recursive
```
Expected: `third_party/openr/` populated with OpenR repo.

- [ ] **Step 3: Pin a specific commit (for reproducibility)**

```bash
cd third_party/openr
git checkout main
git rev-parse HEAD > /tmp/openr_pinned_commit.txt
cd ../..
cat /tmp/openr_pinned_commit.txt   # note this in commit msg
```

- [ ] **Step 4: Read OpenR's README to confirm training entry point exists**

Read: `third_party/openr/README.md` and `third_party/openr/train/mat/README.md`
Confirm: `train/mat/` directory exists with PRM training scripts. Note the entry script name (typically `train_prm.py` or similar).

- [ ] **Step 5: Commit submodule**

```bash
git add .gitmodules third_party/openr
git commit -m "deps: vendor openr at <pinned-commit>"
```

---

### Task 4: Provision the lab box (3090) for training

**Files:**
- Create: `scripts/00_setup_lab_box.sh`

- [ ] **Step 1: Write `scripts/00_setup_lab_box.sh`**

```bash
#!/usr/bin/env bash
# Run this ON the lab box, not locally.
set -euo pipefail

# 1. Check GPU
nvidia-smi | grep "RTX 3090" || { echo "ERROR: no 3090 found"; exit 1; }

# 2. Install miniconda if missing
if ! command -v conda &>/dev/null; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p $HOME/miniconda3
    source $HOME/miniconda3/etc/profile.d/conda.sh
fi

# 3. Clone repo
cd $HOME
if [ ! -d code-prm ]; then
    git clone https://github.com/<your-gh-user>/code-prm.git
fi
cd code-prm
git submodule update --init --recursive

# 4. Create env
conda env create -f environment.yml -n code-prm || true
conda activate code-prm

# 5. Verify
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

echo "Lab box ready."
```

- [ ] **Step 2: Push to GitHub from local Mac**

```bash
git add scripts/00_setup_lab_box.sh
chmod +x scripts/00_setup_lab_box.sh
git commit -m "infra: lab box bootstrap script"
git push origin main
```

- [ ] **Step 3: SSH to lab box and run setup**

Run locally: `ssh $LAB_BOX_SSH`
Then on lab box:
```bash
git clone https://github.com/<you>/code-prm.git
cd code-prm
bash scripts/00_setup_lab_box.sh
```
Expected: ends with `Lab box ready.` and prints `NVIDIA GeForce RTX 3090`.

- [ ] **Step 4: Set up tmux + persistent shell on lab box**

On lab box:
```bash
sudo apt-get install -y tmux
tmux new -s code-prm
```
All long-running commands from here on go inside this tmux session.

- [ ] **Step 5: Sanity-check disk space**

On lab box: `df -h $HOME`
Confirm: ≥ 100GB free.

---

### ⚠️ Tasks 5-7 status update (2026-05-27)

After inspecting OpenR's real PRM training code (`prm/code/finetune_qwen_single_gpu.py`),
we discovered OpenR uses **next-token `+/-` prediction** training, not the scalar-head
MSE training our spec §5.1 specifies. See spec §13 decision log.

**Phase 1 deviation:**
- **Task 5 (PRM800K download)**: KEEP. Math PRM800K is still useful as a Phase 2 ablation
  ("our scalar-head PRM also works on math benchmarks").
- **Task 6 (OpenR baseline training)**: SKIP. We are NOT using OpenR's training paradigm,
  so reproducing its math PRM is no longer on the critical path.
- **Task 7 (OpenR baseline eval)**: SKIP for same reason.

Environment validation is instead served by `pytest tests/ -v` (21 tests) passing on the
lab box. If pytest passes, the env is healthy enough to proceed to Task 9+.

---

### Task 5: Download PRM800K dataset (kept — used as Phase 2 ablation)

**Files:**
- Modify: lab box `~/code-prm/data/prm800k/` (gitignored)

- [ ] **Step 1: Activate env on lab box & clone PRM800K**

```bash
cd ~/code-prm
mkdir -p data/prm800k
cd data/prm800k
git clone https://github.com/openai/prm800k.git .
```
Expected: ~500MB cloned.

- [ ] **Step 2: Decompress phase2 data**

```bash
cd prm800k/data
gunzip -k *.jsonl.gz
ls -lh *.jsonl
```
Expected: ~200MB of jsonl files (phase2_train.jsonl is the main one).

- [ ] **Step 3: Sanity-check first row**

```bash
head -1 phase2_train.jsonl | python -m json.tool | head -50
```
Expected: JSON with `question`, `label.steps[].completions[].rating` structure.

- [ ] **Step 4: Spot-count rows**

```bash
wc -l phase2_train.jsonl phase2_test.jsonl
```
Expected: train ~100k lines, test ~3k lines.

---

### Task 6: ~~Run OpenR math-PRM training baseline~~ — SKIPPED (see header)

The original content below is preserved for reference but **DO NOT RUN**. See Phase 1
deviation note at top of "Week 1" section. Reason: OpenR uses a different PRM paradigm
(`+/-` token prediction) than our spec (scalar head); reproducing it provides little signal
for our actual Phase 2 training code.

**Files:**
- Create: `scripts/01_train_openr_baseline.sh`

- [ ] **Step 1: Write `scripts/01_train_openr_baseline.sh`**

```bash
#!/usr/bin/env bash
# Train OpenR PRM baseline on PRM800K (small subset, smoke test).
# Run on lab box inside tmux.
set -euo pipefail

cd ~/code-prm/third_party/openr/train/mat

# Use only first 5000 examples to verify pipeline (full train takes >1 day on 3090)
python train_prm.py \
    --train_path ~/code-prm/data/prm800k/prm800k/data/phase2_train.jsonl \
    --max_train_samples 5000 \
    --model_name Qwen/Qwen2.5-Math-1.5B \
    --output_dir ~/code-prm/checkpoints/openr-baseline-smoke \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --logging_steps 10 \
    --save_steps 200 \
    --bf16
```

(Note: argument names may differ slightly from OpenR's actual `train_prm.py` — read the OpenR README & adapt before running.)

- [ ] **Step 2: Push & sync to lab box**

```bash
# Local:
git add scripts/01_train_openr_baseline.sh
chmod +x scripts/01_train_openr_baseline.sh
git commit -m "infra: openr baseline training script"
git push

# Lab box:
cd ~/code-prm
git pull
```

- [ ] **Step 3: Run smoke training**

On lab box inside tmux:
```bash
conda activate code-prm
bash scripts/01_train_openr_baseline.sh 2>&1 | tee logs/openr-baseline-smoke.log
```
Expected:
- Training starts within 2 min (model download first time may take 10 min)
- Loss decreases monotonically over ~625 steps (5000 / batch 8)
- Final checkpoint saved to `checkpoints/openr-baseline-smoke/checkpoint-625/`

- [ ] **Step 4: Verify checkpoint loads**

```bash
python -c "
from transformers import AutoModel
m = AutoModel.from_pretrained('checkpoints/openr-baseline-smoke/checkpoint-625')
print('OK,', sum(p.numel() for p in m.parameters())/1e9, 'B params')
"
```
Expected: prints `OK, 1.5 B params` (or close).

- [ ] **Step 5: Commit logs (not weights)**

On local:
```bash
mkdir -p logs/
scp $LAB_BOX_SSH:~/code-prm/logs/openr-baseline-smoke.log logs/
git add logs/openr-baseline-smoke.log
git commit -m "exp: openr smoke training log (5k samples, loss converges)"
```

---

### Task 7: ~~Eval OpenR baseline on PRM800K test set~~ — SKIPPED (see header)

**Files:**
- Create: `scripts/02_eval_openr_baseline.sh`

- [ ] **Step 1: Write eval script**

```bash
#!/usr/bin/env bash
# Eval OpenR baseline PRM on PRM800K test.
set -euo pipefail
cd ~/code-prm/third_party/openr

python -m reason.evaluation.eval_prm \
    --prm_path ~/code-prm/checkpoints/openr-baseline-smoke/checkpoint-625 \
    --test_path ~/code-prm/data/prm800k/prm800k/data/phase2_test.jsonl \
    --output_path ~/code-prm/logs/openr-baseline-eval.json \
    --batch_size 4
```

(Again: read OpenR's actual eval entrypoint name before running. May be `reason/evaluation/run_eval.py` or similar.)

- [ ] **Step 2: Push, sync, run**

```bash
# Local: git add/commit/push
# Lab box: git pull && bash scripts/02_eval_openr_baseline.sh
```
Expected: prints F1, precision, recall to stdout; writes JSON to `logs/openr-baseline-eval.json`.

- [ ] **Step 3: Inspect numbers**

```bash
cat ~/code-prm/logs/openr-baseline-eval.json
```
Expected: F1 in 0.55–0.70 range (small subset, won't match paper but should be in ballpark). If F1 < 0.4, training had a bug — go back to Task 6.

- [ ] **Step 4: Commit eval result**

```bash
scp $LAB_BOX_SSH:~/code-prm/logs/openr-baseline-eval.json logs/
git add logs/openr-baseline-eval.json
git commit -m "exp: openr baseline eval (F1=<actual>)"
```

**🎯 Week 1 Exit:** OpenR end-to-end pipeline works on 3090 with PRM800K. Pipeline confidence verified; can now safely swap math data → code data.

---

## Week 2: Trajectory Collection from codeAgent

### Task 8: Define trajectory schema (Pydantic)

**Files:**
- Create: `src/labeler/trajectory_schema.py`
- Create: `src/utils/jsonl_io.py`
- Create: `tests/test_trajectory_schema.py`
- Create: `src/collector/ts_logger_spec.md`

- [ ] **Step 1: Write the failing test**

`tests/test_trajectory_schema.py`:
```python
import json
import pytest
from src.labeler.trajectory_schema import Trajectory, Step

def test_minimal_trajectory_parses():
    raw = {
        "task_id": "django__django-12345",
        "task_type": "swe-bench-lite",
        "trajectory": [
            {"step": 0, "role": "assistant", "thought": "let me read", "tool": "read_file",
             "tool_args": {"path": "foo.py"}, "tool_result": "file contents"}
        ],
        "outcome": 1,
        "policy_model": "claude-sonnet-4.5",
        "timestamp": "2026-05-27T10:00:00Z",
    }
    t = Trajectory(**raw)
    assert t.task_id == "django__django-12345"
    assert len(t.trajectory) == 1
    assert t.trajectory[0].tool == "read_file"

def test_outcome_must_be_0_or_1():
    raw = {"task_id": "x", "task_type": "x", "trajectory": [],
           "outcome": 2, "policy_model": "x", "timestamp": "x"}
    with pytest.raises(ValueError):
        Trajectory(**raw)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_trajectory_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: src.labeler.trajectory_schema`.

- [ ] **Step 3: Write `src/labeler/trajectory_schema.py`**

```python
"""Pydantic schemas for code-agent trajectories."""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class Step(BaseModel):
    step: int = Field(ge=0)
    role: Literal["assistant", "tool", "user"] = "assistant"
    thought: str = ""
    tool: str | None = None        # None = pure thought step
    tool_args: dict[str, Any] = Field(default_factory=dict)
    tool_result: str = ""
    mc_label: float | None = None  # filled by MC labeler later


class Trajectory(BaseModel):
    task_id: str
    task_type: Literal["swe-bench-lite", "bigcodebench-hard", "other"]
    trajectory: list[Step]
    outcome: int                   # 0 = failed, 1 = passed
    policy_model: str
    timestamp: str

    @field_validator("outcome")
    @classmethod
    def outcome_in_range(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError(f"outcome must be 0 or 1, got {v}")
        return v
```

- [ ] **Step 4: Write `src/utils/jsonl_io.py`**

```python
"""Streaming jsonl read/write."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator
from src.labeler.trajectory_schema import Trajectory


def read_trajectories(path: str | Path) -> Iterator[Trajectory]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Trajectory(**json.loads(line))


def write_trajectories(path: str | Path, trajectories: list[Trajectory]) -> None:
    with open(path, "w") as f:
        for t in trajectories:
            f.write(t.model_dump_json() + "\n")


def append_trajectory(path: str | Path, t: Trajectory) -> None:
    with open(path, "a") as f:
        f.write(t.model_dump_json() + "\n")
```

- [ ] **Step 5: Run test, verify it passes**

Run: `pytest tests/test_trajectory_schema.py -v`
Expected: 2 passed.

- [ ] **Step 6: Write TS integration spec**

`src/collector/ts_logger_spec.md`:
````markdown
# TS Trajectory Logger Integration Spec

This document is the contract between the Python pipeline and the existing
TypeScript codeAgent. The TS side MUST produce jsonl matching this schema.

## File location
Each codeAgent run produces ONE jsonl line appended to:
`$TS_REPO/logs/trajectories/<task_type>_<YYYYMMDD>.jsonl`

## Schema
\```typescript
interface Step {
    step: number;                    // 0-indexed
    role: "assistant" | "tool" | "user";
    thought: string;                 // text before tool call, "" if none
    tool: string | null;             // tool name, null for pure thought
    tool_args: Record<string, any>;
    tool_result: string;             // truncated to 8000 chars
}

interface Trajectory {
    task_id: string;
    task_type: "swe-bench-lite" | "bigcodebench-hard";
    trajectory: Step[];
    outcome: 0 | 1;                  // 0 = test failed, 1 = test passed
    policy_model: string;            // e.g., "claude-sonnet-4-5"
    timestamp: string;               // ISO 8601 UTC
}
\```

## Integration hook in codeAgent
Add a hook into the codeAgent's tool dispatch loop:
1. On agent start, allocate an in-memory `steps: Step[]`.
2. After each tool execution, push `{step, role, thought, tool, tool_args, tool_result}` to `steps`.
3. On task completion (or max steps), run the task's test suite and set `outcome`.
4. Append the assembled Trajectory as a single jsonl line.

## Truncation rules
- `tool_result` longer than 8000 chars → truncate to first 4000 + "...[TRUNC]..." + last 3000.
- `thought` longer than 2000 chars → keep first 2000.
````

- [ ] **Step 7: Create `__init__.py` files and commit**

```bash
touch src/__init__.py src/labeler/__init__.py src/utils/__init__.py src/collector/__init__.py
git add src/__init__.py src/labeler/__init__.py src/utils/__init__.py src/collector/__init__.py
git add src/labeler/trajectory_schema.py src/utils/jsonl_io.py \
        tests/test_trajectory_schema.py src/collector/ts_logger_spec.md
git commit -m "feat: trajectory schema + jsonl io + TS contract"
```

---

### Task 9: Add trajectory logger to TS codeAgent

**Files:**
- Create: `$TS_REPO/src/hooks/trajectory_logger.ts`

(This task is in your existing TS repo, not in `agentrl/`. Adapt paths to match.)

- [ ] **Step 1: Locate the tool dispatch loop in codeAgent**

In `$TS_REPO`, search for where tool calls are dispatched (likely in a file named `agent.ts`, `loop.ts`, or `dispatcher.ts`). Note the function name & file path.

Run (in `$TS_REPO`): `grep -rn "tool_use\|toolCall\|dispatch" src/ | head -20`

Identify the post-tool-call hook point.

- [ ] **Step 2: Write `src/hooks/trajectory_logger.ts`**

```typescript
// Trajectory logger for Code-PRM data collection.
// Activates when env var CODE_PRM_LOG_DIR is set.

import * as fs from "fs";
import * as path from "path";

export interface Step {
  step: number;
  role: "assistant" | "tool" | "user";
  thought: string;
  tool: string | null;
  tool_args: Record<string, any>;
  tool_result: string;
}

export interface Trajectory {
  task_id: string;
  task_type: "swe-bench-lite" | "bigcodebench-hard";
  trajectory: Step[];
  outcome: 0 | 1;
  policy_model: string;
  timestamp: string;
}

const LOG_DIR = process.env.CODE_PRM_LOG_DIR;

function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  const head = Math.floor(max * 0.55);
  const tail = max - head - 14;
  return s.slice(0, head) + "...[TRUNC]..." + s.slice(s.length - tail);
}

export class TrajectoryLogger {
  private steps: Step[] = [];
  private stepIdx = 0;
  constructor(
    private taskId: string,
    private taskType: Trajectory["task_type"],
    private policyModel: string,
  ) {}

  recordStep(args: {
    thought: string;
    tool: string | null;
    toolArgs: Record<string, any>;
    toolResult: string;
  }) {
    if (!LOG_DIR) return;
    this.steps.push({
      step: this.stepIdx++,
      role: "assistant",
      thought: truncate(args.thought, 2000),
      tool: args.tool,
      tool_args: args.toolArgs,
      tool_result: truncate(args.toolResult, 8000),
    });
  }

  finalize(outcome: 0 | 1) {
    if (!LOG_DIR) return;
    const traj: Trajectory = {
      task_id: this.taskId,
      task_type: this.taskType,
      trajectory: this.steps,
      outcome,
      policy_model: this.policyModel,
      timestamp: new Date().toISOString(),
    };
    const date = traj.timestamp.slice(0, 10).replace(/-/g, "");
    const file = path.join(LOG_DIR, `${this.taskType}_${date}.jsonl`);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.appendFileSync(file, JSON.stringify(traj) + "\n");
  }
}
```

- [ ] **Step 3: Wire into dispatch loop**

In your tool dispatch loop (from Step 1), find the place AFTER each tool result is obtained. Add:

```typescript
import { TrajectoryLogger } from "./hooks/trajectory_logger";

// at agent run start:
const logger = new TrajectoryLogger(taskId, taskType, policyModel);

// after each tool call:
logger.recordStep({
  thought: assistantThoughtText,
  tool: toolName,
  toolArgs: toolArgs,
  toolResult: toolResultText,
});

// at agent end (after test suite runs):
logger.finalize(testPassed ? 1 : 0);
```

- [ ] **Step 4: Smoke-test with one task**

```bash
export CODE_PRM_LOG_DIR=/tmp/code-prm-test
# Run codeAgent on any simple task end-to-end
# Then:
cat /tmp/code-prm-test/*.jsonl | python -m json.tool
```
Expected: a valid JSON object matching the Pydantic schema. If `task_id` or `outcome` are missing, fix wiring.

- [ ] **Step 5: Validate against Pydantic schema**

Back in `agentrl/`:
```bash
python -c "
from src.utils.jsonl_io import read_trajectories
import sys
for t in read_trajectories('/tmp/code-prm-test/swe-bench-lite_20260527.jsonl'):
    print('OK:', t.task_id, len(t.trajectory), 'steps')
"
```
Expected: prints `OK: <task_id> N steps` with no Pydantic validation errors.

- [ ] **Step 6: Commit (in TS repo)**

```bash
cd $TS_REPO
git add src/hooks/trajectory_logger.ts
git add <files-modified-for-wiring>
git commit -m "feat: trajectory logger for Code-PRM data collection"
```

---

### Task 10: Stand up SWE-bench Lite runner

**Files:**
- Create: `src/eval/swebench_runner.py`

- [ ] **Step 1: Install SWE-bench harness**

```bash
pip install swebench
```
Verify: `python -c "import swebench; print(swebench.__version__)"`

- [ ] **Step 2: Download SWE-bench Lite task list**

```python
# Run this in `python` REPL:
from datasets import load_dataset
ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
print(len(ds), "tasks")
print(ds[0].keys())
```
Expected: 300 tasks, keys include `instance_id`, `problem_statement`, `patch`, `test_patch`.

- [ ] **Step 3: Write `src/eval/swebench_runner.py`**

```python
"""SWE-bench Lite task launcher.
Provides a function that given a task and a callable that runs your TS codeAgent,
produces a Trajectory with outcome.
"""
from __future__ import annotations
import os, subprocess, json
from pathlib import Path
from datasets import load_dataset


def load_swebench_lite() -> list[dict]:
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    return [dict(row) for row in ds]


def run_task_with_codeagent(task: dict, ts_repo: Path, log_dir: Path) -> bool:
    """Run codeAgent on one task; return True if tests pass.

    Assumes codeAgent CLI: `codeagent run --task-id <id> --task-type swe-bench-lite`
    Adjust if your CLI differs.
    """
    env = os.environ.copy()
    env["CODE_PRM_LOG_DIR"] = str(log_dir)
    env["SWEBENCH_TASK_JSON"] = json.dumps(task)

    result = subprocess.run(
        ["node", str(ts_repo / "dist" / "cli.js"),
         "run", "--task-id", task["instance_id"],
         "--task-type", "swe-bench-lite"],
        env=env, capture_output=True, text=True, timeout=600,
    )
    # The TS side writes outcome into the jsonl via TrajectoryLogger.finalize.
    # We just check the process exit code as a coarse signal here.
    return result.returncode == 0


if __name__ == "__main__":
    tasks = load_swebench_lite()
    print(f"Loaded {len(tasks)} SWE-bench Lite tasks.")
    print("First:", tasks[0]["instance_id"])
```

- [ ] **Step 4: Smoke-run on 1 task**

```bash
python -m src.eval.swebench_runner
```
Expected: prints `Loaded 300 SWE-bench Lite tasks.` plus first instance id.

- [ ] **Step 5: Commit**

```bash
git add src/eval/swebench_runner.py
git commit -m "feat: SWE-bench Lite task loader"
```

---

### Task 11: Pilot collection — 10 trajectories

**Files:**
- Modify: usage of `scripts/10_collect_trajectories.sh` (created in Task 13)

- [ ] **Step 1: Hand-run codeAgent on 10 SWE-bench Lite tasks**

```bash
export CODE_PRM_LOG_DIR=$PWD/data/raw/pilot/
export ANTHROPIC_API_KEY=...
mkdir -p $CODE_PRM_LOG_DIR

python -c "
from src.eval.swebench_runner import load_swebench_lite, run_task_with_codeagent
from pathlib import Path
ts_repo = Path('$TS_REPO_PATH')
log_dir = Path('$CODE_PRM_LOG_DIR')
tasks = load_swebench_lite()[:10]
for t in tasks:
    print('Running', t['instance_id'])
    ok = run_task_with_codeagent(t, ts_repo, log_dir)
    print('  pass:', ok)
"
```
Expected: prints 10 task ids and their pass/fail status. Wall-clock ~30 min.

- [ ] **Step 2: Inspect collected jsonl**

```bash
wc -l data/raw/pilot/*.jsonl
python -c "
from src.utils.jsonl_io import read_trajectories
import glob
for f in glob.glob('data/raw/pilot/*.jsonl'):
    for t in read_trajectories(f):
        print(t.task_id, 'steps=', len(t.trajectory), 'outcome=', t.outcome)
"
```
Expected: 10 trajectories, mix of outcomes (likely 3-5 pass, 5-7 fail with Sonnet at this difficulty).

- [ ] **Step 3: Health-check distributions**

```python
# Quick analysis:
from src.utils.jsonl_io import read_trajectories
import glob
trajs = []
for f in glob.glob('data/raw/pilot/*.jsonl'):
    trajs.extend(read_trajectories(f))
print(f"N={len(trajs)}")
print(f"Pass rate: {sum(t.outcome for t in trajs) / len(trajs):.2%}")
print(f"Avg steps: {sum(len(t.trajectory) for t in trajs) / len(trajs):.1f}")
print(f"Step range: {min(len(t.trajectory) for t in trajs)}-{max(len(t.trajectory) for t in trajs)}")
```
Expected:
- Pass rate: 30-60%
- Avg steps: 6-15
- If pass rate is 0% or 100%, the integration is broken — investigate.

- [ ] **Step 4: Commit pilot results**

```bash
# Only commit the analysis log, not the data (data/ is gitignored)
mkdir -p logs/
cat > logs/pilot-collection.md <<EOF
# Pilot Collection Results (10 tasks)

- N trajectories: <fill>
- Pass rate: <fill>
- Avg steps: <fill>
- Step range: <fill>

Health check: PASSED / NEEDS FIX
EOF
git add logs/pilot-collection.md
git commit -m "exp: pilot collection (10 trajectories) health check"
```

---

### Task 12: Cost monitoring infrastructure

**Files:**
- Create: `src/utils/cost_tracker.py`
- Create: `tests/test_cost_tracker.py`

- [ ] **Step 1: Write failing test**

`tests/test_cost_tracker.py`:
```python
from src.utils.cost_tracker import CostTracker

def test_tracks_sonnet_cost():
    t = CostTracker(budget_usd=10.0)
    t.add("claude-sonnet-4-5", input_tokens=1000, output_tokens=500)
    # Pricing per 1M tokens (as of 2025): Sonnet $3 in / $15 out
    expected = (1000 / 1_000_000) * 3 + (500 / 1_000_000) * 15
    assert abs(t.total_usd - expected) < 1e-6

def test_warns_at_80pct_budget():
    t = CostTracker(budget_usd=0.10)
    # Force 81% spend
    t.add("claude-haiku-4-5", input_tokens=1_000_000, output_tokens=0)  # $1 input
    # Haiku is $1/M input — so 1M input tokens = $1, way over $0.10 budget
    assert t.over_budget()
```

- [ ] **Step 2: Verify test fails**

Run: `pytest tests/test_cost_tracker.py -v`
Expected: FAIL ModuleNotFoundError.

- [ ] **Step 3: Implement**

`src/utils/cost_tracker.py`:
```python
"""Tracks token spend across Anthropic API calls."""
from __future__ import annotations
from dataclasses import dataclass, field
import threading

# Prices in USD per 1M tokens (verify against Anthropic pricing page).
PRICING = {
    "claude-sonnet-4-5":   {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5":    {"input": 1.00,  "output": 5.00},
    "claude-opus-4-7":     {"input": 15.00, "output": 75.00},
}


@dataclass
class CostTracker:
    budget_usd: float
    total_usd: float = 0.0
    per_model: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, model: str, input_tokens: int, output_tokens: int) -> float:
        if model not in PRICING:
            raise KeyError(f"Unknown model pricing for {model!r}")
        p = PRICING[model]
        cost = (input_tokens / 1e6) * p["input"] + (output_tokens / 1e6) * p["output"]
        with self._lock:
            self.total_usd += cost
            self.per_model[model] = self.per_model.get(model, 0.0) + cost
        return cost

    def remaining(self) -> float:
        return self.budget_usd - self.total_usd

    def over_budget(self) -> bool:
        return self.total_usd >= self.budget_usd

    def warn_threshold(self, frac: float = 0.8) -> bool:
        return self.total_usd >= self.budget_usd * frac

    def __str__(self) -> str:
        return (f"CostTracker: ${self.total_usd:.2f} / ${self.budget_usd:.2f} "
                f"({self.total_usd / self.budget_usd:.1%}), per_model={self.per_model}")
```

- [ ] **Step 4: Verify test passes**

Run: `pytest tests/test_cost_tracker.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/utils/cost_tracker.py tests/test_cost_tracker.py
git commit -m "feat: cost tracker with per-model pricing"
```

---

### Task 13: Batched trajectory collection (scale up to 300 SWE-bench tasks)

**Files:**
- Create: `scripts/10_collect_trajectories.sh`

- [ ] **Step 1: Write driver script**

```bash
#!/usr/bin/env bash
# Collect trajectories on SWE-bench Lite full set, 4 rollouts each.
set -euo pipefail

export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?must set}"
export CODE_PRM_LOG_DIR="${CODE_PRM_LOG_DIR:-$PWD/data/raw/swebench-lite}"
export TS_REPO_PATH="${TS_REPO_PATH:?must set in .env}"

mkdir -p "$CODE_PRM_LOG_DIR"

# Use a fan-out python script (concurrency = 4 to respect API rate limits)
python -m src.eval.collect_batch \
    --task_set swebench-lite \
    --num_rollouts 4 \
    --concurrency 4 \
    --log_dir "$CODE_PRM_LOG_DIR" \
    --budget_usd 250
```

- [ ] **Step 2: Implement `src/eval/collect_batch.py`**

```python
"""Batched trajectory collection with concurrency + cost cap."""
from __future__ import annotations
import argparse, asyncio, os
from pathlib import Path
from src.eval.swebench_runner import load_swebench_lite, run_task_with_codeagent
from src.utils.cost_tracker import CostTracker
from rich.progress import Progress


async def collect(task_set: str, num_rollouts: int, concurrency: int,
                  log_dir: Path, budget_usd: float):
    tracker = CostTracker(budget_usd=budget_usd)
    if task_set == "swebench-lite":
        tasks = load_swebench_lite()
    else:
        raise ValueError(task_set)

    ts_repo = Path(os.environ["TS_REPO_PATH"])
    sem = asyncio.Semaphore(concurrency)
    total_runs = len(tasks) * num_rollouts

    async def one_run(task, k):
        async with sem:
            if tracker.over_budget():
                return
            # Subprocess is sync; run in thread
            loop = asyncio.get_running_loop()
            ok = await loop.run_in_executor(
                None, run_task_with_codeagent, task, ts_repo, log_dir / f"rollout_{k}",
            )
            # Cost estimation: assume ~30k tokens per Sonnet trajectory
            tracker.add("claude-sonnet-4-5", input_tokens=25_000, output_tokens=5_000)

    with Progress() as prog:
        bar = prog.add_task("collect", total=total_runs)
        async def wrapped(task, k):
            await one_run(task, k)
            prog.advance(bar)
        coros = [wrapped(t, k) for t in tasks for k in range(num_rollouts)]
        await asyncio.gather(*coros)

    print(tracker)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task_set", required=True)
    p.add_argument("--num_rollouts", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--log_dir", type=Path, required=True)
    p.add_argument("--budget_usd", type=float, default=250.0)
    args = p.parse_args()
    asyncio.run(collect(args.task_set, args.num_rollouts, args.concurrency,
                        args.log_dir, args.budget_usd))
```

- [ ] **Step 3: Dry-run on 5 tasks**

Temporarily edit `load_swebench_lite` call in `collect_batch.py` to slice `[:5]`. Run:
```bash
bash scripts/10_collect_trajectories.sh
```
Expected: 5 tasks × 4 rollouts = 20 trajectory lines. Cost ~$2.

- [ ] **Step 4: Run full SWE-bench Lite collection**

Remove the slice. Run:
```bash
bash scripts/10_collect_trajectories.sh 2>&1 | tee logs/swebench-collection.log
```
Expected wall-clock: ~6-10 hours. Expected cost: $200-250. Final tracker: ≤ $250.

(This is long-running. Use tmux. Check progress every couple hours.)

- [ ] **Step 5: Verify dataset count**

```bash
find data/raw/swebench-lite -name "*.jsonl" -exec cat {} \; | wc -l
```
Expected: ~1200 lines (300 tasks × 4 rollouts). If much less, some tasks timed out — that's OK as long as ≥ 1000.

- [ ] **Step 6: Commit log + summary**

```bash
echo "Collected N=$(find data/raw/swebench-lite -name '*.jsonl' -exec cat {} \; | wc -l) trajectories" >> logs/swebench-collection.log
git add logs/swebench-collection.log
git commit -m "exp: collected ~1200 SWE-bench Lite trajectories"
```

---

### Task 14: Collect BigCodeBench-Hard trajectories

- [ ] **Step 1: Install BigCodeBench**

```bash
pip install bigcodebench
```

- [ ] **Step 2: Add BigCodeBench loader to `src/eval/swebench_runner.py`**

Add function:
```python
def load_bigcodebench_hard() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("bigcode/bigcodebench-hard", split="v0.1.4")
    # Subsample to 300 tasks for budget control
    return [dict(row) for row in ds.select(range(300))]
```

Add `task_set == "bigcodebench-hard"` branch in `collect_batch.py`.

- [ ] **Step 3: Run collection**

```bash
export CODE_PRM_LOG_DIR=$PWD/data/raw/bigcodebench-hard
mkdir -p $CODE_PRM_LOG_DIR
python -m src.eval.collect_batch \
    --task_set bigcodebench-hard \
    --num_rollouts 4 \
    --concurrency 4 \
    --log_dir $CODE_PRM_LOG_DIR \
    --budget_usd 100
```
Expected: ~1200 lines, $80-100, ~4-6 hours.

- [ ] **Step 4: Commit summary**

```bash
echo "BigCodeBench: $(find data/raw/bigcodebench-hard -name '*.jsonl' -exec cat {} \; | wc -l) trajectories" >> logs/swebench-collection.log
git add logs/swebench-collection.log
git commit -m "exp: collected ~1200 BigCodeBench-Hard trajectories"
```

**🎯 Week 2 Exit:** ~2400 raw trajectories under `data/raw/`, total cost ≤ $350.

---

## Week 3: MC Labeling & Dataset Assembly

### Task 15: Anthropic client with rate-limit + retry

**Files:**
- Create: `src/labeler/anthropic_client.py`

- [ ] **Step 1: Implement**

```python
"""Rate-limited Anthropic API client with retries."""
from __future__ import annotations
import os
from anthropic import Anthropic, RateLimitError, APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.utils.cost_tracker import CostTracker


class RateLimitedClient:
    def __init__(self, tracker: CostTracker, model: str = "claude-haiku-4-5"):
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.tracker = tracker
        self.model = model

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception_type((RateLimitError, APIError)),
    )
    def complete(self, messages: list[dict], max_tokens: int = 2048,
                 temperature: float = 0.8) -> tuple[str, int, int]:
        if self.tracker.over_budget():
            raise RuntimeError(f"Over budget: {self.tracker}")
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
            temperature=temperature,
        )
        text = resp.content[0].text
        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        self.tracker.add(self.model, in_tok, out_tok)
        return text, in_tok, out_tok
```

- [ ] **Step 2: Smoke-test**

```bash
python -c "
from src.utils.cost_tracker import CostTracker
from src.labeler.anthropic_client import RateLimitedClient
t = CostTracker(budget_usd=1.0)
c = RateLimitedClient(t)
txt, _, _ = c.complete([{'role':'user','content':'Say hi in 3 words.'}], max_tokens=10)
print(txt)
print(t)
"
```
Expected: prints 3 words + cost line.

- [ ] **Step 3: Commit**

```bash
git add src/labeler/anthropic_client.py
git commit -m "feat: rate-limited anthropic client with retry"
```

---

### Task 16: MC labeler core implementation

**Files:**
- Create: `src/labeler/mc_labeler.py`

- [ ] **Step 1: Implement labeler**

```python
"""MC rollout label generator.

For each step in an outcome=1 trajectory, re-roll K times from that step
using Haiku, run the test suite on the final state, count successes,
and assign mc_i = successes / K.

For outcome=0 trajectories, simplification: set mc_i = 0 for all steps
(Math-Shepherd simplification — avoids noisy MC on failure paths).
"""
from __future__ import annotations
import json
from pathlib import Path
from src.labeler.trajectory_schema import Trajectory, Step
from src.labeler.anthropic_client import RateLimitedClient
from src.utils.cost_tracker import CostTracker


def label_trajectory_simplified(traj: Trajectory) -> Trajectory:
    """For outcome=0: set all mc_i = 0. For outcome=1: defer to MC rollouts."""
    if traj.outcome == 0:
        for s in traj.trajectory:
            s.mc_label = 0.0
        return traj
    return traj  # caller must run MC rollouts for outcome=1


def mc_rollout_for_step(
    traj: Trajectory,
    step_idx: int,
    client: RateLimitedClient,
    test_fn,           # callable: (final_state) -> bool
    K: int = 4,
) -> float:
    """Re-roll from step_idx K times; return success rate.

    NOTE: This is a SIMPLIFIED MC version that asks Haiku to continue from
    the partial trajectory and predict outcome, rather than literally
    re-running tools (which would require sandboxed execution).

    For Phase 1 we use this lightweight surrogate. Phase 2 may upgrade
    to real tool re-execution if signal is too noisy.
    """
    prefix = traj.trajectory[: step_idx + 1]
    prompt = _build_continuation_prompt(prefix, traj.task_id)

    successes = 0
    for _ in range(K):
        text, _, _ = client.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.9,
        )
        if _parses_as_successful(text):
            successes += 1
    return successes / K


def _build_continuation_prompt(prefix: list[Step], task_id: str) -> str:
    lines = [f"Task: {task_id}", "Trajectory so far:"]
    for s in prefix:
        lines.append(f"  Step {s.step}: {s.tool}({json.dumps(s.tool_args)[:200]}) "
                     f"→ {s.tool_result[:200]}")
    lines.append(
        "\nGiven this partial trajectory, predict the final outcome. "
        "Reply with exactly one line: either 'OUTCOME: PASS' or 'OUTCOME: FAIL', "
        "followed by a brief justification (1-2 sentences)."
    )
    return "\n".join(lines)


def _parses_as_successful(text: str) -> bool:
    return "OUTCOME: PASS" in text.upper()


def label_file(
    input_path: Path,
    output_path: Path,
    client: RateLimitedClient,
    K: int = 4,
    only_tool_steps: bool = True,
) -> None:
    """Label every trajectory in input jsonl, write to output jsonl."""
    from src.utils.jsonl_io import read_trajectories, append_trajectory

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()  # fresh write

    for traj in read_trajectories(input_path):
        if traj.outcome == 0:
            label_trajectory_simplified(traj)
        else:
            for i, step in enumerate(traj.trajectory):
                if only_tool_steps and step.tool is None:
                    continue
                step.mc_label = mc_rollout_for_step(traj, i, client, test_fn=None, K=K)
        append_trajectory(output_path, traj)
```

- [ ] **Step 2: Commit**

```bash
git add src/labeler/mc_labeler.py
git commit -m "feat: MC labeler (lightweight LLM-judge surrogate)"
```

---

### Task 17: Unit test the labeler on synthetic data

**Files:**
- Create: `tests/fixtures/synthetic_trajectory.json`
- Create: `tests/test_mc_labeler.py`

- [ ] **Step 1: Create synthetic trajectory fixture**

`tests/fixtures/synthetic_trajectory.json`:
```json
{
  "task_id": "synth-001",
  "task_type": "swe-bench-lite",
  "trajectory": [
    {"step": 0, "role": "assistant", "thought": "read", "tool": "read_file",
     "tool_args": {"path": "x.py"}, "tool_result": "def f(): pass"},
    {"step": 1, "role": "assistant", "thought": "edit", "tool": "edit",
     "tool_args": {"path": "x.py", "diff": "..."}, "tool_result": "ok"},
    {"step": 2, "role": "assistant", "thought": "test", "tool": "bash",
     "tool_args": {"cmd": "pytest"}, "tool_result": "passed"}
  ],
  "outcome": 1,
  "policy_model": "claude-sonnet-4-5",
  "timestamp": "2026-05-27T10:00:00Z"
}
```

- [ ] **Step 2: Write tests**

`tests/test_mc_labeler.py`:
```python
import json
from pathlib import Path
from src.labeler.trajectory_schema import Trajectory
from src.labeler.mc_labeler import label_trajectory_simplified, _build_continuation_prompt

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_trajectory.json"

def test_outcome_zero_gets_all_zero_labels():
    t = Trajectory(**json.loads(FIXTURE.read_text()))
    t.outcome = 0
    labeled = label_trajectory_simplified(t)
    assert all(s.mc_label == 0.0 for s in labeled.trajectory)

def test_outcome_one_passes_through_unchanged():
    t = Trajectory(**json.loads(FIXTURE.read_text()))
    labeled = label_trajectory_simplified(t)
    # No MC labels set yet for outcome=1 path
    assert all(s.mc_label is None for s in labeled.trajectory)

def test_continuation_prompt_includes_steps():
    t = Trajectory(**json.loads(FIXTURE.read_text()))
    p = _build_continuation_prompt(t.trajectory[:2], t.task_id)
    assert "synth-001" in p
    assert "read_file" in p
    assert "OUTCOME: PASS" in p
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_mc_labeler.py -v`
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_mc_labeler.py tests/fixtures/
git commit -m "test: MC labeler unit tests on synthetic data"
```

---

### Task 18: Pilot MC labeling on 10 trajectories

- [ ] **Step 1: Run pilot labeling**

```bash
python -c "
from pathlib import Path
from src.utils.cost_tracker import CostTracker
from src.labeler.anthropic_client import RateLimitedClient
from src.labeler.mc_labeler import label_file
import glob

tracker = CostTracker(budget_usd=10.0)
client = RateLimitedClient(tracker, model='claude-haiku-4-5')

# Take first pilot file
input_files = sorted(glob.glob('data/raw/pilot/*.jsonl'))[:1]
for f in input_files:
    label_file(Path(f), Path('data/labeled/pilot/labeled.jsonl'), client, K=4)
print(tracker)
"
```
Expected: cost < $5; ~10 trajectories labeled.

- [ ] **Step 2: Inspect MC label distribution**

```python
from src.utils.jsonl_io import read_trajectories
import statistics
labels = []
for t in read_trajectories('data/labeled/pilot/labeled.jsonl'):
    for s in t.trajectory:
        if s.mc_label is not None and s.tool is not None:
            labels.append(s.mc_label)
print(f"N labels: {len(labels)}")
print(f"Mean: {statistics.mean(labels):.3f}")
print(f"Distribution:")
for thresh in [0, 0.25, 0.5, 0.75, 1.0]:
    n = sum(1 for l in labels if l == thresh)
    print(f"  =={thresh}: {n}")
```
Expected:
- ≥ 30 labels
- Mean in 0.3-0.7 range (NOT all 0 or all 1 — that would indicate Haiku always says PASS or always FAIL)
- Some non-degenerate distribution (not just {0, 1})

- [ ] **Step 3: Decision point**

If labels are degenerate (all 0 or all 1):
- Tune the continuation prompt in `_build_continuation_prompt` to elicit better reasoning
- Or upgrade K to 8 for more granularity (0/8, 1/8, ..., 8/8 = 9 levels)
- Or switch to Sonnet for labeling (3x cost but better discrimination)
- Document the decision in `logs/pilot-labeling.md`

If labels look reasonable: proceed to Task 19.

- [ ] **Step 4: Commit pilot summary**

```bash
cat > logs/pilot-labeling.md <<EOF
# Pilot MC Labeling Summary

- N labels: <fill>
- Mean MC: <fill>
- Cost: $<fill>
- Decision: PROCEED / REVISIT-PROMPT / UPGRADE-MODEL
EOF
git add logs/pilot-labeling.md
git commit -m "exp: pilot MC labeling distribution check"
```

---

### Task 19: Full-scale MC labeling

**Files:**
- Create: `scripts/20_label_mc.sh`

- [ ] **Step 1: Write driver script**

```bash
#!/usr/bin/env bash
set -euo pipefail
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?must set}"

python -m src.labeler.label_all \
    --input_dir data/raw/swebench-lite \
    --output_dir data/labeled/swebench-lite \
    --budget_usd 80 \
    --K 4 \
    --concurrency 8

python -m src.labeler.label_all \
    --input_dir data/raw/bigcodebench-hard \
    --output_dir data/labeled/bigcodebench-hard \
    --budget_usd 60 \
    --K 4 \
    --concurrency 8
```

- [ ] **Step 2: Implement `src/labeler/label_all.py`**

```python
"""Drive MC labeling across many jsonl files with concurrency."""
from __future__ import annotations
import argparse, asyncio
from pathlib import Path
from src.utils.cost_tracker import CostTracker
from src.labeler.anthropic_client import RateLimitedClient
from src.labeler.mc_labeler import label_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--budget_usd", type=float, required=True)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()

    tracker = CostTracker(budget_usd=args.budget_usd)
    client = RateLimitedClient(tracker, model="claude-haiku-4-5")

    files = sorted(args.input_dir.glob("*.jsonl"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        out = args.output_dir / f.name
        print(f"Labeling {f} -> {out}")
        label_file(f, out, client, K=args.K)
        print(f"  cost so far: {tracker}")
        if tracker.over_budget():
            print("OVER BUDGET — stopping.")
            break

    print(f"\nFINAL: {tracker}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run labeling on SWE-bench Lite**

```bash
bash scripts/20_label_mc.sh 2>&1 | tee logs/mc-labeling.log
```
Expected wall-clock: ~10-15 hours (in tmux). Expected cost: ~$120 (within budget).

- [ ] **Step 4: Verify output**

```bash
find data/labeled -name "*.jsonl" -exec cat {} \; | wc -l
python -c "
from src.utils.jsonl_io import read_trajectories
import glob
labeled = 0
unlabeled = 0
for f in glob.glob('data/labeled/**/*.jsonl', recursive=True):
    for t in read_trajectories(f):
        for s in t.trajectory:
            if s.tool is not None:
                if s.mc_label is not None: labeled += 1
                else: unlabeled += 1
print(f'Labeled steps: {labeled}, Unlabeled: {unlabeled}')
"
```
Expected: ~20k+ labeled steps, < 1k unlabeled.

- [ ] **Step 5: Commit log**

```bash
git add scripts/20_label_mc.sh src/labeler/label_all.py logs/mc-labeling.log
git commit -m "exp: full MC labeling pass (~$120, ~20k labels)"
```

---

### Task 20: Train/val/test dataset assembly

**Files:**
- Create: `scripts/30_assemble_dataset.py`

- [ ] **Step 1: Implement**

```python
"""Combine all labeled trajectories into train/val/test split."""
from __future__ import annotations
import argparse, random
from pathlib import Path
from src.utils.jsonl_io import read_trajectories, write_trajectories


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dirs", nargs="+", type=Path,
                   default=[Path("data/labeled/swebench-lite"),
                            Path("data/labeled/bigcodebench-hard")])
    p.add_argument("--output_dir", type=Path,
                   default=Path("data/code-trajectory-2.4k"))
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--test_frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    all_trajs = []
    for d in args.input_dirs:
        for f in sorted(d.glob("*.jsonl")):
            all_trajs.extend(read_trajectories(f))

    print(f"Total: {len(all_trajs)} trajectories")
    rng = random.Random(args.seed)
    rng.shuffle(all_trajs)

    n_test = int(len(all_trajs) * args.test_frac)
    n_val = int(len(all_trajs) * args.val_frac)
    test = all_trajs[:n_test]
    val = all_trajs[n_test:n_test + n_val]
    train = all_trajs[n_test + n_val:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_trajectories(args.output_dir / "train.jsonl", train)
    write_trajectories(args.output_dir / "val.jsonl", val)
    write_trajectories(args.output_dir / "test.jsonl", test)

    print(f"train={len(train)}  val={len(val)}  test={len(test)}")
    # Report outcome balance
    for split, data in [("train", train), ("val", val), ("test", test)]:
        pass_rate = sum(t.outcome for t in data) / max(len(data), 1)
        print(f"  {split} pass rate: {pass_rate:.2%}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run assembly**

```bash
python scripts/30_assemble_dataset.py
```
Expected:
- `train.jsonl` ~ 1900 lines
- `val.jsonl` ~ 240 lines
- `test.jsonl` ~ 240 lines
- Pass rate consistent across splits (within 5 pp)

- [ ] **Step 3: Final sanity check**

```bash
python -c "
from src.utils.jsonl_io import read_trajectories
from collections import Counter
for split in ['train', 'val', 'test']:
    trajs = list(read_trajectories(f'data/code-trajectory-2.4k/{split}.jsonl'))
    n_steps = sum(len(t.trajectory) for t in trajs)
    types = Counter(t.task_type for t in trajs)
    print(f'{split}: {len(trajs)} traj, {n_steps} steps, types={dict(types)}')
"
```
Expected: distribution roughly even between swe-bench-lite and bigcodebench-hard.

- [ ] **Step 4: Commit**

```bash
git add scripts/30_assemble_dataset.py
git commit -m "feat: dataset assembly script (train/val/test split)"
```

---

### Task 21: Phase 1 closeout — report + spec for Plan 2

**Files:**
- Create: `docs/phase1-report.md`

- [ ] **Step 1: Write phase 1 report**

`docs/phase1-report.md`:
```markdown
# Code-PRM Phase 1 Report

## Status: COMPLETE

## Deliverables
- OpenR baseline reproduced on PRM800K: F1 = <fill>
- Trajectory dataset: 2400 trajectories, ~24k step labels
- Total cost: $<fill> (budget: $400)

## Dataset Statistics
| Split | Trajectories | Steps | Pass rate |
|---|---|---|---|
| train | <fill> | <fill> | <fill>% |
| val | <fill> | <fill> | <fill>% |
| test | <fill> | <fill> | <fill>% |

## MC Label Distribution
- Mean mc_label: <fill>
- Median: <fill>
- {0, 0.25, 0.5, 0.75, 1.0} counts: <fill>

## Key Findings
- <e.g., Haiku labeling correlation with outcome>
- <e.g., trajectory length distribution>
- <e.g., any surprising patterns>

## Risks Carried Forward to Phase 2
- <e.g., MC label noise level X, may need K=8 if PRM doesn't converge>
- <e.g., outcome=0 simplification may hurt — monitor in training>

## Next Phase
Plan 2 (Training) to be written next.
```

- [ ] **Step 2: Commit**

```bash
git add docs/phase1-report.md
git commit -m "docs: phase 1 closeout report"
```

- [ ] **Step 3: Tag the phase**

```bash
git tag -a phase1-complete -m "Phase 1 (data foundation) complete"
git push --tags
```

---

## Phase 1 Exit Criteria

Phase 1 is complete when ALL of these are true:

- [ ] OpenR baseline trains & evals on lab box 3090, F1 ≥ 0.5 on PRM800K small subset
- [ ] `data/code-trajectory-2.4k/{train,val,test}.jsonl` exist
- [ ] Total trajectories ≥ 2000 (target 2400, allow some failures)
- [ ] MC labels: ≥ 80% of tool-call steps have non-None `mc_label`
- [ ] MC label distribution is non-degenerate (mean ∈ [0.2, 0.8], not all-0/all-1)
- [ ] Total Phase 1 cost ≤ $500
- [ ] All Python unit tests pass: `pytest tests/ -v`
- [ ] `git tag phase1-complete` pushed
- [ ] `docs/phase1-report.md` filled in with actual numbers

Once all checkboxes ticked: ping Claude to invoke writing-plans skill again to draft Phase 2 plan (training).
