# Code-PRM Phase 1 Implementation Plan: Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or `superpowers:executing-plans`.
> Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **IMPORTANT:** This plan was rewritten on 2026-05-27 after three review
> rounds. The rewrite removes stale code blocks and points to the actual
> committed source files. The original verbose draft is in git history
> (search for commit `06ff01d`).

**Goal:** Build the data pipeline foundation — ~2400 code-agent trajectories with step-level LLM-judge labels — so that Phase 2 (PRM training) has clean inputs.

**Architecture:** Fork OpenR (reference only); instrument existing TypeScript codeAgent to emit `trajectory.jsonl`; Python step labeler asks an LLM K=4 times per step whether the partial trajectory will succeed, recording the success fraction as `step_label ∈ [0, 1]`. This is a weak-supervision surrogate, NOT Monte-Carlo rollout (see spec §5.3).

**Tech Stack:** Python 3.12, PyTorch 2.5, Transformers, PEFT, OpenR (referenced, not used for training), Anthropic SDK, SWE-bench, BigCodeBench, TypeScript (codeAgent integration).

**Deliverable at end of Phase 1:**
- `data/code-trajectory-2.4k/{train,val,test}.jsonl` with LLM-judge `step_label`
- Real cost ≤ $500 (aggregated from `Trajectory.token_usage` via `cost_aggregator`)

**Out of scope (later phases):** PRM training (Phase 2), CodeProcessBench + end-to-end eval (Phase 3), real MC rollout upgrade (Future Work).

---

## Decisions log (read these before executing)

These choices have been made and committed. Subsequent revisions of this plan
should respect them.

| # | Decision | Rationale | Spec ref |
|---|---|---|---|
| D1 | Skip OpenR baseline training | OpenR uses `+/-` token-prediction PRM (Math-Shepherd style); our spec uses scalar-head MSE (VLM-PRM style). Different paradigms. Env validated by `pytest tests/` passing. | §13 |
| D2 | LLM-judge surrogate for step labels (NOT real MC) | Real MC needs sandbox replay + state checkpoint; 3-4 weeks engineering + 4-8x API cost. Surrogate is honest weak supervision. | §5.3 |
| D3 | Flat data directory layout | Nested `rollout_k/` subdirs caused `glob` bugs in downstream readers. Each trajectory carries its own `rollout_id` instead. | — |
| D4 | Schema includes replay fields (`repo`, `base_commit`, `final_diff`, `test_result`, `token_usage`, `task_prompt`) | Optional in Phase 1; required for Phase 2 future real-MC upgrade. Real cost tracking depends on `token_usage`. | §4.3 |
| D5 | Python 3.12 + PyTorch 2.5 + CUDA 12.4 | Match the rental GPU image (AutoDL `pytorch / 2.5.1 / 3.12(ubuntu22.04) / 12.4`). | — |
| D6 | `label_method` enum distinguishes `"llm_judge"` (outcome=1 path) vs `"outcome_zero_simplification"` (outcome=0 path) | Phase 2 trainer can weight or filter by method. | §5.3 |

---

## Repository structure (current, post all reviews)

```
agentrl/
├── pyproject.toml                          # python>=3.11,<3.13; torch>=2.4,<2.6
├── environment.yml                         # conda: python=3.12, pytorch=2.5, cuda 12.4
├── .gitignore, .env.example, README.md
├── third_party/
│   └── openr/                              # submodule @ 54ae004 (reference only)
├── src/
│   ├── collector/
│   │   └── ts_logger_spec.md               # contract: TS-side trajectory schema
│   ├── eval/
│   │   ├── swebench_runner.py              # task loader + TS launcher (Task 10)
│   │   └── collect_batch.py                # async batched collection (Task 13)
│   ├── labeler/
│   │   ├── trajectory_schema.py            # Pydantic schemas (Task 8)
│   │   ├── anthropic_client.py             # rate-limited + retry (Task 15)
│   │   ├── step_labeler.py                 # LLM-judge labeler (Task 16)
│   │   └── label_all.py                    # batch labeling driver (Task 19)
│   └── utils/
│       ├── jsonl_io.py                     # read/write helpers (Task 8)
│       ├── cost_tracker.py                 # in-process token tracker (Task 12)
│       └── cost_aggregator.py              # real-cost summary from token_usage (Task 12)
├── scripts/
│   ├── 00_setup_lab_box.sh                 # any-NVIDIA-GPU bootstrap (Task 4)
│   ├── 05_collect_pilot.sh                 # 10-task pilot (Task 11)
│   ├── 10_collect_trajectories.sh          # SWE-bench Lite collection (Task 13)
│   ├── 11_collect_bigcodebench.sh          # BigCodeBench collection (Task 14)
│   ├── 20_label_steps.sh                   # LLM-judge labeling (Task 19)
│   └── 30_assemble_dataset.py              # train/val/test split (Task 20)
├── tests/
│   ├── test_trajectory_schema.py
│   ├── test_step_labeler.py
│   ├── test_cost_tracker.py
│   ├── test_cost_aggregator.py
│   └── fixtures/synthetic_trajectory.json
└── docs/superpowers/{specs,plans}/...
```

TS codeAgent side (lives in your separate TS repo, abbreviated `$TS_REPO`):

```
$TS_REPO/
└── src/hooks/trajectory_logger.ts          # implements ts_logger_spec.md (Task 9)
```

---

## Prerequisites (before any task)

- [ ] **Anthropic API key with $500+ budget** — `ANTHROPIC_API_KEY` env var
- [ ] **Lab box (NVIDIA GPU + Linux + ≥ 50 GB free disk)** — SSH + GitHub SSH key on the box
- [ ] **Your TS codeAgent repo cloned somewhere reachable** — its absolute path goes in `.env` as `TS_REPO_PATH`
- [ ] **Docker on lab box** (for any SWE-bench docker grading; optional in Phase 1)

---

## Task status (as of 2026-05-27)

Each task: ✅ committed / ⏳ pending external action / ⚠️ blocked-on-user.

| # | Task | Status | Source files / Notes |
|---|---|---|---|
| 1 | Bootstrap repo (`pyproject.toml`, `.gitignore`, `.env.example`, `README.md`) | ✅ | commit `bccebc4` — see repo root |
| 2 | conda `environment.yml` (Python 3.12 + PyTorch 2.5 + CUDA 12.4) | ✅ | commit `b357d6d` |
| 3 | Fork & vendor OpenR | ✅ | commit `bd073c4` — pinned at `54ae004` |
| 4 | Lab box bootstrap script | ✅ | commit `c8b267c` — `scripts/00_setup_lab_box.sh`, supports any NVIDIA GPU + auto-detects PyTorch |
| 5 | Download PRM800K dataset | ⏳ | LFS pull on lab box (done by user); kept as Phase 2 math ablation |
| 6 | ~~Run OpenR math-PRM baseline~~ | ❌ SKIPPED | see Decision D1 |
| 7 | ~~Eval OpenR baseline~~ | ❌ SKIPPED | see Decision D1 |
| 8 | Trajectory schema + jsonl IO + TS contract | ✅ | commit `5bd1627` (+ later schema enrichment in `5d15159`, `8ee7186`) |
| 9 | TS codeAgent logger | ⚠️ | User must add `trajectory_logger.ts` to TS repo per `src/collector/ts_logger_spec.md` |
| 10 | SWE-bench + BigCodeBench task loaders + TS launcher | ✅ | commit `5134bb0`; multi-task-set dispatch fixed in `8ee7186` |
| 11 | Pilot collection (10 trajectories) | ⏳ | Run on lab box after Task 9 |
| 12 | Cost tracker + aggregator | ✅ | commit `aac08ac` (tracker) + `5d15159` (aggregator) |
| 13 | Batched trajectory collection | ✅ | commit `b3e84c1` (script) + revisions in `5d15159` (flat layout) + this round (timeout handling) |
| 14 | BigCodeBench-Hard collection | ✅ | `scripts/11_collect_bigcodebench.sh` |
| 15 | Anthropic client (rate-limit + retry + budget gate) | ✅ | commit `305ff56` |
| 16 | Step labeler (LLM-judge surrogate) | ✅ | commit `13b8af8` (+ task_prompt + regex parser + symmetric only_tool_steps in `8ee7186`) |
| 17 | Step labeler unit tests | ✅ | `tests/test_step_labeler.py` — covers prompt construction, parsing, simplified path |
| 18 | Pilot step labeling (10 trajectories) | ⏳ | Run on lab box after pilot collection |
| 19 | Full-scale step labeling | ✅ (driver ready) / ⏳ (execution) | `scripts/20_label_steps.sh` + `src/labeler/label_all.py` |
| 20 | Dataset assembly | ✅ (script) / ⏳ (execution) | `scripts/30_assemble_dataset.py` — uses `rglob` |
| 21 | Phase 1 closeout report | ⏳ | `docs/phase1-report.md` — fill in after data is collected |

---

## Execution sequence (what's left to actually run)

```
[on Mac]  ── Already done. All code + scripts committed. ────────────
[on Lab Box]  ↓
   1. git pull origin main                                          (Task 0)
   2. pip install -e .                                              (one-time)
   3. pytest tests/ -v                                              (verify env)
[on Mac, in $TS_REPO]  ↓
   4. Add trajectory_logger.ts per src/collector/ts_logger_spec.md  (Task 9)
   5. git push your TS changes; clone TS repo onto lab box
[on Lab Box, in tmux]  ↓
   6. export TS_REPO_PATH=<...>; export ANTHROPIC_API_KEY=<...>
   7. Pilot collection:  bash scripts/05_collect_pilot.sh            (Task 11)
                          # 10 tasks × 1 rollout, budget $10
   8. Inspect data/raw/pilot/*.jsonl: outcome distribution,
      task_prompt populated, token_usage populated                   (Task 11)
   9. Pilot real-cost check:  python -m src.utils.cost_aggregator --dir data/raw/pilot
  10. Pilot labeling:  python -m src.labeler.label_all \
        --input_dir data/raw/pilot --output_dir data/labeled/pilot \
        --budget_usd 5 --K 4 --clean                                 (Task 18)
                                                                      # Pilot data
                                                                      # likely has
                                                                      # task_prompt
                                                                      # coverage < 95%
                                                                      # before TS
                                                                      # logger is
                                                                      # fully wired,
                                                                      # so you may
                                                                      # also need
                                                                      # --allow_low_task_prompt_coverage
  11. Inspect step_label distribution; commit pilot summary
  12. Full SWE:        bash scripts/10_collect_trajectories.sh       (Task 13)
  13. Full BigCode:    bash scripts/11_collect_bigcodebench.sh       (Task 14)
  14. Real cost:       python -m src.utils.cost_aggregator --dir data/raw
  15. Full labeling:   bash scripts/20_label_steps.sh                (Task 19)
  16. Assemble:        python scripts/30_assemble_dataset.py         (Task 20)
  17. Fill in docs/phase1-report.md with real numbers                (Task 21)
  18. Tag:             git tag phase1-complete && git push --tags
```

---

## Phase 1 Exit Criteria

Phase 1 is complete when ALL of these are true. Numbers are measured on the
lab box, not estimated.

- [ ] `pytest tests/ -v` passes (all unit tests green) on lab box
- [ ] `data/code-trajectory-2.4k/{train,val,test}.jsonl` exist
- [ ] Total trajectories ≥ 2000 (target 2400; allow some task failures)
- [ ] ≥ 80% of tool-call steps in outcome=1 trajectories have non-None `step_label`
- [ ] On outcome=1 trajectories: mean `step_label` ∈ [0.2, 0.8] (non-degenerate)
- [ ] Every labeled trajectory has `label_method` set to either `"llm_judge"` or `"outcome_zero_simplification"`
- [ ] **≥ 95% of outcome=1 trajectories have non-empty `task_prompt`** (without this the LLM judge is near-random)
- [ ] ≥ 80% of trajectories have `token_usage` populated (real cost coverage)
- [ ] Real cost (per `cost_aggregator`) ≤ $500
- [ ] `docs/phase1-report.md` filled in with actual numbers
- [ ] `git tag phase1-complete` pushed

When all boxes ticked: ping Claude to invoke `writing-plans` skill again
for Phase 2 (training) plan.

---

## Phase 1 Report Template

Once data is collected, fill in `docs/phase1-report.md`:

```markdown
# Code-PRM Phase 1 Report

## Status: COMPLETE | INCOMPLETE

## Deliverables
- Trajectory dataset: <N> trajectories, <M> total steps, <K> labeled steps
- Real cost: $<X.XX> (cost_aggregator output; budget $500)
- Pytest: <N>/<N> passing on lab box

## Dataset Statistics
| Split | Trajectories | Steps | Pass rate | Avg steps/traj |
|---|---|---|---|---|
| train | ... | ... | ...% | ... |
| val | ... | ... | ...% | ... |
| test | ... | ... | ...% | ... |

## Step-label distribution (outcome=1 path)
- Mean step_label: ...
- Median: ...
- Histogram at {0, 0.25, 0.5, 0.75, 1.0}: ...

## label_method breakdown
- llm_judge: <N> trajectories
- outcome_zero_simplification: <N> trajectories

## Token-usage coverage
- Trajectories with token_usage: <N> / <total> (...%)
- If < 80%: TS logger needs investigation

## Known Limitations / Carried Risks
- (list anything found during execution that Phase 2 should know about)
```
