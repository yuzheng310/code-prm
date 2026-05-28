# TS Trajectory Logger Integration Spec

This document is the contract between the Python pipeline and the existing
TypeScript codeAgent. The TS side MUST produce jsonl that conforms to
`src/labeler/trajectory_schema.py` (Pydantic-validated).

## Activation

Set `CODE_PRM_LOG_DIR` in the codeAgent's process env. If unset, no
logging happens — codeAgent runs normally.

Additional env vars from the Python collector (`src/eval/collect_batch.py`):

| Env var | Purpose | Default if unset |
|---|---|---|
| `CODE_PRM_LOG_DIR` | Directory to write jsonl into | (no logging) |
| `CODE_PRM_ROLLOUT_ID` | Stamp `rollout_id` on trajectory | `0` |
| `CODE_PRM_RUN_ID` | Stamp `run_id` on trajectory | TS generates UUID |
| `CODE_PRM_TASK_JSON` | Full task payload from HF dataset (preferred) | — |
| `CODE_PRM_TASK_TYPE` | Task set name ("swe-bench-lite" / "bigcodebench-hard") | — |
| `SWEBENCH_TASK_JSON` | Legacy alias of `CODE_PRM_TASK_JSON` (still set, will be removed) | — |

## File location

ONE jsonl line appended per agent run to:

`$CODE_PRM_LOG_DIR/<task_type>_<YYYYMMDD>.jsonl`

All rollouts write to the same file. Use `rollout_id` and `run_id` inside
the trajectory record to distinguish them — do NOT nest into subdirs.

## Schema (mirror of `trajectory_schema.py`)

```typescript
interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens?: number;
  cache_creation_tokens?: number;
  cost_usd: number;
}

interface TestResult {
  passed: boolean;
  command?: string;
  exit_code?: number;
  stdout_tail?: string;            // last ~2k chars
  stderr_tail?: string;            // last ~2k chars
  duration_sec?: number;
}

interface Step {
  step: number;                    // 0-indexed
  role: "assistant" | "tool" | "user";
  thought: string;                 // "" if none
  tool: string | null;             // null for pure-thought step
  tool_args: Record<string, any>;
  tool_result: string;             // truncated to 8000 chars
  step_label?: number | null;      // null at collection; filled by labeler
}

interface Trajectory {
  // identification
  task_id: string;
  task_type: "swe-bench-lite" | "bigcodebench-hard" | "other";
  run_id?: string;                 // uuid v4 per agent run
  rollout_id: number;              // from CODE_PRM_ROLLOUT_ID, default 0

  // task description (REQUIRED by step labeler's judge prompt)
  task_prompt?: string;            // problem_statement (SWE) or prompt (BigCode)
  task_metadata?: Record<string, any>;  // any other benchmark fields you want

  // environment / replay (recommended for future real MC; optional now)
  repo?: string | null;            // e.g. "django/django"
  base_commit?: string | null;     // git SHA before agent ran
  final_diff?: string | null;      // patch agent produced

  // agent execution
  trajectory: Step[];
  outcome: 0 | 1;                  // 0 = test failed, 1 = test passed
  test_result?: TestResult | null; // richer outcome detail

  // meta
  policy_model: string;            // e.g. "claude-sonnet-4-5"
  timestamp: string;               // ISO 8601 UTC
  token_usage?: TokenUsage | null; // REAL spend; required for cost aggregation
  label_method?: null;             // set by Python labeler; leave null on TS side
}
```

## Required fields the TS MUST produce

These cannot default sensibly and must be in the output:
- `task_id`
- `task_type`
- `trajectory` (list of steps)
- `outcome` (the agent must actually run the test suite to compute this)
- `policy_model`
- `timestamp`

## Strongly recommended fields

The pipeline still works without these, but the analysis quality drops:
- **`task_prompt`** — REQUIRED for meaningful step labeling. Extract from
  `JSON.parse(process.env.CODE_PRM_TASK_JSON)`:
  - SWE-bench: `task.problem_statement`
  - BigCodeBench: `task.prompt` (or `task.instruct_prompt`)
  Without this the LLM judge has no idea what task is being solved.
- **`token_usage`** — without it, cost aggregation is impossible; Python
  side has only an estimate
- **`run_id`** — set to `process.env.CODE_PRM_RUN_ID || crypto.randomUUID()`
- **`rollout_id`** — set to `parseInt(process.env.CODE_PRM_ROLLOUT_ID || "0", 10)`
- **`test_result`** — distinguishes test failure modes (compile error vs
  flaky vs real wrong-answer). MUST be consistent with `outcome`:
  if `test_result.passed === true` then `outcome === 1`, else 0.
  Schema enforces this; emitting mismatched values will fail Pydantic
  validation downstream.

## Recommended for Phase 2 (real MC rollout upgrade path)

- `repo`, `base_commit` — required to checkout the right state for replay
- `final_diff` — required to verify the agent's final state

## Integration hook in codeAgent

1. At agent start, allocate `steps: Step[] = []` and a `stepIdx = 0`.
2. Parse `CODE_PRM_TASK_JSON` to extract:
   - `task_prompt` (problem_statement / prompt)
   - `task_metadata` (any other benchmark fields you care about)
3. Capture initial `base_commit` (git SHA of the repo before the agent acts).
4. After each tool execution, push `{step: stepIdx++, role: "assistant", thought, tool, tool_args, tool_result}`.
5. Track cumulative API token usage from each Anthropic response, summing into a `TokenUsage` accumulator.
6. On task completion:
   - Run the task's test suite
   - Capture `outcome` (0/1), `test_result` (richer), `final_diff` (git diff vs base_commit)
   - Verify `test_result.passed === !!outcome` before emitting (consistency).
7. Assemble the Trajectory and append as a single jsonl line.

## Truncation rules

- `tool_result` longer than 8000 chars → keep first 4400 + `"...[TRUNC]..."` + last 3586.
- `thought` longer than 2000 chars → keep first 2000.
- `stdout_tail` / `stderr_tail` (in `test_result`) → keep last ~2000 chars each.

## Process exit code convention

This is crucial for the Python collector's `--max_initial_failed_attempts`
guard. The TS CLI MUST distinguish:

| Outcome | TS exit code | trajectory.outcome | Notes |
|---|---|---|---|
| Agent solved + tests pass | **0** | 1 | normal success |
| Agent attempted + tests fail | **0** | 0 | normal task-level failure — process is healthy |
| Agent crashed mid-run | **0** if a partial trajectory was logged, else non-zero | 0 if logged | best-effort: log what you have, exit 0 |
| Config error (bad CLI args, missing env, code bug) | **non-zero** | (no jsonl written) | aborts the whole batch via launch_error/initial-failure guard |

In short: **exit 0 means "the process ran to completion (regardless of
whether the task passed)"**. Only exit non-zero on infrastructure errors
that genuinely require human intervention.

Violating this convention will make the Python collector either:
(a) silently swallow agent crashes as task failures, or
(b) abort the whole batch on benign task failures.

## Reference implementation

A working implementation against the **pi** agent harness
(github.com/earendil-works/pi) lives in this repo at:

    src/collector/trajectory_logger.ts

It is loaded by pi via its extension system (jiti — no compilation needed).
Install:

```bash
# Global (recommended)
ln -sf $PWD/src/collector/trajectory_logger.ts \
       ~/.pi/agent/extensions/trajectory_logger.ts

# OR project-local (only loaded in pi runs within a given project)
ln -sf $PWD/src/collector/trajectory_logger.ts \
       <pi-project-root>/.pi/extensions/trajectory_logger.ts
```

A bootstrap script automates this:

    bash scripts/01_setup_pi.sh

The reference implementation:
- Activates only when `CODE_PRM_LOG_DIR` is set
- Hooks `session_start` (captures task_id/task_prompt/base_commit),
  `tool_call` + `tool_result` (records each step with cumulative thought),
  `message_end` (accumulates token_usage),
  `agent_end` (runs test command if set, captures final diff, appends jsonl),
  `session_shutdown` (best-effort flush)
- Honors `CODE_PRM_TEST_COMMAND` env to run a shell command as the grader
- Honors `CODE_PRM_ROLLOUT_ID` / `CODE_PRM_RUN_ID` from the Python collector

## Adapting to a different TS codeAgent

The reference depends on pi's specific hook names (`tool_call`, `tool_result`,
`message_end`, `agent_end`). If you wire this up to a non-pi codeAgent:

1. Find the equivalent hook points in your agent's lifecycle.
2. Preserve the env-var contract (CODE_PRM_*).
3. Preserve the jsonl output path convention (`$LOG_DIR/<task_type>_<YYYYMMDD>.jsonl`).
4. Preserve the schema (see `trajectory_schema.py`).
5. Run `pytest tests/test_trajectory_schema.py` on a sample of your output
   before scaling up.
