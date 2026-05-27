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
| `SWEBENCH_TASK_JSON` | Full task payload from HF dataset | — |

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
- **`token_usage`** — without it, cost aggregation is impossible; Python
  side has only an estimate
- **`run_id`** — set to `process.env.CODE_PRM_RUN_ID || crypto.randomUUID()`
- **`rollout_id`** — set to `parseInt(process.env.CODE_PRM_ROLLOUT_ID || "0", 10)`
- **`test_result`** — distinguishes test failure modes (compile error vs
  flaky vs real wrong-answer)

## Recommended for Phase 2 (real MC rollout upgrade path)

- `repo`, `base_commit` — required to checkout the right state for replay
- `final_diff` — required to verify the agent's final state

## Integration hook in codeAgent

1. At agent start, allocate `steps: Step[] = []` and a `stepIdx = 0`.
2. Capture initial `base_commit` (git SHA of the repo before the agent acts).
3. After each tool execution, push `{step: stepIdx++, role: "assistant", thought, tool, tool_args, tool_result}`.
4. Track cumulative API token usage from each Anthropic response, summing into a `TokenUsage` accumulator.
5. On task completion:
   - Run the task's test suite
   - Capture `outcome` (0/1), `test_result` (richer), `final_diff` (git diff vs base_commit)
6. Assemble the Trajectory and append as a single jsonl line.

## Truncation rules

- `tool_result` longer than 8000 chars → keep first 4400 + `"...[TRUNC]..."` + last 3586.
- `thought` longer than 2000 chars → keep first 2000.
- `stdout_tail` / `stderr_tail` (in `test_result`) → keep last ~2000 chars each.

## Sample reference implementation

See `<TS_REPO>/src/hooks/trajectory_logger.ts` for the recommended Node
implementation (created in Task 9 of the Phase 1 plan).
