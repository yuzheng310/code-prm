# TS Trajectory Logger Integration Spec

This document is the contract between the Python pipeline and the existing
TypeScript codeAgent. The TS side MUST produce jsonl that conforms to
`src/labeler/trajectory_schema.py` (Pydantic-validated).

## File location

Each codeAgent run appends ONE jsonl line to:
`$CODE_PRM_LOG_DIR/<task_type>_<YYYYMMDD>.jsonl`

The env var `CODE_PRM_LOG_DIR` is the activation switch. If unset, no
logging happens — codeAgent runs as normal.

## Schema (mirror of `trajectory_schema.py`)

```typescript
interface Step {
  step: number;                    // 0-indexed
  role: "assistant" | "tool" | "user";
  thought: string;                 // "" if none
  tool: string | null;             // null for pure-thought step
  tool_args: Record<string, any>;
  tool_result: string;             // truncated to 8000 chars (see below)
  mc_label?: number | null;        // null at collection time
}

interface Trajectory {
  task_id: string;
  task_type: "swe-bench-lite" | "bigcodebench-hard" | "other";
  trajectory: Step[];
  outcome: 0 | 1;                  // 0 = test failed, 1 = test passed
  policy_model: string;            // e.g. "claude-sonnet-4-5"
  timestamp: string;               // ISO 8601 UTC
}
```

## Integration hook in codeAgent

Add into the codeAgent's tool-dispatch loop:

1. On agent start, allocate `steps: Step[] = []` and `stepIdx = 0`.
2. After each tool execution, push `{step: stepIdx++, role: "assistant", thought, tool, tool_args, tool_result}` to `steps`.
3. On task completion, run the task's test suite to compute `outcome`.
4. Append the assembled Trajectory as a single jsonl line.

## Truncation rules

- `tool_result` longer than 8000 chars → keep first 4400 + `"...[TRUNC]..."` + last 3586.
- `thought` longer than 2000 chars → keep first 2000.
