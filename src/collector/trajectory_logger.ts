/**
 * Code-PRM trajectory logger — pi extension.
 *
 * Logs every agent run to a single jsonl file per (task_type, date) under
 * `$CODE_PRM_LOG_DIR`. Schema matches `src/labeler/trajectory_schema.py`.
 *
 * Activation: requires env `CODE_PRM_LOG_DIR`. Without it the extension is inert.
 *
 * Optional env:
 *   CODE_PRM_ROLLOUT_ID    integer, stamped on trajectory.rollout_id (default 0)
 *   CODE_PRM_RUN_ID        uuid, stamped on trajectory.run_id (default new uuid4)
 *   CODE_PRM_TASK_JSON     full task payload (SWE-bench row / BigCodeBench row)
 *   CODE_PRM_TASK_TYPE     "swe-bench-lite" | "bigcodebench-hard" | "other"
 *   CODE_PRM_TEST_COMMAND  shell command to run as the test suite at agent_end.
 *                          If unset, outcome defaults to 0 (NOT 1) — Phase 1
 *                          callers MUST supply a real grader to get meaningful
 *                          outcome labels.
 *
 * Install (project-local):
 *   ln -sf <agentrl-repo>/src/collector/trajectory_logger.ts \
 *          <pi-repo>/.pi/extensions/trajectory_logger.ts
 *
 * Install (global):
 *   ln -sf <agentrl-repo>/src/collector/trajectory_logger.ts \
 *          ~/.pi/agent/extensions/trajectory_logger.ts
 *
 * Loaded by jiti; no compilation needed.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";

// --- schema (mirror of src/labeler/trajectory_schema.py) ---

interface Step {
  step: number;
  role: "assistant" | "tool" | "user";
  thought: string;
  tool: string | null;
  tool_args: Record<string, unknown>;
  tool_result: string;
}

interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_creation_tokens: number;
  cost_usd: number;
}

interface TestResult {
  passed: boolean;
  command: string;
  exit_code: number;
  stdout_tail: string;
  stderr_tail: string;
  duration_sec: number | null;
}

interface Trajectory {
  task_id: string;
  task_type: "swe-bench-lite" | "bigcodebench-hard" | "other";
  run_id: string;
  rollout_id: number;
  task_prompt: string | null;
  task_metadata: Record<string, unknown>;
  repo: string | null;
  base_commit: string | null;
  final_diff: string | null;
  trajectory: Step[];
  outcome: 0 | 1;
  test_result: TestResult | null;
  policy_model: string;
  timestamp: string;
  token_usage: TokenUsage | null;
  label_method: null;
}

// --- helpers ---

function truncate(s: string, max: number): string {
  if (!s || s.length <= max) return s;
  const head = Math.floor(max * 0.55);
  const tail = max - head - 14; // "...[TRUNC]..." is 14 chars including dots
  return s.slice(0, head) + "...[TRUNC]..." + s.slice(s.length - tail);
}

function parseTaskJson(): Record<string, unknown> {
  const raw = process.env.CODE_PRM_TASK_JSON;
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

function inferTaskId(task: Record<string, unknown>): string {
  if (typeof task.instance_id === "string") return task.instance_id;
  if (typeof task.task_id === "string") return task.task_id;
  return "unknown";
}

function inferTaskPrompt(task: Record<string, unknown>): string | null {
  // SWE-bench uses "problem_statement"; BigCodeBench uses "prompt".
  if (typeof task.problem_statement === "string") return task.problem_statement;
  if (typeof task.prompt === "string") return task.prompt;
  if (typeof task.instruct_prompt === "string") return task.instruct_prompt;
  return null;
}

function readToolResultText(content: unknown): string {
  // pi ToolResultEvent shape (per packages/agent/src/harness/types.ts):
  //   { type: "tool_result", toolCallId, toolName, input,
  //     content: Array<TextContent | ImageContent>, details, isError }
  // We pass `event.content` directly here.
  if (!Array.isArray(content)) return "";
  return (content as Array<{ type?: string; text?: string }>)
    .filter((c) => c && c.type === "text" && typeof c.text === "string")
    .map((c) => c.text as string)
    .join("\n");
}

// --- extension ---

export default function (pi: ExtensionAPI) {
  const LOG_DIR = process.env.CODE_PRM_LOG_DIR;
  if (!LOG_DIR) {
    // Extension is inert when env var is unset — production pi sessions
    // run normally without any logging side-effect.
    return;
  }

  let traj: Trajectory | null = null;
  let stepIdx = 0;
  let policyModel = "unknown";
  const tokenUsage: TokenUsage = {
    input_tokens: 0,
    output_tokens: 0,
    cache_read_tokens: 0,
    cache_creation_tokens: 0,
    cost_usd: 0,
  };

  // Map toolCallId → partial step (between tool_call and tool_result).
  const pendingByCallId = new Map<string, Partial<Step>>();
  // Step indices belonging to the CURRENT assistant message. At message_end
  // we walk back and fill their `thought` field from the message's text
  // content. Pi emits tool_call BEFORE message_end for the same message,
  // so we can't read text at tool_call time — we have to back-fill.
  let currentMessageStepIndices: number[] = [];

  // -------------------------- session lifecycle --------------------------

  pi.on("session_start", async (_event, _ctx) => {
    const task = parseTaskJson();
    const taskType = (process.env.CODE_PRM_TASK_TYPE || "other") as
      | "swe-bench-lite"
      | "bigcodebench-hard"
      | "other";
    const rolloutId = Number.parseInt(process.env.CODE_PRM_ROLLOUT_ID || "0", 10);
    const runId = process.env.CODE_PRM_RUN_ID || crypto.randomUUID();
    const taskId = inferTaskId(task);
    const taskPrompt = inferTaskPrompt(task);

    // Capture initial git state (best-effort; outside-of-repo cases are OK).
    let baseCommit: string | null = null;
    let repoName: string | null = null;
    try {
      const { stdout } = await pi.exec("git", ["rev-parse", "HEAD"]);
      baseCommit = stdout.trim() || null;
    } catch {
      /* not a git repo or git missing — leave null */
    }
    try {
      const { stdout } = await pi.exec("git", ["remote", "get-url", "origin"]);
      repoName = stdout.trim() || null;
    } catch {
      /* no origin remote */
    }

    traj = {
      task_id: taskId,
      task_type: taskType,
      run_id: runId,
      rollout_id: rolloutId,
      task_prompt: taskPrompt,
      task_metadata: task,
      repo: repoName,
      base_commit: baseCommit,
      final_diff: null,
      trajectory: [],
      outcome: 0,
      test_result: null,
      policy_model: policyModel,
      timestamp: new Date().toISOString(),
      token_usage: null,
      label_method: null,
    };
    stepIdx = 0;
  });

  // -------------------------- tool lifecycle --------------------------

  pi.on("tool_call", async (event, _ctx) => {
    if (!traj) return;
    pendingByCallId.set(event.toolCallId, {
      tool: event.toolName,
      tool_args: event.input as Record<string, unknown>,
    });
  });

  pi.on("tool_result", async (event, _ctx) => {
    if (!traj) return;
    const pending = pendingByCallId.get(event.toolCallId) || {};
    // Per packages/agent/src/harness/types.ts ToolResultEvent: result content
    // lives directly on event.content, NOT on event.result.content.
    const ev = event as { content?: unknown };
    const resultText = readToolResultText(ev.content);
    const step: Step = {
      step: stepIdx++,
      role: "assistant",
      thought: "", // back-filled at message_end (pi emits tool_call BEFORE message_end)
      tool: pending.tool || null,
      tool_args: pending.tool_args || {},
      tool_result: truncate(resultText, 8000),
    };
    traj.trajectory.push(step);
    currentMessageStepIndices.push(traj.trajectory.length - 1);
    pendingByCallId.delete(event.toolCallId);
  });

  // -------------------------- message tracking --------------------------

  pi.on("message_end", async (event, _ctx) => {
    if (!traj) return;
    // pi AgentMessage union: UserMessage | AssistantMessage | ToolResultMessage.
    // Only AssistantMessage has model + usage; we care about those.
    const msg = (event as { message?: unknown }).message as
      | {
          role?: string;
          content?: unknown;
          model?: string;
          responseModel?: string;
          provider?: string;
          usage?: {
            input?: number;
            output?: number;
            cacheRead?: number;
            cacheWrite?: number;
            totalTokens?: number;
            cost?: {
              total?: number;
              input?: number;
              output?: number;
              cacheRead?: number;
              cacheWrite?: number;
            };
          };
        }
      | undefined;
    if (!msg || msg.role !== "assistant") return;

    // Stamp policy_model from the message itself. Prefer responseModel
    // (the actually-served model, e.g. when openrouter resolves "auto"),
    // fall back to the requested model.
    if (msg.responseModel) policyModel = msg.responseModel;
    else if (msg.model) policyModel = msg.model;
    if (traj) traj.policy_model = policyModel;

    // Extract assistant text and back-fill `thought` onto the tool steps
    // that came from THIS message. Distribute the text across all those
    // steps (cheap: same thought for all sibling tool calls in one message).
    let assistantText = "";
    if (Array.isArray(msg.content)) {
      const texts = msg.content
        .filter(
          (c): c is { type: "text"; text: string } =>
            !!c && typeof c === "object" && (c as { type?: string }).type === "text" &&
            typeof (c as { text?: string }).text === "string",
        )
        .map((c) => c.text);
      assistantText = texts.join("\n");
    }
    if (assistantText && traj && currentMessageStepIndices.length > 0) {
      const truncated = truncate(assistantText, 2000);
      for (const idx of currentMessageStepIndices) {
        if (idx >= 0 && idx < traj.trajectory.length) {
          traj.trajectory[idx].thought = truncated;
        }
      }
    }
    currentMessageStepIndices = []; // reset for next message

    // Accumulate token usage. Pi's Usage shape (packages/ai/src/types.ts):
    //   { input, output, cacheRead, cacheWrite, totalTokens, cost: {...} }
    const usage = msg.usage;
    if (usage) {
      tokenUsage.input_tokens += usage.input ?? 0;
      tokenUsage.output_tokens += usage.output ?? 0;
      tokenUsage.cache_read_tokens += usage.cacheRead ?? 0;
      tokenUsage.cache_creation_tokens += usage.cacheWrite ?? 0;
      tokenUsage.cost_usd += usage.cost?.total ?? 0;
    }
  });

  // -------------------------- finalize --------------------------

  pi.on("agent_end", async (_event, _ctx) => {
    if (!traj) return;

    // Run test command if supplied; otherwise outcome stays 0.
    const testCmd = process.env.CODE_PRM_TEST_COMMAND;
    if (testCmd) {
      const t0 = Date.now();
      try {
        const { stdout, stderr, code } = await pi.exec("bash", ["-c", testCmd]);
        const passed = code === 0;
        traj.test_result = {
          passed,
          command: testCmd,
          exit_code: code,
          stdout_tail: truncate(stdout || "", 2000),
          stderr_tail: truncate(stderr || "", 2000),
          duration_sec: (Date.now() - t0) / 1000,
        };
        traj.outcome = passed ? 1 : 0;
      } catch (e) {
        traj.test_result = {
          passed: false,
          command: testCmd,
          exit_code: -1,
          stdout_tail: "",
          stderr_tail: String(e).slice(0, 2000),
          duration_sec: (Date.now() - t0) / 1000,
        };
        traj.outcome = 0;
      }
    }

    // Capture final diff vs base_commit (if any).
    if (traj.base_commit) {
      try {
        const { stdout } = await pi.exec("git", ["diff", traj.base_commit]);
        traj.final_diff = stdout || null;
      } catch {
        /* git failed — leave null */
      }
    }

    traj.token_usage = tokenUsage;

    // Atomically append one jsonl line to a per-day file.
    const date = traj.timestamp.slice(0, 10).replace(/-/g, "");
    const file = path.join(LOG_DIR, `${traj.task_type}_${date}.jsonl`);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.appendFileSync(file, JSON.stringify(traj) + "\n");

    // Reset for next agent run within same pi session (if any).
    traj = null;
    stepIdx = 0;
    pendingByCallId.clear();
    currentMessageStepIndices = [];
  });

  // Best-effort flush on session shutdown in case agent_end didn't fire.
  pi.on("session_shutdown", async (_event, _ctx) => {
    if (!traj) return;
    traj.token_usage = tokenUsage;
    const date = traj.timestamp.slice(0, 10).replace(/-/g, "");
    const file = path.join(LOG_DIR, `${traj.task_type}_${date}.jsonl`);
    try {
      fs.mkdirSync(path.dirname(file), { recursive: true });
      fs.appendFileSync(file, JSON.stringify(traj) + "\n");
    } catch {
      /* shutting down anyway */
    }
    traj = null;
  });
}
