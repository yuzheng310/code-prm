"""Pydantic schemas for code-agent trajectories.

This module defines the data contract between:
- TypeScript codeAgent (which emits trajectory.jsonl)
- Python step labeler (which reads, labels, and writes back)
- PRM trainer (which reads labeled trajectories)

Schema design notes:
- `step_label` (formerly `mc_label`) holds a per-step quality score in [0, 1].
  In Phase 1 these are computed by an LLM-judge surrogate (see step_labeler.py).
  Phase 2 may upgrade to real Monte-Carlo rollout (replay state + sandboxed
  re-execution); the field semantics stay the same, only `label_method` changes.
- Replay-supporting fields (`repo`, `base_commit`, `final_diff`, `test_result`)
  are optional in Phase 1 but recommended for Phase 2 to enable real MC.
- `token_usage` should be filled by the TS side from the API response, so
  Python-side cost reporting reflects real spend, not estimates.
"""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class TokenUsage(BaseModel):
    """Per-trajectory API token usage and dollar cost.

    Filled by the TS codeAgent from the Anthropic API response. Allows the
    Python side to aggregate real cost rather than estimating from step counts.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0


class TestResult(BaseModel):
    """Captured test-execution outcome for the final state of a trajectory.

    Richer than `Trajectory.outcome` (which is just 0/1). Useful for analyzing
    flaky tests, distinguishing failure modes, and debugging label drift.
    """
    passed: bool
    command: str = ""
    exit_code: int = 0
    stdout_tail: str = ""
    stderr_tail: str = ""
    duration_sec: float | None = None


class Step(BaseModel):
    """One step in an agent trajectory (one tool call or one thought).

    Attributes:
        step: 0-indexed step number.
        role: Always "assistant" for now; reserved for future expansion.
        thought: Free-form text before the tool call. "" if pure tool step.
        tool: Tool name (e.g. "read_file"). None for a pure-thought step.
        tool_args: Arguments passed to the tool.
        tool_result: Tool output (truncated to 8000 chars upstream).
        step_label: Quality score in [0, 1]. None until step_labeler fills it.
            Phase 1: LLM-judge surrogate. Phase 2 may switch to real MC.
    """
    step: int = Field(ge=0)
    role: Literal["assistant", "tool", "user"] = "assistant"
    thought: str = ""
    tool: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    tool_result: str = ""
    step_label: float | None = Field(default=None, ge=0.0, le=1.0)


class Trajectory(BaseModel):
    """A full multi-turn agent trajectory on one task.

    Required fields are needed by every consumer. Optional fields support
    replay (real MC rollout), richer cost tracking, and richer outcome analysis.
    """
    # --- identification ---
    task_id: str
    task_type: Literal["swe-bench-lite", "bigcodebench-hard", "other"]
    run_id: str | None = None
    rollout_id: int = Field(default=0, ge=0)

    # --- task description (used by step labeler's judge prompt) ---
    # Without this, the LLM judge sees only the tool trace and cannot tell
    # whether the trajectory is solving the right problem. TS logger should
    # populate this from SWE-bench `problem_statement` / BigCodeBench `prompt`.
    task_prompt: str | None = None
    task_metadata: dict[str, Any] = Field(default_factory=dict)

    # --- environment / replay (optional in Phase 1, required for real MC) ---
    repo: str | None = None
    base_commit: str | None = None
    final_diff: str | None = None

    # --- agent execution ---
    trajectory: list[Step]
    outcome: int                 # 0 = test failed, 1 = test passed
    test_result: TestResult | None = None

    # --- meta ---
    policy_model: str
    timestamp: str
    token_usage: TokenUsage | None = None
    label_method: Literal["llm_judge", "mc_rollout", "ground_truth"] | None = None

    @field_validator("outcome")
    @classmethod
    def outcome_in_range(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError(f"outcome must be 0 or 1, got {v}")
        return v

    @model_validator(mode="after")
    def outcome_matches_test_result(self) -> "Trajectory":
        """If `test_result` is present, `outcome` must agree with `test_result.passed`."""
        if self.test_result is not None:
            expected = 1 if self.test_result.passed else 0
            if self.outcome != expected:
                raise ValueError(
                    f"outcome ({self.outcome}) disagrees with "
                    f"test_result.passed ({self.test_result.passed}); "
                    "they must be consistent."
                )
        return self
