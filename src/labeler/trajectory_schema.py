"""Pydantic schemas for code-agent trajectories.

This module defines the data contract between:
- TypeScript codeAgent (which emits trajectory.jsonl)
- Python MC labeler (which reads, labels, and writes back)
- PRM trainer (which reads labeled trajectories)
"""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class Step(BaseModel):
    """One step in an agent trajectory (one tool call or one thought).

    Attributes:
        step: 0-indexed step number.
        role: Always "assistant" for now; reserved for future expansion.
        thought: Free-form text before the tool call. "" if pure tool step.
        tool: Tool name (e.g. "read_file"). None for a pure-thought step.
        tool_args: Arguments passed to the tool.
        tool_result: Tool output (truncated to 8000 chars upstream).
        mc_label: Monte-Carlo soft label in [0, 1]. None until labeler fills it.
    """
    step: int = Field(ge=0)
    role: Literal["assistant", "tool", "user"] = "assistant"
    thought: str = ""
    tool: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    tool_result: str = ""
    mc_label: float | None = Field(default=None, ge=0.0, le=1.0)


class Trajectory(BaseModel):
    """A full multi-turn agent trajectory on one task.

    Attributes:
        task_id: Unique task identifier (e.g. SWE-bench instance_id).
        task_type: Which task set this came from.
        trajectory: Ordered list of steps.
        outcome: 0 = test failed, 1 = test passed.
        policy_model: Name of the LLM that generated the trajectory.
        timestamp: ISO 8601 UTC timestamp string.
    """
    task_id: str
    task_type: Literal["swe-bench-lite", "bigcodebench-hard", "other"]
    trajectory: list[Step]
    outcome: int
    policy_model: str
    timestamp: str

    @field_validator("outcome")
    @classmethod
    def outcome_in_range(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError(f"outcome must be 0 or 1, got {v}")
        return v
