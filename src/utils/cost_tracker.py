"""Tracks token spend across Anthropic API calls.

Used by both the trajectory collector (Sonnet rollouts) and the MC labeler
(Haiku rollouts) to enforce a global budget cap.

Pricing constants are USD per 1M tokens. Update them if Anthropic's
pricing changes — see https://www.anthropic.com/pricing.
"""
from __future__ import annotations
import threading
from dataclasses import dataclass, field


# USD per 1,000,000 tokens. Source: Anthropic public pricing.
# Verify against the pricing page before relying on absolute dollar values.
PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-5":  {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5":   {"input": 1.00,  "output":  5.00},
    "claude-opus-4-7":    {"input": 15.00, "output": 75.00},
}


@dataclass
class CostTracker:
    """Thread-safe accumulator of token spend across multiple API calls.

    Attributes:
        budget_usd: Hard cap. Use `over_budget()` to short-circuit work.
        total_usd: Accumulated spend.
        per_model: Per-model spend breakdown.
    """
    budget_usd: float
    total_usd: float = 0.0
    per_model: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Record one API call. Returns the cost of this call in USD.

        Raises:
            KeyError: if `model` is not in PRICING.
        """
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
        """True once spend crosses `frac` of budget. Use for soft warnings."""
        return self.total_usd >= self.budget_usd * frac

    def __str__(self) -> str:
        pct = (self.total_usd / self.budget_usd * 100.0) if self.budget_usd > 0 else 0.0
        return (f"CostTracker: ${self.total_usd:.2f} / ${self.budget_usd:.2f} "
                f"({pct:.1f}%), per_model={self.per_model}")
