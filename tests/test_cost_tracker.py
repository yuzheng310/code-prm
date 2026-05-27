"""Unit tests for CostTracker."""
from __future__ import annotations
import pytest
from src.utils.cost_tracker import CostTracker, PRICING


def test_tracks_sonnet_cost() -> None:
    t = CostTracker(budget_usd=10.0)
    t.add("claude-sonnet-4-5", input_tokens=1000, output_tokens=500)
    # Sonnet: $3/M input, $15/M output
    expected = (1000 / 1_000_000) * 3.00 + (500 / 1_000_000) * 15.00
    assert abs(t.total_usd - expected) < 1e-9


def test_per_model_breakdown_accumulates() -> None:
    t = CostTracker(budget_usd=100.0)
    t.add("claude-sonnet-4-5", 1_000_000, 0)  # $3
    t.add("claude-haiku-4-5", 1_000_000, 0)   # $1
    t.add("claude-sonnet-4-5", 1_000_000, 0)  # $3 more → $6 sonnet total
    assert abs(t.per_model["claude-sonnet-4-5"] - 6.00) < 1e-9
    assert abs(t.per_model["claude-haiku-4-5"] - 1.00) < 1e-9
    assert abs(t.total_usd - 7.00) < 1e-9


def test_remaining_decreases() -> None:
    t = CostTracker(budget_usd=10.0)
    assert t.remaining() == 10.0
    t.add("claude-haiku-4-5", 1_000_000, 0)  # $1
    assert abs(t.remaining() - 9.0) < 1e-9


def test_over_budget_when_exceeded() -> None:
    t = CostTracker(budget_usd=0.10)
    # 1M input on Haiku = $1, way over $0.10 budget
    t.add("claude-haiku-4-5", 1_000_000, 0)
    assert t.over_budget()
    assert t.remaining() < 0


def test_unknown_model_raises() -> None:
    t = CostTracker(budget_usd=10.0)
    with pytest.raises(KeyError):
        t.add("gpt-5-nonexistent", 100, 100)


def test_warn_threshold_default_80pct() -> None:
    t = CostTracker(budget_usd=10.0)
    # Spend $7.99 — should NOT trigger warn
    t.add("claude-sonnet-4-5", input_tokens=int(7.99e6 / 3), output_tokens=0)
    assert not t.warn_threshold()
    # Push over 80%: add a bit more
    t.add("claude-sonnet-4-5", input_tokens=100_000, output_tokens=0)  # +$0.30
    assert t.warn_threshold()


def test_add_returns_call_cost() -> None:
    t = CostTracker(budget_usd=10.0)
    cost = t.add("claude-haiku-4-5", input_tokens=500_000, output_tokens=200_000)
    # Haiku: 0.5M * $1 + 0.2M * $5 = $0.50 + $1.00 = $1.50
    assert abs(cost - 1.50) < 1e-9


def test_pricing_table_has_expected_models() -> None:
    """Sanity check: the models we plan to use in Phase 1 must be priced."""
    for name in ("claude-sonnet-4-5", "claude-haiku-4-5"):
        assert name in PRICING
        assert PRICING[name]["input"] > 0
        assert PRICING[name]["output"] > 0
