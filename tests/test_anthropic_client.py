"""Tests for the RateLimitedClient wrapper (initialization + relay routing).

We don't actually call the API; we monkeypatch `Anthropic` so we can assert
the right kwargs (base_url) are passed through.
"""
from __future__ import annotations
from unittest.mock import MagicMock

import pytest

from src.utils.cost_tracker import CostTracker


def test_uses_anthropic_base_url_env_var(monkeypatch) -> None:
    """ANTHROPIC_BASE_URL must be forwarded to the Anthropic client."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")

    captured: dict = {}

    def fake_anthropic(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    # Patch BEFORE importing — the import is local to make this clean.
    from src.labeler import anthropic_client
    monkeypatch.setattr(anthropic_client, "Anthropic", fake_anthropic)

    tracker = CostTracker(budget_usd=1.0)
    anthropic_client.RateLimitedClient(tracker, model="claude-haiku-4-5")

    assert captured["api_key"] == "sk-test"
    assert captured["base_url"] == "https://api.deepseek.com/anthropic"


def test_base_url_unset_falls_back_to_default(monkeypatch) -> None:
    """Without ANTHROPIC_BASE_URL set, base_url is NOT passed (SDK default applies)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    captured: dict = {}

    def fake_anthropic(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    from src.labeler import anthropic_client
    monkeypatch.setattr(anthropic_client, "Anthropic", fake_anthropic)

    tracker = CostTracker(budget_usd=1.0)
    anthropic_client.RateLimitedClient(tracker, model="claude-haiku-4-5")

    assert "base_url" not in captured  # SDK uses its own default


def test_explicit_base_url_wins_over_env(monkeypatch) -> None:
    """Constructor-provided base_url overrides the env var."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://env-relay.example")

    captured: dict = {}

    def fake_anthropic(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    from src.labeler import anthropic_client
    monkeypatch.setattr(anthropic_client, "Anthropic", fake_anthropic)

    tracker = CostTracker(budget_usd=1.0)
    anthropic_client.RateLimitedClient(
        tracker, model="claude-haiku-4-5", base_url="https://explicit.example"
    )

    assert captured["base_url"] == "https://explicit.example"


def test_missing_api_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from src.labeler import anthropic_client
    tracker = CostTracker(budget_usd=1.0)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        anthropic_client.RateLimitedClient(tracker, model="claude-haiku-4-5")
