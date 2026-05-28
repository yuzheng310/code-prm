"""Rate-limited Anthropic API client with retries.

Wraps `anthropic.Anthropic` with:
- Tenacity-based exponential backoff on RateLimitError / APIError
- Hard stop when the shared CostTracker exceeds budget
- Token usage automatically reported to the tracker

Designed for use by:
- The step labeler (Haiku by default, LLM-judge surrogate scoring partial
  trajectories). This is the primary in-process Anthropic consumer.
- Any future in-Python rollout code (e.g. real MC rollout in Phase 2);
  the trajectory collector itself subprocesses to the TS codeAgent so
  doesn't use this client directly.

Different instances can wrap different models; the CostTracker is shared.
"""
from __future__ import annotations
import os
from typing import Any

from anthropic import Anthropic, APIError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.cost_tracker import CostTracker


class BudgetExceededError(RuntimeError):
    """Raised when the shared CostTracker is over budget."""


class RateLimitedClient:
    """Anthropic client wrapping one model, sharing a CostTracker.

    Usage:
        tracker = CostTracker(budget_usd=10.0)
        client = RateLimitedClient(tracker, model="claude-haiku-4-5")
        text, in_tok, out_tok = client.complete(
            messages=[{"role": "user", "content": "..."}],
            max_tokens=1024,
        )
    """

    def __init__(
        self,
        tracker: CostTracker,
        model: str = "claude-haiku-4-5",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        # Honor ANTHROPIC_BASE_URL env var for users hitting a relay
        # (DeepSeek's Anthropic-compatible endpoint, OneAPI, AnyRouter, etc.).
        # Same env var the official Anthropic SDK respects.
        endpoint = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set; pass api_key= or export the env var."
            )
        client_kwargs: dict[str, Any] = {"api_key": key}
        if endpoint:
            client_kwargs["base_url"] = endpoint
        self.client = Anthropic(**client_kwargs)
        self.tracker = tracker
        self.model = model
        # When going through a relay, the model name we pass might be
        # ignored or remapped server-side. CostTracker pricing is for
        # direct Anthropic rates; warn the user once.
        if endpoint and endpoint != "https://api.anthropic.com":
            print(
                f"[anthropic_client] Using relay endpoint: {endpoint}\n"
                "  Model name on the wire: " + model + "\n"
                "  Note: CostTracker pricing assumes direct Anthropic rates; "
                "real billing follows your relay's pricing. Trust the relay "
                "dashboard for ground-truth spend.",
                flush=True,
            )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception_type((RateLimitError, APIError)),
        reraise=True,
    )
    def complete(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 2048,
        temperature: float = 0.8,
        system: str | None = None,
    ) -> tuple[str, int, int]:
        """Send one chat-completion request. Returns (text, input_tok, output_tok).

        Raises:
            BudgetExceededError: pre-check before issuing the request.
            RateLimitError / APIError: only if all 5 retries are exhausted.
        """
        # Budget enforcement removed by request (cost monitored via relay
        # dashboard / cost_aggregator instead). CostTracker still accumulates
        # for reporting, but never aborts.

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system is not None:
            kwargs["system"] = system

        resp = self.client.messages.create(**kwargs)
        # Concatenate all text blocks in the response (typically one).
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        self.tracker.add(self.model, in_tok, out_tok)
        return text, in_tok, out_tok
