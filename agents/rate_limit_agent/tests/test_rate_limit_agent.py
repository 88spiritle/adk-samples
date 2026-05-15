"""Tests for the rate-limit agent and its rate-limiter primitives."""

from __future__ import annotations

import time
import pytest

from agents.rate_limit_agent.rate_limiter import RateLimitPolicy, RateLimiter
from agents.rate_limit_agent.agent import RateLimitAgentConfig, RateLimitAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(max_calls: int = 5, period: float = 1.0) -> RateLimitAgent:
    cfg = RateLimitAgentConfig(name="test-agent", max_calls=max_calls, period_seconds=period)
    return RateLimitAgent(cfg)


# ---------------------------------------------------------------------------
# RateLimitPolicy
# ---------------------------------------------------------------------------

class TestRateLimitPolicy:
    def test_defaults(self) -> None:
        p = RateLimitPolicy()
        assert p.max_calls == 10
        assert p.period_seconds == 1.0

    def test_invalid_max_calls_raises(self) -> None:
        with pytest.raises(ValueError, match="max_calls"):
            RateLimitPolicy(max_calls=0)

    def test_invalid_max_calls_negative_raises(self) -> None:
        # Negative values should be rejected just like zero
        with pytest.raises(ValueError, match="max_calls"):
            RateLimitPolicy(max_calls=-5)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period_seconds"):
            RateLimitPolicy(period_seconds=0.0)

    def test_negative_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period_seconds"):
            RateLimitPolicy(period_seconds=-1.0)


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_allows_up_to_max_calls(self) -> None:
        limiter = RateLimiter(policy=RateLimitPolicy(max_calls=3, period_seconds=60))
        assert all(limiter.allow() for _ in range(3))

    def test_blocks_after_max_calls(self) -> None:
        limiter = RateLimiter(policy=RateLimitPolicy(max_calls=2, period_seconds=60))
        limiter.allow()
        limiter.allow()
        assert limiter.allow() is False

    def test_reset_clears_history(self) -> None:
        limiter = RateLimiter(policy=RateLimitPolicy(max_calls=1, period_seconds=60))
        limiter.allow()
        limiter.reset()
        assert limiter.allow() is True

    def test_execute_raises_when_exceeded(self) -> None:
        limiter = RateLimiter(policy=RateLimitPolicy(max_calls=1, period_seconds=60))
        limiter.allow()
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            limiter.execute(lambda: None)

    def test_execute_returns_fn_result(self) -> None:
        limiter = RateLimiter(policy=RateLimitPolicy(max_calls=5, period_seconds=60))
        result = limiter.execute(lambda x: x * 2, 21)
        assert result == 42

    def test_execute_raises_when_exceeded_after_reset(self) -> None:
        # Verify that the limit is re-enforced correctly after a reset cycle.
        limiter = RateLimiter(policy=RateLimitPolicy(max_calls=2, period_seconds=60))
        limiter.execute(lambda: None)
        limiter.execute(lambda: None)
        limiter.reset()
        limiter.execute(lambda: None)
        limiter.execute(lambda: None)
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            limiter.execute(lambda: None)


# ---------------------------------------------------------------------------
# RateLimitAgent
# ---------------------------------------------------------------------------
