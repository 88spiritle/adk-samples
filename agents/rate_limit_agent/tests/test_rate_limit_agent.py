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


# ---------------------------------------------------------------------------
# RateLimitAgent
# ---------------------------------------------------------------------------

class TestRateLimitAgent:
    def test_call_invokes_handler(self) -> None:
        agent = _make_agent(max_calls=5)
        agent.register_handler(lambda x: x + 1)
        assert agent.call(9) == 10

    def test_call_without_handler_raises(self) -> None:
        agent = _make_agent()
        with pytest.raises(RuntimeError, match="No handler registered"):
            agent.call()

    def test_non_callable_handler_raises(self) -> None:
        agent = _make_agent()
        with pytest.raises(TypeError, match="callable"):
            agent.register_handler("not_a_function")  # type: ignore[arg-type]

    def test_rate_limit_enforced(self) -> None:
        agent = _make_agent(max_calls=2, period=60.0)
        agent.register_handler(lambda: "ok")
        agent.call()
        agent.call()
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            agent.call()

    def test_reset_allows_fresh_calls(self) -> None:
        agent = _make_agent(max_calls=1, period=60.0)
        agent.register_handler(lambda: "ok")
        agent.call()
        agent.reset()
        assert agent.call() == "ok"

    def test_config_validation_propagates(self) -> None:
        with pytest.raises(ValueError):
            RateLimitAgentConfig(name="x", max_calls=0).validate()
