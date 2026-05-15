"""Tests for retry.py."""

from __future__ import annotations

import pytest

from agents.retry_agent.retry import RetryPolicy, RetryExecutor, RetryResult


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------

class TestRetryPolicy:
    def test_defaults(self) -> None:
        policy = RetryPolicy()
        assert policy.max_attempts == 3
        assert policy.delay_seconds == 1.0
        assert policy.backoff_factor == 2.0

    def test_invalid_max_attempts_raises(self) -> None:
        with pytest.raises(ValueError, match="max_attempts"):
            RetryPolicy(max_attempts=0)

    def test_negative_delay_raises(self) -> None:
        with pytest.raises(ValueError, match="delay_seconds"):
            RetryPolicy(delay_seconds=-0.1)

    def test_backoff_factor_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="backoff_factor"):
            RetryPolicy(backoff_factor=0.5)


# ---------------------------------------------------------------------------
# RetryExecutor
# ---------------------------------------------------------------------------

class TestRetryExecutor:
    def _fast_policy(self, max_attempts: int = 3) -> RetryPolicy:
        return RetryPolicy(max_attempts=max_attempts, delay_seconds=0.0, backoff_factor=1.0)

    def test_success_on_first_attempt(self) -> None:
        executor = RetryExecutor(self._fast_policy())
        result = executor.execute(lambda: 42)
        assert result.success is True
        assert result.value == 42
        assert result.attempts == 1

    def test_success_after_retries(self) -> None:
        calls = {"count": 0}

        def flaky() -> str:
            calls["count"] += 1
            if calls["count"] < 3:
                raise RuntimeError("not yet")
            return "ok"

        executor = RetryExecutor(self._fast_policy(max_attempts=3))
        result = executor.execute(flaky)
        assert result.success is True
        assert result.value == "ok"
        assert result.attempts == 3

    def test_failure_after_max_attempts(self) -> None:
        executor = RetryExecutor(self._fast_policy(max_attempts=2))
        result = executor.execute(lambda: (_ for _ in ()).throw(ValueError("boom")))
        assert result.success is False
        assert result.attempts == 2
        assert isinstance(result.last_exception, ValueError)

    def test_non_matching_exception_propagates(self) -> None:
        policy = RetryPolicy(
            max_attempts=3,
            delay_seconds=0.0,
            backoff_factor=1.0,
            exceptions=(TypeError,),
        )
        executor = RetryExecutor(policy)
        with pytest.raises(ValueError):
            executor.execute(lambda: (_ for _ in ()).throw(ValueError("unexpected")))

    def test_default_policy_used_when_none_provided(self) -> None:
        executor = RetryExecutor()
        assert executor.policy.max_attempts == 3
