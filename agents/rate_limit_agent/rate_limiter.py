"""Rate limiter primitives for the rate-limit agent."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class RateLimitPolicy:
    """Configuration for a token-bucket rate limiter."""

    max_calls: int = 10
    period_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        if self.period_seconds <= 0:
            raise ValueError("period_seconds must be positive")


@dataclass
class RateLimiter:
    """Simple sliding-window rate limiter."""

    policy: RateLimitPolicy
    _timestamps: list[float] = field(default_factory=list, init=False, repr=False)

    def _evict_old(self, now: float) -> None:
        cutoff = now - self.policy.period_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]

    def allow(self) -> bool:
        """Return True if the call is permitted under the current policy."""
        now = time.monotonic()
        self._evict_old(now)
        if len(self._timestamps) < self.policy.max_calls:
            self._timestamps.append(now)
            return True
        return False

    def reset(self) -> None:
        """Clear all recorded call timestamps."""
        self._timestamps.clear()

    def execute(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute *fn* if the rate limit allows; raise RuntimeError otherwise."""
        if not self.allow():
            raise RuntimeError(
                f"Rate limit exceeded: max {self.policy.max_calls} calls "
                f"per {self.policy.period_seconds}s"
            )
        return fn(*args, **kwargs)
