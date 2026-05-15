"""Retry policy and executor for agent operations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Any, Optional


@dataclass
class RetryPolicy:
    """Configuration for retry behaviour."""

    max_attempts: int = 3
    delay_seconds: float = 1.0
    backoff_factor: float = 2.0
    exceptions: tuple = field(default_factory=lambda: (Exception,))

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.delay_seconds < 0:
            raise ValueError("delay_seconds must be non-negative")
        if self.backoff_factor < 1.0:
            raise ValueError("backoff_factor must be >= 1.0")


@dataclass
class RetryResult:
    """Outcome of a retried operation."""

    success: bool
    value: Any = None
    attempts: int = 0
    last_exception: Optional[Exception] = None


class RetryExecutor:
    """Executes a callable according to a RetryPolicy."""

    def __init__(self, policy: Optional[RetryPolicy] = None) -> None:
        self.policy: RetryPolicy = policy or RetryPolicy()

    def execute(self, fn: Callable[[], Any]) -> RetryResult:
        """Run *fn*, retrying on allowed exceptions.

        Args:
            fn: A zero-argument callable to execute.

        Returns:
            A RetryResult describing the outcome.
        """
        delay = self.policy.delay_seconds
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.policy.max_attempts + 1):
            try:
                value = fn()
                return RetryResult(success=True, value=value, attempts=attempt)
            except self.policy.exceptions as exc:  # type: ignore[misc]
                last_exc = exc
                if attempt < self.policy.max_attempts:
                    time.sleep(delay)
                    delay *= self.policy.backoff_factor

        return RetryResult(
            success=False,
            attempts=self.policy.max_attempts,
            last_exception=last_exc,
        )
