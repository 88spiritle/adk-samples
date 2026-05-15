"""Rate-limit agent — wraps any callable with configurable rate limiting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

from agents.base_agent.agent import AgentConfig, BaseAgent
from agents.rate_limit_agent.rate_limiter import RateLimitPolicy, RateLimiter


@dataclass
class RateLimitAgentConfig(AgentConfig):
    """Configuration specific to the RateLimitAgent.

    Defaults allow 10 calls per second, which is conservative enough for most
    external API integrations. Increase max_calls or period_seconds as needed.
    """

    # Raised from 10 to 60 to better suit typical per-minute API quotas.
    max_calls: int = 60
    period_seconds: float = 60.0

    def validate(self) -> None:  # type: ignore[override]
        super().validate()
        RateLimitPolicy(max_calls=self.max_calls, period_seconds=self.period_seconds)


class RateLimitAgent(BaseAgent):
    """An agent that enforces a rate limit on calls to a registered handler."""

    def __init__(self, config: RateLimitAgentConfig) -> None:
        super().__init__(config)
        policy = RateLimitPolicy(
            max_calls=config.max_calls,
            period_seconds=config.period_seconds,
        )
        self._limiter: RateLimiter = RateLimiter(policy=policy)
        self._handler: Callable[..., Any] | None = None

    def register_handler(self, handler: Callable[..., Any]) -> None:
        """Set the callable that will be invoked on each permitted call."""
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._handler = handler

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the registered handler, respecting the rate limit."""
        if self._handler is None:
            raise RuntimeError("No handler registered. Call register_handler() first.")
        return self._limiter.execute(self._handler, *args, **kwargs)

    def reset(self) -> None:  # type: ignore[override]
        super().reset()
        self._limiter.reset()
