"""Base agent module providing a reusable foundation for ADK sample agents."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for a base agent."""

    name: str
    model: str = "gemini-2.0-flash"
    description: str = ""
    max_iterations: int = 10
    temperature: float = 0.7
    tools: list[Any] = field(default_factory=list)
    system_prompt: Optional[str] = None

    def validate(self) -> None:
        """Validate agent configuration values."""
        if not self.name or not self.name.strip():
            raise ValueError("Agent name must not be empty.")
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")


class BaseAgent:
    """A minimal base agent wrapping common ADK agent setup."""

    def __init__(self, config: AgentConfig) -> None:
        config.validate()
        self.config = config
        self._iteration_count: int = 0
        logger.info("Initialized agent '%s' with model '%s'.", config.name, config.model)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal state between runs."""
        self._iteration_count = 0
        logger.debug("Agent '%s' state reset.", self.config.name)

    def increment_iteration(self) -> bool:
        """Increment the iteration counter.

        Returns:
            True if the agent may continue, False when the limit is reached.
        """
        self._iteration_count += 1
        if self._iteration_count > self.config.max_iterations:
            logger.warning(
                "Agent '%s' reached max_iterations (%d).",
                self.config.name,
                self.config.max_iterations,
            )
            return False
        return True

    def build_generation_config(self) -> dict[str, Any]:
        """Return a generation config dict suitable for Vertex AI / ADK."""
        return {
            "temperature": self.config.temperature,
            "candidate_count": 1,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BaseAgent(name={self.config.name!r}, "
            f"model={self.config.model!r}, "
            f"iterations={self._iteration_count}/{self.config.max_iterations})"
        )
