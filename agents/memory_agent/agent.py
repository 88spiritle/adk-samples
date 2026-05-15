"""MemoryAgent: a conversational agent backed by ConversationMemory."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from agents.base_agent.agent import AgentConfig, BaseAgent
from agents.memory_agent.memory import ConversationMemory, Message


@dataclass
class MemoryAgentConfig(AgentConfig):
    """Configuration for MemoryAgent."""

    max_turns: int = 20
    system_prompt: str = "You are a helpful assistant."

    def validate(self) -> None:  # type: ignore[override]
        super().validate()
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if not self.system_prompt.strip():
            raise ValueError("system_prompt must not be empty")


class MemoryAgent(BaseAgent):
    """Agent that maintains a conversation history across multiple turns."""

    def __init__(self, config: MemoryAgentConfig) -> None:
        config.validate()
        super().__init__(config)
        self.memory: ConversationMemory = ConversationMemory(
            max_turns=config.max_turns
        )
        self._system_prompt: str = config.system_prompt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Process a user message, record it, and return a stub reply.

        In a real implementation this would call an LLM with the full
        history as context.  Here we return a deterministic echo so the
        module is self-contained and testable without external deps.
        """
        if not user_message.strip():
            raise ValueError("user_message must not be empty")

        self.memory.add("user", user_message)
        reply = self._generate_reply(user_message)
        self.memory.add("assistant", reply)
        return reply

    def history(self) -> List[Message]:
        """Return the full conversation history."""
        return self.memory.get_history()

    def reset(self) -> None:  # type: ignore[override]
        """Clear conversation memory and reset base state."""
        super().reset()
        self.memory.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_reply(self, user_message: str) -> str:
        """Stub reply generator — replace with real LLM call."""
        return f"[{self.config.name}] Echo: {user_message}"
