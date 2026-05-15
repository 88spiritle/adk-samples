"""Memory module for MemoryAgent: stores and retrieves conversation turns."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Message:
    """A single conversation message."""

    role: str  # 'user' or 'assistant'
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {self.role!r}")
        if not self.content.strip():
            raise ValueError("content must not be empty")


@dataclass
class ConversationMemory:
    """Stores a bounded history of Messages."""

    max_turns: int = 20
    _history: List[Message] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")

    def add(self, role: str, content: str) -> None:
        """Append a message; evict oldest pair when over capacity."""
        self._history.append(Message(role=role, content=content))
        # Keep at most max_turns * 2 raw messages (user + assistant per turn).
        limit = self.max_turns * 2
        if len(self._history) > limit:
            self._history = self._history[-limit:]

    def get_history(self) -> List[Message]:
        """Return a shallow copy of the current history."""
        return list(self._history)

    def last_n(self, n: int) -> List[Message]:
        """Return the last *n* messages."""
        if n < 0:
            raise ValueError("n must be >= 0")
        return self._history[-n:] if n else []

    def clear(self) -> None:
        """Wipe all stored messages."""
        self._history.clear()

    def find_by_role(self, role: str) -> List[Message]:
        """Return all messages matching *role*."""
        return [m for m in self._history if m.role == role]

    @property
    def size(self) -> int:
        """Number of messages currently stored."""
        return len(self._history)
