"""Tests for memory.py and agent.py in memory_agent."""
import pytest

from agents.memory_agent.memory import ConversationMemory, Message
from agents.memory_agent.agent import MemoryAgent, MemoryAgentConfig


# ---------------------------------------------------------------------------
# Message tests
# ---------------------------------------------------------------------------

class TestMessage:
    def test_valid_user_message(self):
        m = Message(role="user", content="Hello")
        assert m.role == "user"
        assert m.content == "Hello"

    def test_valid_assistant_message(self):
        m = Message(role="assistant", content="Hi there")
        assert m.role == "assistant"

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="role"):
            Message(role="system", content="oops")

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="content"):
            Message(role="user", content="   ")


# ---------------------------------------------------------------------------
# ConversationMemory tests
# ---------------------------------------------------------------------------

class TestConversationMemory:
    def test_add_and_size(self):
        mem = ConversationMemory(max_turns=5)
        mem.add("user", "hi")
        mem.add("assistant", "hello")
        assert mem.size == 2

    def test_eviction_respects_max_turns(self):
        mem = ConversationMemory(max_turns=2)
        for i in range(3):
            mem.add("user", f"msg {i}")
            mem.add("assistant", f"reply {i}")
        assert mem.size == 4  # max_turns=2 => 4 raw messages

    def test_clear_empties_history(self):
        mem = ConversationMemory()
        mem.add("user", "test")
        mem.clear()
        assert mem.size == 0

    def test_last_n(self):
        mem = ConversationMemory()
        mem.add("user", "a")
        mem.add("assistant", "b")
        mem.add("user", "c")
        assert [m.content for m in mem.last_n(2)] == ["b", "c"]

    def test_last_n_zero_returns_empty(self):
        mem = ConversationMemory()
        mem.add("user", "x")
        assert mem.last_n(0) == []

    # Edge case: requesting more messages than exist should return all of them
    def test_last_n_exceeds_size_returns_all(self):
        mem = ConversationMemory()
        mem.add("user", "a")
        mem.add("assistant", "b")
        result = mem.last_n(10)
        assert len(result) == 2

    def test_find_by_role(self):
        mem = ConversationMemory()
        mem.add("user", "q")
        mem.add("assistant", "a")
        mem.add("user", "q2")
        assert len(mem.find_by_role("user")) == 2

    def test_invalid_max_turns_raises(self):
        with pytest.raises(ValueError):
            ConversationMemory(max_turns=0)

    # Personal note: also verify that max_turns=1 is accepted as the minimum
    # valid value (boundary check that the original tests missed).
    def test_min_valid_max_turns(self):
        mem = ConversationMemory(max_turns=1)
        mem.add("user", "only one turn allowed")
        mem.add("assistant", "response")
        # max_turns=1 => at most 2 raw messages kept
        assert mem.size == 2


# ---------------------------------------------------------------------------
# MemoryAgent tests
# ---------------------------------------------------------------------------

def _ma
