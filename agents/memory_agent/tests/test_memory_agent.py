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

    def test_find_by_role(self):
        mem = ConversationMemory()
        mem.add("user", "q")
        mem.add("assistant", "a")
        mem.add("user", "q2")
        assert len(mem.find_by_role("user")) == 2

    def test_invalid_max_turns_raises(self):
        with pytest.raises(ValueError):
            ConversationMemory(max_turns=0)


# ---------------------------------------------------------------------------
# MemoryAgent tests
# ---------------------------------------------------------------------------

def _make_agent(**kwargs) -> MemoryAgent:
    defaults = {"name": "test-agent", "model": "gemini-pro", "max_turns": 5}
    defaults.update(kwargs)
    return MemoryAgent(MemoryAgentConfig(**defaults))


class TestMemoryAgent:
    def test_chat_returns_echo(self):
        agent = _make_agent()
        reply = agent.chat("Hello")
        assert "Hello" in reply

    def test_history_grows_after_chat(self):
        agent = _make_agent()
        agent.chat("first")
        assert len(agent.history()) == 2  # user + assistant

    def test_reset_clears_history(self):
        agent = _make_agent()
        agent.chat("hi")
        agent.reset()
        assert agent.history() == []

    def test_empty_message_raises(self):
        agent = _make_agent()
        with pytest.raises(ValueError):
            agent.chat("   ")

    def test_invalid_max_turns_raises(self):
        with pytest.raises(ValueError):
            _make_agent(max_turns=0)

    def test_empty_system_prompt_raises(self):
        with pytest.raises(ValueError):
            _make_agent(system_prompt="  ")
