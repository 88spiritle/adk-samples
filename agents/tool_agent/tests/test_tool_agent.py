"""Tests for ToolAgent and ToolRegistry."""

import pytest

from agents.tool_agent.agent import ToolAgent, ToolAgentConfig
from agents.tool_agent.tools import ToolDefinition, ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(max_tool_calls: int = 5, allow_unknown: bool = False) -> ToolAgent:
    cfg = ToolAgentConfig(
        name="test-tool-agent",
        max_tool_calls=max_tool_calls,
        allow_unknown_tools=allow_unknown,
    )
    return ToolAgent(cfg)


_add_tool = ToolDefinition(
    name="add",
    description="Adds two numbers.",
    func=lambda a, b: a + b,
    parameters={"a": "int", "b": "int"},
)


# ---------------------------------------------------------------------------
# ToolDefinition tests
# ---------------------------------------------------------------------------

class TestToolDefinition:
    def test_invoke_returns_correct_result(self):
        result = _add_tool.invoke(a=3, b=4)
        assert result == 7

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name must not be empty"):
            ToolDefinition(name="", description="d", func=lambda: None)

    def test_non_callable_func_raises(self):
        with pytest.raises(ValueError, match="must be callable"):
            ToolDefinition(name="t", description="d", func="not_callable")


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        reg.register(_add_tool)
        assert reg.get("add") is _add_tool

    def test_duplicate_registration_raises(self):
        reg = ToolRegistry()
        reg.register(_add_tool)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_add_tool)

    def test_unregister(self):
        reg = ToolRegistry()
        reg.register(_add_tool)
        reg.unregister("add")
        assert reg.get("add") is None

    def test_list_tools_sorted(self):
        reg = ToolRegistry()
        for name in ["zebra", "apple", "mango"]:
            reg.register(ToolDefinition(name=name, description="", func=lambda: None))
        assert reg.list_tools() == ["apple", "mango", "zebra"]


# ---------------------------------------------------------------------------
# ToolAgent tests
# ---------------------------------------------------------------------------

class TestToolAgent:
    def test_run_tool_returns_result(self):
        agent = _make_agent()
        agent.register_tool(_add_tool)
        assert agent.run("add", a=10, b=5) == 15

    def test_history_recorded(self):
        agent = _make_agent()
        agent.register_tool(_add_tool)
        agent.run("add", a=1, b=2)
        # Each run() call should append exactly one entry to the history
        assert len(agent.history) == 1
