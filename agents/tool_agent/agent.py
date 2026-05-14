"""ToolAgent: an agent that dispatches requests to registered tools."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.base_agent.agent import AgentConfig, BaseAgent
from agents.tool_agent.tools import ToolDefinition, ToolRegistry


@dataclass
class ToolAgentConfig(AgentConfig):
    """Configuration for ToolAgent, extending BaseAgent config."""

    max_tool_calls: int = 10
    allow_unknown_tools: bool = False

    def validate(self) -> None:
        super().validate()
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1.")


class ToolAgent(BaseAgent):
    """An agent capable of invoking registered tools by name."""

    def __init__(self, config: ToolAgentConfig):
        super().__init__(config)
        self.config: ToolAgentConfig = config
        self._registry: ToolRegistry = ToolRegistry()
        self._call_count: int = 0
        self._history: List[Dict[str, Any]] = []

    def register_tool(self, tool: ToolDefinition) -> None:
        """Register a tool with the agent's internal registry."""
        self._registry.register(tool)

    def list_tools(self) -> List[str]:
        """Return names of all registered tools."""
        return self._registry.list_tools()

    def run(self, tool_name: str, **kwargs: Any) -> Any:
        """Invoke a tool by name and record the call in history."""
        if self._call_count >= self.config.max_tool_calls:
            raise RuntimeError(
                f"Tool call limit of {self.config.max_tool_calls} reached."
            )

        tool = self._registry.get(tool_name)
        if tool is None:
            if not self.config.allow_unknown_tools:
                raise KeyError(f"Unknown tool: '{tool_name}'.")
            result = None
        else:
            result = tool.invoke(**kwargs)

        self._call_count += 1
        self._history.append(
            {"tool": tool_name, "kwargs": kwargs, "result": result}
        )
        return result

    def reset(self) -> None:
        """Reset call count and history."""
        super().reset()
        self._call_count = 0
        self._history.clear()

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Read-only view of the tool call history."""
        return list(self._history)
