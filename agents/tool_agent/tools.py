"""Tool definitions and registry for the ToolAgent."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolDefinition:
    """Represents a single callable tool available to an agent."""

    name: str
    description: str
    func: Callable[..., Any]
    parameters: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("Tool name must not be empty.")
        if not callable(self.func):
            raise ValueError(f"Tool '{self.name}' func must be callable.")

    def invoke(self, **kwargs: Any) -> Any:
        """Invoke the tool with the provided keyword arguments."""
        return self.func(**kwargs)


class ToolRegistry:
    """Registry that stores and manages ToolDefinitions."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool, raising an error on duplicate names."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Return a tool by name, or None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """Return a sorted list of registered tool names."""
        return sorted(self._tools.keys())

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry.")
        del self._tools[name]

    def __len__(self) -> int:
        return len(self._tools)
