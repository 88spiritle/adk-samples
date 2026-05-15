"""Event-driven agent that reacts to topics published on an EventBus."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from agents.base_agent.agent import AgentConfig, BaseAgent
from agents.event_agent.events import Event, EventBus, Handler


@dataclass
class EventAgentConfig(AgentConfig):
    """Configuration for EventAgent."""

    max_subscriptions: int = 50

    def validate(self) -> None:  # type: ignore[override]
        super().validate()
        if self.max_subscriptions < 1:
            raise ValueError("max_subscriptions must be at least 1.")


class EventAgent(BaseAgent):
    """An agent that subscribes to an EventBus and routes events to handlers."""

    def __init__(self, config: EventAgentConfig, bus: Optional[EventBus] = None) -> None:
        config.validate()
        super().__init__(config)
        self._config: EventAgentConfig = config
        self.bus: EventBus = bus if bus is not None else EventBus()
        self._registered: Dict[str, Handler] = {}

    def on(self, topic: str, handler: Handler) -> None:
        """Subscribe *handler* to *topic* on the shared bus."""
        if len(self._registered) >= self._config.max_subscriptions:
            raise RuntimeError(
                f"Cannot exceed max_subscriptions={self._config.max_subscriptions}."
            )
        self.bus.subscribe(topic, handler)
        self._registered[topic] = handler

    def off(self, topic: str) -> None:
        """Unsubscribe the handler previously registered for *topic*."""
        handler = self._registered.pop(topic, None)
        if handler is None:
            raise KeyError(f"No handler registered for topic '{topic}'.")
        self.bus.unsubscribe(topic, handler)

    def emit(self, topic: str, payload: Any = None) -> Event:
        """Publish an event on the bus and return the resulting Event."""
        return self.bus.publish(topic, payload)

    def reset(self) -> None:  # type: ignore[override]
        """Unsubscribe all handlers and clear bus history."""
        for topic, handler in list(self._registered.items()):
            self.bus.unsubscribe(topic, handler)
        self._registered.clear()
        self.bus.clear_history()
        super().reset()
