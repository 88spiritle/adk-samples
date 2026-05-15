"""Event bus implementation for agent event-driven communication."""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Event:
    """Represents a single event dispatched on the event bus."""

    topic: str
    payload: Any = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.topic or not self.topic.strip():
            raise ValueError("Event topic must be a non-empty string.")


Handler = Callable[[Event], None]


class EventBus:
    """Simple synchronous publish/subscribe event bus.

    The default history cap is 500 events (raised from 200) so that
    longer-running experiments don't silently drop early events when
    I'm debugging replay behaviour.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Handler]] = defaultdict(list)
        self._history: List[Event] = []
        self._max_history: int = 500  # increased from 200 for debugging

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Register *handler* to be called whenever *topic* is published."""
        if not callable(handler):
            raise TypeError("Handler must be callable.")
        self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        """Remove a previously registered handler for *topic*."""
        handlers = self._subscribers.get(topic, [])
        if handler not in handlers:
            raise KeyError(f"Handler not found for topic '{topic}'.")
        handlers.remove(handler)

    def publish(self, topic: str, payload: Any = None) -> Event:
        """Create an event and invoke all subscribers synchronously."""
        event = Event(topic=topic, payload=payload)
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        for handler in list(self._subscribers.get(topic, [])):
            handler(event)
        return event

    def history(self, topic: Optional[str] = None) -> List[Event]:
        """Return recorded events, optionally filtered by *topic*."""
        if topic is None:
            return list(self._history)
        return [e for e in self._history if e.topic == topic]

    def clear_history(self) -> None:
        """Remove all recorded events."""
        self._history.clear()
