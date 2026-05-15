"""Tests for the event_agent module."""
from __future__ import annotations

import pytest

from agents.event_agent.agent import EventAgent, EventAgentConfig
from agents.event_agent.events import Event, EventBus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(**kwargs) -> EventAgent:
    config = EventAgentConfig(name="test-agent", **kwargs)
    return EventAgent(config)


# ---------------------------------------------------------------------------
# Event tests
# ---------------------------------------------------------------------------

class TestEvent:
    def test_valid_event_created(self):
        e = Event(topic="ping", payload={"key": "value"})
        assert e.topic == "ping"
        assert e.payload == {"key": "value"}
        assert e.timestamp > 0

    def test_empty_topic_raises(self):
        with pytest.raises(ValueError, match="topic"):
            Event(topic="")

    def test_whitespace_topic_raises(self):
        with pytest.raises(ValueError, match="topic"):
            Event(topic="   ")


# ---------------------------------------------------------------------------
# EventBus tests
# ---------------------------------------------------------------------------

class TestEventBus:
    def test_publish_calls_subscriber(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe("greet", received.append)
        bus.publish("greet", "hello")
        assert len(received) == 1
        assert received[0].payload == "hello"

    def test_unsubscribe_stops_delivery(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe("x", received.append)
        bus.unsubscribe("x", received.append)
        bus.publish("x")
        assert received == []

    def test_unsubscribe_unknown_handler_raises(self):
        bus = EventBus()
        with pytest.raises(KeyError):
            bus.unsubscribe("missing", lambda e: None)

    def test_history_filtered_by_topic(self):
        bus = EventBus()
        bus.publish("a", 1)
        bus.publish("b", 2)
        bus.publish("a", 3)
        assert len(bus.history("a")) == 2
        assert len(bus.history("b")) == 1

    def test_clear_history(self):
        bus = EventBus()
        bus.publish("z")
        bus.clear_history()
        assert bus.history() == []

    def test_non_callable_handler_raises(self):
        bus = EventBus()
        with pytest.raises(TypeError):
            bus.subscribe("t", "not_a_function")  # type: ignore


# ---------------------------------------------------------------------------
# EventAgent tests
# ---------------------------------------------------------------------------

class TestEventAgentConfig:
    def test_defaults(self):
        cfg = EventAgentConfig(name="a")
        assert cfg.max_subscriptions == 50

    def test_invalid_max_subscriptions_raises(self):
        with pytest.raises(ValueError, match="max_subscriptions"):
            EventAgentConfig(name="a", max_subscriptions=0).validate()


class TestEventAgent:
    def test_on_and_emit(self):
        agent = _make_agent()
        results: list = []
        agent.on("data", lambda e: results.append(e.payload))
        agent.emit("data", 42)
        assert results == [42]

    def test_off_removes_handler(self):
        agent = _make_agent()
        results: list = []
        agent.on("data", lambda e: results.append(e.payload))
        agent.off("data")
        agent.emit("data", 99)
        assert results == []

    def test_off_unknown_topic_raises(self):
        agent = _make_agent()
        with pytest.raises(KeyError):
            agent.off("nonexistent")

    def test_reset_clears_all_subscriptions(self):
        agent = _make_agent()
        results: list = []
        agent.on("evt", lambda e: results.append(e))
        agent.reset()
        agent.emit("evt")
        assert results == []

    def test_max_subscriptions_enforced(self):
        agent = _make_agent(max_subscriptions=2)
        agent.on("t1", lambda e: None)
        agent.on("t2", lambda e: None)
        with pytest.raises(RuntimeError, match="max_subscriptions"):
            agent.on("t3", lambda e: None)

    def test_shared_bus(self):
        bus = EventBus()
        a1 = EventAgent(EventAgentConfig(name="a1"), bus=bus)
        a2 = EventAgent(EventAgentConfig(name="a2"), bus=bus)
        received: list = []
        a2.on("ping", lambda e: received.append(e.payload))
        a1.emit("ping", "pong")
        assert received == ["pong"]
