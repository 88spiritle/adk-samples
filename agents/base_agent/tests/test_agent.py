"""Unit tests for agents/base_agent/agent.py."""

import pytest

from agents.base_agent.agent import AgentConfig, BaseAgent


# ---------------------------------------------------------------------------
# AgentConfig tests
# ---------------------------------------------------------------------------

class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig(name="my-agent")
        assert cfg.model == "gemini-2.0-flash"
        assert cfg.max_iterations == 10
        assert cfg.temperature == 0.7
        assert cfg.tools == []
        assert cfg.system_prompt is None

    def test_validate_passes_for_valid_config(self):
        cfg = AgentConfig(name="valid", temperature=0.5, max_iterations=5)
        cfg.validate()  # should not raise

    def test_validate_raises_for_empty_name(self):
        cfg = AgentConfig(name="   ")
        with pytest.raises(ValueError, match="name"):
            cfg.validate()

    def test_validate_raises_for_temperature_out_of_range(self):
        cfg = AgentConfig(name="t", temperature=1.5)
        with pytest.raises(ValueError, match="Temperature"):
            cfg.validate()

    def test_validate_raises_for_zero_iterations(self):
        cfg = AgentConfig(name="t", max_iterations=0)
        with pytest.raises(ValueError, match="max_iterations"):
            cfg.validate()


# ---------------------------------------------------------------------------
# BaseAgent tests
# ---------------------------------------------------------------------------

class TestBaseAgent:
    @pytest.fixture()
    def agent(self) -> BaseAgent:
        return BaseAgent(AgentConfig(name="test-agent", max_iterations=3))

    def test_initial_iteration_count_is_zero(self, agent):
        assert agent._iteration_count == 0

    def test_increment_iteration_returns_true_within_limit(self, agent):
        assert agent.increment_iteration() is True  # 1
        assert agent.increment_iteration() is True  # 2
        assert agent.increment_iteration() is True  # 3

    def test_increment_iteration_returns_false_over_limit(self, agent):
        for _ in range(3):
            agent.increment_iteration()
        assert agent.increment_iteration() is False  # 4 > max 3

    def test_reset_clears_iteration_count(self, agent):
        agent.increment_iteration()
        agent.increment_iteration()
        agent.reset()
        assert agent._iteration_count == 0

    def test_build_generation_config_contains_temperature(self, agent):
        cfg = agent.build_generation_config()
        assert cfg["temperature"] == agent.config.temperature
        assert cfg["candidate_count"] == 1

    def test_constructor_rejects_invalid_config(self):
        with pytest.raises(ValueError):
            BaseAgent(AgentConfig(name=""))
