"""Tests for pipeline.py and agent.py in pipeline_agent."""
import pytest

from agents.pipeline_agent.pipeline import Pipeline, PipelineStep
from agents.pipeline_agent.agent import PipelineAgent, PipelineAgentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(name="test-pipeline-agent", pipeline_name="test-pipeline"):
    config = PipelineAgentConfig(name=name, pipeline_name=pipeline_name)
    return PipelineAgent(config)


# ---------------------------------------------------------------------------
# PipelineStep tests
# ---------------------------------------------------------------------------

class TestPipelineStep:
    def test_valid_step_runs(self):
        step = PipelineStep(name="double", handler=lambda x: x * 2)
        assert step.run(5) == 10

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            PipelineStep(name="", handler=lambda x: x)

    def test_non_callable_handler_raises(self):
        with pytest.raises(TypeError):
            PipelineStep(name="bad", handler="not_callable")

    # Personal note: also verify that None is rejected as a handler
    def test_none_handler_raises(self):
        with pytest.raises(TypeError):
            PipelineStep(name="none-handler", handler=None)


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_empty_pipeline_raises_on_run(self):
        p = Pipeline(name="empty")
        with pytest.raises(RuntimeError):
            p.run("input")

    def test_single_step_pipeline(self):
        p = Pipeline(name="single")
        p.add_step(PipelineStep(name="upper", handler=str.upper))
        assert p.run("hello") == "HELLO"

    def test_multi_step_pipeline_chains_results(self):
        p = Pipeline(name="chain")
        p.add_step(PipelineStep(name="strip", handler=str.strip))
        p.add_step(PipelineStep(name="upper", handler=str.upper))
        assert p.run("  hello  ") == "HELLO"

    def test_step_names_returns_ordered_list(self):
        p = Pipeline(name="named")
        p.add_step(PipelineStep(name="a", handler=lambda x: x))
        p.add_step(PipelineStep(name="b", handler=lambda x: x))
        assert p.step_names() == ["a", "b"]

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            Pipeline(name="")


# ---------------------------------------------------------------------------
# PipelineAgent tests
# ---------------------------------------------------------------------------

class TestPipelineAgent:
    def test_run_returns_correct_result(self):
        agent = _make_agent()
        agent.add_step(PipelineStep(name="double", handler=lambda x: x * 2))
        assert agent.run(3) == 6

    def test_last_result_cached(self):
        agent = _make_agent()
        agent.add_step(PipelineStep(name="inc", handler=lambda x: x + 1))
        agent.run(10)
        assert agent.last_result == 11

    def test_reset_clears_steps_and_res
