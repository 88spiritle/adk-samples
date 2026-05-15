"""Pipeline agent that wraps a Pipeline and exposes an agent-style interface."""
from dataclasses import dataclass, field
from typing import Any, Optional

from agents.base_agent.agent import AgentConfig, BaseAgent
from agents.pipeline_agent.pipeline import Pipeline, PipelineStep


@dataclass
class PipelineAgentConfig(AgentConfig):
    """Configuration for PipelineAgent."""

    pipeline_name: str = "default_pipeline"

    def validate(self) -> None:
        super().validate()
        if not self.pipeline_name or not self.pipeline_name.strip():
            raise ValueError("pipeline_name must not be empty.")


class PipelineAgent(BaseAgent):
    """An agent that processes input through a configurable pipeline."""

    def __init__(self, config: PipelineAgentConfig) -> None:
        config.validate()
        super().__init__(config)
        self._pipeline = Pipeline(name=config.pipeline_name)
        self._last_result: Optional[Any] = None

    def add_step(self, step: PipelineStep) -> None:
        """Register a new step in the underlying pipeline."""
        self._pipeline.add_step(step)

    def run(self, input_data: Any) -> Any:
        """Execute the pipeline with the provided input and cache the result."""
        self._last_result = self._pipeline.run(input_data)
        return self._last_result

    @property
    def last_result(self) -> Optional[Any]:
        """Return the result of the most recent pipeline execution."""
        return self._last_result

    def reset(self) -> None:
        """Clear cached result and reset pipeline steps."""
        super().reset()
        self._last_result = None
        self._pipeline.steps.clear()

    def step_names(self):
        """Delegate to the underlying pipeline."""
        return self._pipeline.step_names()
