"""Pipeline module for chaining multiple agents sequentially."""
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class PipelineStep:
    """Represents a single step in a pipeline."""

    name: str
    handler: Callable[[Any], Any]
    description: str = ""

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("PipelineStep name must not be empty.")
        if not callable(self.handler):
            raise TypeError("PipelineStep handler must be callable.")

    def run(self, input_data: Any) -> Any:
        """Execute this step with the given input."""
        return self.handler(input_data)


@dataclass
class Pipeline:
    """Executes a sequence of steps, passing output of each to the next."""

    name: str
    steps: List[PipelineStep] = field(default_factory=list)

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("Pipeline name must not be empty.")

    def add_step(self, step: PipelineStep) -> None:
        """Append a step to the pipeline."""
        if not isinstance(step, PipelineStep):
            raise TypeError("Expected a PipelineStep instance.")
        self.steps.append(step)

    def run(self, initial_input: Any) -> Any:
        """Run all steps in order, threading output through each step."""
        if not self.steps:
            raise RuntimeError("Pipeline has no steps to execute.")
        result = initial_input
        for step in self.steps:
            result = step.run(result)
        return result

    def step_names(self) -> List[str]:
        """Return the names of all registered steps."""
        return [step.name for step in self.steps]
