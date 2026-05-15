"""Scheduler module for the scheduler agent.

Provides a simple task scheduling system that supports one-shot and
repeating tasks with configurable intervals and optional max-run limits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class ScheduledTask:
    """Represents a single scheduled task.

    Attributes:
        name: Unique name identifying the task.
        handler: Callable invoked when the task fires.
        interval: Seconds between executions (0 means run once).
        max_runs: Maximum number of times to run; None means unlimited.
    """

    name: str
    handler: Callable[[], None]
    interval: float = 0.0
    max_runs: Optional[int] = None

    # Internal bookkeeping — not part of the public interface.
    _run_count: int = field(default=0, init=False, repr=False)
    _next_run: float = field(default_factory=time.monotonic, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Task name must be a non-empty string.")
        if not callable(self.handler):
            raise TypeError("Task handler must be callable.")
        if self.interval < 0:
            raise ValueError("Task interval must be >= 0.")
        if self.max_runs is not None and self.max_runs < 1:
            raise ValueError("max_runs must be None or a positive integer.")

    @property
    def is_exhausted(self) -> bool:
        """Return True when the task has reached its max_runs limit."""
        return self.max_runs is not None and self._run_count >= self.max_runs

    def is_due(self, now: Optional[float] = None) -> bool:
        """Return True if the task is ready to fire."""
        if self.is_exhausted:
            return False
        return (now if now is not None else time.monotonic()) >= self._next_run

    def run(self) -> None:
        """Execute the handler and update internal counters."""
        self.handler()
        self._run_count += 1
        self._next_run = time.monotonic() + self.interval


class Scheduler:
    """A lightweight in-process task scheduler.

    Tasks are registered by name and executed when :meth:`tick` is called.
    The scheduler does **not** manage its own thread; callers are responsible
    for driving the tick loop at an appropriate frequency.
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, ScheduledTask] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_task(self, task: ScheduledTask) -> None:
        """Register a task with the scheduler.

        Args:
            task: The :class:`ScheduledTask` to register.

        Raises:
            ValueError: If a task with the same name is already registered.
        """
        if task.name in self._tasks:
            raise ValueError(f"A task named '{task.name}' is already registered.")
        self._tasks[task.name] = task

    def remove_task(self, name: str) -> None:
        """Unregister a task by name.

        Args:
            name: The name of the task to remove.

        Raises:
            KeyError: If no task with that name exists.
        """
        if name not in self._tasks:
            raise KeyError(f"No task named '{name}' is registered.")
        del self._tasks[name]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def tick(self) -> List[str]:
        """Run all tasks that are currently due.

        Exhausted (one-shot or max_runs-reached) tasks are automatically
        removed after they fire.

        Returns:
            A list of task names that were executed during this tick.
        """
        now = time.monotonic()
        executed: List[str] = []
        to_remove: List[str] = []

        for name, task in list(self._tasks.items()):
            if task.is_due(now):
                task.run()
                executed.append(name)
                if task.is_exhausted or task.interval == 0:
                    to_remove.append(name)

        for name in to_remove:
            self._tasks.pop(name, None)

        return executed

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def pending_tasks(self) -> List[str]:
        """Return the names of all currently registered tasks."""
        return list(self._tasks.keys())
