"""Scheduler agent that manages and runs scheduled tasks."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from agents.base_agent.agent import AgentConfig, BaseAgent
from agents.scheduler_agent.scheduler import Schedule, ScheduledTask, Scheduler


@dataclass
class SchedulerAgentConfig(AgentConfig):
    """Configuration for SchedulerAgent.

    Attributes:
        tick_interval: Seconds between scheduler ticks (default 1.0).
        max_tasks: Maximum number of tasks allowed (0 = unlimited).
    """

    tick_interval: float = 1.0
    max_tasks: int = 0

    def validate(self) -> None:  # noqa: D102
        super().validate()
        if self.tick_interval <= 0:
            raise ValueError(
                f"tick_interval must be positive, got {self.tick_interval}"
            )
        if self.max_tasks < 0:
            raise ValueError(
                f"max_tasks must be >= 0, got {self.max_tasks}"
            )


class SchedulerAgent(BaseAgent):
    """An agent that manages a collection of scheduled tasks.

    Tasks are registered with a name, a callable handler, and a
    :class:`~agents.scheduler_agent.scheduler.Schedule`.  A background
    thread ticks the internal :class:`~agents.scheduler_agent.scheduler.Scheduler`
    at the configured interval so that due tasks are executed automatically.

    Example::

        cfg = SchedulerAgentConfig(name="cron", tick_interval=0.5)
        agent = SchedulerAgent(cfg)
        agent.add_task("ping", lambda: print("ping"), Schedule(interval=2))
        agent.start()
        time.sleep(5)
        agent.stop()
    """

    def __init__(self, config: SchedulerAgentConfig) -> None:
        super().__init__(config)
        self._config: SchedulerAgentConfig = config
        self._scheduler: Scheduler = Scheduler()
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def add_task(
        self,
        name: str,
        handler: Callable[[], None],
        schedule: Schedule,
        *,
        max_runs: int = 0,
    ) -> None:
        """Register a new scheduled task.

        Args:
            name: Unique task identifier.
            handler: Zero-argument callable to invoke when the task is due.
            schedule: :class:`~agents.scheduler_agent.scheduler.Schedule`
                describing when the task should run.
            max_runs: Maximum number of executions (0 = unlimited).

        Raises:
            ValueError: If *name* is empty, *handler* is not callable, or the
                agent's ``max_tasks`` limit would be exceeded.
        """
        if not name or not name.strip():
            raise ValueError("Task name must not be empty.")
        if not callable(handler):
            raise TypeError(f"handler must be callable, got {type(handler).__name__}")

        limit = self._config.max_tasks
        if limit > 0 and len(self._scheduler.tasks) >= limit:
            raise RuntimeError(
                f"Cannot add task '{name}': max_tasks limit ({limit}) reached."
            )

        task = ScheduledTask(
            name=name,
            handler=handler,
            schedule=schedule,
            max_runs=max_runs,
        )
        self._scheduler.register(task)

    def remove_task(self, name: str) -> bool:
        """Remove a task by name.

        Returns:
            ``True`` if the task was found and removed, ``False`` otherwise.
        """
        return self._scheduler.unregister(name)

    @property
    def task_names(self) -> List[str]:
        """Return names of all currently registered tasks."""
        return [t.name for t in self._scheduler.tasks]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background scheduler thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"scheduler-{self._config.name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the scheduler thread to stop and wait for it to finish.

        Args:
            timeout: Maximum seconds to wait for the thread to join.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    @property
    def is_running(self) -> bool:
        """Return whether the scheduler loop is active."""
        return self._running

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Background loop that ticks the scheduler at the configured interval."""
        while self._running:
            self._scheduler.tick()
            time.sleep(self._config.tick_interval)
