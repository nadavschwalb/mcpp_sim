"""Policy abstraction for MCPP simulator robots."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple

from ..environment.grid import OccupancyGridEnvironment
# from ..visualization.environment_visualizer import EnvironmentVisualizer


class Policy(ABC):
    """Abstract base class for robot policies."""

    def __init__(self, **attr) -> None:
        self._last_move_success = True
        self._env_visualizer : EnvironmentVisualizer = attr.get('visualizer', None)

    @abstractmethod
    def initialize(self, environment: OccupancyGridEnvironment, robot: Any, robots: list) -> Any:
        """initialize the policy using the environment and robot state"""

    @abstractmethod
    def observe(self, environment: OccupancyGridEnvironment, robot: Any, robots: list) -> Any:
        """Observe the environment and robot state."""

    @abstractmethod
    def next_move(self, position: Tuple[int, int], observation: Any) -> str:
        """Return the next action label (up, down, left, right)."""

    def update_last_move_result(self, success: bool) -> None:
        """Record whether the most recent move succeeded."""
        self._last_move_success = success

    @property
    def last_move_success(self) -> bool:
        """Return the success of the last attempted move."""
        return self._last_move_success
