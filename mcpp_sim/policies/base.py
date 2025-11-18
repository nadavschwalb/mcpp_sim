"""Policy abstraction for MCPP simulator robots."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple

from ..environment import OccupancyGridEnvironment


class Policy(ABC):
    """Abstract base class for robot policies."""

    def __init__(self) -> None:
        self._last_move_success = True

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
