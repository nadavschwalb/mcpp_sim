"""Path-following policy implementation."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Iterable, Tuple

from ..environment import OccupancyGridEnvironment
from .base import Policy


class PathPolicy(Policy):
    """Policy that follows a predetermined sequence of moves."""

    def __init__(self, path: Iterable[str]) -> None:
        super().__init__()
        self._path: Deque[str] = deque(path)

    def observe(self, environment: OccupancyGridEnvironment, robot: Any) -> Any:  # noqa: D401
        """This policy does not require observations; returns ``None``."""
        return None

    def next_move(self, position: Tuple[int, int], observation: Any) -> str:
        """Return the next move in the queued path or stay if exhausted."""
        if not self._path:
            return "stay"
        return self._path.popleft()

    def append_moves(self, moves: Iterable[str]) -> None:
        """Append additional moves to the planned path."""
        self._path.extend(moves)
